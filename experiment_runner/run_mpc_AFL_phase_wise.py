# experiment_runner/run_mpc_AFL_phase_wise.py
"""Run phase-wise A-MPC simulation with online prediction-error adaptation.

Example:

python3 experiment_runner/run_mpc_AFL_phase_wise.py --season cold --lr-N 0.00001 --num-simulation-day 31
python3 experiment_runner/run_mpc_AFL_phase_wise.py --season warm --lr-N 0.00001 --num-simulation-day 31
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import math
from pathlib import Path
import shutil
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.builders.builder_digital_twin_one_step import digital_twin_one_step
from src.builders import builder_mpc_model
from src.builders.builder_mpc_model import solve_mpc_base_and_fd_sensitivities
from src.utils.info_handlers_and_plotters import (
    compute_time_axes_from_hist,
    load_outdoor_disturbance_csv,
    load_parameters_with_x_ini,
    plot_mpc_summary_legacy,
    progress_print,
)
from src.utils.vector_order import (
    get_control_indices,
    get_control_names_in_order,
    get_disturbance_indices,
    get_disturbance_names_in_order,
    get_num_controls,
    get_num_disturbances,
    get_num_states,
    get_state_indices,
    get_state_names_in_order,
)

plt.rcParams["font.family"] = "Helvetica"


K = 12
keeper_path = (REPO_ROOT / "configs/var_and_param_keeper.yaml").resolve()

# optional: per-parameter gradient clipping (None disables)
CLIP_FRO_N: Optional[float] = None


def lr_tag(x: float) -> str:
    s = format(Decimal(str(x)), "f")
    s = s.rstrip("0").rstrip(".")
    return s.replace(".", "")


def _clip_by_norm(x: np.ndarray, max_norm: float, eps: float = 1e-12) -> tuple[np.ndarray, float, float]:
    old_norm = float(np.linalg.norm(x))
    if old_norm <= max_norm or old_norm <= eps:
        return x, old_norm, 1.0
    scale = float(max_norm / old_norm)
    return x * scale, old_norm, scale


def _phase_code_from_kappa(
    kappa_0: int,
    num_steps_per_day: int,
    kappa_day_start: int,
    kappa_day_end: int,
) -> int:
    kk = int(kappa_0) % int(num_steps_per_day)
    if int(kappa_day_start) <= kk < int(kappa_day_end):
        return 1
    return 0


def _phase_label_from_code(phase_code: int) -> str:
    return "day" if int(phase_code) == 1 else "night"


def _transition_label_from_kappa(
    kappa_0: int,
    num_steps_per_day: int,
    kappa_transition_start_1: int,
    kappa_transition_end_1: int,
    kappa_transition_start_2: int,
    kappa_transition_end_2: int,
) -> str | None:
    kk = int(kappa_0) % int(num_steps_per_day)
    if int(kappa_transition_start_1) <= kk < int(kappa_transition_end_1):
        return "transition_1"
    if int(kappa_transition_start_2) <= kk < int(kappa_transition_end_2):
        return "transition_2"
    return None


def _num_simulation_day_type(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("--num-simulation-day must be >= 1")
    return ivalue


def _lr_N_type(value: str) -> float:
    fvalue = float(value)
    if (not math.isfinite(fvalue)) or fvalue < 0.0:
        raise argparse.ArgumentTypeError("--lr-N must be a finite float >= 0")
    return fvalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-wise A-MPC simulation.")
    parser.add_argument("--season", choices=("cold", "warm"), default="cold")
    parser.add_argument("--lr-N", type=_lr_N_type, default=0.01)
    parser.add_argument("--num-simulation-day", type=_num_simulation_day_type, default=1)
    return parser.parse_args()


def _save_matrices_snapshot(
    step_dir: Path,
    A_df: pd.DataFrame,
    M_df: pd.DataFrame,
    N_df: pd.DataFrame,
    O_df: pd.DataFrame,
    m_vec_df: pd.DataFrame,
) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    A_df.to_csv(step_dir / "A.csv")
    M_df.to_csv(step_dir / "M.csv")
    N_df.to_csv(step_dir / "N.csv")
    O_df.to_csv(step_dir / "O.csv")
    m_vec_df.to_csv(step_dir / "m_vec.csv")


def hard_reset_matrices_to_baseline(
    matrices_dir: Path,
    matrices_work_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrices_work_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(matrices_dir / "A.csv", matrices_work_dir / "A.csv")
    shutil.copyfile(matrices_dir / "M.csv", matrices_work_dir / "M.csv")
    shutil.copyfile(matrices_dir / "N.csv", matrices_work_dir / "N.csv")
    shutil.copyfile(matrices_dir / "O.csv", matrices_work_dir / "O.csv")
    shutil.copyfile(matrices_dir / "m_vec.csv", matrices_work_dir / "m_vec.csv")

    M_df = pd.read_csv(matrices_work_dir / "M.csv", index_col=0)
    N_df = pd.read_csv(matrices_work_dir / "N.csv", index_col=0)
    O_df = pd.read_csv(matrices_work_dir / "O.csv", index_col=0)
    m_vec_df = pd.read_csv(matrices_work_dir / "m_vec.csv", index_col=0)

    M = M_df.to_numpy(dtype=float).copy()
    N = N_df.to_numpy(dtype=float).copy()
    O = O_df.to_numpy(dtype=float).copy()
    m_vec_col = "value" if "value" in m_vec_df.columns else m_vec_df.columns[0]
    m_vec = m_vec_df[m_vec_col].to_numpy(dtype=float).reshape(-1).copy()
    return M, N, O, m_vec


def _get_state_update_scale(param_keeper: dict, x_names: list[str]) -> np.ndarray:
    scales: list[float] = []
    for name in x_names:
        if name in {"T_in", "H_in", "C_in"}:
            lower_candidates = [
                float(param_keeper[f"{name}_lower_day"]),
                float(param_keeper[f"{name}_lower_night"]),
                float(param_keeper.get(f"{name}_lower_transition_start_1", param_keeper[f"{name}_lower_night"])),
                float(param_keeper.get(f"{name}_lower_transition_end_1", param_keeper[f"{name}_lower_day"])),
                float(param_keeper.get(f"{name}_lower_transition_start_2", param_keeper[f"{name}_lower_day"])),
                float(param_keeper.get(f"{name}_lower_transition_end_2", param_keeper[f"{name}_lower_night"])),
            ]
            upper_candidates = [
                float(param_keeper[f"{name}_upper_day"]),
                float(param_keeper[f"{name}_upper_night"]),
                float(param_keeper.get(f"{name}_upper_transition_start_1", param_keeper[f"{name}_upper_night"])),
                float(param_keeper.get(f"{name}_upper_transition_end_1", param_keeper[f"{name}_upper_day"])),
                float(param_keeper.get(f"{name}_upper_transition_start_2", param_keeper[f"{name}_upper_day"])),
                float(param_keeper.get(f"{name}_upper_transition_end_2", param_keeper[f"{name}_upper_night"])),
            ]
            span = max(upper_candidates) - min(lower_candidates)
            scales.append(max(float(span), 1.0))
        elif name == "L":
            scales.append(max(float(param_keeper.get("L_star_k_max", 1.0)), 1.0))
        else:
            scales.append(1.0)
    return np.asarray(scales, dtype=float).reshape(-1)


def _compute_prediction_error_update_for_N(
    *,
    x_before: np.ndarray,
    x_after: np.ndarray,
    u0: np.ndarray,
    d0_real: np.ndarray,
    M: np.ndarray,
    N: np.ndarray,
    O: np.ndarray,
    m_vec: np.ndarray,
    state_update_scale: np.ndarray,
) -> dict[str, np.ndarray | float]:
    dx_real = np.asarray(x_after, dtype=float).reshape(-1) - np.asarray(x_before, dtype=float).reshape(-1)
    dx_pred = (
        np.asarray(M, dtype=float) @ np.asarray(x_before, dtype=float).reshape(-1)
        + np.asarray(N, dtype=float) @ np.asarray(u0, dtype=float).reshape(-1)
        + np.asarray(O, dtype=float) @ np.asarray(d0_real, dtype=float).reshape(-1)
        + np.asarray(m_vec, dtype=float).reshape(-1)
    )
    dx_error = dx_pred - dx_real
    scale = np.asarray(state_update_scale, dtype=float).reshape(-1)
    inv_scale_sq = 1.0 / np.square(np.clip(scale, 1e-9, None))
    weighted_error = dx_error * inv_scale_sq
    grad_N = np.outer(weighted_error, np.asarray(u0, dtype=float).reshape(-1))
    x_pred = np.asarray(x_before, dtype=float).reshape(-1) + dx_pred
    return {
        "dx_real": dx_real,
        "dx_pred": dx_pred,
        "dx_error": dx_error,
        "x_pred": x_pred,
        "pred_loss": float(0.5 * np.sum((dx_error / scale) ** 2)),
        "pred_rmse": float(np.sqrt(np.mean(np.square(dx_error)))),
        "grad_N": grad_N,
    }


def main() -> None:
    args = parse_args()
    season: str = args.season
    LR_N: float = args.lr_N
    num_simulation_day: int = args.num_simulation_day

    d_pre_csv_path = (REPO_ROOT / f"data/processed/testing_outdoor_prediction_{season}.csv").resolve()
    d_rea_csv_path = (REPO_ROOT / f"data/processed/testing_outdoor_realization_{season}.csv").resolve()
    MATRICES_DIR = (REPO_ROOT / f"experiment_result/r2_fit_state_space/{season}").resolve()

    LR_N_TAG = lr_tag(LR_N)
    OUT_DIR = (REPO_ROOT / f"experiment_result/run_mpc_AFL_phase_wise_{LR_N_TAG}/{season}").resolve()
    FIG_PATH = OUT_DIR / f"mpc_summary_{season}_AFL_phasewise_lrN{LR_N_TAG}_{num_simulation_day}day.pdf"

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matrices_snap_dir = OUT_DIR / "matrices"
    matrices_work_dir = OUT_DIR / "matrices_current"
    matrices_work_M_csv_path = matrices_work_dir / "M.csv"
    matrices_snap_dir.mkdir(parents=True, exist_ok=True)

    matrices_work_dir.mkdir(parents=True, exist_ok=True)
    M, N, O, m_vec = hard_reset_matrices_to_baseline(MATRICES_DIR, matrices_work_dir)

    A_df = pd.read_csv(matrices_work_dir / "A.csv", index_col=0)
    M_df = pd.read_csv(matrices_work_dir / "M.csv", index_col=0)
    N_df = pd.read_csv(matrices_work_dir / "N.csv", index_col=0)
    O_df = pd.read_csv(matrices_work_dir / "O.csv", index_col=0)
    m_vec_df = pd.read_csv(matrices_work_dir / "m_vec.csv", index_col=0)

    param_keeper = load_parameters_with_x_ini(keeper_path)
    num_x = get_num_states(param_keeper)
    num_u = get_num_controls(param_keeper)
    num_d = get_num_disturbances(param_keeper)
    x_names = list(get_state_names_in_order(param_keeper))
    u_names = list(get_control_names_in_order(param_keeper))
    d_names = list(get_disturbance_names_in_order(param_keeper))
    x_idx = get_state_indices(param_keeper)
    u_idx = get_control_indices(param_keeper)
    d_idx = get_disturbance_indices(param_keeper)
    display_x_order = ["T_in", "H_in", "C_in", "L"]
    display_u_order = ["U_heat", "U_fan", "U_nat", "U_ac", "U_dos", "U_LED", "U_hum", "U_deh", "U_shad", "U_warm"]
    display_d_order = ["T_out", "H_out", "C_out", "R_out"]
    x_display_idx = [x_idx[name] for name in display_x_order]
    u_display_idx = [u_idx[name] for name in display_u_order]
    d_display_idx = [d_idx[name] for name in display_d_order]
    state_update_scale = _get_state_update_scale(param_keeper, x_names)
    x0 = np.array(param_keeper["x_ini"], dtype=float)
    kappa_transition_start_1 = int(param_keeper["kappa_transition_start_1"])
    kappa_transition_end_1 = int(param_keeper["kappa_transition_end_1"])
    kappa_transition_start_2 = int(param_keeper["kappa_transition_start_2"])
    kappa_transition_end_2 = int(param_keeper["kappa_transition_end_2"])
    kappa_day_start = int(param_keeper["kappa_day_start"])
    kappa_day_end = int(param_keeper["kappa_day_end"])
    num_steps_per_day = int(param_keeper["kappa_day_night_total_steps"])
    num_steps_in_whole_simulation = int(num_simulation_day) * num_steps_per_day

    t_pred, disturbance_pred = load_outdoor_disturbance_csv(d_pre_csv_path)
    t_real, disturbance_real = load_outdoor_disturbance_csv(d_rea_csv_path)
    raw_disturbance_names = ["T_out", "H_out", "C_out", "R_out"]
    d_reorder = [raw_disturbance_names.index(name) for name in d_names]
    disturbance_pred = disturbance_pred[:, d_reorder]
    disturbance_real = disturbance_real[:, d_reorder]

    x_hist = [x0.copy()]
    u_hist, d_real_hist, d_pred0_hist, t_hist = [], [], [], []
    step_hist: list[int] = []
    kappa_0_hist: list[int] = []
    obj_mpc_hist: list[float | None] = []
    cost_actual_hist: list[float] = []
    control_cost_components_hist: list[dict[str, float]] = []
    slack_cost_components_hist: list[dict[str, float]] = []
    x_before_hist: list[np.ndarray] = []
    x_after_hist: list[np.ndarray] = []
    x_pred_hist: list[np.ndarray] = []
    dx_real_hist: list[np.ndarray] = []
    dx_pred_hist: list[np.ndarray] = []
    dx_error_hist: list[np.ndarray] = []
    pred_loss_hist: list[float] = []
    pred_rmse_hist: list[float] = []
    phase_hist: list[str] = []
    dt_s_step = int(param_keeper["delta_t"])
    prev_phase: int | None = None

    for step in range(num_steps_in_whole_simulation):
        if step >= disturbance_pred.shape[0] or step >= disturbance_real.shape[0]:
            break

        d_forecast = disturbance_pred[step: step + K]
        if d_forecast.shape[0] < K:
            break

        kappa_0 = step % num_steps_per_day
        phase_now = _phase_code_from_kappa(
            kappa_0,
            num_steps_per_day=num_steps_per_day,
            kappa_day_start=kappa_day_start,
            kappa_day_end=kappa_day_end,
        )
        transition_label = _transition_label_from_kappa(
            kappa_0,
            num_steps_per_day=num_steps_per_day,
            kappa_transition_start_1=kappa_transition_start_1,
            kappa_transition_end_1=kappa_transition_end_1,
            kappa_transition_start_2=kappa_transition_start_2,
            kappa_transition_end_2=kappa_transition_end_2,
        )
        in_transition = transition_label is not None
        update_model_this_step = not in_transition
        phase_label = _phase_label_from_code(phase_now)
        is_phase_change = (step > 0) and (prev_phase is not None) and (phase_now != prev_phase)
        d0_real = np.asarray(disturbance_real[step], dtype=float)

        reset_reasons: list[str] = []
        if in_transition:
            reset_reasons.append(f"{transition_label}:baseline")
        elif is_phase_change:
            reset_reasons.append(f"phase_change:{prev_phase}->{phase_now}")
        if reset_reasons:
            M, N, O, m_vec = hard_reset_matrices_to_baseline(
                matrices_dir=MATRICES_DIR,
                matrices_work_dir=matrices_work_dir,
            )
            A_df = pd.read_csv(matrices_work_dir / "A.csv", index_col=0)
            M_df = pd.read_csv(matrices_work_dir / "M.csv", index_col=0)
            N_df = pd.read_csv(matrices_work_dir / "N.csv", index_col=0)
            O_df = pd.read_csv(matrices_work_dir / "O.csv", index_col=0)
            m_vec_df = pd.read_csv(matrices_work_dir / "m_vec.csv", index_col=0)
            print(f"[reset] step={step} kappa_0={kappa_0} reason={'|'.join(reset_reasons)}")

        if hasattr(builder_mpc_model, "_load_dynamics_matrices"):
            builder_mpc_model._load_dynamics_matrices.cache_clear()

        mpc_result = solve_mpc_base_and_fd_sensitivities(
            x_ini=x0,
            d_forecast=d_forecast,
            kappa_ini=kappa_0,
            keeper_path=keeper_path,
            matrices_path=matrices_work_M_csv_path,
            horizon_K=K,
            num_x=num_x,
            num_u=num_u,
            num_d=num_d,
            rel_step=1e-5,
            abs_step_floor=1e-7,
            which_u="u0",
            cost_surrogate_npz_path=None,
            fd_M=False,
            fd_N=False,
            fd_O=False,
            fd_m_vec=False,
        )

        u0 = builder_mpc_model.round_control_vector_for_online_dfl(
            np.asarray(mpc_result["u0"], dtype=float)
        )
        mpc_result["u0"] = u0.copy()
        x_before = x0.copy()

        x1_real = digital_twin_one_step(x=x0, u=u0, d=d0_real, param_keeper=param_keeper)
        x_after_raw = np.asarray(x1_real[0] if isinstance(x1_real, tuple) else x1_real, dtype=float).reshape(-1)

        one_step_cost_act_info = builder_mpc_model.get_cost_at_end_state(
            x_act=x_after_raw,
            u0=u0,
            kappa_act=int(kappa_0),
            keeper_path=keeper_path,
        )
        one_step_total_cost_act = float(one_step_cost_act_info["one_step_total_cost_act"])
        one_step_control_cost_act_components = {
            str(k): float(v)
            for k, v in dict(one_step_cost_act_info["one_step_control_cost_act_components"]).items()
        }
        one_step_slack_cost_act_components = {
            str(k): float(v)
            for k, v in dict(one_step_cost_act_info["one_step_slack_cost_act_components"]).items()
        }
        obj_mpc = mpc_result.get("obj_mpc", None)

        pe_update = _compute_prediction_error_update_for_N(
            x_before=x_before,
            x_after=x_after_raw,
            u0=u0,
            d0_real=d0_real,
            M=M,
            N=N,
            O=O,
            m_vec=m_vec,
            state_update_scale=state_update_scale,
        )

        x0_next = x_after_raw.copy()
        next_time = t_real[step] + pd.Timedelta(seconds=dt_s_step)
        if next_time.hour == 0 and next_time.minute == 0:
            x0_next[x_idx["L"]] = 0.0
        x0 = x0_next

        progress_print(
            step=step,
            total=num_steps_in_whole_simulation - 1,
            ts=t_real[step],
            u0=u0[u_display_idx],
            x_before=x_before[x_display_idx],
            x_after=x0[x_display_idx],
            obj_mpc=obj_mpc,
            cost_actual_step=one_step_total_cost_act,
            param_keeper=param_keeper,
            kappa_0=kappa_0,
            horizon_K=K,
            d0_pred=d_forecast[0][d_display_idx],
            d0_real=d0_real[d_display_idx],
        )
        print(
            f"[prediction-error] phase={phase_label} "
            f"loss={float(pe_update['pred_loss']):.4e} rmse={float(pe_update['pred_rmse']):.4e}"
        )
        if in_transition:
            print(
                f"[transition] step={step} kappa_0={kappa_0} window={transition_label} "
                "using baseline matrices/vectors; skipped parameter updates"
            )

        grad_N = np.asarray(pe_update["grad_N"], dtype=float)
        if LR_N != 0 and CLIP_FRO_N is not None:
            grad_N, normN, scaleN = _clip_by_norm(grad_N, CLIP_FRO_N)
            if scaleN < 1.0:
                print(f"[CLIP] clip grad_N: {normN:.2e}->{CLIP_FRO_N:.2e} (x{scaleN:.2e})")

        if update_model_this_step:
            N = N - LR_N * grad_N

        N_df.iloc[:, :] = N
        A_df.iloc[:, :] = np.eye(M.shape[0], dtype=float) + M
        N_df.to_csv(matrices_work_dir / "N.csv")
        A_df.to_csv(matrices_work_dir / "A.csv")

        step_mats_dir = matrices_snap_dir / f"matrices_step_{step:06d}"
        _save_matrices_snapshot(step_mats_dir, A_df=A_df, M_df=M_df, N_df=N_df, O_df=O_df, m_vec_df=m_vec_df)
        print(
            f"[AFL_Update] ||grad_N||_F = {np.linalg.norm(grad_N):.2e}  "
            f"LR_N = {LR_N:.2e}  updated={update_model_this_step and LR_N != 0}"
        )
        print(f"[MATS] saved snapshot under: {step_mats_dir}")

        u_hist.append(u0.copy())
        d_real_hist.append(d0_real.copy())
        d_pred0_hist.append(d_forecast[0].copy())
        t_hist.append(t_real[step])
        x_hist.append(x0.copy())
        step_hist.append(int(step))
        kappa_0_hist.append(int(kappa_0))
        obj_mpc_hist.append(None if obj_mpc is None else float(obj_mpc))
        cost_actual_hist.append(float(one_step_total_cost_act))
        control_cost_components_hist.append(one_step_control_cost_act_components)
        slack_cost_components_hist.append(one_step_slack_cost_act_components)
        x_before_hist.append(np.asarray(x_before, dtype=float).reshape(-1))
        x_after_hist.append(np.asarray(x0, dtype=float).reshape(-1))
        x_pred_hist.append(np.asarray(pe_update["x_pred"], dtype=float).reshape(-1))
        dx_real_hist.append(np.asarray(pe_update["dx_real"], dtype=float).reshape(-1))
        dx_pred_hist.append(np.asarray(pe_update["dx_pred"], dtype=float).reshape(-1))
        dx_error_hist.append(np.asarray(pe_update["dx_error"], dtype=float).reshape(-1))
        pred_loss_hist.append(float(pe_update["pred_loss"]))
        pred_rmse_hist.append(float(pe_update["pred_rmse"]))
        phase_hist.append(str(phase_label))
        prev_phase = phase_now

    n_roll = len(step_hist)
    t_str = np.asarray([pd.Timestamp(ts).isoformat() for ts in t_hist], dtype=str)
    step_arr = np.asarray(step_hist, dtype=int)
    kappa_arr = np.asarray(kappa_0_hist, dtype=int)
    obj_arr = np.asarray([np.nan if v is None else float(v) for v in obj_mpc_hist], dtype=float)
    total_cost_arr = np.asarray(cost_actual_hist, dtype=float)
    pred_loss_arr = np.asarray(pred_loss_hist, dtype=float)
    pred_rmse_arr = np.asarray(pred_rmse_hist, dtype=float)
    X_before_arr = np.asarray(x_before_hist, dtype=float).reshape(n_roll, num_x)
    X_after_arr = np.asarray(x_after_hist, dtype=float).reshape(n_roll, num_x)
    X_pred_arr = np.asarray(x_pred_hist, dtype=float).reshape(n_roll, num_x)
    DX_real_arr = np.asarray(dx_real_hist, dtype=float).reshape(n_roll, num_x)
    DX_pred_arr = np.asarray(dx_pred_hist, dtype=float).reshape(n_roll, num_x)
    DX_error_arr = np.asarray(dx_error_hist, dtype=float).reshape(n_roll, num_x)
    U0_arr = np.asarray(u_hist, dtype=float).reshape(n_roll, num_u)
    D_real_arr = np.asarray(d_real_hist, dtype=float).reshape(n_roll, num_d)
    D_pred0_arr = np.asarray(d_pred0_hist, dtype=float).reshape(n_roll, num_d)

    rollout_df = pd.DataFrame(
        {
            "step": step_arr,
            "timestamp": t_str,
            "kappa_0": kappa_arr,
            "phase_label": phase_hist,
            "obj": obj_arr,
            "one_step_actual_cost_twin": total_cost_arr,
            "one_step_prediction_error_loss": pred_loss_arr,
            "one_step_prediction_error_rmse": pred_rmse_arr,
        }
    )
    control_component_cols = u_names
    for col in control_component_cols:
        rollout_df[col] = [float(v.get(col, np.nan)) for v in control_cost_components_hist]
    slack_component_cols = [
        "var_S_T_pos",
        "var_S_T_neg",
        "var_S_H_pos",
        "var_S_H_neg",
        "var_S_C_pos",
        "var_S_C_neg",
        "var_S_L",
    ]
    for col in slack_component_cols:
        rollout_df[col] = [float(v.get(col, np.nan)) for v in slack_cost_components_hist]
    for i, name in enumerate(x_names):
        rollout_df[f"x_before_{name}"] = X_before_arr[:, i]
    for i, name in enumerate(x_names):
        rollout_df[f"x_after_{name}"] = X_after_arr[:, i]
    for i, name in enumerate(x_names):
        rollout_df[f"x_pred_{name}"] = X_pred_arr[:, i]
    for i, name in enumerate(x_names):
        rollout_df[f"dx_real_{name}"] = DX_real_arr[:, i]
    for i, name in enumerate(x_names):
        rollout_df[f"dx_pred_{name}"] = DX_pred_arr[:, i]
    for i, name in enumerate(x_names):
        rollout_df[f"dx_error_{name}"] = DX_error_arr[:, i]
    for i, name in enumerate(u_names):
        rollout_df[f"u0_{name}"] = U0_arr[:, i]
    for i, name in enumerate(d_names):
        rollout_df[f"d0_real_{name}"] = D_real_arr[:, i]
    for i, name in enumerate(d_names):
        rollout_df[f"d0_pred0_{name}"] = D_pred0_arr[:, i]

    rollout_csv_path = OUT_DIR / f"rollout_step_log_season={season}__AFL_phasewise_lrN={LR_N_TAG}_days={num_simulation_day}.csv"
    rollout_df.to_csv(rollout_csv_path, index=False)
    print(f"Saved rollout CSV: {rollout_csv_path}")

    if n_roll > 0:
        fig_cmp, ax_cmp = plt.subplots(figsize=(10.0, 4.2))
        ax_cmp.plot(
            step_arr,
            rollout_df["one_step_prediction_error_rmse"].to_numpy(dtype=float),
            color="#BF124D",
            lw=1.2,
            label="One-step prediction RMSE",
        )
        ax_cmp.set_xlabel("Step")
        ax_cmp.set_ylabel("RMSE")
        ax_cmp.set_title("One-step state-prediction RMSE")
        ax_cmp.grid(True, ls="--", lw=0.4, alpha=0.6)
        for spine in ("top", "right"):
            ax_cmp.spines[spine].set_visible(False)
        ax_cmp.legend(loc="upper right", frameon=False)
        fig_cmp.tight_layout()
        rmse_plot_path = OUT_DIR / f"one_step_prediction_rmse_season={season}__AFL_phasewise_lrN={LR_N_TAG}_days={num_simulation_day}.pdf"
        fig_cmp.savefig(rmse_plot_path, bbox_inches="tight", format="pdf")
        plt.close(fig_cmp)
        print(f"Saved plot: {rmse_plot_path}")

    _ = t_pred
    if len(u_hist) > 0:
        t, t_x, dt_s = compute_time_axes_from_hist(t_hist, dt_default_s=int(param_keeper["delta_t"]))
        plot_mpc_summary_legacy(
            FIG_PATH,
            x_hist=[np.asarray(x, dtype=float).reshape(-1)[x_display_idx] for x in x_hist],
            u_hist=[np.asarray(u, dtype=float).reshape(-1)[u_display_idx] for u in u_hist],
            d_pred0_hist=[np.asarray(d0, dtype=float).reshape(-1)[d_display_idx] for d0 in d_pred0_hist],
            d_real_hist=[np.asarray(d0, dtype=float).reshape(-1)[d_display_idx] for d0 in d_real_hist],
            param_keeper=param_keeper,
            t=t,
            t_x=t_x,
            dt_s=dt_s,
            show=False,
        )
        print(f"Saved plot: {FIG_PATH}")


if __name__ == "__main__":
    main()
