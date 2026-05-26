# experiment_runner/run_mpc_DFL_phase_wise.py
"""Run day/night phase-wise MPC-DFL simulation using phase-specific surrogate models.

Example:


python3 experiment_runner/run_mpc_DFL_phase_wise.py --season cold --surrogate-model DNN --lr-N 0.0000 --num-simulation-day 3
python3 experiment_runner/run_mpc_DFL_phase_wise.py --season warm --surrogate-model DNN --lr-N 0.0000 --num-simulation-day 3


python3 experiment_runner/run_mpc_DFL_phase_wise.py --season cold --surrogate-model DNN --lr-N 0.00001 --num-simulation-day 31
python3 experiment_runner/run_mpc_DFL_phase_wise.py --season warm --surrogate-model DNN --lr-N 0.00001 --num-simulation-day 31

python3 experiment_runner/run_mpc_DFL_phase_wise.py --season cold --surrogate-model DNN --lr-N 0.1 --num-simulation-day 2 &
python3 experiment_runner/run_mpc_DFL_phase_wise.py --season warm --surrogate-model DNN --lr-N 0.1 --num-simulation-day 2 &

wait

Supported season:
- cold
- warm

Supported surrogate-model:
- DNN
- REG

Supported lr-N:
- 0.001
- ...
- 0.00001

Supported num-simulation-day:
- 1
- ...
- 14

"""
import argparse
import json
from pathlib import Path
import sys
import shutil
from decimal import Decimal
import math
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


LR_M = 0
LR_O = 0
LR_m_vec = 0

K = 12

keeper_path = (REPO_ROOT / "configs/var_and_param_keeper.yaml").resolve()

_PHASE_SURROGATE_DIR_MAP = {
    "DNN": ("r4_fit_J_act_DNN_phase_wise",),
    "REG": ("r4_fit_J_act_REG_phase_wise", "r4_fit_J_act_reg_phase_wise"),
}

_PHASE_MODEL_ORDER = ("night", "day")

# optional: per-parameter gradient clipping (None disables)
CLIP_FRO_M: Optional[float] = None
CLIP_FRO_N: Optional[float] = None
CLIP_FRO_O: Optional[float] = None
CLIP_L2_m_vec: Optional[float] = None
# Examples (enable only when needed; defaults stay disabled):
# CLIP_FRO_M = 1e3
# CLIP_FRO_N = 1e3
# CLIP_FRO_O = 1e3
# CLIP_L2_m_vec = 1e2

def lr_tag(x: float) -> str:
    # Use Decimal(str(x)) to avoid "1e-05" formatting surprises
    s = format(Decimal(str(x)), "f")     # e.g., "0.01", "0.001", "1.0"
    s = s.rstrip("0").rstrip(".")        # e.g., "0.01", "0.001", "1"
    return s.replace(".", "")            # e.g., "001", "0001", "1"


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


def _load_and_validate_phase_surrogates(phase_root_dir: Path) -> dict[str, dict]:
    phase_surrogates: dict[str, dict] = {}
    ref_feature_names: list[str] | None = None
    ref_cost_target_kind: str | None = None

    for phase_name in _PHASE_MODEL_ORDER:
        npz_path = (phase_root_dir / phase_name / "model_and_data.npz").resolve()
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Missing phase-wise surrogate artifact for phase={phase_name!r}: {npz_path}"
            )
        surrogate_meta = builder_mpc_model._load_cost_surrogate_npz(str(npz_path))
        feature_names = [str(name) for name in surrogate_meta["feature_names"]]
        cost_target_kind = str(surrogate_meta.get("cost_target_kind", ""))

        if ref_feature_names is None:
            ref_feature_names = feature_names
        elif feature_names != ref_feature_names:
            raise ValueError(
                "Phase-wise surrogate feature-name mismatch:\n"
                f"  reference={ref_feature_names}\n"
                f"  phase={phase_name}: {feature_names}"
            )

        if ref_cost_target_kind is None:
            ref_cost_target_kind = cost_target_kind
        elif cost_target_kind != ref_cost_target_kind:
            raise ValueError(
                "Phase-wise surrogate target-kind mismatch:\n"
                f"  reference={ref_cost_target_kind!r}\n"
                f"  phase={phase_name}: {cost_target_kind!r}"
            )

        phase_surrogates[phase_name] = {
            "npz_path": npz_path,
            "meta": surrogate_meta,
            "feature_names": feature_names,
        }

    return phase_surrogates


def _resolve_default_phase_surrogate_root(
    surrogate_model: str,
    season: str,
) -> Path:
    candidate_dirs = _PHASE_SURROGATE_DIR_MAP.get(
        surrogate_model,
        (f"r4_fit_J_act_{surrogate_model}_phase_wise",),
    )
    candidate_paths = [
        (REPO_ROOT / "experiment_result" / candidate_dir / season).resolve()
        for candidate_dir in candidate_dirs
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    resolved_from_find_best = _resolve_phase_surrogate_root_from_find_best(
        surrogate_model=surrogate_model,
        season=season,
    )
    if resolved_from_find_best is not None:
        return resolved_from_find_best
    return candidate_paths[0]


def _resolve_phase_surrogate_root_from_find_best(
    *,
    surrogate_model: str,
    season: str,
) -> Path | None:
    if surrogate_model != "DNN":
        return None

    phase_npz_paths: dict[str, Path] = {}
    for phase_name in _PHASE_MODEL_ORDER:
        find_best_root = (REPO_ROOT / "experiment_result" / f"find_best_{season}_{phase_name}_DNN").resolve()
        if not find_best_root.exists():
            return None

        run_dirs = sorted(
            [candidate for candidate in find_best_root.iterdir() if candidate.is_dir() and candidate.name.startswith("run_")],
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        if not run_dirs:
            return None

        latest_run_dir = run_dirs[-1]
        best_hparams_path = (latest_run_dir / "best_hparams.json").resolve()
        if not best_hparams_path.exists():
            return None

        with best_hparams_path.open("r", encoding="utf-8") as fh:
            best_payload = json.load(fh)
        best_trial_dir_raw = best_payload.get("best_trial_dir", None)
        if not best_trial_dir_raw:
            return None

        best_trial_dir = Path(str(best_trial_dir_raw)).expanduser().resolve()
        npz_path = (best_trial_dir / "model_and_data.npz").resolve()
        if not npz_path.exists():
            return None
        phase_npz_paths[phase_name] = npz_path

    resolved_root = (REPO_ROOT / "experiment_result" / "_resolved_phase_surrogates" / surrogate_model / season).resolve()
    for phase_name, npz_path in phase_npz_paths.items():
        phase_dir = (resolved_root / phase_name).resolve()
        phase_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(npz_path, phase_dir / "model_and_data.npz")

    print(
        f"[surrogate] auto-resolved consolidated phase artifacts from find_best_* under: {resolved_root}",
        flush=True,
    )
    return resolved_root


def _cost_target_label(cost_target_kind: str) -> str:
    if cost_target_kind == builder_mpc_model.DFL_COST_TARGET_KIND:
        return "one-step actual cost"
    if cost_target_kind == builder_mpc_model.CONTROL_COST_TARGET_KIND:
        return "one-step control cost"
    return f"one-step surrogate target ({cost_target_kind})"


def _cost_target_reference_column(cost_target_kind: str) -> str:
    if cost_target_kind == builder_mpc_model.CONTROL_COST_TARGET_KIND:
        return "control_cost_actual"
    return "cost_actual"


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
    parser = argparse.ArgumentParser(description="Run phase-wise MPC-DFL simulation.")
    parser.add_argument("--season", choices=("cold", "warm"), default="cold")
    parser.add_argument("--surrogate-model", choices=("DNN", "REG"), default="DNN")
    parser.add_argument("--lr-N", type=_lr_N_type, default=0.01)
    parser.add_argument("--num-simulation-day", type=_num_simulation_day_type, default=1)
    parser.add_argument(
        "--keeper-path",
        type=Path,
        default=keeper_path,
        help="Path to var_and_param_keeper.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for figures, rollout logs, and gradients.",
    )
    return parser.parse_args()


def _resolve_default_paths_for_season(season: str) -> tuple[Path, Path, Path]:
    d_pre_csv_path = (REPO_ROOT / f"data/processed/testing_outdoor_prediction_{season}.csv").resolve()
    d_rea_csv_path = (REPO_ROOT / f"data/processed/testing_outdoor_realization_{season}.csv").resolve()
    matrices_dir = (REPO_ROOT / f"experiment_result/r2_fit_state_space/{season}").resolve()
    return d_pre_csv_path, d_rea_csv_path, matrices_dir



def _save_fd_grads_as_csv(step_dir: Path,
                          mpc_result_and_gradients: dict,
                          step: int,
                          kappa_0: int,
                          x_ini: np.ndarray,
                          *,
                          u_names: list[str],
                          x_names: list[str],
                          d_names: list[str]) -> None:
    """
    Save full FD gradient tensors (u0-only) to multiple CSVs under step_dir.

    Files:
      - u0.csv           : (num_u,)
      - dJ_du0.csv       : (num_u,)
      - du0_dm_vec.csv   : (num_u, num_x)
      - du0_dM.csv       : (num_u, num_x*num_x)
      - du0_dN.csv       : (num_u, num_x*num_u)
      - du0_dO.csv       : (num_u, num_x*num_d)
      - meta.csv         : one row with metadata
    """
    step_dir.mkdir(parents=True, exist_ok=True)

    u0 = np.asarray(mpc_result_and_gradients["u0"], dtype=float)        # (num_u,)
    dJ_du0 = np.asarray(mpc_result_and_gradients["d_cost_du0"], dtype=float).reshape(-1)  # (num_u,)
    du0_dM = np.asarray(mpc_result_and_gradients["du0_dM"], dtype=float)          # (num_u, num_x, num_x)
    du0_dN = np.asarray(mpc_result_and_gradients["du0_dN"], dtype=float)          # (num_u, num_x, num_u)
    du0_dO = np.asarray(mpc_result_and_gradients["du0_dO"], dtype=float)          # (num_u, num_x, num_d)
    du0_dm_vec = np.asarray(mpc_result_and_gradients["du0_dm_vec"], dtype=float)  # (num_u, num_x)

    num_u_local = u0.shape[0]
    num_x_local = du0_dm_vec.shape[1]
    num_d_local = du0_dO.shape[2]

    # u0.csv
    if len(u_names) != num_u_local or len(x_names) != num_x_local or len(d_names) != num_d_local:
        raise ValueError(
            "FD CSV naming mismatch: "
            f"u_names={len(u_names)} vs num_u={num_u_local}, "
            f"x_names={len(x_names)} vs num_x={num_x_local}, "
            f"d_names={len(d_names)} vs num_d={num_d_local}"
        )

    dx_row_labels = [f"d{name}" for name in x_names]
    m_vec_labels = [f"m_{name}" for name in x_names]

    pd.DataFrame([u0], columns=u_names).to_csv(step_dir / "u0.csv", index=False)
    # dJ_du0.csv
    pd.DataFrame([dJ_du0], columns=[f"dJ_du0_{name}" for name in u_names]).to_csv(step_dir / "dJ_du0.csv", index=False)

    # du0_dm_vec.csv: (num_u, num_x)
    pd.DataFrame(du0_dm_vec, columns=m_vec_labels, index=u_names).to_csv(step_dir / "du0_dm_vec.csv")

    # du0_dM.csv: (num_u, num_x*num_x)
    du0_dM_2d = du0_dM.reshape(num_u_local, num_x_local * num_x_local)
    cols_M = [f"{row}__{col}" for row in dx_row_labels for col in x_names]
    pd.DataFrame(du0_dM_2d, columns=cols_M, index=u_names).to_csv(step_dir / "du0_dM.csv")

    # du0_dN.csv: (num_u, num_x*num_u)
    du0_dN_2d = du0_dN.reshape(num_u_local, num_x_local * num_u_local)
    cols_N = [f"{row}__{col}" for row in dx_row_labels for col in u_names]
    pd.DataFrame(du0_dN_2d, columns=cols_N, index=u_names).to_csv(step_dir / "du0_dN.csv")

    # du0_dO.csv: (num_u, num_x*num_d)
    du0_dO_2d = du0_dO.reshape(num_u_local, num_x_local * num_d_local)
    cols_O = [f"{row}__{col}" for row in dx_row_labels for col in d_names]
    pd.DataFrame(du0_dO_2d, columns=cols_O, index=u_names).to_csv(step_dir / "du0_dO.csv")

    # meta.csv
    meta = {
        "step": int(step),
        "kappa_0": int(kappa_0),
        "obj_mpc": float(mpc_result_and_gradients["obj_mpc"]),
        "solving_condition": str(mpc_result_and_gradients["solving_condition"]),
        "rel_step": float(mpc_result_and_gradients["rel_step"]),
        "abs_step_floor": float(mpc_result_and_gradients["abs_step_floor"]),
        "which_u": str(mpc_result_and_gradients.get("which_u", "u0")),
        "u_names": ";".join(u_names),
        "x_names": ";".join(x_names),
        "d_names": ";".join(d_names),
    }
    x_ini_vec = np.asarray(x_ini, float).reshape(-1)
    for name, value in zip(x_names, x_ini_vec, strict=True):
        meta[f"x_ini_{name}"] = float(value)
    pd.DataFrame([meta]).to_csv(step_dir / "meta.csv", index=False)




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



def run_simulation(
    *,
    season: str,
    surrogate_model: str,
    LR_N: float,
    num_simulation_day: int,
    keeper_path_override: Path | None = None,
    prediction_csv_path: Path | None = None,
    realization_csv_path: Path | None = None,
    matrices_dir: Path | None = None,
    phase_surrogate_root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    default_prediction_csv_path, default_realization_csv_path, default_matrices_dir = _resolve_default_paths_for_season(season)
    d_pre_csv_path = default_prediction_csv_path if prediction_csv_path is None else prediction_csv_path.resolve()
    d_rea_csv_path = default_realization_csv_path if realization_csv_path is None else realization_csv_path.resolve()
    MATRICES_DIR = default_matrices_dir if matrices_dir is None else matrices_dir.resolve()
    keeper_path_resolved = keeper_path if keeper_path_override is None else keeper_path_override.resolve()
    if phase_surrogate_root is None:
        phase_surrogate_root = _resolve_default_phase_surrogate_root(
            surrogate_model=surrogate_model,
            season=season,
        )
    else:
        phase_surrogate_root = phase_surrogate_root.resolve()
    phase_surrogates = _load_and_validate_phase_surrogates(phase_surrogate_root)
    surrogate_cost_target_kind = str(phase_surrogates[_PHASE_MODEL_ORDER[0]]["meta"]["cost_target_kind"])
    surrogate_cost_label = _cost_target_label(surrogate_cost_target_kind)
    surrogate_cost_reference_column = _cost_target_reference_column(surrogate_cost_target_kind)
    for phase_name in _PHASE_MODEL_ORDER:
        surrogate_meta = phase_surrogates[phase_name]["meta"]
        surrogate_feature_names = phase_surrogates[phase_name]["feature_names"]
        surrogate_output_activation = str(phase_surrogates[phase_name].get("torch_output_activation", "linear"))
        surrogate_uses_refs = any(
            name in surrogate_feature_names
            for name in ("T_in_star", "H_in_star", "C_in_star", "L_star")
        )
        surrogate_uses_kappa = "kappa_k" in surrogate_feature_names
        surrogate_has_scaler = surrogate_meta.get("feature_scaler", None) is not None
        print(
            f"[surrogate:{phase_name}] n_features={len(surrogate_feature_names)} "
            f"refs={surrogate_uses_refs} kappa={surrogate_uses_kappa} "
            f"scaler={surrogate_has_scaler} output_activation={surrogate_output_activation} "
            f"target={surrogate_meta.get('cost_target_kind', '')}"
        )
    print(
        f"[surrogate] model={surrogate_model} root={phase_surrogate_root} "
        f"target_kind={surrogate_cost_target_kind} compare_against={surrogate_cost_reference_column}",
        flush=True,
    )

    LR_N_TAG = lr_tag(LR_N)
    if output_dir is None:
        OUT_DIR = (REPO_ROOT / f"experiment_result/run_mpc_DFL_phase_wise_{surrogate_model}_{LR_N_TAG}/{season}").resolve()
    else:
        OUT_DIR = output_dir.resolve()
    FIG_PATH = OUT_DIR / f"mpc_summary_{season}_{surrogate_model}_phasewise_lrN{LR_N_TAG}_{num_simulation_day}day.pdf"

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grad_dir = OUT_DIR / "gradients_and_u0"  # define here, not globally
    matrices_snap_dir = OUT_DIR / "matrices"
    matrices_work_dir = OUT_DIR / "matrices_current"
    matrices_work_M_csv_path = matrices_work_dir / "M.csv"

    # Clear Gradients folder every run
    if grad_dir.exists():
        shutil.rmtree(grad_dir)
    grad_dir.mkdir(parents=True, exist_ok=True)
    matrices_snap_dir.mkdir(parents=True, exist_ok=True)

    matrices_work_dir.mkdir(parents=True, exist_ok=True)
    M, N, O, m_vec = hard_reset_matrices_to_baseline(MATRICES_DIR, matrices_work_dir)

    A_df = pd.read_csv(matrices_work_dir / "A.csv", index_col=0)
    M_df = pd.read_csv(matrices_work_dir / "M.csv", index_col=0)
    N_df = pd.read_csv(matrices_work_dir / "N.csv", index_col=0)
    O_df = pd.read_csv(matrices_work_dir / "O.csv", index_col=0)
    m_vec_df = pd.read_csv(matrices_work_dir / "m_vec.csv", index_col=0)

    param_keeper = load_parameters_with_x_ini(keeper_path_resolved)
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
    cost_hat_hist: list[float] = []
    control_cost_components_hist: list[dict[str, float]] = []
    slack_cost_components_hist: list[dict[str, float]] = []
    x_before_hist: list[np.ndarray] = []
    x_after_hist: list[np.ndarray] = []
    surrogate_phase_hist: list[str] = []
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
        surrogate_phase = _phase_label_from_code(phase_now)
        surrogate_npz_path = phase_surrogates[surrogate_phase]["npz_path"]
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

        # ---- 1) Solve MPC and do FD sensitivities ----
        mpc_result_and_gradients = solve_mpc_base_and_fd_sensitivities(
            x_ini=x0,
            d_forecast=d_forecast,
            kappa_ini=kappa_0,
            keeper_path=keeper_path_resolved,
            matrices_path=matrices_work_M_csv_path,
            horizon_K=K,
            num_x=num_x,
            num_u=num_u,
            num_d=num_d,
            rel_step=1e-5,
            abs_step_floor=1e-7,
            which_u="u0",
            cost_surrogate_npz_path=surrogate_npz_path,
            cost_u_names=u_names,
            cost_x0_names=[f"X_ini_{name}" for name in x_names],
            cost_d_names=[f"D_{name}" for name in d_names],
            d0_for_cost=d0_real,
            fd_M=(LR_M != 0) and update_model_this_step,
            fd_N=(LR_N != 0) and update_model_this_step,
            fd_O=(LR_O != 0) and update_model_this_step,
            fd_m_vec=(LR_m_vec != 0) and update_model_this_step,
        )

        # ---- 2) Apply first control and simulate one step ----
        u0 = builder_mpc_model.round_control_vector_for_online_dfl(
            np.asarray(mpc_result_and_gradients["u0"], dtype=float)
        )
        mpc_result_and_gradients["u0"] = u0.copy()
        x_before = x0.copy()

        x1_real = digital_twin_one_step(x=x0, u=u0, d=d0_real, param_keeper=param_keeper)
        x_after_raw = np.asarray(x1_real[0] if isinstance(x1_real, tuple) else x1_real, dtype=float).reshape(-1)

        one_step_cost_act_info = builder_mpc_model.get_cost_at_end_state(
            x_act=x_after_raw,
            u0=u0,
            kappa_act=int(kappa_0),
            keeper_path=keeper_path_resolved,
        )
        one_step_total_cost_act = float(one_step_cost_act_info["one_step_total_cost_act"])
        one_step_control_cost_act = float(one_step_cost_act_info["one_step_control_cost_act"])
        one_step_control_cost_act_components = {str(k): float(v) for k, v in dict(one_step_cost_act_info["one_step_control_cost_act_components"]).items()}
        one_step_slack_cost_act_components = {str(k): float(v) for k, v in dict(one_step_cost_act_info["one_step_slack_cost_act_components"]).items()}
        obj_mpc = mpc_result_and_gradients.get("obj_mpc", None)

        x0_next = x_after_raw.copy()
        next_time = t_real[step] + pd.Timedelta(seconds=dt_s_step)
        if next_time.hour == 0 and next_time.minute == 0:
            x0_next[x_idx["L"]] = 0.0
        x0 = x0_next

        progress_print(step=step, total=num_steps_in_whole_simulation - 1, ts=t_real[step], u0=u0[u_display_idx], x_before=x_before[x_display_idx],
                       x_after=x0[x_display_idx], obj_mpc=obj_mpc, cost_actual_step=one_step_total_cost_act,
                       param_keeper=param_keeper, kappa_0=kappa_0, horizon_K=K,
                       d0_pred=d_forecast[0][d_display_idx], d0_real=d0_real[d_display_idx])
        one_step_cost_pre = mpc_result_and_gradients.get("one_step_cost_pre", None)
        print(f"Predicted {surrogate_cost_label}: {one_step_cost_pre} (phase={surrogate_phase})")
        if in_transition:
            print(
                f"[transition] step={step} kappa_0={kappa_0} window={transition_label} "
                "using baseline matrices/vectors; skipped parameter updates"
            )

        # ---- 3) Chain-rule gradients wrt model matrices/vectors ----
        dJ_du0 = np.asarray(mpc_result_and_gradients["d_cost_du0"], float)  # (num_u,)
        du0_dM = np.asarray(mpc_result_and_gradients["du0_dM"], float)
        du0_dN = np.asarray(mpc_result_and_gradients["du0_dN"], float)
        du0_dO = np.asarray(mpc_result_and_gradients["du0_dO"], float)
        du0_dm_vec = np.asarray(mpc_result_and_gradients["du0_dm_vec"], float)
        dJ_dM = np.tensordot(dJ_du0, du0_dM, axes=(0, 0))      # (num_x, num_x)
        dJ_dN = np.tensordot(dJ_du0, du0_dN, axes=(0, 0))      # (num_x, num_u)
        dJ_dO = np.tensordot(dJ_du0, du0_dO, axes=(0, 0))      # (num_x, num_d)
        dJ_dm_vec = np.tensordot(dJ_du0, du0_dm_vec, axes=(0, 0))  # (num_x,)

        clip_logs = []
        if LR_M != 0 and CLIP_FRO_M is not None:
            dJ_dM, normM, scaleM = _clip_by_norm(dJ_dM, CLIP_FRO_M)
            if scaleM < 1.0:
                clip_logs.append(f"clip dJ_dM: {normM:.2e}->{CLIP_FRO_M:.2e} (x{scaleM:.2e})")
        if LR_N != 0 and CLIP_FRO_N is not None:
            dJ_dN, normN, scaleN = _clip_by_norm(dJ_dN, CLIP_FRO_N)
            if scaleN < 1.0:
                clip_logs.append(f"clip dJ_dN: {normN:.2e}->{CLIP_FRO_N:.2e} (x{scaleN:.2e})")
        if LR_O != 0 and CLIP_FRO_O is not None:
            dJ_dO, normO, scaleO = _clip_by_norm(dJ_dO, CLIP_FRO_O)
            if scaleO < 1.0:
                clip_logs.append(f"clip dJ_dO: {normO:.2e}->{CLIP_FRO_O:.2e} (x{scaleO:.2e})")
        if LR_m_vec != 0 and CLIP_L2_m_vec is not None:
            dJ_dm_vec, normm, scalem = _clip_by_norm(dJ_dm_vec, CLIP_L2_m_vec)
            if scalem < 1.0:
                clip_logs.append(f"clip dJ_dm_vec: {normm:.2e}->{CLIP_L2_m_vec:.2e} (x{scalem:.2e})")
        if clip_logs:
            print("[CLIP] " + " | ".join(clip_logs))

        # ---- 4) Gradient descent update (lower cost) ----
        M = M - LR_M * dJ_dM
        N = N - LR_N * dJ_dN
        O = O - LR_O * dJ_dO
        m_vec = m_vec - LR_m_vec * dJ_dm_vec

        # Persist updated matrices so next MPC iteration uses latest model parameters.
        M_df.iloc[:, :] = M
        N_df.iloc[:, :] = N
        O_df.iloc[:, :] = O
        m_vec_col = "value" if "value" in m_vec_df.columns else m_vec_df.columns[0]
        m_vec_df.loc[:, m_vec_col] = m_vec
        A_df.iloc[:, :] = np.eye(M.shape[0], dtype=float) + M
        M_df.to_csv(matrices_work_dir / "M.csv")
        N_df.to_csv(matrices_work_dir / "N.csv")
        O_df.to_csv(matrices_work_dir / "O.csv")
        m_vec_df.to_csv(matrices_work_dir / "m_vec.csv")
        A_df.to_csv(matrices_work_dir / "A.csv")

        step_mats_dir = matrices_snap_dir / f"matrices_step_{step:06d}"
        _save_matrices_snapshot(step_mats_dir, A_df=A_df, M_df=M_df, N_df=N_df, O_df=O_df, m_vec_df=m_vec_df)

        dJ_dM_text = f"{np.linalg.norm(dJ_dM):.2e}" if LR_M != 0 else "unused"
        dJ_dN_text = f"{np.linalg.norm(dJ_dN):.2e}" if LR_N != 0 else "unused"
        dJ_dO_text = f"{np.linalg.norm(dJ_dO):.2e}" if LR_O != 0 else "unused"
        dJ_dm_vec_text = f"{np.linalg.norm(dJ_dm_vec):.2e}" if LR_m_vec != 0 else "unused"
        du0_dM_text = f"{np.linalg.norm(du0_dM):.2e}" if LR_M != 0 else "unused"
        du0_dN_text = f"{np.linalg.norm(du0_dN):.2e}" if LR_N != 0 else "unused"
        du0_dO_text = f"{np.linalg.norm(du0_dO):.2e}" if LR_O != 0 else "unused"
        du0_dm_vec_text = f"{np.linalg.norm(du0_dm_vec):.2e}" if LR_m_vec != 0 else "unused"

        g2_pairs = [
            ("||du0_dM||_F", du0_dM_text),
            ("||du0_dN||_F", du0_dN_text),
            ("||du0_dO||_F", du0_dO_text),
            ("||du0_dm_vec||_F", du0_dm_vec_text),
        ]
        g3_pairs = [
            ("||dJ_dM||_F", dJ_dM_text),
            ("||dJ_dN||_F", dJ_dN_text),
            ("||dJ_dO||_F", dJ_dO_text),
            ("||dJ_dm_vec||_2", dJ_dm_vec_text),
        ]
        lr_pairs = [
            ("LR_M", f"{LR_M:.2e}"),
            ("LR_N", f"{LR_N:.2e}"),
            ("LR_O", f"{LR_O:.2e}"),
            ("LR_m_vec", f"{LR_m_vec:.2e}"),
        ]
        col_lhs_w = [
            max(len(g2_pairs[i][0]), len(g3_pairs[i][0]))
            for i in range(len(g2_pairs))
        ]
        col_rhs_w = [
            max(len(g2_pairs[i][1]), len(g3_pairs[i][1]))
            for i in range(len(g2_pairs))
        ]
        lr_lhs_w = max(len(lhs) for lhs, _ in lr_pairs)
        lr_rhs_w = max(len(rhs) for _, rhs in lr_pairs)

        def _fmt_row(pairs: list[tuple[str, str]]) -> str:
            parts: list[str] = []
            for i, (lhs, rhs) in enumerate(pairs):
                parts.append(f"{lhs:>{col_lhs_w[i]}} = {rhs:<{col_rhs_w[i]}}")
            return "  ".join(parts)

        def _fmt_lr_row(pairs: list[tuple[str, str]]) -> str:
            parts = [f"{lhs:>{lr_lhs_w}} = {rhs:<{lr_rhs_w}}" for lhs, rhs in pairs]
            return "          ".join(parts)

        print(f"[Gradient_Part1] ||dJ_du0||_2 = {np.linalg.norm(dJ_du0):.2e}")

        print(f"[Gradient_Part2] {_fmt_row(g2_pairs)}")
        print(f"[Gradient_Final] {_fmt_row(g3_pairs)}")
        print(f"[Learning_Rate] {_fmt_lr_row(lr_pairs)}")

        print(f"[MATS] saved snapshot under: {step_mats_dir}")

        # ---- 2) Save gradients as CSVs (including dJ_du0) under a per-step folder ----
        step_csv_dir = grad_dir / f"step_{step:06d}"
        _save_fd_grads_as_csv(step_csv_dir, mpc_result_and_gradients=mpc_result_and_gradients, step=step,
                              kappa_0=kappa_0, x_ini=x_before, u_names=u_names, x_names=x_names, d_names=d_names)

        print(f"[FD] saved CSVs under: {step_csv_dir}")

        u_hist.append(u0.copy())
        d_real_hist.append(d0_real.copy())
        d_pred0_hist.append(d_forecast[0].copy())
        t_hist.append(t_real[step])
        x_hist.append(x0.copy())
        step_hist.append(int(step))
        kappa_0_hist.append(int(kappa_0))
        obj_mpc_hist.append(None if obj_mpc is None else float(obj_mpc))
        cost_actual_hist.append(float(one_step_total_cost_act))
        try:
            cost_hat_hist.append(float(one_step_cost_pre) if one_step_cost_pre is not None else np.nan)
        except Exception:
            cost_hat_hist.append(np.nan)
        control_cost_components_hist.append(one_step_control_cost_act_components)
        slack_cost_components_hist.append(one_step_slack_cost_act_components)
        x_before_hist.append(np.asarray(x_before, dtype=float).reshape(-1))
        x_after_hist.append(np.asarray(x0, dtype=float).reshape(-1))
        surrogate_phase_hist.append(str(surrogate_phase))
        prev_phase = phase_now

    n_roll = len(step_hist)
    t_str = np.asarray([pd.Timestamp(ts).isoformat() for ts in t_hist], dtype=str)
    step_arr = np.asarray(step_hist, dtype=int)
    kappa_arr = np.asarray(kappa_0_hist, dtype=int)
    obj_arr = np.asarray([np.nan if v is None else float(v) for v in obj_mpc_hist], dtype=float)
    total_cost_arr = np.asarray(cost_actual_hist, dtype=float)
    cost_hat_arr = np.asarray(cost_hat_hist, dtype=float)
    X_before_arr = np.asarray(x_before_hist, dtype=float).reshape(n_roll, num_x)
    X_after_arr = np.asarray(x_after_hist, dtype=float).reshape(n_roll, num_x)
    U0_arr = np.asarray(u_hist, dtype=float).reshape(n_roll, num_u)
    D_real_arr = np.asarray(d_real_hist, dtype=float).reshape(n_roll, num_d)
    D_pred0_arr = np.asarray(d_pred0_hist, dtype=float).reshape(n_roll, num_d)

    rollout_df = pd.DataFrame(
        {
            "step": step_arr,
            "timestamp": t_str,
            "kappa_0": kappa_arr,
            "surrogate_phase": surrogate_phase_hist,
            "obj": obj_arr,
            "one_step_actual_cost_twin": total_cost_arr,
            "one_step_actual_cost_surrogate": cost_hat_arr,
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
    for i, name in enumerate(u_names):
        rollout_df[f"u0_{name}"] = U0_arr[:, i]
    for i, name in enumerate(d_names):
        rollout_df[f"d0_real_{name}"] = D_real_arr[:, i]
    for i, name in enumerate(d_names):
        rollout_df[f"d0_pred0_{name}"] = D_pred0_arr[:, i]

    rollout_csv_path = OUT_DIR / f"rollout_step_log_season={season}__sur={surrogate_model}_phasewise_lrN={LR_N_TAG}_days={num_simulation_day}.csv"
    rollout_df.to_csv(rollout_csv_path, index=False)

    if n_roll > 0:
        actual_ref_label = "One-step actual cost (twin)"
        predicted_label = f"Predicted {surrogate_cost_label} (surrogate)"
        fig_cmp, ax_cmp = plt.subplots(figsize=(10.0, 4.2))
        ax_cmp.plot(
            step_arr,
            rollout_df["one_step_actual_cost_twin"].to_numpy(dtype=float),
            color="#1C6EA4",
            lw=1.2,
            label=actual_ref_label,
        )
        ax_cmp.plot(
            step_arr,
            rollout_df["one_step_actual_cost_surrogate"].to_numpy(dtype=float),
            color="#BF124D",
            lw=1.2,
            ls="--",
            label=predicted_label,
        )
        ax_cmp.set_xlabel("Step")
        ax_cmp.set_ylabel("Cost")
        ax_cmp.set_title(f"{actual_ref_label} vs {predicted_label}")
        ax_cmp.grid(True, ls="--", lw=0.4, alpha=0.6)
        for spine in ("top", "right"):
            ax_cmp.spines[spine].set_visible(False)
        ax_cmp.legend(loc="upper right", frameon=False)
        fig_cmp.tight_layout()
        cost_cmp_path = OUT_DIR / f"one_step_actual_vs_predicted_cost_season={season}__sur={surrogate_model}_phasewise_lrN={LR_N_TAG}_days={num_simulation_day}.pdf"
        fig_cmp.savefig(cost_cmp_path, bbox_inches="tight", format="pdf")
        plt.close(fig_cmp)
        print(f"Saved plot: {cost_cmp_path}")

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
        print(f"Saved FD gradients (CSVs) under: {grad_dir}")

    return {
        "output_dir": OUT_DIR,
        "figure_path": FIG_PATH,
        "rollout_csv_path": rollout_csv_path,
        "prediction_csv_path": d_pre_csv_path,
        "realization_csv_path": d_rea_csv_path,
        "matrices_dir": MATRICES_DIR,
        "phase_surrogate_root": phase_surrogate_root,
    }


def main() -> None:
    args = parse_args()
    run_simulation(
        season=args.season,
        surrogate_model=args.surrogate_model,
        LR_N=args.lr_N,
        num_simulation_day=args.num_simulation_day,
        keeper_path_override=args.keeper_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
