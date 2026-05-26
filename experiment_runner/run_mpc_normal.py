# experiment_runner/run_mpc_normal.py
"""Run MPC simulation with switchable MPC method.

Example:

python3 experiment_runner/run_mpc_normal.py --season cold --mpc-method base       --num-simulation-day 31
python3 experiment_runner/run_mpc_normal.py --season cold --mpc-method robust     --num-simulation-day 31
python3 experiment_runner/run_mpc_normal.py --season cold --mpc-method stochastic --num-simulation-day 31
python3 experiment_runner/run_mpc_normal.py --season warm --mpc-method base       --num-simulation-day 31
python3 experiment_runner/run_mpc_normal.py --season warm --mpc-method robust     --num-simulation-day 31
python3 experiment_runner/run_mpc_normal.py --season warm --mpc-method stochastic --num-simulation-day 31

python3 experiment_runner/run_mpc_normal.py --season cold --mpc-method base       --num-simulation-day 31 &
python3 experiment_runner/run_mpc_normal.py --season cold --mpc-method robust     --num-simulation-day 31 &
python3 experiment_runner/run_mpc_normal.py --season cold --mpc-method stochastic --num-simulation-day 31 &
python3 experiment_runner/run_mpc_normal.py --season warm --mpc-method base       --num-simulation-day 31 &
python3 experiment_runner/run_mpc_normal.py --season warm --mpc-method robust     --num-simulation-day 31 &
python3 experiment_runner/run_mpc_normal.py --season warm --mpc-method stochastic --num-simulation-day 31 &

wait


Supported season:
- cold
- warm

Supported mpc-method:
- base
- robust
- stochastic

Supported num-simulation-day:
- 1
- ...
- 14

"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Callable, Dict, Literal, Any

import numpy as np
import pandas as pd

K = 12

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.builders.builder_digital_twin_one_step import digital_twin_one_step
from src.builders import builder_mpc_model
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
from src.utils.info_handlers_and_plotters import (
    compute_time_axes_from_hist,
    load_outdoor_disturbance_csv,
    load_parameters_with_x_ini,
    plot_mpc_summary_legacy,
    progress_print,
)

keeper_path = (REPO_ROOT / "configs/var_and_param_keeper.yaml").resolve()


def _num_simulation_day_type(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("--num-simulation-day must be >= 1")
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MPC simulation with selectable method and season.",
    )
    parser.add_argument(
        "--season",
        choices=("cold", "warm"),
        default="cold",
        help="Season for disturbance inputs and model matrices.",
    )
    parser.add_argument(
        "--mpc-method",
        choices=("base", "robust", "stochastic"),
        default="base",
        help="MPC method to solve at each step.",
    )
    parser.add_argument(
        "--num-simulation-day",
        type=_num_simulation_day_type,
        default=1,
        help="Number of simulation days (must be >= 1).",
    )
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
        help="Override output directory for figures and rollout logs.",
    )
    return parser.parse_args()


def _resolve_mpc_solver(method: str) -> Callable[..., Dict[str, Any]]:
    solvers: Dict[str, Callable[..., Dict[str, Any]]] = {
        "base": builder_mpc_model.build_then_solve_mpc_base,
        "robust": builder_mpc_model.build_then_solve_mpc_robust,
        "stochastic": builder_mpc_model.build_then_solve_mpc_stochastic,
    }
    return solvers[method]


def _resolve_default_paths_for_season(season: str) -> tuple[Path, Path, Path]:
    if season == "warm":
        d_pre_csv_path = (REPO_ROOT / "data/processed/testing_outdoor_prediction_warm.csv").resolve()
        d_rea_csv_path = (REPO_ROOT / "data/processed/testing_outdoor_realization_warm.csv").resolve()
        matrices_path = (REPO_ROOT / "experiment_result/r2_fit_state_space/warm/M.csv").resolve()
    else:
        d_pre_csv_path = (REPO_ROOT / "data/processed/testing_outdoor_prediction_cold.csv").resolve()
        d_rea_csv_path = (REPO_ROOT / "data/processed/testing_outdoor_realization_cold.csv").resolve()
        matrices_path = (REPO_ROOT / "experiment_result/r2_fit_state_space/cold/M.csv").resolve()
    return d_pre_csv_path, d_rea_csv_path, matrices_path


def _out_paths(method: str, season: str, num_simulation_day: int, *, output_dir: Path | None = None) -> tuple[Path, Path]:
    if output_dir is None:
        out_dir = (REPO_ROOT / f"experiment_result/run_mpc_normal_{method}" / season).resolve()
    else:
        out_dir = output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"mpc_summary_{season}_{method}_{num_simulation_day}day.pdf"
    return out_dir, fig_path


def run_simulation(
    *,
    season: Literal["cold", "warm"],
    mpc_method: Literal["base", "robust", "stochastic"],
    num_simulation_day: int,
    keeper_path_override: Path | None = None,
    prediction_csv_path: Path | None = None,
    realization_csv_path: Path | None = None,
    matrices_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    default_prediction_csv_path, default_realization_csv_path, default_matrices_path = _resolve_default_paths_for_season(season)
    d_pre_csv_path = default_prediction_csv_path if prediction_csv_path is None else prediction_csv_path.resolve()
    d_rea_csv_path = default_realization_csv_path if realization_csv_path is None else realization_csv_path.resolve()
    matrices_path = default_matrices_path if matrices_path is None else matrices_path.resolve()
    keeper_path_resolved = keeper_path if keeper_path_override is None else keeper_path_override.resolve()
    solve_mpc = _resolve_mpc_solver(mpc_method)
    out_dir, fig_path = _out_paths(mpc_method, season, num_simulation_day, output_dir=output_dir)

    param_keeper = load_parameters_with_x_ini(keeper_path_resolved)
    num_x = get_num_states(param_keeper)
    num_u = get_num_controls(param_keeper)
    num_d = get_num_disturbances(param_keeper)
    x_names = get_state_names_in_order(param_keeper)
    u_names = get_control_names_in_order(param_keeper)
    d_names = get_disturbance_names_in_order(param_keeper)
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
    num_steps_per_day = int(param_keeper["kappa_day_night_total_steps"])
    num_steps_in_whole_simulation = int(num_simulation_day) * num_steps_per_day

    t_pred, disturbance_pred = load_outdoor_disturbance_csv(d_pre_csv_path)
    t_real, disturbance_real = load_outdoor_disturbance_csv(d_rea_csv_path)
    raw_disturbance_names = ["T_out", "H_out", "C_out", "R_out"]
    d_reorder = [raw_disturbance_names.index(name) for name in d_names]
    disturbance_pred = disturbance_pred[:, d_reorder]
    disturbance_real = disturbance_real[:, d_reorder]

    x_hist = [x0.copy()]
    u_hist: list[np.ndarray] = []
    d_real_hist: list[np.ndarray] = []
    d_pred0_hist: list[np.ndarray] = []
    t_hist: list[pd.Timestamp] = []
    # NEW: step-level rollout logging containers
    step_hist: list[int] = []
    kappa_0_hist: list[int] = []
    obj_hist: list[float | None] = []
    cost_actual_hist: list[float] = []
    control_cost_components_hist: list[dict[str, float]] = []
    slack_cost_components_hist: list[dict[str, float]] = []
    x_before_hist: list[np.ndarray] = []
    x_after_hist: list[np.ndarray] = []
    dt_s_step = int(param_keeper["delta_t"])

    for step in range(num_steps_in_whole_simulation):
        if step >= disturbance_pred.shape[0] or step >= disturbance_real.shape[0]:
            break

        d_forecast = disturbance_pred[step : step + K]

        kappa_0 = step % int(param_keeper["kappa_day_night_total_steps"])
        mpc_result = solve_mpc(
            x_ini=x0,
            d_forecast=d_forecast,
            kappa_ini=kappa_0,
            keeper_path=keeper_path_resolved,
            matrices_path=matrices_path,
            horizon_K=K,
            num_x=num_x,
            num_u=num_u,
            num_d=num_d,
        )

        u0 = np.asarray(mpc_result["u"][0], dtype=float)
        d0_real = np.asarray(disturbance_real[step], dtype=float)
        x_before = x0.copy()

        x1_real = digital_twin_one_step(x=x0, u=u0, d=d0_real, param_keeper=param_keeper)
        x_after_raw = np.asarray(x1_real[0] if isinstance(x1_real, tuple) else x1_real, dtype=float).reshape(-1)

        # NEW: realized one-step actual control cost at end state (same function used by fit_relationship script)
        one_step_cost_act_info = builder_mpc_model.get_cost_at_end_state(
            x_act=x_after_raw,
            u0=u0,
            kappa_act=int(kappa_0),
            keeper_path=keeper_path_resolved,
        )
        one_step_total_cost_act = float(one_step_cost_act_info["one_step_total_cost_act"])
        one_step_control_cost_act_components = {
            str(k): float(v) for k, v in dict(one_step_cost_act_info["one_step_control_cost_act_components"]).items()
        }
        one_step_slack_cost_act_components = {
            str(k): float(v) for k, v in dict(one_step_cost_act_info["one_step_slack_cost_act_components"]).items()
        }

        x0_next = x_after_raw.copy()
        next_time = t_real[step] + pd.Timedelta(seconds=dt_s_step)
        if next_time.hour == 0 and next_time.minute == 0:
            x0_next[3] = 0.0
        x0 = x0_next

        obj_mpc = mpc_result.get("obj", None)

        u_hist.append(u0.copy())
        d_real_hist.append(d0_real.copy())
        d_pred0_hist.append(np.asarray(d_forecast[0], dtype=float).copy())
        t_hist.append(t_real[step])
        x_hist.append(x0.copy())
        step_hist.append(int(step))
        kappa_0_hist.append(int(kappa_0))
        obj_hist.append(None if obj_mpc is None else float(obj_mpc))
        cost_actual_hist.append(float(one_step_total_cost_act))
        control_cost_components_hist.append(one_step_control_cost_act_components)
        slack_cost_components_hist.append(one_step_slack_cost_act_components)
        x_before_hist.append(np.asarray(x_before, dtype=float).reshape(-1))
        x_after_hist.append(np.asarray(x0, dtype=float).reshape(-1))

        progress_print(step=step, total=num_steps_in_whole_simulation, ts=t_real[step], u0=u0[u_display_idx], x_before=x_before[x_display_idx],
                       x_after=x0[x_display_idx], obj_mpc=obj_mpc, cost_actual_step=one_step_total_cost_act,
                       param_keeper=param_keeper, kappa_0=kappa_0, horizon_K=K,
                       d0_pred=d_forecast[0][d_display_idx], d0_real=d0_real[d_display_idx])

    # NEW: save per-step rollout logs to CSV under the existing output directory.
    n_roll = len(step_hist)
    t_str = np.asarray([pd.Timestamp(ts).isoformat() for ts in t_hist], dtype=str)
    step_arr = np.asarray(step_hist, dtype=int)
    kappa_arr = np.asarray(kappa_0_hist, dtype=int)
    obj_arr = np.asarray([np.nan if v is None else float(v) for v in obj_hist], dtype=float)
    cost_arr = np.asarray(cost_actual_hist, dtype=float)
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
            "obj": obj_arr,
            "cost_actual": cost_arr,
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

    rollout_csv_path = out_dir / f"rollout_step_log_{season}_{mpc_method}_{num_simulation_day}day.csv"
    rollout_df.to_csv(rollout_csv_path, index=False)

    _ = t_pred
    if u_hist:
        t, t_x, dt_s = compute_time_axes_from_hist(t_hist, dt_default_s=int(param_keeper["delta_t"]))
        x_hist_display = [np.asarray(x, dtype=float).reshape(-1)[x_display_idx] for x in x_hist]
        u_hist_display = [np.asarray(u, dtype=float).reshape(-1)[u_display_idx] for u in u_hist]
        d_pred0_hist_display = [np.asarray(d0, dtype=float).reshape(-1)[d_display_idx] for d0 in d_pred0_hist]
        d_real_hist_display = [np.asarray(d0, dtype=float).reshape(-1)[d_display_idx] for d0 in d_real_hist]
        plot_mpc_summary_legacy(
            fig_path,
            x_hist=x_hist_display,
            u_hist=u_hist_display,
            d_pred0_hist=d_pred0_hist_display,
            d_real_hist=d_real_hist_display,
            param_keeper=param_keeper,
            t=t,
            t_x=t_x,
            dt_s=dt_s,
            show=False,
        )
        print(f"Saved plot: {fig_path}")

    return {
        "output_dir": out_dir,
        "figure_path": fig_path,
        "rollout_csv_path": rollout_csv_path,
        "prediction_csv_path": d_pre_csv_path,
        "realization_csv_path": d_rea_csv_path,
        "matrices_path": matrices_path,
    }


def main() -> None:
    args = parse_args()
    run_simulation(
        season=args.season,
        mpc_method=args.mpc_method,
        num_simulation_day=args.num_simulation_day,
        keeper_path_override=args.keeper_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
