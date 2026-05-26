"""Generate base-MPC rollout data as an independent r3 dataset.


Example:
python3 experiment_runner/r3_generate_data_from_base.py --season cold --num-simulation-day 59 --chunk-days 4 --max-workers 8
python3 experiment_runner/r3_generate_data_from_base.py --season warm --num-simulation-day 90 --chunk-days 4 --max-workers 8


python3 experiment_runner/r3_generate_data_from_base.py --season cold --num-simulation-day 30 --chunk-days 2 --max-workers 8 &
python3 experiment_runner/r3_generate_data_from_base.py --season warm --num-simulation-day 30 --chunk-days 2 --max-workers 8 &

wait

Run order in pipeline:
1) experiment_runner/r1_generate_data_from_twin.py
2) experiment_runner/r2_fit_state_space.py
3) this script (r3_generate_data_from_base.py), if you want on-policy base-MPC samples
4) experiment_runner/r4_fit_J_act_DNN_phase_wise.py or experiment_runner/run_mpc_normal.py

This script no longer appends into r1 data. It writes a separate dataset under:
experiment_result/r3_generate_data_from_base/<season>/training_data.csv

Default behavior:
- split the requested horizon into independent small chunks
- reset x_ini at the start of each chunk
- write one shard per chunk
- merge shards into the final r3 training_data.csv
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.builders import builder_mpc_model
from src.builders.builder_digital_twin_one_step import digital_twin_one_step, load_parameters
from src.utils.info_handlers_and_plotters import load_outdoor_disturbance_csv
from src.utils.vector_order import (
    get_control_names_in_order,
    get_disturbance_indices,
    get_disturbance_names_in_order,
    get_num_controls,
    get_num_disturbances,
    get_num_states,
    get_state_indices,
    get_state_names_in_order,
)


KEEPER_PATH = (REPO_ROOT / "configs/var_and_param_keeper.yaml").resolve()
FIT_BASE_DIR = (REPO_ROOT / "experiment_result/r2_fit_state_space").resolve()
SCHEMA_TEMPLATE_BASE_DIR = (REPO_ROOT / "experiment_result/r1_generate_data_from_twin").resolve()
OUTPUT_BASE_DIR = (REPO_ROOT / "experiment_result/r3_generate_data_from_base").resolve()
OUTDOOR_REAL_CSV_BY_SEASON = {
    "cold": (REPO_ROOT / "data/processed/training_outdoor_realization_cold.csv").resolve(),
    "warm": (REPO_ROOT / "data/processed/training_outdoor_realization_warm.csv").resolve(),
}
OUTDOOR_PRED_CSV_BY_SEASON = {
    "cold": (REPO_ROOT / "data/processed/training_outdoor_prediction_cold.csv").resolve(),
    "warm": (REPO_ROOT / "data/processed/training_outdoor_prediction_warm.csv").resolve(),
}

RAW_DISTURBANCE_NAMES = ["T_out", "H_out", "C_out", "R_out"]
DEFAULT_CHUNK_DAYS = 2
DEFAULT_MAX_WORKERS_CAP = 8
OUTPUT_TRAINING_FILENAME = "training_data.csv"
ENRICHED_SCHEMA_EXTRA_COLS = [
    "cost_actual",
    "cost",
    "control_cost",
    "slack_cost",
    "target_output_cost",
    "surrogate_phase",
    "T_in_star",
    "H_in_star",
    "C_in_star",
    "L_star",
]


def _positive_int(value: str, flag_name: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{flag_name} must be >= 1")
    return ivalue


def _num_simulation_day_type(value: str) -> int:
    return _positive_int(value, "--num-simulation-day")


def _horizon_type(value: str) -> int:
    return _positive_int(value, "--mpc-horizon-K")


def _chunk_days_type(value: str) -> int:
    return _positive_int(value, "--chunk-days")


def _max_workers_type(value: str) -> int:
    return _positive_int(value, "--max-workers")


def _default_horizon_k() -> int:
    try:
        import run_mpc_normal as run_mpc_normal  # same folder

        return int(getattr(run_mpc_normal, "K"))
    except Exception:
        return 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate base-MPC rollout data as an independent r3 dataset.")
    parser.add_argument("--season", choices=("cold", "warm"), required=True)
    parser.add_argument("--num-simulation-day", type=_num_simulation_day_type, default=1)
    parser.add_argument("--mpc-horizon-K", type=_horizon_type, default=None)
    parser.add_argument("--chunk-days", type=_chunk_days_type, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument("--max-workers", type=_max_workers_type, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--append-to", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _slice_with_pad(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    n = int(arr.shape[0])
    if n < 1:
        raise ValueError("Disturbance array is empty.")
    s = int(start)
    e = min(s + int(length), n)
    if s < n:
        block = np.asarray(arr[s:e], dtype=float)
    else:
        block = np.empty((0, arr.shape[1]), dtype=float)
    if block.shape[0] < length:
        pad = np.repeat(arr[-1:, :], repeats=(length - block.shape[0]), axis=0)
        block = np.vstack([block, pad])
    return block


def _infer_hhmm(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%H:%M")


def _infer_rh_and_ppm_from_d(d_vec: np.ndarray, d_idx: dict[str, int]) -> tuple[float, float]:
    t_out = float(d_vec[d_idx["T_out"]])
    h_out = float(d_vec[d_idx["H_out"]])
    c_out = float(d_vec[d_idx["C_out"]])

    e_s_hpa = 6.112 * math.exp((17.67 * t_out) / (t_out + 243.5))
    e_h_hpa = h_out * (t_out + 273.15) / 216.7
    rh_out = float(e_h_hpa / e_s_hpa) if e_s_hpa > 0 else float("nan")

    m_co2 = 44.01e-3
    r_gas = 8.314462618
    t_k = t_out + 273.15
    rho_co2 = c_out / 1000.0
    x_mole = (rho_co2 * r_gas * t_k) / (101325.0 * m_co2)
    ppm_out = float(x_mole * 1e6)
    return rh_out, ppm_out


def _schema_template_path(season: str) -> Path:
    return (SCHEMA_TEMPLATE_BASE_DIR / season / "data.csv").resolve()


def _resolve_output_path(season: str, output_csv: Path | None, append_to_alias: Path | None) -> Path:
    chosen = output_csv if output_csv is not None else append_to_alias
    if chosen is not None:
        return chosen.expanduser().resolve()
    return (OUTPUT_BASE_DIR / season / OUTPUT_TRAINING_FILENAME).resolve()


def _chunk_dir_for_output(output_csv: Path) -> Path:
    return (output_csv.parent / "_chunks").resolve()


def _load_initial_state(param_keeper: dict, num_x: int) -> np.ndarray:
    x_ini = np.asarray(param_keeper.get("x_ini", []), dtype=float).reshape(-1)
    if x_ini.size != int(num_x) or not np.all(np.isfinite(x_ini)):
        raise ValueError("Cannot infer initial state x0 from keeper yaml.")
    return x_ini.astype(float)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _build_column_groups(
    x_names: list[str],
    u_names: list[str],
    d_names: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    x_ini_cols = [f"X_ini_{name}" for name in x_names]
    u_cols = list(u_names)
    d_cols = [f"D_{name}" for name in d_names]
    dx_cols = [f"DX_d{name}" for name in x_names]
    x1_cols = [f"X1_{name}" for name in x_names]
    return x_ini_cols, u_cols, d_cols, dx_cols, x1_cols


def _get_cost_field(cost_info: dict, new_key: str, legacy_key: str) -> float:
    val = cost_info.get(new_key, cost_info.get(legacy_key, np.nan))
    try:
        return float(val)
    except Exception:
        return float("nan")


def _build_rollout_row(
    x_ini: np.ndarray,
    u0: np.ndarray,
    d0_actual: np.ndarray,
    x_next: np.ndarray,
    kappa_k: int,
    hhmm: str,
    rh_out: float,
    ppm_out: float,
    surrogate_phase: str,
    cost_info: dict,
    x_ini_cols: list[str],
    u_cols: list[str],
    d_cols: list[str],
    dx_cols: list[str],
    x1_cols: list[str],
) -> dict:
    dx = x_next - x_ini
    row: dict[str, float | int | str] = {}
    row.update(dict(zip(x_ini_cols, x_ini.tolist())))
    row.update(dict(zip(u_cols, u0.tolist())))
    row.update(dict(zip(d_cols, d0_actual.tolist())))
    row.update(dict(zip(dx_cols, dx.tolist())))
    row.update(dict(zip(x1_cols, x_next.tolist())))
    row["kappa_k"] = int(kappa_k)
    row["HHMM"] = str(hhmm)
    row["RH_out"] = float(rh_out)
    row["ppm_out"] = float(ppm_out)
    row["surrogate_phase"] = str(surrogate_phase)

    row["cost_actual"] = _get_cost_field(cost_info, "one_step_total_cost_act", "cost")
    row["cost"] = _get_cost_field(cost_info, "one_step_total_cost_act", "cost")
    row["control_cost"] = _get_cost_field(cost_info, "one_step_control_cost_act", "control_cost")
    row["slack_cost"] = _get_cost_field(cost_info, "one_step_slack_cost_act", "slack_cost")
    row["target_output_cost"] = row["cost_actual"]
    control_cost_components = cost_info.get("one_step_control_cost_act_components", cost_info.get("control_cost_components", {}))
    slack_cost_components = cost_info.get("one_step_slack_cost_act_components", cost_info.get("slack_cost_components", {}))
    for k, v in dict(control_cost_components).items():
        row[str(k)] = float(v)
    for k, v in dict(slack_cost_components).items():
        row[str(k)] = float(v)
    for k, v in dict(cost_info.get("slacks", {})).items():
        row[str(k)] = float(v)
    for k, v in dict(cost_info.get("refs", {})).items():
        row[str(k)] = float(v)
    return row


def _load_schema_columns(schema_path: Path) -> list[str]:
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema template not found: {schema_path}\n"
            "Run experiment_runner/r1_generate_data_from_twin.py first."
        )
    schema_df = pd.read_csv(schema_path, nrows=0)
    schema_cols = list(schema_df.columns)
    if not schema_cols:
        raise ValueError(f"Schema template has no columns: {schema_path}")
    for col in ENRICHED_SCHEMA_EXTRA_COLS:
        if col not in schema_cols:
            schema_cols.append(col)
    return schema_cols


def _load_season_disturbances(season: str, d_names: list[str]) -> tuple[pd.DatetimeIndex, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    t_pred, disturbance_pred = load_outdoor_disturbance_csv(OUTDOOR_PRED_CSV_BY_SEASON[season])
    t_real, disturbance_real = load_outdoor_disturbance_csv(OUTDOOR_REAL_CSV_BY_SEASON[season])
    d_reorder = [RAW_DISTURBANCE_NAMES.index(name) for name in d_names]
    disturbance_pred = disturbance_pred[:, d_reorder]
    disturbance_real = disturbance_real[:, d_reorder]
    return t_pred, disturbance_pred, t_real, disturbance_real


def _build_chunk_specs(total_steps: int, chunk_steps: int) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    chunk_id = 0
    start = 0
    while start < total_steps:
        steps = min(int(chunk_steps), int(total_steps - start))
        specs.append({"chunk_id": int(chunk_id), "start_step": int(start), "num_steps": int(steps)})
        start += steps
        chunk_id += 1
    return specs


def _resolve_max_workers(requested: int | None, num_chunks: int) -> int:
    if num_chunks <= 0:
        return 1
    if requested is not None:
        return max(1, min(int(requested), int(num_chunks)))
    cpu_count = os.cpu_count() or 1
    default_workers = min(DEFAULT_MAX_WORKERS_CAP, cpu_count, num_chunks)
    return max(1, int(default_workers))


def _prepare_output_locations(output_csv: Path, dry_run: bool) -> Path:
    chunk_dir = _chunk_dir_for_output(output_csv)
    if dry_run:
        return chunk_dir
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    return chunk_dir


def _generate_chunk_to_csv(
    *,
    season: str,
    start_step: int,
    num_steps: int,
    horizon_k: int,
    schema_cols: list[str],
    shard_path: Path,
    dry_run: bool,
) -> dict:
    param_keeper = load_parameters(KEEPER_PATH)
    num_x = get_num_states(param_keeper)
    num_u = get_num_controls(param_keeper)
    num_d = get_num_disturbances(param_keeper)
    x_names = get_state_names_in_order(param_keeper)
    u_names = get_control_names_in_order(param_keeper)
    d_names = get_disturbance_names_in_order(param_keeper)
    x_idx = get_state_indices(param_keeper)
    d_idx = get_disturbance_indices(param_keeper)
    x_ini_cols, u_cols, d_cols, dx_cols, x1_cols = _build_column_groups(x_names, u_names, d_names)
    steps_per_day = int(param_keeper["kappa_day_night_total_steps"])
    dt_s = int(param_keeper["delta_t"])
    matrices_path = (FIT_BASE_DIR / season / "M.csv").resolve()
    if not matrices_path.exists():
        raise FileNotFoundError(
            f"Fitted matrices not found at {matrices_path}. "
            "Run experiment_runner/r2_fit_state_space.py first."
        )

    _, disturbance_pred, t_real, disturbance_real = _load_season_disturbances(season, d_names)
    x_t = _load_initial_state(param_keeper, num_x)

    new_rows: list[dict] = []
    t0 = time.time()
    for local_step in range(int(num_steps)):
        global_step = int(start_step + local_step)
        d_forecast = _slice_with_pad(disturbance_pred, start=global_step, length=horizon_k)
        d0_actual = np.asarray(disturbance_real[global_step], dtype=float).reshape(-1)
        if d0_actual.size != int(num_d):
            raise ValueError(f"Disturbance row must be size {num_d}, got {d0_actual.size}")

        kappa_0 = int(global_step % steps_per_day)
        mpc_result = builder_mpc_model.build_then_solve_mpc_base(
            x_ini=x_t,
            d_forecast=d_forecast,
            kappa_ini=kappa_0,
            keeper_path=KEEPER_PATH,
            matrices_path=matrices_path,
            horizon_K=horizon_k,
            num_x=num_x,
            num_u=num_u,
            num_d=num_d,
        )
        u0 = np.asarray(mpc_result["u"][0], dtype=float).reshape(-1)
        if u0.size != int(num_u):
            raise ValueError(f"u0 must be size {num_u}, got {u0.size}")

        x_next, _ = digital_twin_one_step(x=x_t, u=u0, d=d0_actual, param_keeper=param_keeper)
        x_next = np.asarray(x_next, dtype=float).reshape(-1)
        if x_next.size != int(num_x):
            raise ValueError(f"x_next must be size {num_x}, got {x_next.size}")

        cost_info = builder_mpc_model.get_cost_at_end_state(
            x_act=x_next,
            u0=u0,
            kappa_act=kappa_0,
            keeper_path=KEEPER_PATH,
        )

        ts = pd.Timestamp(t_real[global_step])
        hhmm = _infer_hhmm(ts)
        rh_out, ppm_out = _infer_rh_and_ppm_from_d(d0_actual, d_idx)
        climate_refs = builder_mpc_model.get_bounds_and_refs(param_keeper, np.asarray([kappa_0], dtype=int))
        is_day = bool(np.asarray(climate_refs["is_day"], dtype=bool).reshape(-1)[0])
        surrogate_phase = "day" if is_day else "night"

        row = _build_rollout_row(
            x_ini=x_t,
            u0=u0,
            d0_actual=d0_actual,
            x_next=x_next,
            kappa_k=kappa_0,
            hhmm=hhmm,
            rh_out=rh_out,
            ppm_out=ppm_out,
            surrogate_phase=surrogate_phase,
            cost_info=cost_info,
            x_ini_cols=x_ini_cols,
            u_cols=u_cols,
            d_cols=d_cols,
            dx_cols=dx_cols,
            x1_cols=x1_cols,
        )
        new_rows.append(row)

        x_next_use = x_next.copy()
        next_time = ts + pd.Timedelta(seconds=dt_s)
        if next_time.hour == 0 and next_time.minute == 0:
            x_next_use[x_idx["L"]] = 0.0
        x_t = x_next_use

    new_df = pd.DataFrame(new_rows)
    for col in schema_cols:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[schema_cols]

    for col in new_df.columns:
        if col in ("kappa_k", "HHMM"):
            continue
        if pd.api.types.is_numeric_dtype(new_df[col]):
            new_df[col] = np.round(new_df[col].astype(float), 2)

    if not dry_run:
        _atomic_write_csv(new_df, shard_path)

    key_cols = [c for c in (x_ini_cols + u_cols + d_cols + dx_cols + x1_cols) if c in new_df.columns]
    finite_fraction = float("nan")
    if key_cols:
        vals = new_df[key_cols].to_numpy(dtype=float)
        finite_fraction = float(np.isfinite(vals).mean()) if vals.size > 0 else float("nan")

    elapsed_s = time.time() - t0
    return {
        "start_step": int(start_step),
        "num_steps": int(num_steps),
        "rows": int(len(new_df)),
        "finite_fraction": float(finite_fraction),
        "elapsed_s": float(elapsed_s),
        "shard_path": str(shard_path),
    }


def main() -> None:
    args = parse_args()
    season = str(args.season)
    num_simulation_day = int(args.num_simulation_day)
    horizon_k = int(args.mpc_horizon_K) if args.mpc_horizon_K is not None else int(_default_horizon_k())
    chunk_days = int(args.chunk_days)
    output_csv = _resolve_output_path(season, args.output_csv, args.append_to)
    schema_path = _schema_template_path(season)
    schema_cols = _load_schema_columns(schema_path)

    param_keeper = load_parameters(KEEPER_PATH)
    steps_per_day = int(param_keeper["kappa_day_night_total_steps"])
    total_steps_requested = int(num_simulation_day) * steps_per_day
    chunk_steps = int(chunk_days) * steps_per_day

    d_names = get_disturbance_names_in_order(param_keeper)
    t_pred, disturbance_pred, t_real, disturbance_real = _load_season_disturbances(season, d_names)
    available_steps = int(min(disturbance_pred.shape[0], disturbance_real.shape[0]))
    total_steps = int(min(total_steps_requested, available_steps))
    if total_steps < total_steps_requested:
        print(
            f"[WARN] truncating requested steps from {total_steps_requested} to {total_steps} "
            f"because disturbance data only has {available_steps} usable rows."
        )

    chunk_specs = _build_chunk_specs(total_steps, chunk_steps)
    max_workers = _resolve_max_workers(args.max_workers, len(chunk_specs))
    chunk_dir = _prepare_output_locations(output_csv, args.dry_run)

    print(
        f"[START] season={season} days={num_simulation_day} requested_steps={total_steps_requested} "
        f"usable_steps={total_steps} K={horizon_k} chunk_days={chunk_days} chunks={len(chunk_specs)} "
        f"max_workers={max_workers}"
    )
    print(f"[START] schema_template={schema_path}")
    print(f"[START] output_csv={output_csv}")
    if not args.dry_run:
        print(f"[START] chunk_dir={chunk_dir}")
    _ = t_pred, t_real

    results: list[dict] = []
    t0 = time.time()
    if len(chunk_specs) == 0:
        print("[DONE] nothing to generate; no usable disturbance rows.")
        return

    worker_kwargs = [
        {
            "season": season,
            "start_step": spec["start_step"],
            "num_steps": spec["num_steps"],
            "horizon_k": horizon_k,
            "schema_cols": schema_cols,
            "shard_path": (chunk_dir / f"chunk_{spec['start_step']:06d}.csv").resolve(),
            "dry_run": bool(args.dry_run),
        }
        for spec in chunk_specs
    ]

    if max_workers == 1:
        for kwargs in worker_kwargs:
            result = _generate_chunk_to_csv(**kwargs)
            results.append(result)
            print(
                f"[CHUNK] start_step={result['start_step']} steps={result['num_steps']} rows={result['rows']} "
                f"finite_fraction={result['finite_fraction']:.4f} elapsed={result['elapsed_s']:.1f}s"
            )
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            future_map = {executor.submit(_generate_chunk_to_csv, **kwargs): kwargs for kwargs in worker_kwargs}
            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                print(
                    f"[CHUNK] start_step={result['start_step']} steps={result['num_steps']} rows={result['rows']} "
                    f"finite_fraction={result['finite_fraction']:.4f} elapsed={result['elapsed_s']:.1f}s"
                )

    results.sort(key=lambda item: int(item["start_step"]))
    total_rows = int(sum(int(item["rows"]) for item in results))

    if not args.dry_run:
        shard_paths = [Path(str(item["shard_path"])).resolve() for item in results if int(item["rows"]) > 0]
        shard_dfs = [pd.read_csv(path) for path in shard_paths]
        merged_df = pd.concat(shard_dfs, axis=0, ignore_index=True) if shard_dfs else pd.DataFrame(columns=schema_cols)
        for col in schema_cols:
            if col not in merged_df.columns:
                merged_df[col] = np.nan
        merged_df = merged_df[schema_cols]
        _atomic_write_csv(merged_df, output_csv)

    finite_values = [float(item["finite_fraction"]) for item in results if not np.isnan(float(item["finite_fraction"]))]
    avg_finite_fraction = float(np.mean(finite_values)) if finite_values else float("nan")
    elapsed_total_s = time.time() - t0
    mode = "DRY-RUN" if args.dry_run else "WRITTEN"
    print(f"[{mode}] season={season} K={horizon_k} num_simulation_day={num_simulation_day} chunk_days={chunk_days}")
    print(f"[{mode}] output_csv={output_csv}")
    print(f"[{mode}] chunks={len(results)} rows={total_rows} avg_chunk_finite_fraction={avg_finite_fraction:.4f}")
    print(f"[{mode}] elapsed_total={elapsed_total_s:.1f}s")


if __name__ == "__main__":
    main()
