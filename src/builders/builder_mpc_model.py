# src/builders/builder_mpc_model.py

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import yaml

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
from src.utils.surrogate_feature_scaler import (
    feature_scaler_from_npz,
    transform_surrogate_feature_vector,
)
from src.utils.torch_surrogate_mlp import load_torch_mlp_checkpoint, predict_torch_mlp

keeper_path = (Path(__file__).resolve().parents[2] / "configs/var_and_param_keeper.yaml").resolve()
matrices_path = (Path(__file__).resolve().parents[2] / "experiment_result/r2_fit_state_space").resolve()

num_x = get_num_states()
num_u = get_num_controls()
num_d = get_num_disturbances()

_DISPLAY_X_NAMES = ["T_in", "H_in", "C_in", "L"]
_DISPLAY_D_NAMES = ["T_out", "H_out", "C_out", "R_out"]
_DISPLAY_U_NAMES = ["U_heat", "U_fan", "U_nat", "U_ac", "U_dos", "U_LED", "U_hum", "U_deh", "U_shad", "U_warm"]
DFL_COST_TARGET_KIND = "reference_quad_v1"
CONTROL_COST_TARGET_KIND = "one_step_control_cost_act"
SUPPORTED_COST_TARGET_KINDS = {
    DFL_COST_TARGET_KIND,
    CONTROL_COST_TARGET_KIND,
}
_FD_N_PARALLEL_ENV = "CEA_DFL_FD_N_WORKERS"
_FD_N_WORKER_GUROBI_THREADS_ENV = "CEA_DFL_FD_GUROBI_THREADS"
_FD_N_DEFAULT_MAX_WORKERS = max(1, min(8, os.cpu_count() or 1))


def _get_vector_meta(param_keeper: dict) -> dict[str, object]:
    x_idx = get_state_indices(param_keeper)
    d_idx = get_disturbance_indices(param_keeper)
    u_idx = get_control_indices(param_keeper)
    x_names = get_state_names_in_order(param_keeper)
    d_names = get_disturbance_names_in_order(param_keeper)
    u_names = get_control_names_in_order(param_keeper)

    required_x = set(_DISPLAY_X_NAMES)
    required_d = set(_DISPLAY_D_NAMES)
    required_u = set(_DISPLAY_U_NAMES)
    if not required_x.issubset(x_idx):
        missing = sorted(required_x.difference(x_idx))
        raise KeyError(f"Missing required state names in vector order: {missing}")
    if not required_d.issubset(d_idx):
        missing = sorted(required_d.difference(d_idx))
        raise KeyError(f"Missing required disturbance names in vector order: {missing}")
    if not required_u.issubset(u_idx):
        missing = sorted(required_u.difference(u_idx))
        raise KeyError(f"Missing required control names in vector order: {missing}")

    return {
        "num_x": get_num_states(param_keeper),
        "num_u": get_num_controls(param_keeper),
        "num_d": get_num_disturbances(param_keeper),
        "x_idx": x_idx,
        "d_idx": d_idx,
        "u_idx": u_idx,
        "x_names": x_names,
        "d_names": d_names,
        "u_names": u_names,
    }


def _validate_problem_sizes(num_x_arg: int, num_u_arg: int, num_d_arg: int, meta: dict[str, object]) -> None:
    num_x_cfg = int(meta["num_x"])
    num_u_cfg = int(meta["num_u"])
    num_d_cfg = int(meta["num_d"])
    if int(num_x_arg) != num_x_cfg or int(num_u_arg) != num_u_cfg or int(num_d_arg) != num_d_cfg:
        raise ValueError(
            "Problem size mismatch against vector_order/yaml: "
            f"(num_x, num_u, num_d)=({num_x_arg}, {num_u_arg}, {num_d_arg}) "
            f"vs yaml=({num_x_cfg}, {num_u_cfg}, {num_d_cfg})"
        )


def _get_fd_n_parallel_max_workers(num_tasks: int) -> int:
    raw = os.getenv(_FD_N_PARALLEL_ENV, "").strip()
    if raw:
        try:
            configured = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {_FD_N_PARALLEL_ENV} must be an integer >= 1; got {raw!r}"
            ) from exc
        if configured < 1:
            raise ValueError(
                f"Environment variable {_FD_N_PARALLEL_ENV} must be >= 1; got {configured}"
            )
        return max(1, min(int(num_tasks), configured))
    return max(1, min(int(num_tasks), _FD_N_DEFAULT_MAX_WORKERS))


def _get_fd_worker_gurobi_threads() -> int:
    raw = os.getenv(_FD_N_WORKER_GUROBI_THREADS_ENV, "").strip()
    if not raw:
        return 1
    try:
        configured = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {_FD_N_WORKER_GUROBI_THREADS_ENV} must be an integer >= 1; got {raw!r}"
        ) from exc
    if configured < 1:
        raise ValueError(
            f"Environment variable {_FD_N_WORKER_GUROBI_THREADS_ENV} must be >= 1; got {configured}"
        )
    return configured


def _build_control_cost_vector(param_keeper: dict, u_idx: dict[str, int]) -> np.ndarray:
    delta_t = float(param_keeper["delta_t"])
    c_u = np.zeros(get_num_controls(param_keeper), dtype=float)
    c_u[u_idx["U_heat"]] = float(param_keeper["alpha_heat"]) * float(param_keeper["Q_bar_heat"]) * delta_t / 3.6e6
    c_u[u_idx["U_fan"]] = float(param_keeper["alpha_fan"]) * float(param_keeper["S_fan"]) * float(param_keeper["V_bar_fan"]) * delta_t / 3.6e6
    c_u[u_idx["U_ac"]] = float(param_keeper["alpha_ac"]) * float(param_keeper["Q_bar_ac"]) * delta_t / 3.6e6
    c_u[u_idx["U_dos"]] = float(param_keeper["alpha_dos"]) * float(param_keeper["D_bar_dos"]) * delta_t
    c_u[u_idx["U_LED"]] = float(param_keeper["alpha_LED"]) * float(param_keeper["P_bar_LED"]) * delta_t / 3.6e6
    c_u[u_idx["U_hum"]] = float(param_keeper["alpha_hum"]) * float(param_keeper["F_bar_hum"]) * delta_t
    c_u[u_idx["U_deh"]] = float(param_keeper["alpha_deh"]) * float(param_keeper["F_bar_deh"]) * delta_t
    return c_u


def round_control_vector_for_online_dfl(u: np.ndarray, decimals: int = 2) -> np.ndarray:
    u_vec = np.asarray(u, dtype=float).reshape(-1)
    u_round = np.round(u_vec, int(decimals))
    u_round = np.clip(u_round, 0.0, 1.0)
    u_round[np.abs(u_round) < (10.0 ** (-int(decimals))) * 0.5] = 0.0
    return np.asarray(u_round, dtype=float)


def _linear_ramp_array(k: np.ndarray, start: int, end: int, v0: float, v1: float) -> np.ndarray:
    k_arr = np.asarray(k, dtype=float)
    start_f = float(start)
    end_f = float(end)
    if end_f <= start_f:
        return np.full(k_arr.shape, float(v0), dtype=float)
    frac = np.clip((k_arr - start_f) / (end_f - start_f), 0.0, 1.0)
    return float(v0) + frac * (float(v1) - float(v0))


def get_bounds_and_refs(param_keeper: dict, kappa: int | np.ndarray) -> dict[str, np.ndarray | float]:
    kappa_raw = np.asarray(kappa, dtype=int)
    is_scalar = kappa_raw.ndim == 0
    total_steps = int(param_keeper["kappa_day_night_total_steps"])
    kappa_arr = np.atleast_1d(kappa_raw).reshape(-1) % total_steps

    kappa_day_start = int(param_keeper["kappa_day_start"])
    kappa_day_end = int(param_keeper["kappa_day_end"])
    kappa_transition_start_1 = int(param_keeper["kappa_transition_start_1"])
    kappa_transition_end_1 = int(param_keeper["kappa_transition_end_1"])
    kappa_transition_start_2 = int(param_keeper["kappa_transition_start_2"])
    kappa_transition_end_2 = int(param_keeper["kappa_transition_end_2"])

    is_day = (kappa_arr >= kappa_day_start) & (kappa_arr < kappa_day_end)
    is_transition_1 = (kappa_arr >= kappa_transition_start_1) & (kappa_arr < kappa_transition_end_1)
    is_transition_2 = (kappa_arr >= kappa_transition_start_2) & (kappa_arr < kappa_transition_end_2)

    def _build_series(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower_day = float(param_keeper[f"{prefix}_lower_day"])
        upper_day = float(param_keeper[f"{prefix}_upper_day"])
        lower_night = float(param_keeper[f"{prefix}_lower_night"])
        upper_night = float(param_keeper[f"{prefix}_upper_night"])
        star_day = float(param_keeper[f"{prefix}_star_day"])
        star_night = float(param_keeper[f"{prefix}_star_night"])

        lower = np.where(is_day, lower_day, lower_night).astype(float)
        upper = np.where(is_day, upper_day, upper_night).astype(float)
        star = np.where(is_day, star_day, star_night).astype(float)

        lower_transition_start_1 = float(param_keeper.get(f"{prefix}_lower_transition_start_1", lower_night))
        lower_transition_end_1 = float(param_keeper.get(f"{prefix}_lower_transition_end_1", lower_day))
        upper_transition_start_1 = float(param_keeper.get(f"{prefix}_upper_transition_start_1", upper_night))
        upper_transition_end_1 = float(param_keeper.get(f"{prefix}_upper_transition_end_1", upper_day))
        lower_transition_start_2 = float(param_keeper.get(f"{prefix}_lower_transition_start_2", lower_day))
        lower_transition_end_2 = float(param_keeper.get(f"{prefix}_lower_transition_end_2", lower_night))
        upper_transition_start_2 = float(param_keeper.get(f"{prefix}_upper_transition_start_2", upper_day))
        upper_transition_end_2 = float(param_keeper.get(f"{prefix}_upper_transition_end_2", upper_night))

        if np.any(is_transition_1):
            lower[is_transition_1] = _linear_ramp_array(
                kappa_arr[is_transition_1],
                kappa_transition_start_1,
                kappa_transition_end_1,
                lower_transition_start_1,
                lower_transition_end_1,
            )
            upper[is_transition_1] = _linear_ramp_array(
                kappa_arr[is_transition_1],
                kappa_transition_start_1,
                kappa_transition_end_1,
                upper_transition_start_1,
                upper_transition_end_1,
            )
            star[is_transition_1] = _linear_ramp_array(
                kappa_arr[is_transition_1],
                kappa_transition_start_1,
                kappa_transition_end_1,
                star_night,
                star_day,
            )

        if np.any(is_transition_2):
            lower[is_transition_2] = _linear_ramp_array(
                kappa_arr[is_transition_2],
                kappa_transition_start_2,
                kappa_transition_end_2,
                lower_transition_start_2,
                lower_transition_end_2,
            )
            upper[is_transition_2] = _linear_ramp_array(
                kappa_arr[is_transition_2],
                kappa_transition_start_2,
                kappa_transition_end_2,
                upper_transition_start_2,
                upper_transition_end_2,
            )
            star[is_transition_2] = _linear_ramp_array(
                kappa_arr[is_transition_2],
                kappa_transition_start_2,
                kappa_transition_end_2,
                star_day,
                star_night,
            )

        return lower, upper, star

    T_in_lower, T_in_upper, T_in_star = _build_series("T_in")
    H_in_lower, H_in_upper, H_in_star = _build_series("H_in")
    C_in_lower, C_in_upper, C_in_star = _build_series("C_in")

    L_star_k_start = int(param_keeper["L_star_k_start"])
    L_star_k_end = int(param_keeper["L_star_k_end"])
    L_star_k_max = float(param_keeper["L_star_k_max"])
    L_star_k_slope = float(param_keeper["L_star_k_slope"])
    L_star = np.where(
        kappa_arr < L_star_k_start,
        0.0,
        np.where(
            kappa_arr >= L_star_k_end,
            L_star_k_max,
            L_star_k_slope * (kappa_arr - L_star_k_start),
        ),
    ).astype(float)

    bounds_and_refs_keeper: dict[str, np.ndarray | float] = {
        "kappa": kappa_arr.astype(int),
        "is_day": is_day,
        "is_transition_1": is_transition_1,
        "is_transition_2": is_transition_2,
        "T_in_lower": T_in_lower,
        "T_in_upper": T_in_upper,
        "T_in_star": T_in_star,
        "H_in_lower": H_in_lower,
        "H_in_upper": H_in_upper,
        "H_in_star": H_in_star,
        "C_in_lower": C_in_lower,
        "C_in_upper": C_in_upper,
        "C_in_star": C_in_star,
        "L_star": L_star,
    }
    if is_scalar:
        return {key: (float(val[0]) if isinstance(val, np.ndarray) and val.ndim == 1 and val.size == 1 else bool(val[0]) if isinstance(val, np.ndarray) and val.dtype == bool and val.size == 1 else int(val[0]) if key == "kappa" else val) for key, val in bounds_and_refs_keeper.items()}
    return bounds_and_refs_keeper


def _solve_base_mpc_and_get_u0(
    *,
    x_ini: np.ndarray,
    d_forecast: np.ndarray,
    kappa_ini: int,
    keeper_path: Path,
    matrices_path: Path | str,
    horizon_K: int,
    num_x: int,
    num_u: int,
    num_d: int,
    M: np.ndarray,
    N: np.ndarray,
    O: np.ndarray,
    m_vec: np.ndarray,
    gurobi_threads: int | None = None,
) -> tuple[np.ndarray, float, str]:
    result = build_then_solve_mpc_base(
        x_ini=x_ini,
        d_forecast=d_forecast,
        kappa_ini=kappa_ini,
        keeper_path=keeper_path,
        matrices_path=Path(matrices_path),
        horizon_K=horizon_K,
        num_x=num_x,
        num_u=num_u,
        num_d=num_d,
        M_override=M,
        N_override=N,
        O_override=O,
        m_vec_override=m_vec,
        gurobi_threads=gurobi_threads,
    )
    u0 = np.asarray(result["u"][0], dtype=float).reshape(-1)
    obj_mpc = float(result["obj"])
    solving_condition = str(result.get("termination_condition", "unknown"))
    return u0, obj_mpc, solving_condition


def _fd_n_entry_worker(task: tuple) -> tuple[int, int, np.ndarray]:
    (
        i,
        j,
        x_ini,
        d_forecast,
        kappa_ini,
        keeper_path_str,
        matrices_path_str,
        horizon_K,
        num_x,
        num_u,
        num_d,
        M_incumbent,
        N_incumbent,
        O_incumbent,
        m_vec_incumbent,
        rel_step,
        abs_step_floor,
        gurobi_threads,
    ) = task

    theta = float(N_incumbent[i, j])
    delta = max(float(abs_step_floor), float(rel_step) * max(1.0, abs(theta)))
    Np = np.asarray(N_incumbent, dtype=float).copy()
    Nm = np.asarray(N_incumbent, dtype=float).copy()
    Np[i, j] += delta
    Nm[i, j] -= delta

    up, _, _ = _solve_base_mpc_and_get_u0(
        x_ini=np.asarray(x_ini, dtype=float),
        d_forecast=np.asarray(d_forecast, dtype=float),
        kappa_ini=int(kappa_ini),
        keeper_path=Path(keeper_path_str),
        matrices_path=Path(matrices_path_str),
        horizon_K=int(horizon_K),
        num_x=int(num_x),
        num_u=int(num_u),
        num_d=int(num_d),
        M=np.asarray(M_incumbent, dtype=float),
        N=np.asarray(Np, dtype=float),
        O=np.asarray(O_incumbent, dtype=float),
        m_vec=np.asarray(m_vec_incumbent, dtype=float),
        gurobi_threads=int(gurobi_threads),
    )
    um, _, _ = _solve_base_mpc_and_get_u0(
        x_ini=np.asarray(x_ini, dtype=float),
        d_forecast=np.asarray(d_forecast, dtype=float),
        kappa_ini=int(kappa_ini),
        keeper_path=Path(keeper_path_str),
        matrices_path=Path(matrices_path_str),
        horizon_K=int(horizon_K),
        num_x=int(num_x),
        num_u=int(num_u),
        num_d=int(num_d),
        M=np.asarray(M_incumbent, dtype=float),
        N=np.asarray(Nm, dtype=float),
        O=np.asarray(O_incumbent, dtype=float),
        m_vec=np.asarray(m_vec_incumbent, dtype=float),
        gurobi_threads=int(gurobi_threads),
    )
    grad = (up - um) / (2.0 * delta)
    return int(i), int(j), np.asarray(grad, dtype=float)


def _control_cost_component_dict(u0: np.ndarray, c_u: np.ndarray, u_names: list[str]) -> dict[str, float]:
    u_vec = np.asarray(u0, dtype=float).reshape(-1)
    if u_vec.size != c_u.size or u_vec.size != len(u_names):
        raise ValueError(
            f"Control cost component size mismatch: u0={u_vec.size}, c_u={c_u.size}, u_names={len(u_names)}"
        )
    return {name: float(c_u[j] * u_vec[j]) for j, name in enumerate(u_names)}


@lru_cache(maxsize=8)
def _load_cost_surrogate_npz(npz_path_str: str) -> dict:
    npz_path = Path(npz_path_str).resolve()
    npz = np.load(npz_path, allow_pickle=True)

    if "feature_names" not in npz:
        raise KeyError(f"Missing 'feature_names' in surrogate npz: {npz_path}")
    if "cost_target_kind" not in npz:
        raise KeyError(
            f"Missing 'cost_target_kind' in surrogate npz: {npz_path}. "
            "This surrogate artifact predates the current target-tagged format. "
            "Retrain it with experiment_runner/r4_fit_J_act_DNN_phase_wise.py "
            "or experiment_runner/r4_fit_J_act_REG_phase_wise.py."
        )
    cost_target_kind = str(np.asarray(npz["cost_target_kind"]).reshape(-1)[0]).strip()
    if cost_target_kind not in SUPPORTED_COST_TARGET_KINDS:
        raise ValueError(
            f"Incompatible surrogate target in {npz_path}: expected one of "
            f"{sorted(SUPPORTED_COST_TARGET_KINDS)!r}, got {cost_target_kind!r}. "
            "Retrain the surrogate with experiment_runner/r4_fit_J_act_DNN_phase_wise.py "
            "or experiment_runner/r4_fit_J_act_REG_phase_wise.py."
        )
    feature_names = [str(s) for s in npz["feature_names"].tolist()]
    if len(feature_names) == 0:
        raise ValueError(f"Empty 'feature_names' in surrogate npz: {npz_path}")
    feature_scaler = feature_scaler_from_npz(
        npz,
        expected_feature_names=feature_names,
    )

    has_linear = ("coef" in npz) and ("intercept" in npz)
    has_model_path = ("model_torch_path" in npz)
    if not has_linear and not has_model_path:
        raise KeyError(
            f"Surrogate npz {npz_path} must contain either ('coef','intercept') "
            f"or 'model_torch_path'."
        )

    coef = np.asarray([], float)
    intercept = 0.0
    model = None
    model_torch_path = ""
    kind = ""

    if has_linear:
        coef = np.asarray(npz["coef"], float).ravel()
        intercept = float(np.asarray(npz["intercept"]).reshape(-1)[0])
        if coef.size != len(feature_names):
            raise ValueError(
                f"Linear surrogate size mismatch in {npz_path}: "
                f"coef size={coef.size}, feature_names size={len(feature_names)}"
            )
        kind = "linear_terms"
    else:
        raw_path = str(np.asarray(npz["model_torch_path"]).reshape(-1)[0]).strip()
        if not raw_path:
            raise ValueError(f"Empty model_torch_path in surrogate npz: {npz_path}")
        p = Path(raw_path)
        if not p.is_absolute():
            p = (npz_path.parent / p).resolve()
        model_torch_path = str(p)
        if not p.exists():
            raise FileNotFoundError(f"Surrogate model file not found: {p}")
        loaded = load_torch_mlp_checkpoint(p, map_location="cpu")
        model = loaded["model"]
        if model is None:
            raise ValueError(f"Loaded surrogate model is None: {p}")
        output_activation = str(loaded.get("output_activation", "linear"))
        kind = "torch_model"
    if not has_linear:
        torch_output_activation = output_activation
    else:
        torch_output_activation = "linear"

    has_x0_mu = "x0_mu" in npz
    has_x0_sd = "x0_sd" in npz
    if has_x0_mu != has_x0_sd:
        raise KeyError(f"Surrogate npz {npz_path} must contain both x0_mu and x0_sd, or neither.")
    has_d_mu = "d_mu" in npz
    has_d_sd = "d_sd" in npz
    if has_d_mu != has_d_sd:
        raise KeyError(f"Surrogate npz {npz_path} must contain both d_mu and d_sd, or neither.")

    x0_mu = np.asarray(npz["x0_mu"], float).ravel() if has_x0_mu else np.asarray([], float)
    x0_sd = np.asarray(npz["x0_sd"], float).ravel() if has_x0_sd else np.asarray([], float)
    d_mu = np.asarray(npz["d_mu"], float).ravel() if has_d_mu else np.asarray([], float)
    d_sd = np.asarray(npz["d_sd"], float).ravel() if has_d_sd else np.asarray([], float)
    if x0_mu.size != x0_sd.size:
        raise ValueError(f"x0_mu/x0_sd size mismatch in surrogate npz: {npz_path}")
    if d_mu.size != d_sd.size:
        raise ValueError(f"d_mu/d_sd size mismatch in surrogate npz: {npz_path}")

    return dict(
        kind=kind,
        coef=coef,
        intercept=intercept,
        model=model,
        model_torch_path=model_torch_path,
        torch_output_activation=torch_output_activation,
        feature_names=feature_names,
        cost_target_kind=cost_target_kind,
        x0_mu=x0_mu,
        x0_sd=x0_sd,
        d_mu=d_mu,
        d_sd=d_sd,
        feature_scaler=feature_scaler,
    )

def _compute_feature_value(term_name: str, var_values: dict[str, float]) -> float:
    term = str(term_name).strip()

    # Reject unsupported patterns early
    bad_patterns = ["*", ":", "(", ")", "+", "-", "/"]
    if any(p in term for p in bad_patterns):
        raise ValueError(
            f"Unsupported feature name syntax: {term!r}. "
            f"Only single-variable names with optional '^' power are supported. "
            f"Example: 'U_heat' or 'X_ini_T_in^2'."
        )

    if term == "":
        raise ValueError("Empty feature name encountered.")

    # Optional strictness: if you truly never use cross terms, fail if multiple tokens
    tokens = term.split()
    if len(tokens) != 1:
        raise ValueError(
            f"Unexpected multi-token feature name (cross term?): {term!r}. "
            f"You said cross terms are not used, so this is likely a bug in feature_names."
        )

    token = tokens[0]
    if "^" in token:
        base, power = token.split("^", 1)
        base = base.strip()
        power = power.strip()

        if base not in var_values:
            raise KeyError(
                f"Feature base {base!r} not found in var_values for term {term!r}. "
                f"Available keys (first 50): {list(var_values.keys())[:50]}"
            )

        base_val = float(var_values[base])
        if not np.isfinite(base_val):
            raise FloatingPointError(
                f"Non-finite var_values[{base!r}]={base_val} for term {term!r}."
            )

        try:
            p = float(power)
        except Exception as e:
            raise ValueError(f"Bad power in feature token {token!r}: {e}") from e

        val = base_val ** p
    else:
        base = token.strip()

        if base not in var_values:
            raise KeyError(
                f"Feature {base!r} not found in var_values for term {term!r}. "
                f"Available keys (first 50): {list(var_values.keys())[:50]}"
            )

        base_val = float(var_values[base])
        if not np.isfinite(base_val):
            raise FloatingPointError(
                f"Non-finite var_values[{base!r}]={base_val} for term {term!r}."
            )

        val = base_val

    if not np.isfinite(val):
        raise FloatingPointError(f"Non-finite term value for {term!r}: {val}")

    return float(val)


def _build_feature_dict(
    u0: np.ndarray,
    x_ini: np.ndarray | None,
    d0: np.ndarray | None,
    surrogate: dict,
    u_names: list[str],
    x0_names: list[str] | None,
    d_names: list[str] | None,
    kappa_k: float | int | None = None,
    param_keeper: dict | None = None,
) -> dict[str, float]:
    """
    Build var_values using ONLY the provided official names.
    No aliases are generated.
    """
    u_vec = np.asarray(u0, float).reshape(-1)
    x_vec = np.asarray(x_ini, float).reshape(-1) if x_ini is not None else np.asarray([], float)
    d_vec = np.asarray(d0, float).reshape(-1) if d0 is not None else np.asarray([], float)

    # ----- standardization (keep same behavior) -----
    x0_mu = np.asarray(surrogate.get("x0_mu", np.asarray([], float)), float).ravel()
    x0_sd = np.asarray(surrogate.get("x0_sd", np.asarray([], float)), float).ravel()
    d_mu = np.asarray(surrogate.get("d_mu", np.asarray([], float)), float).ravel()
    d_sd = np.asarray(surrogate.get("d_sd", np.asarray([], float)), float).ravel()

    x_use = x_vec.copy()
    if x_vec.size > 0:
        if x0_mu.size == 0 and x0_sd.size == 0:
            pass
        elif x0_mu.size == x_vec.size and x0_sd.size == x_vec.size:
            if np.any(x0_sd == 0.0):
                raise ValueError("x0_sd contains zero(s); cannot standardize x_ini safely.")
            x_use = (x_vec - x0_mu) / x0_sd
        else:
            raise ValueError(
                f"x0 standardization size mismatch: x_ini={x_vec.size}, x0_mu={x0_mu.size}, x0_sd={x0_sd.size}"
            )

    d_use = d_vec.copy()
    if d_vec.size > 0:
        if d_mu.size == 0 and d_sd.size == 0:
            pass
        elif d_mu.size == d_vec.size and d_sd.size == d_vec.size:
            if np.any(d_sd == 0.0):
                raise ValueError("d_sd contains zero(s); cannot standardize disturbance safely.")
            d_use = (d_vec - d_mu) / d_sd
        else:
            raise ValueError(
                f"d standardization size mismatch: d0={d_vec.size}, d_mu={d_mu.size}, d_sd={d_sd.size}"
            )

    # ----- strict naming checks -----
    if len(u_names) != u_vec.size:
        raise ValueError(f"u_names length {len(u_names)} must match u0 length {u_vec.size}.")
    if x0_names is not None and len(x0_names) != x_use.size:
        raise ValueError(f"x0_names length {len(x0_names)} must match x_ini length {x_use.size}.")
    if d_names is not None and len(d_names) != d_use.size:
        raise ValueError(f"d_names length {len(d_names)} must match d0 length {d_use.size}.")

    # ----- ONLY official keys -----
    var_values: dict[str, float] = {}

    def _add_key(key: str, val: float, src: str) -> None:
        k = str(key).strip()
        if not k:
            raise ValueError(f"Empty feature key from {src}.")
        if k in var_values:
            raise ValueError(f"Duplicate feature key {k!r} from {src}.")
        if not np.isfinite(val):
            raise FloatingPointError(f"Non-finite value for key {k!r} from {src}: {val}")
        var_values[k] = float(val)

    # controls
    for j, nm in enumerate(u_names):
        _add_key(nm, float(u_vec[j]), src=f"u_names[{j}]")

    # x0
    if x0_names is not None:
        for i, nm in enumerate(x0_names):
            _add_key(nm, float(x_use[i]), src=f"x0_names[{i}]")

    # d0
    if d_names is not None:
        for i, nm in enumerate(d_names):
            _add_key(nm, float(d_use[i]), src=f"d_names[{i}]")

    # extra scalar feature used by your model
    if kappa_k is not None:
        _add_key("kappa_k", float(kappa_k), src="kappa_k")
        if param_keeper is not None:
            climate_refs = get_bounds_and_refs(param_keeper, int(kappa_k))
            _add_key("T_in_star", float(climate_refs["T_in_star"]), src="T_in_star")
            _add_key("H_in_star", float(climate_refs["H_in_star"]), src="H_in_star")
            _add_key("C_in_star", float(climate_refs["C_in_star"]), src="C_in_star")
            _add_key("L_star", float(climate_refs["L_star"]), src="L_star")

    return var_values


def _predict_surrogate_cost(
    u0: np.ndarray,
    x_ini: np.ndarray | None,
    d0: np.ndarray | None,
    surrogate: dict,
    u_names: list[str],
    x0_names: list[str] | None,
    d_names: list[str] | None,
    kappa_k: float | int | None = None,
    param_keeper: dict | None = None,
) -> float:
    feature_names = [str(s) for s in surrogate.get("feature_names", [])]
    if len(feature_names) == 0:
        raise RuntimeError("Surrogate has empty feature_names.")

    var_values = _build_feature_dict(
        u0=u0,
        x_ini=x_ini,
        d0=d0,
        surrogate=surrogate,
        u_names=u_names,
        x0_names=x0_names,
        d_names=d_names,
        kappa_k=kappa_k,
        param_keeper=param_keeper,
    )

    # HARD check: all features must exist (no aliases allowed)
    missing = [nm for nm in feature_names if nm.split("^", 1)[0].strip() not in var_values]
    if missing:
        raise KeyError(
            "Surrogate feature_names contain keys not provided by var_values (no-alias mode).\n"
            f"Missing (first 50): {missing[:50]}\n"
            f"Available keys: {sorted(var_values.keys())}"
        )

    z = np.asarray([_compute_feature_value(nm, var_values) for nm in feature_names], dtype=float)
    if not np.all(np.isfinite(z)):
        raise FloatingPointError("Non-finite surrogate feature vector z.")
    feature_scaler = surrogate.get("feature_scaler", None)
    if feature_scaler is not None:
        z = transform_surrogate_feature_vector(
            z,
            feature_scaler,
            feature_names=feature_names,
        )
        if not np.all(np.isfinite(z)):
            raise FloatingPointError("Non-finite scaled surrogate feature vector z.")

    kind = str(surrogate.get("kind", ""))
    if kind == "linear_terms":
        coef = np.asarray(surrogate.get("coef", np.asarray([], float)), float).ravel()
        intercept = float(surrogate.get("intercept", 0.0))
        if coef.size != z.size:
            raise ValueError(f"Linear surrogate size mismatch: coef size={coef.size}, feature size={z.size}")
        y_lin = float(np.dot(coef, z) + intercept)
        if not np.isfinite(y_lin):
            raise FloatingPointError("Linear surrogate produced non-finite prediction.")
        return y_lin

    if kind == "torch_model":
        model = surrogate.get("model", None)
        if model is None:
            raise RuntimeError("Surrogate kind=torch_model but no model object was loaded.")
        y = predict_torch_mlp(model, z.reshape(1, -1), device="cpu").reshape(-1)
        if y.size == 0:
            raise RuntimeError("Torch surrogate prediction returned empty output.")
        if not np.isfinite(y[0]):
            raise FloatingPointError(f"Surrogate model produced non-finite prediction: {y[0]}")
        return float(y[0])

    raise ValueError(f"Unknown surrogate kind: {kind!r}")


def _get_d_predicted_actual_cost_d_u0(
    u0: np.ndarray,
    x_ini: np.ndarray | None,
    d0: np.ndarray | None,
    surrogate: dict,
    u_names: list[str],
    x0_names: list[str] | None,
    d_names: list[str] | None,
    kappa_k: float | int | None = None,
    param_keeper: dict | None = None,
    rel_step: float = 1e-3,
    abs_step_floor: float = 1e-7,
) -> tuple[float, np.ndarray]:
    u_vec = np.asarray(u0, float).reshape(-1)
    if len(u_names) != u_vec.size:
        raise ValueError(f"u_names length {len(u_names)} must match u0 length {u_vec.size}.")

    y_hat = _predict_surrogate_cost(
        u0=u_vec,
        x_ini=x_ini,
        d0=d0,
        surrogate=surrogate,
        u_names=u_names,
        x0_names=x0_names,
        d_names=d_names,
        kappa_k=kappa_k,
        param_keeper=param_keeper,
    )

    grad_u = np.zeros(u_vec.size, dtype=float)
    for j in range(u_vec.size):
        delta = max(float(abs_step_floor), float(rel_step) * max(1.0, abs(float(u_vec[j]))))
        up = u_vec.copy(); um = u_vec.copy()
        up[j] += delta
        um[j] -= delta

        yp = _predict_surrogate_cost(
            u0=up,
            x_ini=x_ini,
            d0=d0,
            surrogate=surrogate,
            u_names=u_names,
            x0_names=x0_names,
            d_names=d_names,
            kappa_k=kappa_k,
            param_keeper=param_keeper,
        )
        ym = _predict_surrogate_cost(
            u0=um,
            x_ini=x_ini,
            d0=d0,
            surrogate=surrogate,
            u_names=u_names,
            x0_names=x0_names,
            d_names=d_names,
            kappa_k=kappa_k,
            param_keeper=param_keeper,
        )

        if not np.isfinite(yp) or not np.isfinite(ym):
            raise FloatingPointError(
                f"Non-finite surrogate prediction during FD gradient at index {j}: yp={yp}, ym={ym}"
            )
        grad_u[j] = float((yp - ym) / (2.0 * delta))

    if not np.isfinite(y_hat):
        raise FloatingPointError("Surrogate y_hat is non-finite.")
    if not np.all(np.isfinite(grad_u)):
        raise FloatingPointError("Surrogate gradient contains non-finite values.")
    return float(y_hat), grad_u


def solve_mpc_base_and_fd_sensitivities(
    x_ini: np.ndarray,
    d_forecast: np.ndarray,
    kappa_ini: int,
    keeper_path: Path,
    matrices_path: Path,
    horizon_K: int,
    num_x: int,
    num_u: int,
    num_d: int,
    # --- step size control ---
    rel_step: float = 1e-5,
    abs_step_floor: float = 1e-7,
    # --- what part of u to return sensitivities for ---
    which_u: str = "u0",  # keep this arg, but only "u0" is supported here
    cost_surrogate_npz_path: str | Path | None = None,
    cost_u_names: list[str] | None = None,
    cost_x0_names: list[str] | None = None,
    cost_d_names: list[str] | None = None,
    d0_for_cost: np.ndarray | None = None,
    # NEW: control which FD blocks run
    fd_M: bool = True,
    fd_N: bool = True,
    fd_O: bool = True,
    fd_m_vec: bool = True,
) -> dict:
    """
    QP MPC baseline solve + full central-difference sensitivities of u0 wrt (M, N, O, m_vec).

    Central difference:
        du0/dtheta ≈ (u0(theta+δ) - u0(theta-δ)) / (2δ)

    Returns full tensors:
        du0_dM: (num_u, num_x, num_x)
        du0_dN: (num_u, num_x, num_u)
        du0_dO: (num_u, num_x, num_d)
        du0_dm_vec: (num_u, num_x)

    NOTE: This runs MANY solves: 1 + 2*(num_x*num_x + num_x*num_u + num_x*num_d + num_x).
          For (4,10,4) it is 1 + 2*(16+40+16+4)=153 solves.
    """

    if which_u != "u0":
        raise ValueError("This function computes sensitivities for u0 only. Set which_u='u0'.")

    K = int(horizon_K)

    # -------- load base matrices (no overrides) --------
    M_incumbent, N_incumbent, O_incumbent, m_vec_incumbent = _load_dynamics_matrices(str(Path(matrices_path).resolve()))

    # -------- shared data --------
    d_forecast_full = np.asarray(d_forecast, dtype=float)
    d = d_forecast_full[:K]

    param_keeper = load_parameters(keeper_path)
    meta = _get_vector_meta(param_keeper)
    _validate_problem_sizes(num_x, num_u, num_d, meta)
    x_ini = np.asarray(x_ini, dtype=float).reshape(-1)
    if x_ini.size != num_x:
        raise ValueError(f"x_ini length mismatch: expected {num_x}, got {x_ini.size}")
    if d.ndim != 2 or d.shape[1] != num_d:
        raise ValueError(f"d_forecast shape mismatch: expected (*, {num_d}), got {d.shape}")

    x_names = list(meta["x_names"])
    d_names = list(meta["d_names"])
    u_names = list(meta["u_names"])
    kappa_day_night_total_steps = int(param_keeper["kappa_day_night_total_steps"])

    def _step(theta: float) -> float:
        return max(abs_step_floor, rel_step * max(1.0, abs(theta)))

    # -------- inner solve function (same MPC as the base pipeline) --------
    def _solve_mpc_and_get_u0(M: np.ndarray, N: np.ndarray, O: np.ndarray, m_vec: np.ndarray) -> tuple[np.ndarray, float, str]:
        return _solve_base_mpc_and_get_u0(
            x_ini=x_ini,
            d_forecast=d,
            kappa_ini=kappa_ini,
            keeper_path=keeper_path,
            matrices_path=matrices_path,
            horizon_K=K,
            num_x=num_x,
            num_u=num_u,
            num_d=num_d,
            M=M,
            N=N,
            O=O,
            m_vec=m_vec,
            gurobi_threads=None,
        )

    # -------- baseline solve --------
    u0_raw, obj_mpc, solving_condition = _solve_mpc_and_get_u0(
        M_incumbent,
        N_incumbent,
        O_incumbent,
        m_vec_incumbent,
    )
    u0 = round_control_vector_for_online_dfl(u0_raw)

    # Allocate full sensitivity tensors
    du0_dM = np.zeros((num_u, num_x, num_x), dtype=float)
    du0_dN = np.zeros((num_u, num_x, num_u), dtype=float)
    du0_dO = np.zeros((num_u, num_x, num_d), dtype=float)
    du0_dm_vec = np.zeros((num_u, num_x), dtype=float)

    # -------- full sensitivities wrt M --------
    if fd_M:
        for i in range(num_x):
            for j in range(num_x):
                delta = _step(float(M_incumbent[i, j]))
                Mp = M_incumbent.copy(); Mn = M_incumbent.copy()
                Mp[i, j] += delta
                Mn[i, j] -= delta
                up, obj_p, term_p = _solve_mpc_and_get_u0(Mp, N_incumbent, O_incumbent, m_vec_incumbent)
                um, obj_m, term_m = _solve_mpc_and_get_u0(Mn, N_incumbent, O_incumbent, m_vec_incumbent)
                du0_dM[:, i, j] = (up - um) / (2.0 * delta)

    # -------- full sensitivities wrt N --------
    if fd_N:
        n_tasks = [(i, j) for i in range(num_x) for j in range(num_u)]
        max_workers = _get_fd_n_parallel_max_workers(len(n_tasks))
        worker_gurobi_threads = _get_fd_worker_gurobi_threads()
        if max_workers <= 1 or len(n_tasks) <= 1:
            for i, j in n_tasks:
                delta = _step(float(N_incumbent[i, j]))
                Np = N_incumbent.copy(); Nm = N_incumbent.copy()
                Np[i, j] += delta
                Nm[i, j] -= delta
                up, obj_p, term_p = _solve_mpc_and_get_u0(M_incumbent, Np, O_incumbent, m_vec_incumbent)
                um, obj_m, term_m = _solve_mpc_and_get_u0(M_incumbent, Nm, O_incumbent, m_vec_incumbent)
                du0_dN[:, i, j] = (up - um) / (2.0 * delta)
        else:
            tasks = [
                (
                    i,
                    j,
                    x_ini,
                    d,
                    int(kappa_ini),
                    str(Path(keeper_path).resolve()),
                    str(Path(matrices_path).resolve()),
                    K,
                    num_x,
                    num_u,
                    num_d,
                    M_incumbent,
                    N_incumbent,
                    O_incumbent,
                    m_vec_incumbent,
                    float(rel_step),
                    float(abs_step_floor),
                    int(worker_gurobi_threads),
                )
                for i, j in n_tasks
            ]
            mp_context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
                for i, j, grad in executor.map(_fd_n_entry_worker, tasks, chunksize=1):
                    du0_dN[:, i, j] = grad

    # -------- full sensitivities wrt O --------
    if fd_O:
        for i in range(num_x):
            for j in range(num_d):
                delta = _step(float(O_incumbent[i, j]))
                Op = O_incumbent.copy(); Om = O_incumbent.copy()
                Op[i, j] += delta
                Om[i, j] -= delta
                up, obj_p, term_p = _solve_mpc_and_get_u0(M_incumbent, N_incumbent, Op, m_vec_incumbent)
                um, obj_m, term_m = _solve_mpc_and_get_u0(M_incumbent, N_incumbent, Om, m_vec_incumbent)
                du0_dO[:, i, j] = (up - um) / (2.0 * delta)

    # -------- full sensitivities wrt m --------
    if fd_m_vec:
        for i in range(num_x):
            delta = _step(float(m_vec_incumbent[i]))
            m_plus = m_vec_incumbent.copy(); m_minus = m_vec_incumbent.copy()
            m_plus[i] += delta
            m_minus[i] -= delta
            up, obj_p, term_p = _solve_mpc_and_get_u0(M_incumbent, N_incumbent, O_incumbent, m_plus)
            um, obj_m, term_m = _solve_mpc_and_get_u0(M_incumbent, N_incumbent, O_incumbent, m_minus)
            du0_dm_vec[:, i] = (up - um) / (2.0 * delta)

    one_step_cost_pre = None
    d_cost_du0 = None
    d0_use = None

    if cost_surrogate_npz_path is not None:
        surrogate = _load_cost_surrogate_npz(str(cost_surrogate_npz_path))

        if cost_u_names is None:
            cost_u_names = list(u_names)
        if cost_x0_names is None:
            cost_x0_names = [f"X_ini_{name}" for name in x_names]
        if cost_d_names is None:
            cost_d_names = [f"D_{name}" for name in d_names]

        d0_use = np.asarray(
            d0_for_cost if d0_for_cost is not None else d_forecast_full[0],
            float,
        ).reshape(-1)
        if d0_use.size != num_d:
            raise ValueError(f"d0_for_cost length mismatch: expected {num_d}, got {d0_use.size}")
        x_ini_use = np.asarray(x_ini, float).reshape(-1)

        # kappa feature used by your surrogate (use current step's kappa)
        kp0 = int(kappa_ini) % int(kappa_day_night_total_steps)

        one_step_cost_pre, d_cost_du0 = _get_d_predicted_actual_cost_d_u0(
            u0=u0,
            x_ini=x_ini_use,
            d0=d0_use,
            surrogate=surrogate,
            u_names=cost_u_names,
            x0_names=cost_x0_names,
            d_names=cost_d_names,
            kappa_k=kp0,
            param_keeper=param_keeper,
            rel_step=rel_step,
            abs_step_floor=abs_step_floor,
        )

        d_cost_du0 = np.asarray(d_cost_du0, dtype=float).reshape(-1)
    return {
        "u0": u0,
        "obj_mpc": float(obj_mpc),
        "solving_condition": solving_condition,
        "du0_dM": du0_dM,
        "du0_dN": du0_dN,
        "du0_dO": du0_dO,
        "du0_dm_vec": du0_dm_vec,
        "one_step_cost_pre": one_step_cost_pre,
        "d_cost_du0": d_cost_du0,
        "d0_for_cost_used": d0_use if cost_surrogate_npz_path is not None else None,
        "rel_step": float(rel_step),
        "abs_step_floor": float(abs_step_floor),
        "which_u": which_u,
    }


def build_then_solve_mpc_base(
    x_ini: np.ndarray,
    d_forecast: np.ndarray,
    kappa_ini: int,
    keeper_path: Path,
    matrices_path: Path,
    horizon_K: int,
    num_x: int,
    num_u: int,
    num_d: int,
    M_override: np.ndarray | None = None,
    N_override: np.ndarray | None = None,
    O_override: np.ndarray | None = None,
    m_vec_override: np.ndarray | None = None,
    gurobi_threads: int | None = None,
) -> dict:

    # Confirm total steps in this MPC-- follow horizon_K
    K = int(horizon_K)

    # Get disturbance vector
    d_forecast_full = np.asarray(d_forecast, dtype=float)
    d = d_forecast_full[:K]

    # Get matrices
    if any(v is not None for v in (M_override, N_override, O_override, m_vec_override)):
        if not all(v is not None for v in (M_override, N_override, O_override, m_vec_override)):
            raise ValueError("M_override, N_override, O_override, and m_vec_override must be provided together.")
        M = np.asarray(M_override, dtype=float).copy()
        N = np.asarray(N_override, dtype=float).copy()
        O = np.asarray(O_override, dtype=float).copy()
        m_vec = np.asarray(m_vec_override, dtype=float).reshape(-1).copy()
    else:
        M, N, O, m_vec = _load_dynamics_matrices(str(Path(matrices_path).resolve()))

    # Get parameters
    param_keeper = load_parameters(keeper_path)
    meta = _get_vector_meta(param_keeper)
    _validate_problem_sizes(num_x, num_u, num_d, meta)
    x_ini = np.asarray(x_ini, dtype=float).reshape(-1)

    x_idx = meta["x_idx"]
    u_idx = meta["u_idx"]
    T_IN_IDX = x_idx["T_in"]
    H_IN_IDX = x_idx["H_in"]
    C_IN_IDX = x_idx["C_in"]
    L_IDX = x_idx["L"]
    WARM_IDX = u_idx["U_warm"]

    climate = get_bounds_and_refs(param_keeper, int(kappa_ini) + np.arange(K, dtype=int))
    kappa = np.asarray(climate["kappa"], dtype=int)
    is_day = np.asarray(climate["is_day"], dtype=bool)
    T_in_lower_k = np.asarray(climate["T_in_lower"], dtype=float)
    T_in_upper_k = np.asarray(climate["T_in_upper"], dtype=float)
    T_in_star_k = np.asarray(climate["T_in_star"], dtype=float)
    H_in_lower_k = np.asarray(climate["H_in_lower"], dtype=float)
    H_in_upper_k = np.asarray(climate["H_in_upper"], dtype=float)
    H_in_star_k = np.asarray(climate["H_in_star"], dtype=float)
    C_in_lower_k = np.asarray(climate["C_in_lower"], dtype=float)
    C_in_upper_k = np.asarray(climate["C_in_upper"], dtype=float)
    C_in_star_k = np.asarray(climate["C_in_star"], dtype=float)
    L_star = np.asarray(climate["L_star"], dtype=float)

    gamma = float(param_keeper["gamma"])
    c_u = _build_control_cost_vector(param_keeper, u_idx)
    lambda_T_pos = float(param_keeper["lambda_T_pos"])
    lambda_T_neg = float(param_keeper["lambda_T_neg"])
    lambda_H_pos = float(param_keeper["lambda_H_pos"])
    lambda_H_neg = float(param_keeper["lambda_H_neg"])
    lambda_C_pos = float(param_keeper["lambda_C_pos"])
    lambda_C_neg = float(param_keeper["lambda_C_neg"])
    lambda_L = float(param_keeper["lambda_L"])

    # Build model
    mpc_base_model = pyo.ConcreteModel()

    # Define the index sets (the “axes” of variables)
    mpc_base_model.index_steps = pyo.RangeSet(0, K - 1)         # Index set of control steps  u_0, u_1, ..., u_K-1
    mpc_base_model.index_states = pyo.RangeSet(0, K)            # Index set of all state vectors x_0, x_1, ..., x_K
    mpc_base_model.index_com_of_x = pyo.RangeSet(0, num_x - 1)  # Index set of components in one state vector x
    mpc_base_model.index_com_of_u = pyo.RangeSet(0, num_u - 1)  # Index set of components in one state vector u

    # Define variables
    mpc_base_model.var_x = pyo.Var(mpc_base_model.index_states, mpc_base_model.index_com_of_x)
    mpc_base_model.var_u = pyo.Var(mpc_base_model.index_steps, mpc_base_model.index_com_of_u, domain=pyo.NonNegativeReals)
    mpc_base_model.var_S_T_pos = pyo.Var(mpc_base_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_base_model.var_S_T_neg = pyo.Var(mpc_base_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_base_model.var_S_H_pos = pyo.Var(mpc_base_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_base_model.var_S_H_neg = pyo.Var(mpc_base_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_base_model.var_S_C_pos = pyo.Var(mpc_base_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_base_model.var_S_C_neg = pyo.Var(mpc_base_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_base_model.var_S_L = pyo.Var(mpc_base_model.index_steps, domain=pyo.NonNegativeReals)
    WARM_MAX_DAY = 0.0

    def _warm_day_cap_rule(model, k):
        if bool(is_day[k]):
            return model.var_u[k, WARM_IDX] <= WARM_MAX_DAY
        return pyo.Constraint.Skip

    mpc_base_model.con_warm_day_cap = pyo.Constraint(mpc_base_model.index_steps, rule=_warm_day_cap_rule)

    # Define constraints: Initial state
    mpc_base_model.con_x_ini = pyo.Constraint(mpc_base_model.index_com_of_x,
                                              rule=lambda model, i: model.var_x[0, i] == float(x_ini[i]))

    # Define constraints: State-space model
    mpc_base_model.con_dyn = pyo.Constraint(
        mpc_base_model.index_steps,
        mpc_base_model.index_com_of_x,
        rule=lambda model, k, i: (
                model.var_x[k + 1, i]
                == model.var_x[k, i]
                + sum(M[i, j] * model.var_x[k, j] for j in range(num_x))
                + sum(N[i, j] * model.var_u[k, j] for j in range(num_u))
                + sum(O[i, j] * d[k, j] for j in range(num_d))
                + float(m_vec[i])
        ),
    )

    #######
    # Climate security: Bounds-based
    # mpc_base_model.con_T_low = pyo.Constraint(mpc_base_model.index_steps,
    #                                           rule=lambda model, k: model.var_x[k, T_IN_IDX]  >= T_in_lower_k[k] - model.var_S_T_neg[k])
    # mpc_base_model.con_T_high = pyo.Constraint(mpc_base_model.index_steps,
    #                                            rule=lambda model, k: model.var_x[k, T_IN_IDX] <= T_in_upper_k[k] + model.var_S_T_pos[k])
    #
    # mpc_base_model.con_H_low = pyo.Constraint(mpc_base_model.index_steps,
    #                                           rule=lambda model, k: model.var_x[k, H_IN_IDX]  >= H_in_lower_k[k] - model.var_S_H_neg[k])
    # mpc_base_model.con_H_high = pyo.Constraint(mpc_base_model.index_steps,
    #                                            rule=lambda model, k: model.var_x[k, H_IN_IDX] <= H_in_upper_k[k] + model.var_S_H_pos[k])
    #
    # mpc_base_model.con_C_low = pyo.Constraint(mpc_base_model.index_steps,
    #                                           rule=lambda model, k: model.var_x[k, C_IN_IDX]  >= C_in_lower_k[k] - model.var_S_C_neg[k])
    # mpc_base_model.con_C_high = pyo.Constraint(mpc_base_model.index_steps,
    #                                            rule=lambda model, k: model.var_x[k, C_IN_IDX] <= C_in_upper_k[k] + model.var_S_C_pos[k])

    # Climate security: Reference-based
    mpc_base_model.con_T_in_ref = pyo.Constraint(mpc_base_model.index_steps,
                                                 rule=lambda model, k: model.var_x[k, T_IN_IDX] == T_in_star_k[k] + model.var_S_T_pos[k] - model.var_S_T_neg[k])

    mpc_base_model.con_H_in_ref = pyo.Constraint(mpc_base_model.index_steps,
                                                 rule=lambda model, k: model.var_x[k, H_IN_IDX] == H_in_star_k[k] + model.var_S_H_pos[k] - model.var_S_H_neg[k])

    mpc_base_model.con_C_in_ref = pyo.Constraint(mpc_base_model.index_steps,
                                                 rule=lambda model, k: model.var_x[k, C_IN_IDX] == C_in_star_k[k] + model.var_S_C_pos[k] - model.var_S_C_neg[k])

    # DLI tracking
    mpc_base_model.con_L_shortage = pyo.Constraint(mpc_base_model.index_steps, rule=lambda model, k: model.var_x[k, L_IDX] >= L_star[k] - model.var_S_L[k])

    # Define constraints: Gates
    mpc_base_model.con_u_leq_1 = pyo.Constraint(mpc_base_model.index_steps, mpc_base_model.index_com_of_u,
                                                rule=lambda model, k, j: model.var_u[k, j] <= 1)

    mpc_base_model.obj = pyo.Objective(
        expr=sum(
            (gamma**k)
            * (
                    sum(c_u[j] * mpc_base_model.var_u[k, j] for j in range(num_u))
                    + lambda_T_pos * mpc_base_model.var_S_T_pos[k] * mpc_base_model.var_S_T_pos[k]
                    + lambda_T_neg * mpc_base_model.var_S_T_neg[k] * mpc_base_model.var_S_T_neg[k]
                    + lambda_H_pos * mpc_base_model.var_S_H_pos[k] * mpc_base_model.var_S_H_pos[k]
                    + lambda_H_neg * mpc_base_model.var_S_H_neg[k] * mpc_base_model.var_S_H_neg[k]
                    + lambda_C_pos * mpc_base_model.var_S_C_pos[k] * mpc_base_model.var_S_C_pos[k]
                    + lambda_C_neg * mpc_base_model.var_S_C_neg[k] * mpc_base_model.var_S_C_neg[k]
                    + lambda_L * mpc_base_model.var_S_L[k] * mpc_base_model.var_S_L[k]
            )
            for k in mpc_base_model.index_steps
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("gurobi")
    if gurobi_threads is not None:
        threads = int(gurobi_threads)
        if threads < 1:
            raise ValueError(f"gurobi_threads must be >= 1 when provided; got {threads}")
        solver.options["Threads"] = threads
    result = solver.solve(mpc_base_model, tee=False)
    term = str(result.solver.termination_condition)

    x_sol = np.array([[pyo.value(mpc_base_model.var_x[k, i]) for i in range(num_x)] for k in range(K + 1)], dtype=float)
    u_sol = np.array([[pyo.value(mpc_base_model.var_u[k, j]) for j in range(num_u)] for k in range(K)], dtype=float)
    slacks = {
        "var_S_T_pos": np.array([pyo.value(mpc_base_model.var_S_T_pos[k]) for k in range(K)], dtype=float),
        "var_S_T_neg": np.array([pyo.value(mpc_base_model.var_S_T_neg[k]) for k in range(K)], dtype=float),
        "var_S_H_pos": np.array([pyo.value(mpc_base_model.var_S_H_pos[k]) for k in range(K)], dtype=float),
        "var_S_H_neg": np.array([pyo.value(mpc_base_model.var_S_H_neg[k]) for k in range(K)], dtype=float),
        "var_S_C_pos": np.array([pyo.value(mpc_base_model.var_S_C_pos[k]) for k in range(K)], dtype=float),
        "var_S_C_neg": np.array([pyo.value(mpc_base_model.var_S_C_neg[k]) for k in range(K)], dtype=float),
        "var_S_L": np.array([pyo.value(mpc_base_model.var_S_L[k]) for k in range(K)], dtype=float),
    }
    return {
        "x": x_sol,
        "u": u_sol,
        "slacks": slacks,
        "obj": float(pyo.value(mpc_base_model.obj)),
        "termination_condition": term,
        "kappa": kappa.astype(int),
    }


def build_then_solve_mpc_robust(
    x_ini: np.ndarray,
    d_forecast: np.ndarray,
    kappa_ini: int,
    keeper_path: Path,
    matrices_path: Path,
    horizon_K: int,
    num_x: int,
    num_u: int,
    num_d: int,
) -> dict:

    # Confirm total steps in this MPC-- follow horizon_K
    K = int(horizon_K)

    # Get disturbance vector
    d_forecast_full = np.asarray(d_forecast, dtype=float)
    d = d_forecast_full[:K]

    # Get matrices
    M, N, O, m_vec = _load_dynamics_matrices(str(Path(matrices_path).resolve()))

    # Get parameters
    param_keeper = load_parameters(keeper_path)
    meta = _get_vector_meta(param_keeper)
    _validate_problem_sizes(num_x, num_u, num_d, meta)
    x_ini = np.asarray(x_ini, dtype=float).reshape(-1)

    x_idx = meta["x_idx"]
    u_idx = meta["u_idx"]
    T_IN_IDX = x_idx["T_in"]
    H_IN_IDX = x_idx["H_in"]
    C_IN_IDX = x_idx["C_in"]
    L_IDX = x_idx["L"]
    WARM_IDX = u_idx["U_warm"]

    climate_bounds_and_refs = get_bounds_and_refs(param_keeper, int(kappa_ini) + np.arange(K, dtype=int))
    kappa = np.asarray(climate_bounds_and_refs["kappa"], dtype=int)
    is_day = np.asarray(climate_bounds_and_refs["is_day"], dtype=bool)
    T_in_lower_k = np.asarray(climate_bounds_and_refs["T_in_lower"], dtype=float)
    T_in_upper_k = np.asarray(climate_bounds_and_refs["T_in_upper"], dtype=float)
    T_in_star_k = np.asarray(climate_bounds_and_refs["T_in_star"], dtype=float)
    H_in_lower_k = np.asarray(climate_bounds_and_refs["H_in_lower"], dtype=float)
    H_in_upper_k = np.asarray(climate_bounds_and_refs["H_in_upper"], dtype=float)
    H_in_star_k = np.asarray(climate_bounds_and_refs["H_in_star"], dtype=float)
    C_in_lower_k = np.asarray(climate_bounds_and_refs["C_in_lower"], dtype=float)
    C_in_upper_k = np.asarray(climate_bounds_and_refs["C_in_upper"], dtype=float)
    C_in_star_k = np.asarray(climate_bounds_and_refs["C_in_star"], dtype=float)
    L_star = np.asarray(climate_bounds_and_refs["L_star"], dtype=float)


    T_in_ref_corridor_neg_k = 0.2 * (T_in_star_k - T_in_lower_k)
    T_in_ref_corridor_pos_k = 0.2 * (T_in_upper_k - T_in_star_k)
    H_in_ref_corridor_neg_k = 0.2 * (H_in_star_k - H_in_lower_k)
    H_in_ref_corridor_pos_k = 0.2 * (H_in_upper_k - H_in_star_k)
    C_in_ref_corridor_neg_k = 0.2 * (C_in_star_k - C_in_lower_k)
    C_in_ref_corridor_pos_k = 0.2 * (C_in_upper_k - C_in_star_k)

    gamma = float(param_keeper["gamma"])
    c_u = _build_control_cost_vector(param_keeper, u_idx)
    lambda_T_pos = float(param_keeper["lambda_T_pos"])
    lambda_T_neg = float(param_keeper["lambda_T_neg"])
    lambda_H_pos = float(param_keeper["lambda_H_pos"])
    lambda_H_neg = float(param_keeper["lambda_H_neg"])
    lambda_C_pos = float(param_keeper["lambda_C_pos"])
    lambda_C_neg = float(param_keeper["lambda_C_neg"])
    lambda_L = float(param_keeper["lambda_L"])

    # Build model
    mpc_robust_model = pyo.ConcreteModel()

    # Define the index sets (the “axes” of variables)
    mpc_robust_model.index_steps = pyo.RangeSet(0, K - 1)         # Index set of control steps  u_0, u_1, ..., u_K-1
    mpc_robust_model.index_states = pyo.RangeSet(0, K)            # Index set of all state vectors x_0, x_1, ..., x_K
    mpc_robust_model.index_com_of_x = pyo.RangeSet(0, num_x - 1)  # Index set of components in one state vector x
    mpc_robust_model.index_com_of_u = pyo.RangeSet(0, num_u - 1)  # Index set of components in one state vector u

    # Define variables
    mpc_robust_model.var_x = pyo.Var(mpc_robust_model.index_states, mpc_robust_model.index_com_of_x)
    mpc_robust_model.var_u = pyo.Var(mpc_robust_model.index_steps, mpc_robust_model.index_com_of_u, domain=pyo.NonNegativeReals)
    mpc_robust_model.var_S_T_pos = pyo.Var(mpc_robust_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_robust_model.var_S_T_neg = pyo.Var(mpc_robust_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_robust_model.var_S_H_pos = pyo.Var(mpc_robust_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_robust_model.var_S_H_neg = pyo.Var(mpc_robust_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_robust_model.var_S_C_pos = pyo.Var(mpc_robust_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_robust_model.var_S_C_neg = pyo.Var(mpc_robust_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_robust_model.var_S_L = pyo.Var(mpc_robust_model.index_steps, domain=pyo.NonNegativeReals)
    U_WARM_MAX_DAYTIME = 0.0

    def _U_warm_daytime_cap_rule(model, k):
        if bool(is_day[k]):
            return model.var_u[k, WARM_IDX] <= U_WARM_MAX_DAYTIME
        return pyo.Constraint.Skip

    mpc_robust_model.con_warm_day_cap = pyo.Constraint(mpc_robust_model.index_steps, rule=_U_warm_daytime_cap_rule)

    # Define constraints: Initial state
    mpc_robust_model.con_x_ini = pyo.Constraint(mpc_robust_model.index_com_of_x,
                                                rule=lambda model, i: model.var_x[0, i] == float(x_ini[i]))

    # Define constraints: State-space model
    mpc_robust_model.con_dyn = pyo.Constraint(
        mpc_robust_model.index_steps,
        mpc_robust_model.index_com_of_x,
        rule=lambda model, k, i: (
                model.var_x[k + 1, i]
                == model.var_x[k, i]
                + sum(M[i, j] * model.var_x[k, j] for j in range(num_x))
                + sum(N[i, j] * model.var_u[k, j] for j in range(num_u))
                + sum(O[i, j] * d[k, j] for j in range(num_d))
                + float(m_vec[i])
        ),
    )

    # Climate security: Bound-based
    # mpc_robust_model.con_T_low = pyo.Constraint(
    #     mpc_robust_model.index_steps,
    #     rule=lambda model, k: model.var_x[k, T_IN_IDX] >= 1.0 * T_in_lower_k[k] - model.var_S_T_neg[k])
    # mpc_robust_model.con_T_high = pyo.Constraint(
    #     mpc_robust_model.index_steps,
    #     rule=lambda model, k: model.var_x[k, T_IN_IDX] <= 1.0 * T_in_upper_k[k] + model.var_S_T_pos[k])
    #
    # mpc_robust_model.con_H_low = pyo.Constraint(
    #     mpc_robust_model.index_steps,
    #     rule=lambda model, k: model.var_x[k, H_IN_IDX] >= 1.0 * H_in_lower_k[k] - model.var_S_H_neg[k])
    # mpc_robust_model.con_H_high = pyo.Constraint(
    #     mpc_robust_model.index_steps,
    #     rule=lambda model, k: model.var_x[k, H_IN_IDX] <= 1.0 * H_in_upper_k[k] + model.var_S_H_pos[k])
    #
    # mpc_robust_model.con_C_low = pyo.Constraint(
    #     mpc_robust_model.index_steps,
    #     rule=lambda model, k: model.var_x[k, C_IN_IDX] >= 1.0 * C_in_lower_k[k] - model.var_S_C_neg[k])
    # mpc_robust_model.con_C_high = pyo.Constraint(
    #     mpc_robust_model.index_steps,
    #     rule=lambda model, k: model.var_x[k, C_IN_IDX] <= 1.0 * C_in_upper_k[k] + model.var_S_C_pos[k])

    # Climate security: Reference-centered tightened corridor
    mpc_robust_model.con_T_ref_low = pyo.Constraint(
        mpc_robust_model.index_steps,
        rule=lambda model, k: model.var_x[k, T_IN_IDX] >= T_in_star_k[k] - T_in_ref_corridor_neg_k[k] - model.var_S_T_neg[k],
    )
    mpc_robust_model.con_T_ref_high = pyo.Constraint(
        mpc_robust_model.index_steps,
        rule=lambda model, k: model.var_x[k, T_IN_IDX] <= T_in_star_k[k] + T_in_ref_corridor_pos_k[k] + model.var_S_T_pos[k],
    )

    mpc_robust_model.con_H_ref_low = pyo.Constraint(
        mpc_robust_model.index_steps,
        rule=lambda model, k: model.var_x[k, H_IN_IDX] >= H_in_star_k[k] - H_in_ref_corridor_neg_k[k] - model.var_S_H_neg[k],
    )
    mpc_robust_model.con_H_ref_high = pyo.Constraint(
        mpc_robust_model.index_steps,
        rule=lambda model, k: model.var_x[k, H_IN_IDX] <= H_in_star_k[k] + H_in_ref_corridor_pos_k[k] + model.var_S_H_pos[k],
    )

    mpc_robust_model.con_C_ref_low = pyo.Constraint(
        mpc_robust_model.index_steps,
        rule=lambda model, k: model.var_x[k, C_IN_IDX] >= C_in_star_k[k] - C_in_ref_corridor_neg_k[k] - model.var_S_C_neg[k],
    )
    mpc_robust_model.con_C_ref_high = pyo.Constraint(
        mpc_robust_model.index_steps,
        rule=lambda model, k: model.var_x[k, C_IN_IDX] <= C_in_star_k[k] + C_in_ref_corridor_pos_k[k] + model.var_S_C_pos[k],
    )

    # DLI shortage
    mpc_robust_model.con_L_shortage = pyo.Constraint(mpc_robust_model.index_steps,
                                                     rule=lambda model, k: model.var_S_L[k] >= L_star[k] - model.var_x[k, L_IDX])

    # Define constraints: Gates
    mpc_robust_model.con_u_leq_1 = pyo.Constraint(mpc_robust_model.index_steps, mpc_robust_model.index_com_of_u,
                                                  rule=lambda model, k, j: model.var_u[k, j] <= 1)

    mpc_robust_model.obj = pyo.Objective(
        expr=sum(
            (gamma**k)
            * (
                    sum(c_u[j] * mpc_robust_model.var_u[k, j] for j in range(num_u))
                    + lambda_T_pos * mpc_robust_model.var_S_T_pos[k] * mpc_robust_model.var_S_T_pos[k]
                    + lambda_T_neg * mpc_robust_model.var_S_T_neg[k] * mpc_robust_model.var_S_T_neg[k]
                    + lambda_H_pos * mpc_robust_model.var_S_H_pos[k] * mpc_robust_model.var_S_H_pos[k]
                    + lambda_H_neg * mpc_robust_model.var_S_H_neg[k] * mpc_robust_model.var_S_H_neg[k]
                    + lambda_C_pos * mpc_robust_model.var_S_C_pos[k] * mpc_robust_model.var_S_C_pos[k]
                    + lambda_C_neg * mpc_robust_model.var_S_C_neg[k] * mpc_robust_model.var_S_C_neg[k]
                    + lambda_L * mpc_robust_model.var_S_L[k] * mpc_robust_model.var_S_L[k]
            )
            for k in mpc_robust_model.index_steps
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("gurobi")
    result = solver.solve(mpc_robust_model, tee=False)
    term = str(result.solver.termination_condition)

    x_sol = np.array([[pyo.value(mpc_robust_model.var_x[k, i]) for i in range(num_x)] for k in range(K + 1)], dtype=float)
    u_sol = np.array([[pyo.value(mpc_robust_model.var_u[k, j]) for j in range(num_u)] for k in range(K)], dtype=float)
    slacks = {
        "var_S_T_pos": np.array([pyo.value(mpc_robust_model.var_S_T_pos[k]) for k in range(K)], dtype=float),
        "var_S_T_neg": np.array([pyo.value(mpc_robust_model.var_S_T_neg[k]) for k in range(K)], dtype=float),
        "var_S_H_pos": np.array([pyo.value(mpc_robust_model.var_S_H_pos[k]) for k in range(K)], dtype=float),
        "var_S_H_neg": np.array([pyo.value(mpc_robust_model.var_S_H_neg[k]) for k in range(K)], dtype=float),
        "var_S_C_pos": np.array([pyo.value(mpc_robust_model.var_S_C_pos[k]) for k in range(K)], dtype=float),
        "var_S_C_neg": np.array([pyo.value(mpc_robust_model.var_S_C_neg[k]) for k in range(K)], dtype=float),
        "var_S_L": np.array([pyo.value(mpc_robust_model.var_S_L[k]) for k in range(K)], dtype=float),
    }
    return {
        "x": x_sol,
        "u": u_sol,
        "slacks": slacks,
        "obj": float(pyo.value(mpc_robust_model.obj)),
        "termination_condition": term,
        "kappa": kappa.astype(int),
    }


def build_then_solve_mpc_stochastic(
    x_ini: np.ndarray,
    d_forecast: np.ndarray,
    kappa_ini: int,
    keeper_path: Path,
    matrices_path: Path,
    horizon_K: int,
    num_x: int,
    num_u: int,
    num_d: int,
) -> dict:
    """Scenario-based stochastic MPC with shared controls across disturbance scenarios."""

    # Confirm total steps in this MPC-- follow horizon_K
    K = int(horizon_K)
    default_num_scen = 10

    SIGMA_T_OUT = 1.0
    SIGMA_H_OUT = 0.5
    SIGMA_C_OUT = 0.1
    SIGMA_R_OUT = 50.0

    # Get disturbance vector
    d_forecast_full = np.asarray(d_forecast, dtype=float)
    d = d_forecast_full[:K, :num_d]

    # Build scenario disturbances
    param_keeper = load_parameters(keeper_path)
    meta = _get_vector_meta(param_keeper)
    _validate_problem_sizes(num_x, num_u, num_d, meta)
    x_ini = np.asarray(x_ini, dtype=float).reshape(-1)

    x_idx = meta["x_idx"]
    d_idx = meta["d_idx"]
    u_idx = meta["u_idx"]
    T_IN_IDX = x_idx["T_in"]
    H_IN_IDX = x_idx["H_in"]
    C_IN_IDX = x_idx["C_in"]
    L_IDX = x_idx["L"]
    WARM_IDX = u_idx["U_warm"]
    num_scen = default_num_scen
    sigma_d = np.zeros(num_d, dtype=float)
    sigma_d[d_idx["T_out"]] = SIGMA_T_OUT
    sigma_d[d_idx["H_out"]] = SIGMA_H_OUT
    sigma_d[d_idx["C_out"]] = SIGMA_C_OUT
    sigma_d[d_idx["R_out"]] = SIGMA_R_OUT
    rng = np.random.default_rng(123)
    d_scen = d[np.newaxis, :, :] + rng.normal(
        loc=0.0,
        scale=sigma_d.reshape(1, 1, num_d),
        size=(num_scen, K, num_d),
    )
    pi_n = 1.0 / float(num_scen)

    # Get matrices
    M, N, O, m_vec = _load_dynamics_matrices(str(Path(matrices_path).resolve()))

    climate = get_bounds_and_refs(param_keeper, int(kappa_ini) + np.arange(K, dtype=int))
    kappa = np.asarray(climate["kappa"], dtype=int)
    is_day = np.asarray(climate["is_day"], dtype=bool)
    T_in_lower_k = np.asarray(climate["T_in_lower"], dtype=float)
    T_in_upper_k = np.asarray(climate["T_in_upper"], dtype=float)
    T_in_star_k = np.asarray(climate["T_in_star"], dtype=float)
    H_in_lower_k = np.asarray(climate["H_in_lower"], dtype=float)
    H_in_upper_k = np.asarray(climate["H_in_upper"], dtype=float)
    H_in_star_k = np.asarray(climate["H_in_star"], dtype=float)
    C_in_lower_k = np.asarray(climate["C_in_lower"], dtype=float)
    C_in_upper_k = np.asarray(climate["C_in_upper"], dtype=float)
    C_in_star_k = np.asarray(climate["C_in_star"], dtype=float)
    L_star = np.asarray(climate["L_star"], dtype=float)


    gamma = float(param_keeper["gamma"])
    c_u = _build_control_cost_vector(param_keeper, u_idx)
    lambda_T_pos = float(param_keeper["lambda_T_pos"])
    lambda_T_neg = float(param_keeper["lambda_T_neg"])
    lambda_H_pos = float(param_keeper["lambda_H_pos"])
    lambda_H_neg = float(param_keeper["lambda_H_neg"])
    lambda_C_pos = float(param_keeper["lambda_C_pos"])
    lambda_C_neg = float(param_keeper["lambda_C_neg"])
    lambda_L = float(param_keeper["lambda_L"])

    # Build model
    mpc_stochastic_model = pyo.ConcreteModel()

    # Define the index sets (the “axes” of variables)
    mpc_stochastic_model.index_scen = pyo.RangeSet(0, num_scen - 1)   # Index set of scenarios n_0, n_1, ..., n_num_scen-1
    mpc_stochastic_model.index_steps = pyo.RangeSet(0, K - 1)         # Index set of control steps  u_0, u_1, ..., u_K-1
    mpc_stochastic_model.index_states = pyo.RangeSet(0, K)            # Index set of all state vectors x_0, x_1, ..., x_K
    mpc_stochastic_model.index_com_of_x = pyo.RangeSet(0, num_x - 1)  # Index set of components in one state vector x
    mpc_stochastic_model.index_com_of_u = pyo.RangeSet(0, num_u - 1)  # Index set of components in one state vector u

    # Define variables
    mpc_stochastic_model.var_x = pyo.Var(
        mpc_stochastic_model.index_scen,
        mpc_stochastic_model.index_states,
        mpc_stochastic_model.index_com_of_x,
    )
    mpc_stochastic_model.var_u = pyo.Var(mpc_stochastic_model.index_steps, mpc_stochastic_model.index_com_of_u, domain=pyo.NonNegativeReals)
    mpc_stochastic_model.var_S_T_pos = pyo.Var(mpc_stochastic_model.index_scen, mpc_stochastic_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_stochastic_model.var_S_T_neg = pyo.Var(mpc_stochastic_model.index_scen, mpc_stochastic_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_stochastic_model.var_S_H_pos = pyo.Var(mpc_stochastic_model.index_scen, mpc_stochastic_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_stochastic_model.var_S_H_neg = pyo.Var(mpc_stochastic_model.index_scen, mpc_stochastic_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_stochastic_model.var_S_C_pos = pyo.Var(mpc_stochastic_model.index_scen, mpc_stochastic_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_stochastic_model.var_S_C_neg = pyo.Var(mpc_stochastic_model.index_scen, mpc_stochastic_model.index_steps, domain=pyo.NonNegativeReals)
    mpc_stochastic_model.var_S_L = pyo.Var(mpc_stochastic_model.index_scen, mpc_stochastic_model.index_steps, domain=pyo.NonNegativeReals)
    WARM_MAX_DAY = 0.0

    def _warm_day_cap_rule(model, k):
        if bool(is_day[k]):
            return model.var_u[k, WARM_IDX] <= WARM_MAX_DAY
        return pyo.Constraint.Skip

    mpc_stochastic_model.con_warm_day_cap = pyo.Constraint(mpc_stochastic_model.index_steps, rule=_warm_day_cap_rule)

    # Define constraints: Initial state
    mpc_stochastic_model.con_x_ini = pyo.Constraint(
        mpc_stochastic_model.index_scen,
        mpc_stochastic_model.index_com_of_x,
        rule=lambda model, n, i: model.var_x[n, 0, i] == float(x_ini[i]),
    )

    # Define constraints: State-space model
    mpc_stochastic_model.con_dyn = pyo.Constraint(
        mpc_stochastic_model.index_scen,
        mpc_stochastic_model.index_steps,
        mpc_stochastic_model.index_com_of_x,
        rule=lambda model, n, k, i: (
                model.var_x[n, k + 1, i]
                == model.var_x[n, k, i]
                + sum(M[i, j] * model.var_x[n, k, j] for j in range(num_x))
                + sum(N[i, j] * model.var_u[k, j] for j in range(num_u))
                + sum(O[i, j] * d_scen[n, k, j] for j in range(num_d))
                + float(m_vec[i])
        ),
    )

    #######
    # Climate security: Bounds-based
    # mpc_stochastic_model.con_T_low = pyo.Constraint(
    #     mpc_stochastic_model.index_scen,
    #     mpc_stochastic_model.index_steps,
    #     rule=lambda model, n, k: model.var_x[n, k, T_IN_IDX] >= T_in_lower_k[k] - model.var_S_T_neg[n, k])
    # mpc_stochastic_model.con_T_high = pyo.Constraint(
    #     mpc_stochastic_model.index_scen,
    #     mpc_stochastic_model.index_steps,
    #     rule=lambda model, n, k: model.var_x[n, k, T_IN_IDX] <= T_in_upper_k[k] + model.var_S_T_pos[n, k])

    # mpc_stochastic_model.con_H_low = pyo.Constraint(
    #     mpc_stochastic_model.index_scen,
    #     mpc_stochastic_model.index_steps,
    #     rule=lambda model, n, k: model.var_x[n, k, H_IN_IDX] >= H_in_lower_k[k] - model.var_S_H_neg[n, k])
    # mpc_stochastic_model.con_H_high = pyo.Constraint(
    #     mpc_stochastic_model.index_scen,
    #     mpc_stochastic_model.index_steps,
    #     rule=lambda model, n, k: model.var_x[n, k, H_IN_IDX] <= H_in_upper_k[k] + model.var_S_H_pos[n, k])

    # mpc_stochastic_model.con_C_low = pyo.Constraint(
    #     mpc_stochastic_model.index_scen,
    #     mpc_stochastic_model.index_steps,
    #     rule=lambda model, n, k: model.var_x[n, k, C_IN_IDX] >= C_in_lower_k[k] - model.var_S_C_neg[n, k])
    # mpc_stochastic_model.con_C_high = pyo.Constraint(
    #     mpc_stochastic_model.index_scen,
    #     mpc_stochastic_model.index_steps,
    #     rule=lambda model, n, k: model.var_x[n, k, C_IN_IDX] <= C_in_upper_k[k] + model.var_S_C_pos[n, k])
    

    # Climate security: Reference-based
    mpc_stochastic_model.con_T_in_ref = pyo.Constraint(
        mpc_stochastic_model.index_scen,
        mpc_stochastic_model.index_steps,
        rule=lambda model, n, k: model.var_x[n, k, T_IN_IDX] == T_in_star_k[k] + model.var_S_T_pos[n, k] - model.var_S_T_neg[n, k],
    )

    mpc_stochastic_model.con_H_in_ref = pyo.Constraint(
        mpc_stochastic_model.index_scen,
        mpc_stochastic_model.index_steps,
        rule=lambda model, n, k: model.var_x[n, k, H_IN_IDX] == H_in_star_k[k] + model.var_S_H_pos[n, k] - model.var_S_H_neg[n, k],
    )

    mpc_stochastic_model.con_C_in_ref = pyo.Constraint(
        mpc_stochastic_model.index_scen,
        mpc_stochastic_model.index_steps,
        rule=lambda model, n, k: model.var_x[n, k, C_IN_IDX] == C_in_star_k[k] + model.var_S_C_pos[n, k] - model.var_S_C_neg[n, k],
    )

    # DLI shortage
    mpc_stochastic_model.con_L_shortage = pyo.Constraint(
        mpc_stochastic_model.index_scen,
        mpc_stochastic_model.index_steps,
        rule=lambda model, n, k: model.var_x[n, k, L_IDX] >= L_star[k] - model.var_S_L[n, k])

    # Define constraints: Gates
    mpc_stochastic_model.con_u_leq_1 = pyo.Constraint(
        mpc_stochastic_model.index_steps,
        mpc_stochastic_model.index_com_of_u,
        rule=lambda model, k, j: model.var_u[k, j] <= 1,
    )

    mpc_stochastic_model.obj = pyo.Objective(
        expr=sum(
            pi_n
            * sum(
                (gamma**k)
                * (
                        sum(c_u[j] * mpc_stochastic_model.var_u[k, j] for j in range(num_u))
                        + lambda_T_pos * mpc_stochastic_model.var_S_T_pos[n, k] * mpc_stochastic_model.var_S_T_pos[n, k]
                        + lambda_T_neg * mpc_stochastic_model.var_S_T_neg[n, k] * mpc_stochastic_model.var_S_T_neg[n, k]
                        + lambda_H_pos * mpc_stochastic_model.var_S_H_pos[n, k] * mpc_stochastic_model.var_S_H_pos[n, k]
                        + lambda_H_neg * mpc_stochastic_model.var_S_H_neg[n, k] * mpc_stochastic_model.var_S_H_neg[n, k]
                        + lambda_C_pos * mpc_stochastic_model.var_S_C_pos[n, k] * mpc_stochastic_model.var_S_C_pos[n, k]
                        + lambda_C_neg * mpc_stochastic_model.var_S_C_neg[n, k] * mpc_stochastic_model.var_S_C_neg[n, k]
                        + lambda_L * mpc_stochastic_model.var_S_L[n, k] * mpc_stochastic_model.var_S_L[n, k]
                )
                for k in mpc_stochastic_model.index_steps
            )
            for n in mpc_stochastic_model.index_scen
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("gurobi")
    result = solver.solve(mpc_stochastic_model, tee=False)
    term = str(result.solver.termination_condition)

    x_scen = np.array(
        [
            [[pyo.value(mpc_stochastic_model.var_x[n, k, i]) for i in range(num_x)] for k in range(K + 1)]
            for n in range(num_scen)
        ],
        dtype=float,
    )
    x_sol = np.mean(x_scen, axis=0)

    u_sol = np.array([[pyo.value(mpc_stochastic_model.var_u[k, j]) for j in range(num_u)] for k in range(K)], dtype=float)

    S_T_pos_all = np.array(
        [[pyo.value(mpc_stochastic_model.var_S_T_pos[n, k]) for k in range(K)] for n in range(num_scen)],
        dtype=float,
    )
    S_T_neg_all = np.array(
        [[pyo.value(mpc_stochastic_model.var_S_T_neg[n, k]) for k in range(K)] for n in range(num_scen)],
        dtype=float,
    )
    S_H_pos_all = np.array(
        [[pyo.value(mpc_stochastic_model.var_S_H_pos[n, k]) for k in range(K)] for n in range(num_scen)],
        dtype=float,
    )
    S_H_neg_all = np.array(
        [[pyo.value(mpc_stochastic_model.var_S_H_neg[n, k]) for k in range(K)] for n in range(num_scen)],
        dtype=float,
    )
    S_C_pos_all = np.array(
        [[pyo.value(mpc_stochastic_model.var_S_C_pos[n, k]) for k in range(K)] for n in range(num_scen)],
        dtype=float,
    )
    S_C_neg_all = np.array(
        [[pyo.value(mpc_stochastic_model.var_S_C_neg[n, k]) for k in range(K)] for n in range(num_scen)],
        dtype=float,
    )
    S_L_all = np.array(
        [[pyo.value(mpc_stochastic_model.var_S_L[n, k]) for k in range(K)] for n in range(num_scen)],
        dtype=float,
    )

    slacks = {
        "var_S_T_pos": np.mean(S_T_pos_all, axis=0),
        "var_S_T_neg": np.mean(S_T_neg_all, axis=0),
        "var_S_H_pos": np.mean(S_H_pos_all, axis=0),
        "var_S_H_neg": np.mean(S_H_neg_all, axis=0),
        "var_S_C_pos": np.mean(S_C_pos_all, axis=0),
        "var_S_C_neg": np.mean(S_C_neg_all, axis=0),
        "var_S_L": np.mean(S_L_all, axis=0),
    }
    return {
        "x": x_sol,
        "u": u_sol,
        "slacks": slacks,
        "obj": float(pyo.value(mpc_stochastic_model.obj)),
        "termination_condition": term,
        "kappa": kappa.astype(int),
    }


def get_cost_at_end_state(
    x_act: np.ndarray,
    u0: np.ndarray,
    kappa_act: int,
    keeper_path: Path = keeper_path,
) -> dict:
    """Compute one-step realized stage cost using base-style reference tracking for T/H/C and shortage tracking for L."""

    param_keeper = load_parameters(keeper_path)
    meta = _get_vector_meta(param_keeper)
    x_act = np.asarray(x_act, dtype=float).reshape(-1)
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    if x_act.size != int(meta["num_x"]):
        raise ValueError(f"x_act size mismatch: expected {int(meta['num_x'])}, got {x_act.size}.")
    if u0.size != int(meta["num_u"]):
        raise ValueError(f"u0 size mismatch: expected {int(meta['num_u'])}, got {u0.size}.")

    x_idx = meta["x_idx"]
    u_names = meta["u_names"]
    T_IN_IDX = x_idx["T_in"]
    H_IN_IDX = x_idx["H_in"]
    C_IN_IDX = x_idx["C_in"]
    L_IDX = x_idx["L"]

    climate = get_bounds_and_refs(param_keeper, int(kappa_act))
    T_in_star = float(climate["T_in_star"])
    H_in_star = float(climate["H_in_star"])
    C_in_star = float(climate["C_in_star"])
    L_star = float(climate["L_star"])

    c_u = _build_control_cost_vector(param_keeper, meta["u_idx"])
    lambda_T_pos = float(param_keeper["lambda_T_pos"])
    lambda_T_neg = float(param_keeper["lambda_T_neg"])
    lambda_H_pos = float(param_keeper["lambda_H_pos"])
    lambda_H_neg = float(param_keeper["lambda_H_neg"])
    lambda_C_pos = float(param_keeper["lambda_C_pos"])
    lambda_C_neg = float(param_keeper["lambda_C_neg"])
    lambda_L = float(param_keeper["lambda_L"])

    S_T_pos = max(float(x_act[T_IN_IDX]) - T_in_star, 0.0)
    S_T_neg = max(T_in_star - float(x_act[T_IN_IDX]), 0.0)
    S_H_pos = max(float(x_act[H_IN_IDX]) - H_in_star, 0.0)
    S_H_neg = max(H_in_star - float(x_act[H_IN_IDX]), 0.0)
    S_C_pos = max(float(x_act[C_IN_IDX]) - C_in_star, 0.0)
    S_C_neg = max(C_in_star - float(x_act[C_IN_IDX]), 0.0)
    S_L = max(L_star - float(x_act[L_IDX]), 0.0)

    one_step_control_cost_act = float(np.dot(c_u, u0))
    one_step_control_cost_act_components = _control_cost_component_dict(u0, c_u, u_names)
    one_step_slack_cost_act = (
        lambda_T_pos * S_T_pos * S_T_pos
        + lambda_T_neg * S_T_neg * S_T_neg
        + lambda_H_pos * S_H_pos * S_H_pos
        + lambda_H_neg * S_H_neg * S_H_neg
        + lambda_C_pos * S_C_pos * S_C_pos
        + lambda_C_neg * S_C_neg * S_C_neg
        + lambda_L * S_L * S_L
    )
    one_step_slack_cost_act_components = {
        "var_S_T_pos": float(lambda_T_pos * S_T_pos * S_T_pos),
        "var_S_T_neg": float(lambda_T_neg * S_T_neg * S_T_neg),
        "var_S_H_pos": float(lambda_H_pos * S_H_pos * S_H_pos),
        "var_S_H_neg": float(lambda_H_neg * S_H_neg * S_H_neg),
        "var_S_C_pos": float(lambda_C_pos * S_C_pos * S_C_pos),
        "var_S_C_neg": float(lambda_C_neg * S_C_neg * S_C_neg),
        "var_S_L": float(lambda_L * S_L * S_L),
    }
    one_step_total_cost_act = float(one_step_control_cost_act + one_step_slack_cost_act)

    slacks = {
        "var_S_T_pos": float(S_T_pos),
        "var_S_T_neg": float(S_T_neg),
        "var_S_H_pos": float(S_H_pos),
        "var_S_H_neg": float(S_H_neg),
        "var_S_C_pos": float(S_C_pos),
        "var_S_C_neg": float(S_C_neg),
        "var_S_L": float(S_L),
    }
    refs = {
        "T_in_star": float(T_in_star),
        "H_in_star": float(H_in_star),
        "C_in_star": float(C_in_star),
        "L_star": float(L_star),
    }
    return {
        "one_step_total_cost_act": one_step_total_cost_act,
        "one_step_control_cost_act": float(one_step_control_cost_act),
        "one_step_slack_cost_act": float(one_step_slack_cost_act),
        "one_step_control_cost_act_components": one_step_control_cost_act_components,
        "one_step_slack_cost_act_components": one_step_slack_cost_act_components,
        "slacks": slacks,
        "refs": refs,
    }


def get_one_step_total_cost_act_batch(
    x_act: np.ndarray,
    u0: np.ndarray,
    kappa_act: np.ndarray,
    keeper_path: Path = keeper_path,
) -> np.ndarray:
    param_keeper = load_parameters(keeper_path)
    meta = _get_vector_meta(param_keeper)

    x_arr = np.asarray(x_act, dtype=float)
    u_arr = np.asarray(u0, dtype=float)
    kappa_arr = np.asarray(kappa_act, dtype=int).reshape(-1)

    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(1, -1)
    if x_arr.ndim != 2:
        raise ValueError(f"x_act must be 1D or 2D, got shape {x_arr.shape}")
    if u_arr.ndim != 2:
        raise ValueError(f"u0 must be 1D or 2D, got shape {u_arr.shape}")
    if x_arr.shape[0] != u_arr.shape[0] or x_arr.shape[0] != kappa_arr.size:
        raise ValueError(
            f"Batch size mismatch: x_act={x_arr.shape[0]}, u0={u_arr.shape[0]}, kappa_act={kappa_arr.size}"
        )
    if x_arr.shape[1] != int(meta["num_x"]):
        raise ValueError(f"x_act width mismatch: expected {int(meta['num_x'])}, got {x_arr.shape[1]}.")
    if u_arr.shape[1] != int(meta["num_u"]):
        raise ValueError(f"u0 width mismatch: expected {int(meta['num_u'])}, got {u_arr.shape[1]}.")

    x_idx = meta["x_idx"]
    climate = get_bounds_and_refs(param_keeper, kappa_arr)

    T_in_star = np.asarray(climate["T_in_star"], dtype=float).reshape(-1)
    H_in_star = np.asarray(climate["H_in_star"], dtype=float).reshape(-1)
    C_in_star = np.asarray(climate["C_in_star"], dtype=float).reshape(-1)
    L_star = np.asarray(climate["L_star"], dtype=float).reshape(-1)

    T_in = x_arr[:, x_idx["T_in"]]
    H_in = x_arr[:, x_idx["H_in"]]
    C_in = x_arr[:, x_idx["C_in"]]
    L = x_arr[:, x_idx["L"]]

    S_T_pos = np.maximum(T_in - T_in_star, 0.0)
    S_T_neg = np.maximum(T_in_star - T_in, 0.0)
    S_H_pos = np.maximum(H_in - H_in_star, 0.0)
    S_H_neg = np.maximum(H_in_star - H_in, 0.0)
    S_C_pos = np.maximum(C_in - C_in_star, 0.0)
    S_C_neg = np.maximum(C_in_star - C_in, 0.0)
    S_L = np.maximum(L_star - L, 0.0)

    c_u = _build_control_cost_vector(param_keeper, meta["u_idx"])
    one_step_control_cost_act = np.einsum("ij,j->i", u_arr, c_u)
    one_step_slack_cost_act = (
        float(param_keeper["lambda_T_pos"]) * S_T_pos * S_T_pos
        + float(param_keeper["lambda_T_neg"]) * S_T_neg * S_T_neg
        + float(param_keeper["lambda_H_pos"]) * S_H_pos * S_H_pos
        + float(param_keeper["lambda_H_neg"]) * S_H_neg * S_H_neg
        + float(param_keeper["lambda_C_pos"]) * S_C_pos * S_C_pos
        + float(param_keeper["lambda_C_neg"]) * S_C_neg * S_C_neg
        + float(param_keeper["lambda_L"]) * S_L * S_L
    )
    total_cost = one_step_control_cost_act + one_step_slack_cost_act
    if not np.all(np.isfinite(total_cost)):
        raise FloatingPointError("Non-finite values detected in batch realized cost computation.")
    return np.asarray(total_cost, dtype=float).reshape(-1)

def load_parameters(yaml_path: Path) -> dict:
    data_keeper = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    param_keeper = data_keeper["parameters"]
    param_keeper["x_ini"] = data_keeper["variables"]["x_ini"]
    return param_keeper


@lru_cache(maxsize=16)
def _load_dynamics_matrices(matrices_path_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(matrices_path_str)

    base_dir = path.parent if path.is_file() else path

    x_names = get_state_names_in_order()
    d_names = get_disturbance_names_in_order()
    u_names = get_control_names_in_order()
    row_labels = [f"d{name}" for name in x_names]
    m_vec_labels = [f"m_{name}" for name in x_names]

    M_df = pd.read_csv(base_dir / "M.csv", index_col=0)
    N_df = pd.read_csv(base_dir / "N.csv", index_col=0)
    O_df = pd.read_csv(base_dir / "O.csv", index_col=0)
    m_vec_df = pd.read_csv(base_dir / "m_vec.csv", index_col=0)

    if list(M_df.index) != row_labels or list(M_df.columns) != x_names:
        raise ValueError(
            f"M.csv labels mismatch against vector_order. Expected rows={row_labels}, cols={x_names}; "
            f"got rows={list(M_df.index)}, cols={list(M_df.columns)}"
        )
    if list(N_df.index) != row_labels or list(N_df.columns) != u_names:
        raise ValueError(
            f"N.csv labels mismatch against vector_order. Expected rows={row_labels}, cols={u_names}; "
            f"got rows={list(N_df.index)}, cols={list(N_df.columns)}"
        )
    if list(O_df.index) != row_labels or list(O_df.columns) != d_names:
        raise ValueError(
            f"O.csv labels mismatch against vector_order. Expected rows={row_labels}, cols={d_names}; "
            f"got rows={list(O_df.index)}, cols={list(O_df.columns)}"
        )
    if list(m_vec_df.index) != m_vec_labels or "value" not in m_vec_df.columns:
        raise ValueError(
            f"m_vec.csv labels mismatch against vector_order. Expected rows={m_vec_labels} and column=['value']; "
            f"got rows={list(m_vec_df.index)}, cols={list(m_vec_df.columns)}"
        )

    M = M_df.to_numpy(dtype=float)
    N = N_df.to_numpy(dtype=float)
    O = O_df.to_numpy(dtype=float)
    m_vec = m_vec_df["value"].to_numpy(dtype=float).reshape(-1)
    return M, N, O, m_vec
