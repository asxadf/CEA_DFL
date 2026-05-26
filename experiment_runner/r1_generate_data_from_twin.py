# experiment_runner/r1_generate_data_from_twin.py

"""Run MPC simulation with switchable MPC method.

Example:
python3 -u experiment_runner/r1_generate_data_from_twin.py --season cold --num-samples 50000 2>&1 | grep --line-buffered -i --color=always -E "warning|error|exception|traceback|convergencewarning|userwarning|runtimewarning|$"
python3 -u experiment_runner/r1_generate_data_from_twin.py --season warm --num-samples 50000 2>&1 | grep --line-buffered -i --color=always -E "warning|error|exception|traceback|convergencewarning|userwarning|runtimewarning|$"

python3 -u experiment_runner/r1_generate_data_from_twin.py --season cold --num-samples 50000 2>&1 | grep --line-buffered -i --color=always -E "warning|error|exception|traceback|convergencewarning|userwarning|runtimewarning|$" &
python3 -u experiment_runner/r1_generate_data_from_twin.py --season warm --num-samples 50000 2>&1 | grep --line-buffered -i --color=always -E "warning|error|exception|traceback|convergencewarning|userwarning|runtimewarning|$" &

wait

"""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.builders.builder_digital_twin_one_step import digital_twin_one_step, load_parameters
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
from src.utils.physical_checks import hard_checks

SEED = 123
PARAM_YAML_PATH = (Path(__file__).resolve().parents[1] / "configs/var_and_param_keeper.yaml").resolve()
OUT_BASE_DIR = (Path(__file__).resolve().parents[1] / "experiment_result/r1_generate_data_from_twin").resolve()
OUTDOOR_REAL_CSV_BY_SEASON = {
    "warm": (Path(__file__).resolve().parents[1] / "data/processed/training_outdoor_realization_warm.csv").resolve(),
    "cold": (Path(__file__).resolve().parents[1] / "data/processed/training_outdoor_realization_cold.csv").resolve(),
}
OUTDOOR_DATETIME_COL = "Datetime"
OUTDOOR_T_COL = "T_Outdoor(C)"
OUTDOOR_H_COL = "H_Outdoor(g/m3)"
OUTDOOR_C_COL = "CO2_Outdoor(g/m3)"
OUTDOOR_R_COL = "Radiation_Outdoor(w/m2)"
prob_random_u = 0.1   # probability of fully random U, tunable
PRINT_EVERY_ACCEPTED = 2000
PRINT_EVERY_ATTEMPTS = 10000
START_TIME = None

EXCITE_HOLD_MIN = 2     # Minimum number of samples to keep the same excitation pattern (same selected actuators + levels)
EXCITE_HOLD_MAX = 12    # Maximum number of samples to keep the same excitation pattern before resampling

EXCITE_NUM_ACT_MIN = 1  # Minimum number of actuators to excite at the same time
EXCITE_NUM_ACT_MAX = 4  # Maximum number of actuators to excite at the same time

EXCITE_LEVEL_MIN = 0.10 # Minimum excitation level assigned to a selected actuator (avoid exactly 0)
EXCITE_LEVEL_MAX = 1.00 # Maximum excitation level assigned to a selected actuator (near full-scale)

# Weights used to choose WHICH actuator(s) to excite (normalized to probabilities).
# Larger weight => more likely to be selected for excitation.
# Order:                   [heat, fan, nat, ac, dos, LED, hum, deh, shad, warm]
EXCITE_WEIGHTS_BY_SEASON = {
    "cold": np.array([30,   18,  14,  4,  40,  20,  40,  12,    6,   24], dtype=float),
    "warm": np.array([ 4,   18,  18, 32,  30,  18,   6,  24,   24,    4], dtype=float),
}

prob_out_of_bound = 0.10 # Probability of out-of-bound initial indoor state for T_in / H_in / C_in
T_in_margin = 3.0
H_in_margin = 2.0 # 3.89 (LB) - 2.0 (Marg) = 1.89 --> min
C_in_margin = 0.2 # 0.50 (LB) - 0.2 (Marg) = 0.30 --> min

def _num_samples_type(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("--num-samples must be >= 1")
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data from one-step digital twin.")
    parser.add_argument("--season", choices=("cold", "warm"), default="warm")
    parser.add_argument("--num-samples", type=_num_samples_type, default=100000)
    return parser.parse_args()


def _require_named_entries(name_to_idx: dict[str, int], required_names: tuple[str, ...], vector_label: str) -> None:
    missing = [name for name in required_names if name not in name_to_idx]
    if missing:
        raise KeyError(f"Missing required {vector_label} names in vector order: {missing}")


def _rh_from_absolute_humidity(T_out: float, H_out: float) -> float:
    e_s_hPa = 6.112 * math.exp((17.67 * T_out) / (T_out + 243.5))
    H_sat = 216.7 * (e_s_hPa / (T_out + 273.15))
    if H_sat <= 0.0:
        return 0.0
    return float(np.clip(100.0 * H_out / H_sat, 0.0, 100.0))


def _ppm_from_co2_gm3(T_out: float, C_out: float) -> float:
    M_CO2 = 44.01e-3
    R = 8.314462618
    T_K = T_out + 273.15
    rho_CO2 = C_out / 1000.0
    mole_fraction = rho_CO2 * R * T_K / (101325.0 * M_CO2)
    return float(max(mole_fraction * 1e6, 0.0))


def _load_outdoor_realization(
    csv_path: Path,
    min_per_step: float,
    step_per_day: int,
) -> dict[str, object]:
    outdoor_df = pd.read_csv(csv_path)
    required_cols = (
        OUTDOOR_DATETIME_COL,
        OUTDOOR_T_COL,
        OUTDOOR_H_COL,
        OUTDOOR_C_COL,
        OUTDOOR_R_COL,
    )
    missing_cols = [col for col in required_cols if col not in outdoor_df.columns]
    if missing_cols:
        raise KeyError(f"Missing required outdoor realization columns in {csv_path}: {missing_cols}")
    if outdoor_df.empty:
        raise ValueError(f"Outdoor realization CSV is empty: {csv_path}")

    datetimes = pd.to_datetime(outdoor_df[OUTDOOR_DATETIME_COL], errors="raise")
    minutes_since_midnight = datetimes.dt.hour.to_numpy(dtype=float) * 60.0 + datetimes.dt.minute.to_numpy(dtype=float)
    kappa_k = np.rint(minutes_since_midnight / float(min_per_step)).astype(int) % int(step_per_day)

    T_out = outdoor_df[OUTDOOR_T_COL].to_numpy(dtype=float)
    H_out = outdoor_df[OUTDOOR_H_COL].to_numpy(dtype=float)
    C_out = outdoor_df[OUTDOOR_C_COL].to_numpy(dtype=float)
    R_out = outdoor_df[OUTDOOR_R_COL].to_numpy(dtype=float)
    RH_out = np.asarray([_rh_from_absolute_humidity(t, h) for t, h in zip(T_out, H_out)], dtype=float)
    ppm_out = np.asarray([_ppm_from_co2_gm3(t, c) for t, c in zip(T_out, C_out)], dtype=float)

    return {
        "kappa_k": kappa_k,
        "hhmm": datetimes.dt.strftime("%H:%M").tolist(),
        "T_out": T_out,
        "H_out": H_out,
        "C_out": C_out,
        "R_out": R_out,
        "RH_out": RH_out,
        "ppm_out": ppm_out,
        "R_out_day_max": float(max(np.max(R_out), 1.0)),
    }


def _linear_ramp_scalar(kappa_k: int, start_k: int, end_k: int, v_start: float, v_end: float) -> float:
    start = float(start_k)
    end = float(end_k)
    if end <= start:
        return float(v_start)
    frac = min(max((float(kappa_k) - start) / (end - start), 0.0), 1.0)
    return float(v_start) + frac * (float(v_end) - float(v_start))


def _get_climate_bounds(kappa_k: int, param_keeper: dict) -> tuple[float, float, float, float, float, float]:
    total_steps = int(param_keeper["kappa_day_night_total_steps"])
    kp = int(kappa_k) % total_steps

    day_start = int(param_keeper["kappa_day_start"])
    day_end = int(param_keeper["kappa_day_end"])
    has_transition_window = all(
        key in param_keeper
        for key in (
            "kappa_transition_start_1",
            "kappa_transition_end_1",
            "kappa_transition_start_2",
            "kappa_transition_end_2",
        )
    )
    has_h_day_night = all(
        key in param_keeper
        for key in (
            "H_in_lower_day",
            "H_in_upper_day",
            "H_in_lower_night",
            "H_in_upper_night",
        )
    )

    if has_transition_window:
        t1_start = int(param_keeper["kappa_transition_start_1"])
        t1_end = int(param_keeper["kappa_transition_end_1"])
        t2_start = int(param_keeper["kappa_transition_start_2"])
        t2_end = int(param_keeper["kappa_transition_end_2"])

        is_transition_1 = t1_start <= kp < t1_end
        is_transition_2 = t2_start <= kp < t2_end
    else:
        is_transition_1 = False
        is_transition_2 = False

    is_day = (day_start <= kp < day_end) and (not is_transition_1) and (not is_transition_2)

    if is_transition_1:
        T_in_LB = _linear_ramp_scalar(
            kp,
            t1_start,
            t1_end,
            float(param_keeper.get("T_in_lower_transition_start_1", param_keeper["T_in_lower_night"])),
            float(param_keeper.get("T_in_lower_transition_end_1", param_keeper["T_in_lower_day"])),
        )
        T_in_UB = _linear_ramp_scalar(
            kp,
            t1_start,
            t1_end,
            float(param_keeper.get("T_in_upper_transition_start_1", param_keeper["T_in_upper_night"])),
            float(param_keeper.get("T_in_upper_transition_end_1", param_keeper["T_in_upper_day"])),
        )
        C_in_LB = _linear_ramp_scalar(
            kp,
            t1_start,
            t1_end,
            float(param_keeper.get("C_in_lower_transition_start_1", param_keeper["C_in_lower_night"])),
            float(param_keeper.get("C_in_lower_transition_end_1", param_keeper["C_in_lower_day"])),
        )
        C_in_UB = _linear_ramp_scalar(
            kp,
            t1_start,
            t1_end,
            float(param_keeper.get("C_in_upper_transition_start_1", param_keeper["C_in_upper_night"])),
            float(param_keeper.get("C_in_upper_transition_end_1", param_keeper["C_in_upper_day"])),
        )
        if has_h_day_night:
            H_in_LB = _linear_ramp_scalar(
                kp,
                t1_start,
                t1_end,
                float(param_keeper.get("H_in_lower_transition_start_1", param_keeper["H_in_lower_night"])),
                float(param_keeper.get("H_in_lower_transition_end_1", param_keeper["H_in_lower_day"])),
            )
            H_in_UB = _linear_ramp_scalar(
                kp,
                t1_start,
                t1_end,
                float(param_keeper.get("H_in_upper_transition_start_1", param_keeper["H_in_upper_night"])),
                float(param_keeper.get("H_in_upper_transition_end_1", param_keeper["H_in_upper_day"])),
            )
        else:
            H_in_LB = float(param_keeper["H_in_lower"])
            H_in_UB = float(param_keeper["H_in_upper"])
    elif is_transition_2:
        T_in_LB = _linear_ramp_scalar(
            kp,
            t2_start,
            t2_end,
            float(param_keeper.get("T_in_lower_transition_start_2", param_keeper["T_in_lower_day"])),
            float(param_keeper.get("T_in_lower_transition_end_2", param_keeper["T_in_lower_night"])),
        )
        T_in_UB = _linear_ramp_scalar(
            kp,
            t2_start,
            t2_end,
            float(param_keeper.get("T_in_upper_transition_start_2", param_keeper["T_in_upper_day"])),
            float(param_keeper.get("T_in_upper_transition_end_2", param_keeper["T_in_upper_night"])),
        )
        C_in_LB = _linear_ramp_scalar(
            kp,
            t2_start,
            t2_end,
            float(param_keeper.get("C_in_lower_transition_start_2", param_keeper["C_in_lower_day"])),
            float(param_keeper.get("C_in_lower_transition_end_2", param_keeper["C_in_lower_night"])),
        )
        C_in_UB = _linear_ramp_scalar(
            kp,
            t2_start,
            t2_end,
            float(param_keeper.get("C_in_upper_transition_start_2", param_keeper["C_in_upper_day"])),
            float(param_keeper.get("C_in_upper_transition_end_2", param_keeper["C_in_upper_night"])),
        )
        if has_h_day_night:
            H_in_LB = _linear_ramp_scalar(
                kp,
                t2_start,
                t2_end,
                float(param_keeper.get("H_in_lower_transition_start_2", param_keeper["H_in_lower_day"])),
                float(param_keeper.get("H_in_lower_transition_end_2", param_keeper["H_in_lower_night"])),
            )
            H_in_UB = _linear_ramp_scalar(
                kp,
                t2_start,
                t2_end,
                float(param_keeper.get("H_in_upper_transition_start_2", param_keeper["H_in_upper_day"])),
                float(param_keeper.get("H_in_upper_transition_end_2", param_keeper["H_in_upper_night"])),
            )
        else:
            H_in_LB = float(param_keeper["H_in_lower"])
            H_in_UB = float(param_keeper["H_in_upper"])
    elif is_day:
        T_in_LB = float(param_keeper["T_in_lower_day"])
        T_in_UB = float(param_keeper["T_in_upper_day"])
        C_in_LB = float(param_keeper["C_in_lower_day"])
        C_in_UB = float(param_keeper["C_in_upper_day"])
        if has_h_day_night:
            H_in_LB = float(param_keeper["H_in_lower_day"])
            H_in_UB = float(param_keeper["H_in_upper_day"])
        else:
            H_in_LB = float(param_keeper["H_in_lower"])
            H_in_UB = float(param_keeper["H_in_upper"])
    else:
        T_in_LB = float(param_keeper["T_in_lower_night"])
        T_in_UB = float(param_keeper["T_in_upper_night"])
        C_in_LB = float(param_keeper["C_in_lower_night"])
        C_in_UB = float(param_keeper["C_in_upper_night"])
        if has_h_day_night:
            H_in_LB = float(param_keeper["H_in_lower_night"])
            H_in_UB = float(param_keeper["H_in_upper_night"])
        else:
            H_in_LB = float(param_keeper["H_in_lower"])
            H_in_UB = float(param_keeper["H_in_upper"])

    return T_in_LB, T_in_UB, H_in_LB, H_in_UB, C_in_LB, C_in_UB


def _sample_L_ini(
    rng: np.random.Generator,
    kappa_k: int,
    KAPPA_DAY_START: int,
    KAPPA_DAY_END: int,
    L_max: float,
) -> float:
    if kappa_k < KAPPA_DAY_START:
        return 0.0
    if kappa_k >= KAPPA_DAY_END:
        return rng.uniform(0.8 * L_max, L_max)
    frac = (kappa_k - KAPPA_DAY_START) / (KAPPA_DAY_END - KAPPA_DAY_START)
    return rng.uniform(0.0, frac * L_max)


def _sample_x_ini(
    rng: np.random.Generator,
    bounds: tuple[float, float, float, float, float, float],
    kappa_k: int,
    KAPPA_DAY_START: int,
    KAPPA_DAY_END: int,
    x_idx: dict[str, int],
    num_x: int,
    param_keeper: dict,
) -> np.ndarray:
    T_in_LB, T_in_UB, H_in_LB, H_in_UB, C_in_LB, C_in_UB = bounds
    L_max = float(param_keeper.get("L_star_k_max"))
    L_ini = _sample_L_ini(rng, kappa_k, KAPPA_DAY_START, KAPPA_DAY_END, L_max)

    def sample_var(low: float, high: float, margin: float) -> float:
        if rng.random() < (1.0 - prob_out_of_bound):
            return rng.uniform(low, high)
        if rng.random() < 0.5:
            return rng.uniform(low - margin, low)
        return rng.uniform(high, high + margin)

    T_in_ini = sample_var(T_in_LB, T_in_UB, T_in_margin)
    H_in_ini = max(sample_var(H_in_LB, H_in_UB, H_in_margin), H_in_LB - H_in_margin)
    C_in_ini = max(sample_var(C_in_LB, C_in_UB, C_in_margin), C_in_LB - C_in_margin)

    x_ini = np.asarray(param_keeper["x_ini"], dtype=float).reshape(-1).copy()
    if x_ini.size != num_x:
        raise ValueError(f"x_ini size mismatch: expected {num_x}, got {x_ini.size}")
    x_ini[x_idx["T_in"]] = T_in_ini
    x_ini[x_idx["H_in"]] = H_in_ini
    x_ini[x_idx["C_in"]] = C_in_ini
    x_ini[x_idx["L"]] = L_ini
    return x_ini


def _build_disturbance(
    T_out: float,
    H_out: float,
    C_out: float,
    R_out: float,
    d_idx: dict[str, int],
    num_d: int,
) -> np.ndarray:
    d = np.zeros(num_d, dtype=float)
    d[d_idx["T_out"]] = T_out
    d[d_idx["H_out"]] = H_out
    d[d_idx["C_out"]] = C_out
    d[d_idx["R_out"]] = R_out
    return d


def _sample_disturbance(
    rng: np.random.Generator,
    outdoor_realization: dict[str, object],
    kappa_day_start: int,
    kappa_day_end: int,
    d_idx: dict[str, int],
    num_d: int,
) -> tuple[np.ndarray, int, str, float, float, bool]:
    n_rows = len(outdoor_realization["kappa_k"])
    row_idx = int(rng.integers(0, n_rows))
    kappa_k = int(outdoor_realization["kappa_k"][row_idx])
    hhmm = str(outdoor_realization["hhmm"][row_idx])
    T_out = float(outdoor_realization["T_out"][row_idx])
    H_out = float(outdoor_realization["H_out"][row_idx])
    C_out = float(outdoor_realization["C_out"][row_idx])
    R_out = float(outdoor_realization["R_out"][row_idx])
    RH_out = float(outdoor_realization["RH_out"][row_idx])
    ppm_out = float(outdoor_realization["ppm_out"][row_idx])
    d = _build_disturbance(T_out, H_out, C_out, R_out, d_idx, num_d)
    is_night = (kappa_k < kappa_day_start) or (kappa_k >= kappa_day_end)
    return d, kappa_k, hhmm, RH_out, ppm_out, is_night

def _sample_rule_of_thumb_u(
    x_ini: np.ndarray,
    d: np.ndarray,
    is_night: bool,
    bounds: tuple[float, float, float, float, float, float],
    num_u,
    u_idx: dict[str, int],
    x_idx: dict[str, int],
    d_idx: dict[str, int],
    R_out_day_max: float,
) -> np.ndarray:
    T_in_ini = float(x_ini[x_idx["T_in"]])
    H_in_ini = float(x_ini[x_idx["H_in"]])
    C_in_ini = float(x_ini[x_idx["C_in"]])
    T_out = float(d[d_idx["T_out"]])
    H_out = float(d[d_idx["H_out"]])
    R_out = float(d[d_idx["R_out"]])
    T_in_LB, T_in_UB, H_in_LB, H_in_UB, C_in_LB, _ = bounds

    u = np.zeros(num_u, dtype=float)

    # Too cold → heating
    if T_in_ini < T_in_LB:
        u[u_idx["U_heat"]] = (T_in_LB - T_in_ini) / 10.0

    # Too hot → ventilate if outdoors helps; otherwise use AC cooling.
    if T_in_ini > T_in_UB:
        e = (T_in_ini - T_in_UB) / 10.0
        if T_out < T_in_ini:
            u[u_idx["U_fan"]] = e
            u[u_idx["U_nat"]] = e
        else:
            u[u_idx["U_ac"]] = e

    # Too humid → vent if outside is drier, else dehumidify
    if H_in_ini > H_in_UB:
        e = (H_in_ini - H_in_UB) / 3.0
        if H_out < H_in_ini:
            u[u_idx["U_fan"]] = max(u[u_idx["U_fan"]], e)
            u[u_idx["U_nat"]] = max(u[u_idx["U_nat"]], e)
        else:
            u[u_idx["U_deh"]] = e

    # Too dry → humidify
    if H_in_ini < H_in_LB:
        u[u_idx["U_hum"]] = (H_in_LB - H_in_ini) / 3.0

    # LED baseline schedule
    if is_night:
        u[u_idx["U_LED"]] = 0.5
    else:
        u[u_idx["U_LED"]] = 1.0

    # If indoor CO₂ is below the lower bound, dose CO₂ and stop venting.
    if C_in_ini < C_in_LB:
        u[u_idx["U_fan"]] = 0.0
        u[u_idx["U_nat"]] = 0.0
        u[u_idx["U_dos"]] = 1.0

    # If outside radiation is strong, pull shade more.
    if R_out > R_out_day_max*0.95:
        u[u_idx["U_shad"]] = 1.0

    # At night, if outside is colder than inside, activate the warm curtain.
    if is_night and T_in_ini > T_out:
        u[u_idx["U_warm"]] = 1.0

    return u


def _apply_supervisor(
    u: np.ndarray,
    x_ini: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    u_idx: dict[str, int],
    x_idx: dict[str, int],
) -> np.ndarray:
    T_in_ini = float(x_ini[x_idx["T_in"]])
    H_in_ini = float(x_ini[x_idx["H_in"]])
    T_in_LB, T_in_UB, H_in_LB, H_in_UB, _, _ = bounds

    u = u.copy()

    # Heating and AC are mutually exclusive.
    if u[u_idx["U_heat"]] > 0.0:
        u[u_idx["U_ac"]] = 0.0
    elif u[u_idx["U_ac"]] > 0.0:
        u[u_idx["U_heat"]] = 0.0

    # If humidifier is on, force dehumidifier off.
    # If dehumidifier is on, force humidifier off.
    if u[u_idx["U_hum"]] > 0.0:
        u[u_idx["U_deh"]] = 0.0
    elif u[u_idx["U_deh"]] > 0.0:
        u[u_idx["U_hum"]] = 0.0

    # If temperature or humidity is more than 2 units beyond its bounds, treat it as emergency.
    emergency = (
        (T_in_ini < T_in_LB - 2.0) # For temperature: 2°C beyond.
        or (T_in_ini > T_in_UB + 2.0)
        or (H_in_ini < H_in_LB - 2.0) # For humidity: 2 g/m³ beyond.
        or (H_in_ini > H_in_UB + 2.0)
    )

    # If you are dosing CO₂ (U_dos > 0) and it’s not an emergency,
    # cap fan ventilation (U_fan) at 0.2;
    # cap natural ventilation (U_nat) at 0.2
    if u[u_idx["U_dos"]] > 0.0 and not emergency:
        if u[u_idx["U_fan"]] > 0.2:
            u[u_idx["U_fan"]] = 0.2
        if u[u_idx["U_nat"]] > 0.2:
            u[u_idx["U_nat"]] = 0.2

    return np.clip(u, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    season = args.season
    num_samples = int(args.num_samples)

    rng = np.random.default_rng(SEED)
    param_keeper = load_parameters(PARAM_YAML_PATH)
    excite_weights = np.asarray(EXCITE_WEIGHTS_BY_SEASON[season], dtype=float)
    u_idx = get_control_indices(param_keeper)
    x_idx = get_state_indices(param_keeper)
    d_idx = get_disturbance_indices(param_keeper)
    u_names = get_control_names_in_order(param_keeper)
    x_names = get_state_names_in_order(param_keeper)
    d_names = get_disturbance_names_in_order(param_keeper)
    num_u = get_num_controls(param_keeper)
    num_x = get_num_states(param_keeper)
    num_d = get_num_disturbances(param_keeper)

    _require_named_entries(u_idx, ("U_heat", "U_fan", "U_nat", "U_ac", "U_dos", "U_LED", "U_hum", "U_deh", "U_shad", "U_warm"), "control")
    _require_named_entries(x_idx, ("T_in", "H_in", "C_in", "L"), "state")
    _require_named_entries(d_idx, ("T_out", "H_out", "C_out", "R_out"), "disturbance")

    delta_t = float(param_keeper["delta_t"])
    min_per_step = delta_t / 60.0
    step_per_day = int(param_keeper["kappa_day_night_total_steps"])
    kappa_day_start = int(param_keeper["kappa_day_start"])
    kappa_day_end = int(param_keeper["kappa_day_end"])
    outdoor_realization = _load_outdoor_realization(OUTDOOR_REAL_CSV_BY_SEASON[season], min_per_step, step_per_day)
    R_out_day_max = float(outdoor_realization["R_out_day_max"])

    out_dir = OUT_BASE_DIR / season
    out_dir.mkdir(parents=True, exist_ok=True)

    X_ini_list = []
    U_list = []
    D_list = []
    DX_list = []
    X_end_list = []
    kappa_list = []
    hhmm_list = []
    rh_list = []
    ppm_out_list = []

    u_exc = np.full(num_u, np.nan, dtype=float)
    hold_left = 0
    attempted_samples = 0
    rejected_by_hard_checks = 0
    random_u_count = 0
    nonrandom_u_count = 0
    t0 = time.time()
    last_print_accepted = 0
    last_print_attempts = 0

    while len(X_ini_list) < num_samples:
        attempted_samples += 1
        if attempted_samples - last_print_attempts >= PRINT_EVERY_ATTEMPTS:
            elapsed = time.time() - t0
            accepted = len(X_ini_list)
            acc_rate = accepted / attempted_samples if attempted_samples else 0.0
            rej_rate = rejected_by_hard_checks / attempted_samples if attempted_samples else 0.0
            speed = accepted / elapsed if elapsed > 0 else 0.0
            remaining = num_samples - accepted
            eta_sec = remaining / speed if speed > 0 else float("inf")
            print(
                f"[attempt {attempted_samples:>9d}] accepted={accepted:>7d}/{num_samples} "
                f"acc_rate={acc_rate:6.3f} rej_rate={rej_rate:6.3f} "
                f"speed={speed:7.2f} samp/s ETA={eta_sec/60.0:7.1f} min "
                f"rand_u={random_u_count:>6d} nonrand_u={nonrandom_u_count:>6d}"
            )
            last_print_attempts = attempted_samples
        d, kappa_k, hhmm, RH_out, ppm_out, is_night = _sample_disturbance(
            rng,
            outdoor_realization,
            kappa_day_start,
            kappa_day_end,
            d_idx=d_idx,
            num_d=num_d,
        )
        bounds = _get_climate_bounds(kappa_k, param_keeper)
        x_ini = _sample_x_ini(rng, bounds, kappa_k, kappa_day_start, kappa_day_end, x_idx, num_x, param_keeper)

        if prob_random_u <= 0.0:
            use_random_u = False
        elif prob_random_u >= 1.0:
            use_random_u = True
        else:
            use_random_u = bool(rng.random() < prob_random_u)

        if use_random_u:
            u = rng.uniform(0.0, 1.0, size=(num_u,))
        else:
            u_rule_of_thumb = _sample_rule_of_thumb_u(
                x_ini,
                d,
                is_night,
                bounds,
                num_u,
                u_idx,
                x_idx,
                d_idx,
                R_out_day_max,
            )
            u_rule_of_thumb = _apply_supervisor(u_rule_of_thumb, x_ini, bounds, u_idx, x_idx)
            u = u_rule_of_thumb.copy()

            if hold_left == 0:
                n_act = int(rng.integers(EXCITE_NUM_ACT_MIN, EXCITE_NUM_ACT_MAX + 1))
                w = excite_weights / excite_weights.sum()
                idx = list(rng.choice(num_u, size=n_act, replace=False, p=w))
                for a, b in (
                    (u_idx["U_heat"], u_idx["U_ac"]),
                    (u_idx["U_fan"], u_idx["U_nat"]),
                    (u_idx["U_hum"], u_idx["U_deh"]),
                ):
                    if a in idx and b in idx:
                        idx.remove(a if rng.random() < 0.5 else b)
                idx = np.asarray(idx, dtype=int)
                u_exc[:] = np.nan
                u_exc[idx] = rng.uniform(EXCITE_LEVEL_MIN, EXCITE_LEVEL_MAX, size=idx.size)
                hold_left = int(rng.integers(EXCITE_HOLD_MIN, EXCITE_HOLD_MAX + 1))
            mask = ~np.isnan(u_exc)
            u[mask] = u_exc[mask]
            hold_left -= 1
        u = _apply_supervisor(u, x_ini, bounds, u_idx, x_idx)

        x_ini_full = np.asarray(x_ini, dtype=float).reshape(-1)
        x_end_full, _ = digital_twin_one_step(x_ini_full, u, d, param_keeper)
        x_end_full = np.asarray(x_end_full, dtype=float).reshape(-1)

        if hard_checks(x_ini_full, u, d, x_end_full, param_keeper):
            rejected_by_hard_checks += 1
            continue

        x_end = x_end_full
        dx = x_end_full - x_ini_full

        if use_random_u:
            random_u_count += 1
        else:
            nonrandom_u_count += 1

        X_ini_list.append(x_ini_full)
        U_list.append(u)
        D_list.append(d)
        DX_list.append(dx)
        X_end_list.append(x_end)
        kappa_list.append(int(kappa_k))
        hhmm_list.append(hhmm)
        rh_list.append(float(RH_out))
        ppm_out_list.append(float(ppm_out))
        accepted = len(X_ini_list)
        if accepted - last_print_accepted >= PRINT_EVERY_ACCEPTED:
            elapsed = time.time() - t0
            acc_rate = accepted / attempted_samples if attempted_samples else 0.0
            rej_rate = rejected_by_hard_checks / attempted_samples if attempted_samples else 0.0
            speed = accepted / elapsed if elapsed > 0 else 0.0
            remaining = num_samples - accepted
            eta_sec = remaining / speed if speed > 0 else float("inf")
            print(
                f"[accepted {accepted:>7d}/{num_samples}] attempt={attempted_samples:>9d} "
                f"acc_rate={acc_rate:6.3f} rej_rate={rej_rate:6.3f} "
                f"speed={speed:7.2f} samp/s ETA={eta_sec/60.0:7.1f} min "
                f"rand_u={random_u_count:>6d} nonrand_u={nonrandom_u_count:>6d} "
                f"last_kappa={kappa_k:>4d} night={str(is_night):>5}"
            )
            last_print_accepted = accepted

    X_ini = np.asarray(X_ini_list, dtype=float)
    U = np.asarray(U_list, dtype=float)
    D = np.asarray(D_list, dtype=float)
    DX = np.asarray(DX_list, dtype=float)
    X_end = np.asarray(X_end_list, dtype=float)

    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    x_ini_cols = [f"X_ini_{name}" for name in x_names]
    u_cols = u_names
    d_cols = [f"D_{name}" for name in d_names]
    dx_names = [f"d{name}" for name in x_names]
    dx_cols = [f"DX_{name}" for name in dx_names]
    x1_cols = [f"X1_{name}" for name in x_names]
    data_df = pd.DataFrame(
        np.hstack([X_ini, U, D, DX, X_end]),
        columns=[*x_ini_cols, *u_cols, *d_cols, *dx_cols, *x1_cols],
    )
    data_df["kappa_k"] = kappa_list
    data_df["HHMM"] = hhmm_list
    data_df["RH_out"] = rh_list
    data_df["ppm_out"] = ppm_out_list
    rounded_cols = [c for c in data_df.columns if c not in ("kappa_k", "HHMM")]
    data_df[rounded_cols] = data_df[rounded_cols].round(2)
    data_df.to_csv(out_dir / "data.csv", index=False)

    meta = {
        "kappa_k": kappa_list,
        "HHMM": hhmm_list,
        "RH_out": rh_list,
        "ppm_out": ppm_out_list,
        "seed": SEED,
        "num_samples": int(num_samples),
        "season": str(season),
        "excite_weights": excite_weights.tolist(),
        "prob_random_u": float(prob_random_u),
        "random_u_count": int(random_u_count),
        "nonrandom_u_count": int(nonrandom_u_count),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def print_stats_block(name: str, arr: np.ndarray, cols: list[str]) -> None:
        vmin = np.min(arr, axis=0)
        v05 = np.quantile(arr, 0.05, axis=0)
        vmean = np.mean(arr, axis=0)
        v95 = np.quantile(arr, 0.95, axis=0)
        vmax = np.max(arr, axis=0)
        print(f"\n{name}")
        print(f"{'col':<10} {'min':>10} {'p05':>10} {'mean':>10} {'p95':>10} {'max':>10}")
        for i, col in enumerate(cols):
            print(f"{col:<10} {vmin[i]:10.4f} {v05[i]:10.4f} {vmean[i]:10.4f} {v95[i]:10.4f} {vmax[i]:10.4f}")

    accepted_samples = int(X_ini.shape[0])
    rej_rate = rejected_by_hard_checks / attempted_samples if attempted_samples > 0 else 0.0
    kappa_arr = np.asarray(kappa_list, dtype=int)
    night_count = int(np.sum((kappa_arr < kappa_day_start) | (kappa_arr >= kappa_day_end)))
    day_count = int(accepted_samples - night_count)

    print("\n========== DATA SUMMARY ==========")
    print(f"{'attempted_samples':<28}{attempted_samples:>10d}")
    print(f"{'accepted_samples':<28}{accepted_samples:>10d}")
    print(f"{'rejected_by_hard_checks':<28}{rejected_by_hard_checks:>10d}")
    print(f"{'rejection_rate':<28}{rej_rate:>10.4f}")
    print(f"{'accepted_night_count':<28}{night_count:>10d}")
    print(f"{'accepted_day_count':<28}{day_count:>10d}")
    print_stats_block(f"X_ini [{', '.join(x_names)}]", X_ini, x_names)
    print_stats_block(f"X_end [{', '.join(x_names)}]", X_end, x_names)
    print_stats_block(
        f"U [{', '.join(u_names)}]",
        U,
        u_names,
    )
    print_stats_block(f"D [{', '.join(d_names)}]", D, d_names)
    print_stats_block(f"DX [{', '.join(dx_names)}]", DX, dx_names)
    frac_005 = np.mean(U > 0.05, axis=0)
    frac_020 = np.mean(U > 0.20, axis=0)
    frac_050 = np.mean(U > 0.50, axis=0)
    print("\nU coverage (fraction of samples)")
    print(f"{'act':<10} {'>0.05':>10} {'>0.20':>10} {'>0.50':>10}")
    for i, name in enumerate(u_names):
        print(f"{name:<10} {frac_005[i]:10.4f} {frac_020[i]:10.4f} {frac_050[i]:10.4f}")


if __name__ == "__main__":
    main()
