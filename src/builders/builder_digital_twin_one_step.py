# src/builders/builder_digital_twin_one_step.py
"""Digital twin one-step builder using RK4 integration with hops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from src.utils.vector_order import (
    get_control_indices,
    get_disturbance_indices,
    get_state_indices,
    get_num_controls,
    get_num_disturbances,
    get_num_states,
)

EPS_SAT = 1e-9

DEFAULT_PARAMS_PATH = (Path(__file__).resolve().parents[2] / "configs/var_and_param_keeper.yaml")


def load_parameters(yaml_path: Path) -> dict:
    data_keeper = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    param_keeper = data_keeper["parameters"]
    param_keeper["x_ini"] = data_keeper["variables"]["x_ini"]
    return param_keeper


def digital_twin_one_step(
    x: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    param_keeper: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """One-step digital twin propagation using RK4 with hops.

    Over each control interval, RK4 approximates the integral of the continuous-time dynamics to produce the one-step state update.

    State, disturbance, and control orders are defined by
    `x_idx_*`, `d_idx_*`, and `u_idx_*` entries in `var_and_param_keeper.yaml`.
    """
    num_hops = param_keeper["num_hops"]

    x_vec = _as_flat_array(x)
    u_vec = _as_flat_array(u)
    d_vec = _as_flat_array(d)
    num_x = get_num_states(param_keeper)
    num_u = get_num_controls(param_keeper)
    num_d = get_num_disturbances(param_keeper)
    if x_vec.size != num_x:
        raise ValueError(f"x size mismatch: expected {num_x}, got {x_vec.size}")
    if u_vec.size != num_u:
        raise ValueError(f"u size mismatch: expected {num_u}, got {u_vec.size}")
    if d_vec.size != num_d:
        raise ValueError(f"d size mismatch: expected {num_d}, got {d_vec.size}")
    _, info_ini = _dynamics(x_vec, u_vec, d_vec, param_keeper)

    delta_t = float(param_keeper["delta_t"])
    h = delta_t / float(num_hops)

    x_next = x_vec.astype(float).copy()
    for _ in range(num_hops):
        k1, _ = _dynamics(x_next, u_vec, d_vec, param_keeper)
        k2, _ = _dynamics(x_next + 0.5 * h * k1, u_vec, d_vec, param_keeper)
        k3, _ = _dynamics(x_next + 0.5 * h * k2, u_vec, d_vec, param_keeper)
        k4, _ = _dynamics(x_next + h * k3, u_vec, d_vec, param_keeper)
        x_next = x_next + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_next = _project_state_physical(x_next, param_keeper)

    _, info_end = _dynamics(x_next, u_vec, d_vec, param_keeper)
    return x_next, {"initial": info_ini, "end": info_end}


def _as_flat_array(vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec, dtype=float).reshape(-1)


def _project_state_physical(x: np.ndarray, param_keeper: Dict[str, Any]) -> np.ndarray:
    x_idx = get_state_indices(param_keeper)
    idx_T_in = x_idx["T_in"]
    idx_H_in = x_idx["H_in"]
    idx_C_in = x_idx["C_in"]
    idx_L = x_idx["L"]

    x_proj = np.asarray(x, dtype=float).reshape(-1).copy()

    T_in = float(x_proj[idx_T_in])
    H_in = float(x_proj[idx_H_in])
    C_in = float(x_proj[idx_C_in])
    L = float(x_proj[idx_L])

    # Rule 1: C_in >= 0
    if C_in < 0.0:
        C_in = 0.0

    # Rule 2: H_in >= 0
    if H_in < 0.0:
        H_in = 0.0

    # Rule 3: L >= 0
    if L < 0.0:
        L = 0.0

    # Rule 4: H_in <= H_in_sat
    H_tilde = float(param_keeper["H_tilde"])
    delta_sat = float(param_keeper["delta_sat"])
    H_in_sat = H_tilde * np.exp(delta_sat * T_in)
    H_max = H_in_sat - EPS_SAT
    if H_in > H_max:
        H_in = H_max

    x_proj[idx_T_in] = T_in
    x_proj[idx_H_in] = H_in
    x_proj[idx_C_in] = C_in
    x_proj[idx_L] = L
    return x_proj


def _smooth_pos(z: float, sigma: float) -> float:
    return 0.5 * (z + np.sqrt(z * z + sigma * sigma))

def _H_sat(T_C: float, H_tilde: float, delta_sat: float) -> float:
    return float(H_tilde * np.exp(delta_sat * float(T_C)))

def _rh_from_T_H(T_C: float, H_gm3: float, H_tilde: float, delta_sat: float, eps: float = 1e-12) -> float:
    H_sat = _H_sat(T_C, H_tilde, delta_sat)
    RH = 100.0 * float(H_gm3) / max(H_sat, eps)
    return float(np.clip(RH, 0.0, 100.0))

def _dynamics(
    x: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    param_keeper: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, float]]:
    # Unpack state, control, disturbances
    x_idx = get_state_indices(param_keeper)
    d_idx = get_disturbance_indices(param_keeper)
    T_in = float(x[x_idx["T_in"]])
    H_in = float(x[x_idx["H_in"]])
    C_in = float(x[x_idx["C_in"]])
    L = float(x[x_idx["L"]])
    u_idx = get_control_indices(param_keeper)

    U_heat = float(u[u_idx["U_heat"]])
    U_fan = float(u[u_idx["U_fan"]])
    U_nat = float(u[u_idx["U_nat"]])
    U_ac = float(u[u_idx["U_ac"]])
    U_dos = float(u[u_idx["U_dos"]])
    U_LED = float(u[u_idx["U_LED"]])
    U_hum = float(u[u_idx["U_hum"]])
    U_deh = float(u[u_idx["U_deh"]])
    U_shad = float(u[u_idx["U_shad"]])
    U_warm = float(u[u_idx["U_warm"]])

    T_out = float(d[d_idx["T_out"]])
    H_out = float(d[d_idx["H_out"]])
    C_out = float(d[d_idx["C_out"]])
    R_out = float(d[d_idx["R_out"]])

    # Unpack parameters
    A_floor = param_keeper["A_floor"]
    A_roof = param_keeper["A_roof"]
    A_wall = param_keeper["A_wall"]
    c_vh = param_keeper["c_vh"]
    C_hat_in = param_keeper["C_hat_in"]
    D_bar_dos = param_keeper["D_bar_dos"]
    D_bar_ass = param_keeper["D_bar_ass"]
    F_bar_hum = param_keeper["F_bar_hum"]
    F_bar_deh = param_keeper["F_bar_deh"]
    H_tilde = param_keeper["H_tilde"]
    I = param_keeper["I"]
    rho = param_keeper["rho"]
    P_bar_LED = param_keeper["P_bar_LED"]
    q_evap = param_keeper["q_evap"]
    q_deh = param_keeper["q_deh"]
    Q_bar_heat = param_keeper["Q_bar_heat"]
    Q_bar_ac = param_keeper["Q_bar_ac"]
    r_b = param_keeper["r_b"]
    r_bar_s = param_keeper["r_bar_s"]
    r_under_s = param_keeper["r_under_s"]
    tau = param_keeper["tau"]
    T_sr = param_keeper["T_sr"]
    upsilon_roof = param_keeper["upsilon_roof"]
    upsilon_wall = param_keeper["upsilon_wall"]
    nu_surf = param_keeper["nu_surf"]
    V_gh = param_keeper["V_gh"]
    V_bar_fan = param_keeper["V_bar_fan"]
    V_bar_nat = param_keeper["V_bar_nat"]
    omega_T = param_keeper["omega_T"]
    omega_H = param_keeper["omega_H"]
    omega_C = param_keeper["omega_C"]
    chi = param_keeper["chi"]
    phi_solar = param_keeper["phi_solar"]
    phi_LED = param_keeper["phi_LED"]
    lambda_leak = param_keeper["lambda_leak"]
    zeta = param_keeper["zeta"]
    sigma = param_keeper["sigma"]
    delta_tran = param_keeper["delta_tran"]
    delta_sat = param_keeper["delta_sat"]
    delta_vc = param_keeper["delta_vc"]
    delta_sr = param_keeper["delta_sr"]
    eta_LED_r = param_keeper["eta_LED_r"]
    eta_evap = param_keeper["eta_evap"]
    eta_cover = param_keeper["eta_cover"]
    eta_tran = param_keeper["eta_tran"]
    eta_shad = param_keeper["eta_shad"]
    eta_warm = param_keeper["eta_warm"]

    # Intermediate variables
    T_o_i = T_out - T_in
    T_cover = (2.0 * T_out + T_in) / 3.0  # Cover temperature

    V_fan = V_bar_fan * U_fan
    V_nat = V_bar_nat * U_nat
    V_leak = (lambda_leak * V_gh) / 3600.0
    V_vent = V_fan + V_nat + V_leak

    R_roof_LED = (eta_LED_r * P_bar_LED * U_LED) / A_floor
    R_roof_solar = eta_cover * (1.0 - eta_shad * U_shad) * (1.0 - U_warm) * R_out
    R_cano_glo = 0.86 * (1 - np.exp(-(0.7 * I))) * (R_roof_solar + R_roof_LED)

    X_cano = phi_solar * R_roof_solar + phi_LED * P_bar_LED * U_LED # PPFD

    H_in_sat = _H_sat(T_in, H_tilde, delta_sat) - EPS_SAT             # Saturation humidity
    H_cano = H_in_sat + chi * (r_b * R_cano_glo) / (2.0 * I * q_evap) # Canopy humidity

    r_s = (r_under_s + r_bar_s * np.exp(-(tau * R_cano_glo) / I)) * ( 1.0 + delta_sr * (T_in - T_sr) ** 2) # Stomatal resistance

    g_tran = (2.0 * I) / ((1.0 + eta_tran * np.exp(delta_tran * T_in)) * r_b + r_s) # Transpiration conductance
    g_vc = nu_surf * (_smooth_pos(T_in - T_cover, sigma) ** (1.0 / 3.0))            # Condensation conductance

    # Temperature fluxes Q
    Q_heat = Q_bar_heat * U_heat
    Q_vent = c_vh * T_o_i * V_vent
    Q_cool = Q_bar_ac * U_ac + q_evap * eta_evap * F_bar_hum * U_hum
    Q_solar = A_floor * R_roof_solar
    Q_o_i = (upsilon_roof * A_roof * (1.0 - eta_warm * U_warm) + upsilon_wall * A_wall) * T_o_i
    Q_LED = P_bar_LED * U_LED
    Q_tran = q_evap * g_tran * (H_cano - H_in) * A_floor
    Q_deh = q_deh * F_bar_deh * U_deh

    # Humidity fluxes F
    F_hum = eta_evap * F_bar_hum * U_hum
    F_deh = F_bar_deh * U_deh
    F_vent = (H_out - H_in) * V_vent
    F_tran = Q_tran / q_evap
    s_vc = zeta * np.exp(delta_vc * T_in) * (-1.0 * T_o_i) - (H_in_sat - H_in)
    F_vc = g_vc * _smooth_pos(s_vc, sigma) * A_floor

    # CO2 fluxes D
    D_dos = D_bar_dos * U_dos
    D_vent = (C_out - C_in) * V_vent
    D_ass = (D_bar_ass * (C_in / (C_in + C_hat_in)) * (1.0 - np.exp(-rho * R_cano_glo)) * A_floor)

    # ODEs
    dT_in_dt = (Q_heat + Q_vent - Q_cool + Q_solar + Q_o_i + Q_LED - Q_tran + Q_deh) / (omega_T * V_gh * c_vh)
    dH_in_dt = (F_hum - F_deh + F_vent + F_tran - F_vc) / (omega_H * V_gh)
    dC_in_dt = (D_dos + D_vent - D_ass) / (omega_C * V_gh)
    dL_dt = X_cano / 1e6

    info = {
        "D_dos": float(D_dos),
        "D_vent": float(D_vent),
        "D_ass": float(D_ass),
        "F_hum": float(F_hum),
        "F_deh": float(F_deh),
        "F_vent": float(F_vent),
        "F_tran": float(F_tran),
        "F_vc": float(F_vc),
        "g_vc": float(g_vc),
        "g_tran": float(g_tran),
        "H_in_sat": float(H_in_sat),
        "H_cano": float(H_cano),
        "Q_heat": float(Q_heat),
        "Q_vent": float(Q_vent),
        "Q_cool": float(Q_cool),
        "Q_solar": float(Q_solar),
        "Q_o_i": float(Q_o_i),
        "Q_LED": float(Q_LED),
        "Q_tran": float(Q_tran),
        "Q_deh": float(Q_deh),
        "r_s": float(r_s),
        "R_roof_solar": float(R_roof_solar),
        "R_roof_LED": float(R_roof_LED),
        "R_cano_glo": float(R_cano_glo),
        "T_cover": float(T_cover),
        "T_o_i": float(T_o_i),
        "V_fan": float(V_fan),
        "V_nat": float(V_nat),
        "V_leak": float(V_leak),
        "V_vent": float(V_vent),
        "X_cano": float(X_cano),
    }

    dxdt = np.zeros(get_num_states(param_keeper), dtype=float)
    dxdt[x_idx["T_in"]] = dT_in_dt
    dxdt[x_idx["H_in"]] = dH_in_dt
    dxdt[x_idx["C_in"]] = dC_in_dt
    dxdt[x_idx["L"]] = dL_dt
    return dxdt, info


__all__ = ["digital_twin_one_step", "load_parameters"]
