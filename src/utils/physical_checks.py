# src/utils/physical_checks.py
from dataclasses import dataclass
from typing import Any
from math import exp
import numpy as np

from src.utils.vector_order import get_control_indices, get_disturbance_indices, get_num_states, get_state_indices


@dataclass
class Violation:
    rule: str
    message: str
    value: float | None
    threshold: float | None
    state: str | None


# ---------------- Helpers ----------------

def _H_sat(T_C: float, params: dict) -> float:
    """Saturation absolute humidity (g/m^3) using your model: H_sat = H_tilde * exp(delta_sat * T)."""
    a = float(params.get("H_sat_tilde", params.get("H_tilde", 1.0)))
    b = float(params.get("H_sat_delta", params.get("delta_sat", 0.0)))
    return a * exp(b * float(T_C))


def _rh_from_T_H(T_C: float, H_gm3: float, params: dict, eps: float = 1e-12) -> float:
    """RH (%) computed from absolute humidity and saturation curve, clipped to [0, 100]."""
    Hsat = _H_sat(T_C, params)
    RH = 100.0 * float(H_gm3) / max(Hsat, eps)
    return float(np.clip(RH, 0.0, 100.0))

def hard_checks(
    x0: Any,
    u: Any,
    d: Any,
    x1: Any,
    params: dict,
    eps: float = 1e-9,
) -> list[Violation]:
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    num_x = get_num_states(params)
    if x0.size != num_x or x1.size != num_x:
        return []

    x_idx = get_state_indices(params)
    T_in = float(x1[x_idx["T_in"]])
    H_in = float(x1[x_idx["H_in"]])
    C_in = float(x1[x_idx["C_in"]])
    L = float(x1[x_idx["L"]])
    dx = x1 - x0
    dT = float(dx[x_idx["T_in"]])
    dH = float(dx[x_idx["H_in"]])
    dC = float(dx[x_idx["C_in"]])
    dL = float(dx[x_idx["L"]])

    violations: list[Violation] = []

    def add(rule: str, message: str, value: float | None, threshold: float | None, state: str | None):
        violations.append(Violation(rule=rule, message=message, value=value, threshold=threshold, state=state))

    if u.size > 0:
        u_idx = get_control_indices(params)
        max_required_u_idx = max(
            u_idx["U_ac"],
            u_idx["U_heat"],
            u_idx["U_hum"],
            u_idx["U_deh"],
        )
        if u.size > max_required_u_idx:
            U_ac = float(u[u_idx["U_ac"]])
            U_heat = float(u[u_idx["U_heat"]])
            U_hum = float(u[u_idx["U_hum"]])
            U_deh = float(u[u_idx["U_deh"]])

            if U_ac > eps and U_heat > eps:
                add(
                    "u_ac_heat_mutex",
                    "U_ac and U_heat are both active",
                    float(min(U_ac, U_heat)),
                    0.0,
                    None,
                )
            if U_hum > eps and U_deh > eps:
                add(
                    "u_hum_deh_mutex",
                    "U_hum and U_deh are both active",
                    float(min(U_hum, U_deh)),
                    0.0,
                    None,
                )

    if H_in < -eps:
        add("nonneg_H_in", "H_in < 0", float(H_in), 0.0, "H_in")
    if C_in < -eps:
        add("nonneg_C_in", "C_in < 0", float(C_in), 0.0, "C_in")
    if L < -eps:
        add("nonneg_L", "L < 0", float(L), 0.0, "L")

    if T_in < -20 - eps:
        add("T_in_min", "T_in < -20", float(T_in), -20.0, "T_in")
    if T_in > 60 + eps:
        add("T_in_max", "T_in > 60", float(T_in), 60.0, "T_in")
    if H_in < 0 - eps:
        add("H_in_min", "H_in < 0", float(H_in), 0.0, "H_in")
    if H_in > 50 + eps:
        add("H_in_max", "H_in > 50", float(H_in), 50.0, "H_in")
    if C_in < 0 - eps:
        add("C_in_min", "C_in < 0", float(C_in), 0.0, "C_in")
    if C_in > 20 + eps:
        add("C_in_max", "C_in > 20", float(C_in), 20.0, "C_in")
    if L < 0 - eps:
        add("L_min", "L < 0", float(L), 0.0, "L")
    if L > 1e6 + eps:
        add("L_max", "L > 1e6", float(L), 1e6, "L")

    if abs(dT) > 10 + eps:
        add("dT_in_max", "|ΔT_in| > 10", float(dT), 10.0, "T_in")
    if abs(dH) > 10 + eps:
        add("dH_in_max", "|ΔH_in| > 10", float(dH), 10.0, "H_in")
    if abs(dC) > 10 + eps:
        add("dC_in_max", "|ΔC_in| > 10", float(dC), 10.0, "C_in")
    if dL < -1e-6 - eps:
        add("dL_min", "ΔL < 0", float(dL), -1e-6, "L")

    a = float(params.get("H_sat_tilde", params.get("H_tilde", 1.0)))
    b = float(params.get("H_sat_delta", params.get("delta_sat", 0.0)))
    H_sat = a * exp(b * float(T_in))
    if H_in > H_sat + eps:
        add("H_in_sat", "H_in > H_sat", float(H_in), float(H_sat), "H_in")

    return violations

def soft_checks(
    x0: Any,
    u: Any,
    d: Any,
    x1: Any,
    params: dict | None = None,
) -> list[Violation]:
    params = params or {}

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    d = np.asarray(d, dtype=float).reshape(-1)
    num_x = get_num_states(params)
    if x0.size != num_x or x1.size != num_x:
        return []

    x_idx = get_state_indices(params)
    d_idx = get_disturbance_indices(params)
    T_in = float(x1[x_idx["T_in"]])
    H_in = float(x1[x_idx["H_in"]])
    C_in = float(x1[x_idx["C_in"]])
    d_state = x1 - x0
    dT = float(d_state[x_idx["T_in"]])
    dH = float(d_state[x_idx["H_in"]])
    dC = float(d_state[x_idx["C_in"]])
    dL = float(d_state[x_idx["L"]])

    violations: list[Violation] = []

    def add(rule: str, message: str, value: float | None, threshold: float | None, state: str | None):
        violations.append(Violation(rule=rule, message=message, value=value, threshold=threshold, state=state))

    # basic plausibility
    if T_in < -5:
        add("plausible_T_in_min", "T_in < -5", float(T_in), -5.0, "T_in")
    if T_in > 50:
        add("plausible_T_in_max", "T_in > 50", float(T_in), 50.0, "T_in")
    if C_in < 0:
        add("plausible_C_in_min", "C_in < 0", float(C_in), 0.0, "C_in")
    if C_in > 10:
        add("plausible_C_in_max", "C_in > 10", float(C_in), 10.0, "C_in")
    if dL > 5:
        add("dL_large", "ΔL > 5", float(dL), 5.0, "L")

    # fan/nat direction checks
    if u.size >= 4 and d.size >= 2:
        u_idx = get_control_indices(params)
        max_required_idx = max(u_idx["U_fan"], u_idx["U_nat"], u_idx["U_ac"])
        if u.size <= max_required_idx:
            return violations

        U_fan = float(u[u_idx["U_fan"]])
        U_nat = float(u[u_idx["U_nat"]])
        T_out = float(d[d_idx["T_out"]])
        H_out = float(d[d_idx["H_out"]])

        # natural stream compares to outdoor air
        if (float(x0[x_idx["T_in"]]) - T_out) > 5 and U_nat > 0.8 and dT > 0:
            add("nat_T_dir", "ΔT_in > 0 with strong natural vent and T_in > T_out", float(dT), 0.0, "T_in")
        if (float(x0[x_idx["H_in"]]) - H_out) > 3 and U_nat > 0.8 and dH > 0:
            add("nat_H_dir", "ΔH_in > 0 with strong natural vent and H_in > H_out", float(dH), 0.0, "H_in")

    return violations
