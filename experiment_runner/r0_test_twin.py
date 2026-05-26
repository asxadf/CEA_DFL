# experiment_runner/r0_test_twin.py

"""
Smoke test and analyze one-step digital twin behavior.

CLI:
python3 experiment_runner/r0_test_twin.py
python3 experiment_runner/r0_test_twin.py --num-tests 500 --seed 123
python3 experiment_runner/r0_test_twin.py --no-run-scenarios
"""

import argparse
import json
from pathlib import Path
import sys
import shutil
from typing import Iterable, Sequence

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.builders.builder_digital_twin_one_step import digital_twin_one_step, load_parameters
from src.utils.vector_order import (
    get_control_names_in_order,
    get_disturbance_indices,
    get_disturbance_names_in_order,
    get_state_indices,
    get_state_names_in_order,
)
from src.utils.physical_checks import hard_checks, soft_checks

PARAM_YAML_PATH = Path(__file__).resolve().parents[1] / "configs/var_and_param_keeper.yaml"
DEFAULT_NUM_TESTS = 500
DEFAULT_SEED = 23
DEFAULT_PROFILE = "wide"
DEFAULT_WRITE_HARD_VIOLATIONS_JSONL = True
DEFAULT_WRITE_SOFT_VIOLATIONS_JSONL = True
DEFAULT_RUN_SCENARIOS = True
DEFAULT_SCENARIO_JITTER_N = 1

U_RANGES = {
    "wide": {
        "U_heat": (0.0, 1.0),
        "U_fan": (0.0, 1.0),
        "U_nat": (0.0, 1.0),
        "U_ac":  (0.0, 1.0),
        "U_dos": (0.0, 1.0),
        "U_LED": (0.0, 1.0),
        "U_hum": (0.0, 1.0),
        "U_deh": (0.0, 1.0),
        "U_shad": (0.0, 1.0),
        "U_warm": (0.0, 1.0),
    }
}

D_RANGES = {
    "wide": {
        "T_out": (-5.0, 40.0),
        "H_out": (1.0, 15.0),
        "C_out": (0.1, 5.0),
        "R_out": (0.0, 1500.0),
    }
}

SCENARIOS = [
    # --- Baselines ---
    {"name": "all_off_at_normal_day", "u": {}, "d": {"T_out": 17.5}},
    {"name": "all_off_at_normal_night", "u": {}, "d": {"R_out": 0.0}},
    # --- Shade vs no-shade (same d) ---
    {"name": "no_shade_at_normal_day", "u": {"U_shad": 0.0}},
    {"name": "full_shade_at_normal_day", "u": {"U_shad": 1.0}},
    # --- Blackout vs no-blackout (normal) ---
    {"name": "no_blackout_at_normal_day", "u": {"U_warm": 0.0}},
    {"name": "full_blackout_at_normal_day", "u": {"U_warm": 1.0}},
    # --- Blackout vs no-blackout (cold night) ---
    {"name": "no_blackout_at_cold_night", "u": {"U_warm": 0.0}, "d": {"T_out": 1.0, "R_out": 0.0}},
    {"name": "full_blackout_at_cold_night", "u": {"U_warm": 1.0}, "d": {"T_out": 1.0, "R_out": 0.0}},
    # --- Vent modes (warm wet lowCO2 night) ---
    {
        "name": "vent_leak_only_at_warm_wet_lowCO2_night",
        "u": {"U_fan": 0.0, "U_nat": 0.0},
        "d": {"T_out": 20.0, "H_out": 10.0, "C_out": 1.0, "R_out": 0.0},
    },
    {
        "name": "vent_nat_only_at_warm_wet_lowCO2_night",
        "u": {"U_nat": 1.0, "U_fan": 0.0},
        "d": {"T_out": 20.0, "H_out": 10.0, "C_out": 1.0, "R_out": 0.0},
    },
    {
        "name": "vent_fan_only_at_warm_wet_lowCO2_night",
        "u": {"U_nat": 0.0, "U_fan": 1.0},
        "d": {"T_out": 20.0, "H_out": 10.0, "C_out": 1.0, "R_out": 0.0},
    },
    # --- Vent modes (warm wet lowCO2 night) ---
    {
        "name": "vent_leak_only_at_cold_dry_highCO2_night",
        "u": {"U_fan": 0.0, "U_nat": 0.0},
        "d": {"T_out": 1.0, "H_out": 1.0, "C_out": 5.0, "R_out": 0.0},
    },
    {
        "name": "vent_nat_only_at_cold_dry_highCO2_night",
        "u": {"U_nat": 1.0, "U_fan": 0.0},
        "d": {"T_out": 1.0, "H_out": 1.0, "C_out": 5.0, "R_out": 0.0},
    },
    {
        "name": "vent_fan_only_at_cold_dry_highCO2_night",
        "u": {"U_nat": 0.0, "U_fan": 1.0},
        "d": {"T_out": 1.0, "H_out": 1.0, "C_out": 5.0, "R_out": 0.0},
    },
    {"name": "led_only_at_normal_day", "u": {"U_LED": 1.0}, "d": {}},
    {"name": "led_only_at_normal_night", "u": {"U_LED": 1.0}, "d": {"R_out": 0.0}},
    {"name": "co2_dose_only_at_normal_night", "u": {"U_dos": 1.0}, "d": {"R_out": 0.0}},
    {"name": "humidify_only_at_normal_night", "u": {"U_hum": 1.0}, "d": {"R_out": 0.0}},
    {"name": "dehumidify_only_at_normal_night", "u": {"U_deh": 1.0}, "d": {"R_out": 0.0}},
    # --- Single-actuator probes (day / normal) ---
    {"name": "heaters_only_at_normal_day", "u": {"U_heat": 1.0}},  # normal d
    {"name": "ac_only_at_normal_day", "u": {"U_ac": 1.0}},  # normal d
    {"name": "humidify_only_at_normal_day", "u": {"U_hum": 1.0}},  # normal d
    # --- Vent stress tests ---
    {"name": "vent_max_at_cold_night", "u": {"U_fan": 1.0, "U_nat": 1.0}, "d": {"T_out": 1.0, "R_out": 0.0}},
    {
        "name": "fan_max_at_cold_dry_lowCO2_night",
        "u": {"U_fan": 1.0},
        "d": {"T_out": 1.0, "H_out": 1.0, "C_out": 0.5, "R_out": 0.0},
    },
    {
        "name": "fan_max_at_hot_wet_highCO2_night",
        "u": {"U_fan": 1.0},
        "d": {"T_out": 35.0, "H_out": 10.0, "C_out": 5.0, "R_out": 0.0},
    },
    {
        "name": "ac_at_hot_wet_night",
        "u": {"U_ac": 1.0},
        "d": {"T_out": 25.0, "H_out": 15.0, "C_out": 1.0, "R_out": 0.0},
    },
]


def _int_ge_1(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test one-step digital twin and summarize stability/physical-check diagnostics."
    )
    parser.add_argument("--num-tests", type=_int_ge_1, default=DEFAULT_NUM_TESTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--profile", choices=tuple(U_RANGES.keys()), default=DEFAULT_PROFILE)
    parser.add_argument("--param-yaml-path", type=Path, default=PARAM_YAML_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiment_result" / "smoke_test_one_step",
    )
    parser.add_argument(
        "--write-hard-violations-jsonl",
        dest="write_hard_violations_jsonl",
        action="store_true",
        default=DEFAULT_WRITE_HARD_VIOLATIONS_JSONL,
    )
    parser.add_argument(
        "--no-write-hard-violations-jsonl",
        dest="write_hard_violations_jsonl",
        action="store_false",
    )
    parser.add_argument(
        "--write-soft-violations-jsonl",
        dest="write_soft_violations_jsonl",
        action="store_true",
        default=DEFAULT_WRITE_SOFT_VIOLATIONS_JSONL,
    )
    parser.add_argument(
        "--no-write-soft-violations-jsonl",
        dest="write_soft_violations_jsonl",
        action="store_false",
    )
    parser.add_argument(
        "--run-scenarios",
        dest="run_scenarios",
        action="store_true",
        default=DEFAULT_RUN_SCENARIOS,
    )
    parser.add_argument("--no-run-scenarios", dest="run_scenarios", action="store_false")
    parser.add_argument("--scenario-jitter-n", type=_int_ge_1, default=DEFAULT_SCENARIO_JITTER_N)
    return parser.parse_args()
def _fmt(x: float, width: int = 12, prec: int = 4) -> str:
    if np.isnan(x):
        return f"{'nan':>{width}}"
    if np.isinf(x):
        return f"{'inf':>{width}}"
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:>{width}.{prec}e}"
    return f"{x:>{width}.{prec}f}"


def _print_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    widths = [len(h) for h in headers]
    row_list = []
    for r in rows:
        r = list(r)
        widths = [max(w, len(c)) for w, c in zip(widths, r)]
        row_list.append(r)

    def line(ch: str = "-") -> str:
        return "+".join(ch * (w + 2) for w in widths)

    print(line("-"))
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print(line("="))
    for r in row_list:
        print("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
    print(line("-"))


def _append_info_value(target: list[float], info: dict, *keys: str) -> None:
    for key in keys:
        if key in info:
            target.append(float(info[key]))
            return


def _summarize_arrays(
    X1: np.ndarray,
    DX: np.ndarray,
    state_names: Sequence[str],
    top_k: int = 5,
) -> dict:
    nan_rows = np.isnan(X1).any(axis=1) | np.isnan(DX).any(axis=1)
    inf_rows = np.isinf(X1).any(axis=1) | np.isinf(DX).any(axis=1)

    safe_mask = ~(nan_rows | inf_rows)
    safe_X1 = X1[safe_mask]
    safe_DX = DX[safe_mask]

    summary: dict = {
        "num_tests": int(X1.shape[0]),
        "num_states": int(X1.shape[1]),
        "nan_count": int(nan_rows.sum()),
        "inf_count": int(inf_rows.sum()),
        "safe_count": int(safe_mask.sum()),
    }

    if safe_X1.size == 0:
        summary["note"] = "All rows contain NaN/Inf; stats skipped."
        return summary

    x1_min = np.min(safe_X1, axis=0)
    x1_mean = np.mean(safe_X1, axis=0)
    x1_max = np.max(safe_X1, axis=0)
    x1_p05 = np.percentile(safe_X1, 5, axis=0)
    x1_p95 = np.percentile(safe_X1, 95, axis=0)

    dx_abs_max = np.max(np.abs(safe_DX), axis=0)

    safe_ids = np.flatnonzero(safe_mask)
    dx_l2 = np.linalg.norm(safe_DX, axis=1)
    top_idx = np.argsort(dx_l2)[::-1][: min(top_k, dx_l2.size)]
    top_tests = [{"id": int(safe_ids[j]), "dx_l2": float(dx_l2[j])} for j in top_idx]

    summary["x1_stats"] = {
        "min": [float(v) for v in x1_min],
        "mean": [float(v) for v in x1_mean],
        "max": [float(v) for v in x1_max],
        "p05": [float(v) for v in x1_p05],
        "p95": [float(v) for v in x1_p95],
    }
    summary["dx_abs_max"] = [float(v) for v in dx_abs_max]
    summary["top_dx_l2_tests"] = top_tests
    summary["state_names"] = list(state_names)
    return summary


def _print_summary(summary: dict) -> None:
    print("\n=== Smoke test summary ===")
    print(
        f"tests={summary['num_tests']} states={summary['num_states']} "
        f"safe={summary.get('safe_count', 0)} nan={summary['nan_count']} inf={summary['inf_count']}"
    )

    if "note" in summary:
        print(f"NOTE: {summary['note']}")
        return

    names = summary["state_names"]
    x1 = summary["x1_stats"]
    dx_abs_max = summary["dx_abs_max"]

    rows = []
    for i, name in enumerate(names):
        rows.append(
            [
                name,
                _fmt(x1["min"][i]),
                _fmt(x1["p05"][i]),
                _fmt(x1["mean"][i]),
                _fmt(x1["p95"][i]),
                _fmt(x1["max"][i]),
                _fmt(dx_abs_max[i]),
            ]
        )

    _print_table(
        headers=["state", "x1_min", "x1_p05", "x1_mean", "x1_p95", "x1_max", "max|dx|"],
        rows=rows,
    )

    top = summary.get("top_dx_l2_tests", [])
    if top:
        print("\nTop outlier tests by ||dx||₂:")
        for t in top:
            print(f"  - id={t['id']:<4d}  ||dx||₂={t['dx_l2']:.6g}")

    if "hard_fail_count" in summary:
        rules = summary.get("hard_rule_counts", {})
        top_rules = sorted(rules.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_rules_str = ", ".join([f"{k}:{v}" for k, v in top_rules])
        first_ids = summary.get("hard_fail_ids", [])[:3]
        print(
            f"hard_fail_count={summary['hard_fail_count']} "
            f"top_rules=[{top_rules_str}] first_fail_ids={first_ids}"
        )

    if "soft_fail_count" in summary:
        rules = summary.get("soft_rule_counts", {})
        top_rules = sorted(rules.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_rules_str = ", ".join([f"{k}:{v}" for k, v in top_rules])
        first_ids = summary.get("soft_fail_ids", [])[:3]
        print(
            f"soft_fail_count={summary['soft_fail_count']} "
            f"top_rules=[{top_rules_str}] first_fail_ids={first_ids}"
        )


def main() -> None:
    args = parse_args()
    params = load_parameters(args.param_yaml_path)
    num_hops = int(params["num_hops"])
    u_keys = get_control_names_in_order(params)
    x_keys = get_state_names_in_order(params)
    d_keys = get_disturbance_names_in_order(params)
    x_idx = get_state_indices(params)
    d_idx = get_disturbance_indices(params)

    x0 = np.array(params["x_ini"], float)
    rng = np.random.default_rng(args.seed)
    u_ranges = U_RANGES[args.profile]
    d_ranges = D_RANGES[args.profile]

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Smoke test config ===")
    print(f"profile={args.profile} tests={args.num_tests} seed={args.seed} num_hops={num_hops}")
    print(f"out_dir={out_dir}")

    X0 = np.zeros((args.num_tests, x0.size), float)
    U = np.zeros((args.num_tests, len(u_keys)), float)
    D = np.zeros((args.num_tests, len(d_keys)), float)
    X1 = np.zeros((args.num_tests, x0.size), float)
    DX = np.zeros((args.num_tests, x0.size), float)

    jsonl_path = out_dir / "experiment_result.jsonl"
    viol_path = out_dir / "violations_hard.jsonl"
    viol_f = viol_path.open("w", encoding="utf-8") if args.write_hard_violations_jsonl else None
    soft_path = out_dir / "violations_soft.jsonl"
    soft_f = soft_path.open("w", encoding="utf-8") if args.write_soft_violations_jsonl else None
    hard_rule_counts = {}
    hard_fail_ids: list[int] = []
    soft_rule_counts = {}
    soft_fail_ids: list[int] = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_tests):
            u = np.array([rng.uniform(*u_ranges[k]) for k in u_keys], float)
            d = np.array([rng.uniform(*d_ranges[k]) for k in d_keys], float)
            x1, info = digital_twin_one_step(x0, u, d, params)
            dx = x1 - x0

            X0[i] = x0
            U[i] = u
            D[i] = d
            X1[i] = x1
            DX[i] = dx

            violations = hard_checks(x0, u, d, x1, params)
            hard_ok = len(violations) == 0
            if not hard_ok:
                hard_fail_ids.append(i)
                for v in violations:
                    hard_rule_counts[v.rule] = hard_rule_counts.get(v.rule, 0) + 1

            soft_violations = soft_checks(x0, u, d, x1, params)
            soft_ok = len(soft_violations) == 0
            if not soft_ok:
                soft_fail_ids.append(i)
                for v in soft_violations:
                    soft_rule_counts[v.rule] = soft_rule_counts.get(v.rule, 0) + 1

            info_initial = info.get("initial", {}) if isinstance(info, dict) else {}
            info_end = info.get("end", {}) if isinstance(info, dict) else {}
            record = {
                "id": i,
                "x0": x0.tolist(),
                "u": u.tolist(),
                "d": d.tolist(),
                "x1": x1.tolist(),
                "dx": dx.tolist(),
                "any_nan": bool(np.isnan(x1).any() or np.isnan(dx).any()),
                "any_inf": bool(np.isinf(x1).any() or np.isinf(dx).any()),
                "info": {
                    "initial": {k: float(v) for k, v in info_initial.items()},
                    "end": {k: float(v) for k, v in info_end.items()},
                },
                "hard_ok": bool(hard_ok),
                "hard_violations": [v.__dict__ for v in violations],
                "soft_ok": bool(soft_ok),
                "soft_violations": [v.__dict__ for v in soft_violations],
            }
            f.write(json.dumps(record) + "\n")
            if viol_f and not hard_ok:
                viol_f.write(json.dumps(record) + "\n")
            if soft_f and not soft_ok:
                soft_f.write(json.dumps(record) + "\n")

    if viol_f:
        viol_f.close()
    if soft_f:
        soft_f.close()

    np.savez(out_dir / "arrays.npz", X0=X0, U=U, D=D, X1=X1, DX=DX)

    summary = _summarize_arrays(X1=X1, DX=DX, state_names=x_keys, top_k=5)
    summary["hard_fail_count"] = len(hard_fail_ids)
    summary["hard_rule_counts"] = hard_rule_counts
    summary["hard_fail_ids"] = hard_fail_ids
    summary["soft_fail_count"] = len(soft_fail_ids)
    summary["soft_rule_counts"] = soft_rule_counts
    summary["soft_fail_ids"] = soft_fail_ids
    _print_summary(summary)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'arrays.npz'}")
    print(f"Wrote: {out_dir / 'experiment_result.jsonl'}")

    if args.run_scenarios:
        d_mid = {k: sum(d_ranges[k]) / 2 for k in d_keys}
        rng_s = np.random.default_rng(args.seed + 1)
        scen_rows = []
        scen_rows_begin = []
        scen_rows_end = []
        scen_rows_delta = []
        V_vent_by_scen: dict[str, list[float]] = {}
        R_roof_solar_by_scen: dict[str, list[float]] = {}
        R_roof_LED_by_scen: dict[str, list[float]] = {}
        R_cano_glo_by_scen: dict[str, list[float]] = {}
        X_cano_by_scen: dict[str, list[float]] = {}
        g_tran_by_scen: dict[str, list[float]] = {}
        Q_tran_by_scen: dict[str, list[float]] = {}
        Q_deh_by_scen: dict[str, list[float]] = {}
        F_tran_by_scen: dict[str, list[float]] = {}
        F_hum_by_scen: dict[str, list[float]] = {}
        F_deh_by_scen: dict[str, list[float]] = {}
        F_vent_by_scen: dict[str, list[float]] = {}
        D_ass_by_scen: dict[str, list[float]] = {}
        D_dos_by_scen: dict[str, list[float]] = {}
        D_vent_by_scen: dict[str, list[float]] = {}
        Q_heat_by_scen: dict[str, list[float]] = {}
        Q_vent_by_scen: dict[str, list[float]] = {}
        Q_cool_by_scen: dict[str, list[float]] = {}
        Q_solar_by_scen: dict[str, list[float]] = {}
        Q_o_i_by_scen: dict[str, list[float]] = {}
        Q_LED_by_scen: dict[str, list[float]] = {}
        T_o_i_by_scen: dict[str, list[float]] = {}
        dL_by_scen: dict[str, list[float]] = {}
        H_in_sat_minus_H_in_by_scen: dict[str, list[float]] = {}
        scen_path = out_dir / "scenarios.jsonl"
        with scen_path.open("w", encoding="utf-8") as sf:
            for sc in SCENARIOS:
                u_base = np.array([sc.get("u", {}).get(k, 0.0) for k in u_keys], float)
                d_base = np.array([sc.get("d", {}).get(k, d_mid[k]) for k in d_keys], float)
                runs = max(1, int(args.scenario_jitter_n))
                dts, dhs, dcs, dls = [], [], [], []
                V_vent_vals = V_vent_by_scen.setdefault(sc["name"], [])
                R_roof_solar_vals = R_roof_solar_by_scen.setdefault(sc["name"], [])
                R_roof_LED_vals = R_roof_LED_by_scen.setdefault(sc["name"], [])
                R_cano_glo_vals = R_cano_glo_by_scen.setdefault(sc["name"], [])
                X_cano_vals = X_cano_by_scen.setdefault(sc["name"], [])
                g_tran_vals = g_tran_by_scen.setdefault(sc["name"], [])
                Q_tran_vals = Q_tran_by_scen.setdefault(sc["name"], [])
                Q_deh_vals = Q_deh_by_scen.setdefault(sc["name"], [])
                F_tran_vals = F_tran_by_scen.setdefault(sc["name"], [])
                F_hum_vals = F_hum_by_scen.setdefault(sc["name"], [])
                F_deh_vals = F_deh_by_scen.setdefault(sc["name"], [])
                F_vent_vals = F_vent_by_scen.setdefault(sc["name"], [])
                D_ass_vals = D_ass_by_scen.setdefault(sc["name"], [])
                D_dos_vals = D_dos_by_scen.setdefault(sc["name"], [])
                D_vent_vals = D_vent_by_scen.setdefault(sc["name"], [])
                Q_heat_vals = Q_heat_by_scen.setdefault(sc["name"], [])
                Q_vent_vals = Q_vent_by_scen.setdefault(sc["name"], [])
                Q_cool_vals = Q_cool_by_scen.setdefault(sc["name"], [])
                Q_solar_vals = Q_solar_by_scen.setdefault(sc["name"], [])
                Q_o_i_vals = Q_o_i_by_scen.setdefault(sc["name"], [])
                Q_LED_vals = Q_LED_by_scen.setdefault(sc["name"], [])
                T_o_i_vals = T_o_i_by_scen.setdefault(sc["name"], [])
                T_in_0_vals: list[float] = []
                T_in_1_vals: list[float] = []
                T_out_vals: list[float] = []
                T_o_i_0_vals: list[float] = []
                T_o_i_1_vals: list[float] = []
                C_in_0_vals: list[float] = []
                C_in_1_vals: list[float] = []
                C_out_vals: list[float] = []
                L_0_vals: list[float] = []
                L_1_vals: list[float] = []
                H_in_0_vals: list[float] = []
                H_in_1_vals: list[float] = []
                H_out_vals: list[float] = []
                g_tran_i_vals: list[float] = []
                H_cano_i_vals: list[float] = []
                H_cano_vals: list[float] = []
                H_cano_minus_H_in_0_vals: list[float] = []
                H_cano_minus_H_in_1_vals: list[float] = []
                Q_tran_i_vals: list[float] = []
                Q_deh_i_vals: list[float] = []
                F_tran_i_vals: list[float] = []
                F_hum_i_vals: list[float] = []
                F_deh_i_vals: list[float] = []
                F_vent_i_vals: list[float] = []
                F_vc_i_vals: list[float] = []
                F_vc_vals: list[float] = []
                g_vc_i_vals: list[float] = []
                g_vc_vals: list[float] = []
                D_ass_i_vals: list[float] = []
                D_dos_i_vals: list[float] = []
                D_vent_i_vals: list[float] = []
                Q_heat_i_vals: list[float] = []
                Q_vent_i_vals: list[float] = []
                Q_cool_i_vals: list[float] = []
                Q_solar_i_vals: list[float] = []
                Q_o_i_i_vals: list[float] = []
                Q_LED_i_vals: list[float] = []
                R_roof_solar_i_vals: list[float] = []
                R_roof_LED_i_vals: list[float] = []
                R_cano_glo_i_vals: list[float] = []
                r_s_i_vals: list[float] = []
                r_s_vals: list[float] = []
                T_cover_i_vals: list[float] = []
                T_cover_vals: list[float] = []
                X_cano_i_vals: list[float] = []
                H_in_sat_minus_H_in_0_vals: list[float] = []
                H_in_sat_minus_H_in_1_vals = H_in_sat_minus_H_in_by_scen.setdefault(sc["name"], [])
                r_b_vals: list[float] = []
                for _ in range(runs):
                    jitter = 1.0 + (rng_s.normal(0.0, 0.01, size=u_base.size) if runs > 1 else 0.0)
                    u = u_base * jitter if runs > 1 else u_base
                    jitter_d = 1.0 + (rng_s.normal(0.0, 0.01, size=d_base.size) if runs > 1 else 0.0)
                    d = d_base * jitter_d if runs > 1 else d_base
                    x1, info = digital_twin_one_step(x0, u, d, params)
                    info_i = info.get("initial", {}) if isinstance(info, dict) else {}
                    info_e = info.get("end", {}) if isinstance(info, dict) else {}
                    dx = x1 - x0
                    if x0.size == 4:
                        T_in_0_vals.append(float(x0[x_idx["T_in"]]))
                        T_in_1_vals.append(float(x1[x_idx["T_in"]]))
                        T_out_vals.append(float(d[d_idx["T_out"]]))
                        T_o_i_0_vals.append(float(d[d_idx["T_out"]]) - float(x0[x_idx["T_in"]]))
                        T_o_i_1_vals.append(float(d[d_idx["T_out"]]) - float(x1[x_idx["T_in"]]))
                        C_in_0_vals.append(float(x0[x_idx["C_in"]]))
                        C_in_1_vals.append(float(x1[x_idx["C_in"]]))
                        C_out_vals.append(float(d[d_idx["C_out"]]))
                        L_0_vals.append(float(x0[x_idx["L"]]))
                        L_1_vals.append(float(x1[x_idx["L"]]))
                        H_in_0_vals.append(float(x0[x_idx["H_in"]]))
                        H_in_1_vals.append(float(x1[x_idx["H_in"]]))
                        H_out_vals.append(float(d[d_idx["H_out"]]))
                        r_b = params.get("r_b")
                        if r_b is not None:
                            r_b_vals.append(float(r_b))

                    V_vent_vals.append(float(info_e["V_vent"]))

                    if info_i or info_e:
                        _append_info_value(R_roof_solar_i_vals, info_i, "R_roof_solar")
                        _append_info_value(R_roof_solar_vals, info_e, "R_roof_solar")

                        _append_info_value(R_roof_LED_i_vals, info_i, "R_roof_LED")
                        _append_info_value(R_roof_LED_vals, info_e, "R_roof_LED")

                        _append_info_value(R_cano_glo_i_vals, info_i, "R_cano_glo")
                        _append_info_value(R_cano_glo_vals, info_e, "R_cano_glo")

                        _append_info_value(X_cano_i_vals, info_i, "X_cano")
                        _append_info_value(X_cano_vals, info_e, "X_cano")

                        _append_info_value(g_tran_i_vals, info_i, "g_tran")
                        _append_info_value(g_tran_vals, info_e, "g_tran")

                        if "H_cano" in info_i:
                            H_cano_i_vals.append(float(info_i["H_cano"]))
                            H_cano_minus_H_in_0_vals.append(float(info_i["H_cano"]) - float(x0[x_idx["H_in"]]))
                        if "H_cano" in info_e:
                            H_cano_vals.append(float(info_e["H_cano"]))
                            H_cano_minus_H_in_1_vals.append(float(info_e["H_cano"]) - float(x1[x_idx["H_in"]]))

                        _append_info_value(r_s_i_vals, info_i, "r_s")
                        _append_info_value(r_s_vals, info_e, "r_s")

                        _append_info_value(T_cover_i_vals, info_i, "T_cover")
                        _append_info_value(T_cover_vals, info_e, "T_cover")

                        _append_info_value(Q_tran_i_vals, info_i, "Q_tran")
                        _append_info_value(Q_tran_vals, info_e, "Q_tran")
                        _append_info_value(F_tran_i_vals, info_i, "F_tran")
                        _append_info_value(F_vc_i_vals, info_i, "F_vc")
                        _append_info_value(F_vc_vals, info_e, "F_vc")
                        _append_info_value(F_hum_i_vals, info_i, "F_hum", "F_cool")
                        _append_info_value(F_hum_vals, info_e, "F_hum", "F_cool")
                        _append_info_value(F_deh_i_vals, info_i, "F_deh")
                        _append_info_value(F_deh_vals, info_e, "F_deh")
                        _append_info_value(Q_deh_i_vals, info_i, "Q_deh")
                        _append_info_value(Q_deh_vals, info_e, "Q_deh")
                        _append_info_value(F_vent_i_vals, info_i, "F_vent")
                        _append_info_value(F_vent_vals, info_e, "F_vent")
                        _append_info_value(g_vc_i_vals, info_i, "g_vc")
                        _append_info_value(g_vc_vals, info_e, "g_vc")
                        _append_info_value(F_tran_vals, info_e, "F_tran")
                        _append_info_value(D_ass_i_vals, info_i, "D_ass")
                        _append_info_value(D_ass_vals, info_e, "D_ass")
                        _append_info_value(D_dos_i_vals, info_i, "D_dos")
                        _append_info_value(D_dos_vals, info_e, "D_dos")
                        _append_info_value(D_vent_i_vals, info_i, "D_vent")
                        _append_info_value(D_vent_vals, info_e, "D_vent")
                        _append_info_value(Q_heat_i_vals, info_i, "Q_heat")
                        _append_info_value(Q_heat_vals, info_e, "Q_heat")
                        _append_info_value(Q_vent_i_vals, info_i, "Q_vent")
                        _append_info_value(Q_vent_vals, info_e, "Q_vent")
                        _append_info_value(Q_cool_i_vals, info_i, "Q_cool")
                        _append_info_value(Q_cool_vals, info_e, "Q_cool")
                        _append_info_value(Q_solar_i_vals, info_i, "Q_solar")
                        _append_info_value(Q_solar_vals, info_e, "Q_solar")
                        _append_info_value(Q_o_i_i_vals, info_i, "Q_o_i")
                        _append_info_value(Q_o_i_vals, info_e, "Q_o_i")
                        _append_info_value(Q_LED_i_vals, info_i, "Q_LED")
                        _append_info_value(Q_LED_vals, info_e, "Q_LED")
                        _append_info_value(T_o_i_vals, info_e, "T_o_i")

                        if x0.size == 4 and "H_in_sat" in info_i:
                            H_in_sat_minus_H_in_0_vals.append(float(info_i["H_in_sat"]) - float(x0[x_idx["H_in"]]))
                        if x0.size == 4 and "H_in_sat" in info_e:
                            H_in_sat_minus_H_in_1_vals.append(float(info_e["H_in_sat"]) - float(x1[x_idx["H_in"]]))

                    if x0.size == 4:
                        dT = dx[x_idx["T_in"]]
                        dH = dx[x_idx["H_in"]]
                        dC = dx[x_idx["C_in"]]
                        dL = dx[x_idx["L"]]
                        dts.append(dT)
                        dhs.append(dH)
                        dcs.append(dC)
                        dls.append(dL)

                    sf.write(
                        json.dumps(
                            {
                                "scenario": sc["name"],
                                "u": u.tolist(),
                                "d": d.tolist(),
                                "x1": x1.tolist(),
                                "dx": dx.tolist(),
                            }
                        )
                        + "\n"
                    )

                if x0.size == 4:
                    dL_by_scen[sc["name"]] = dls
                    scen_rows_begin.append(
                        [
                            sc["name"],
                            _fmt(float(np.mean(C_in_0_vals)) if C_in_0_vals else float("nan")),
                            _fmt(float(np.mean(C_out_vals)) if C_out_vals else float("nan")),
                            _fmt(float(np.mean(D_ass_i_vals)) if D_ass_i_vals else float("nan")),
                            _fmt(float(np.mean(D_dos_i_vals)) if D_dos_i_vals else float("nan")),
                            _fmt(float(np.mean(D_vent_i_vals)) if D_vent_i_vals else float("nan")),
                            _fmt(float(np.mean(F_hum_i_vals)) if F_hum_i_vals else float("nan")),
                            _fmt(float(np.mean(F_deh_i_vals)) if F_deh_i_vals else float("nan")),
                            _fmt(float(np.mean(F_tran_i_vals)) if F_tran_i_vals else float("nan")),
                            _fmt(float(np.mean(F_vent_i_vals)) if F_vent_i_vals else float("nan")),
                            _fmt(float(np.mean(F_vc_i_vals)) if F_vc_i_vals else float("nan")),
                            _fmt(float(np.mean(g_tran_i_vals)) if g_tran_i_vals else float("nan")),
                            _fmt(float(np.mean(g_vc_i_vals)) if g_vc_i_vals else float("nan")),
                            _fmt(float(np.mean(H_cano_i_vals)) if H_cano_i_vals else float("nan")),
                            _fmt(float(np.mean(H_cano_minus_H_in_0_vals)) if H_cano_minus_H_in_0_vals else float("nan")),
                            _fmt(float(np.mean(H_in_0_vals)) if H_in_0_vals else float("nan")),
                            _fmt(float(np.mean(H_in_sat_minus_H_in_0_vals)) if H_in_sat_minus_H_in_0_vals else float("nan")),
                            _fmt(float(np.mean(H_out_vals)) if H_out_vals else float("nan")),
                            _fmt(float(np.mean(L_0_vals)) if L_0_vals else float("nan")),
                            _fmt(float(np.mean(Q_cool_i_vals)) if Q_cool_i_vals else float("nan")),
                            _fmt(float(np.mean(Q_heat_i_vals)) if Q_heat_i_vals else float("nan")),
                            _fmt(float(np.mean(Q_LED_i_vals)) if Q_LED_i_vals else float("nan")),
                            _fmt(float(np.mean(Q_o_i_i_vals)) if Q_o_i_i_vals else float("nan")),
                            _fmt(float(np.mean(Q_solar_i_vals)) if Q_solar_i_vals else float("nan")),
                            _fmt(float(np.mean(Q_tran_i_vals)) if Q_tran_i_vals else float("nan")),
                            _fmt(float(np.mean(Q_deh_i_vals)) if Q_deh_i_vals else float("nan")),
                            _fmt(float(np.mean(Q_vent_i_vals)) if Q_vent_i_vals else float("nan")),
                            _fmt(float(np.mean(R_cano_glo_i_vals)) if R_cano_glo_i_vals else float("nan")),
                            _fmt(float(np.mean(R_roof_LED_i_vals)) if R_roof_LED_i_vals else float("nan")),
                            _fmt(float(np.mean(R_roof_solar_i_vals)) if R_roof_solar_i_vals else float("nan")),
                            _fmt(float(np.mean(r_b_vals)) if r_b_vals else float("nan")),
                            _fmt(float(np.mean(r_s_i_vals)) if r_s_i_vals else float("nan")),
                            _fmt(float(np.mean(T_cover_i_vals)) if T_cover_i_vals else float("nan")),
                            _fmt(float(np.mean(T_in_0_vals)) if T_in_0_vals else float("nan")),
                            _fmt(float(np.mean(T_o_i_0_vals)) if T_o_i_0_vals else float("nan")),
                            _fmt(float(np.mean(T_out_vals)) if T_out_vals else float("nan")),
                            _fmt(float(np.mean(X_cano_i_vals)) if X_cano_i_vals else float("nan")),
                        ]
                    )
                    scen_rows_end.append(
                        [
                            sc["name"],
                            _fmt(float(np.mean(C_in_1_vals)) if C_in_1_vals else float("nan")),
                            _fmt(float(np.mean(C_out_vals)) if C_out_vals else float("nan")),
                            _fmt(float(np.mean(D_ass_vals)) if D_ass_vals else float("nan")),
                            _fmt(float(np.mean(D_dos_vals)) if D_dos_vals else float("nan")),
                            _fmt(float(np.mean(D_vent_vals)) if D_vent_vals else float("nan")),
                            _fmt(float(np.mean(F_hum_vals)) if F_hum_vals else float("nan")),
                            _fmt(float(np.mean(F_deh_vals)) if F_deh_vals else float("nan")),
                            _fmt(float(np.mean(F_tran_vals)) if F_tran_vals else float("nan")),
                            _fmt(float(np.mean(F_vent_vals)) if F_vent_vals else float("nan")),
                            _fmt(float(np.mean(F_vc_vals)) if F_vc_vals else float("nan")),
                            _fmt(float(np.mean(g_tran_vals)) if g_tran_vals else float("nan")),
                            _fmt(float(np.mean(g_vc_vals)) if g_vc_vals else float("nan")),
                            _fmt(float(np.mean(H_cano_vals)) if H_cano_vals else float("nan")),
                            _fmt(float(np.mean(H_cano_minus_H_in_1_vals)) if H_cano_minus_H_in_1_vals else float("nan")),
                            _fmt(float(np.mean(H_in_1_vals)) if H_in_1_vals else float("nan")),
                            _fmt(float(np.mean(H_in_sat_minus_H_in_1_vals)) if H_in_sat_minus_H_in_1_vals else float("nan")),
                            _fmt(float(np.mean(H_out_vals)) if H_out_vals else float("nan")),
                            _fmt(float(np.mean(L_1_vals)) if L_1_vals else float("nan")),
                            _fmt(float(np.mean(Q_cool_vals)) if Q_cool_vals else float("nan")),
                            _fmt(float(np.mean(Q_heat_vals)) if Q_heat_vals else float("nan")),
                            _fmt(float(np.mean(Q_LED_vals)) if Q_LED_vals else float("nan")),
                            _fmt(float(np.mean(Q_o_i_vals)) if Q_o_i_vals else float("nan")),
                            _fmt(float(np.mean(Q_solar_vals)) if Q_solar_vals else float("nan")),
                            _fmt(float(np.mean(Q_tran_vals)) if Q_tran_vals else float("nan")),
                            _fmt(float(np.mean(Q_deh_vals)) if Q_deh_vals else float("nan")),
                            _fmt(float(np.mean(Q_vent_vals)) if Q_vent_vals else float("nan")),
                            _fmt(float(np.mean(R_cano_glo_vals)) if R_cano_glo_vals else float("nan")),
                            _fmt(float(np.mean(R_roof_LED_vals)) if R_roof_LED_vals else float("nan")),
                            _fmt(float(np.mean(R_roof_solar_vals)) if R_roof_solar_vals else float("nan")),
                            _fmt(float(np.mean(r_b_vals)) if r_b_vals else float("nan")),
                            _fmt(float(np.mean(r_s_vals)) if r_s_vals else float("nan")),
                            _fmt(float(np.mean(T_cover_vals)) if T_cover_vals else float("nan")),
                            _fmt(float(np.mean(T_in_1_vals)) if T_in_1_vals else float("nan")),
                            _fmt(float(np.mean(T_o_i_1_vals)) if T_o_i_1_vals else float("nan")),
                            _fmt(float(np.mean(T_out_vals)) if T_out_vals else float("nan")),
                            _fmt(float(np.mean(X_cano_vals)) if X_cano_vals else float("nan")),
                        ]
                    )
                    scen_rows_delta.append(
                        [
                            sc["name"],
                            _fmt(float(np.mean(dts)) if dts else float("nan")),
                            _fmt(float(np.mean(dhs)) if dhs else float("nan")),
                            _fmt(float(np.mean(dcs)) if dcs else float("nan")),
                            _fmt(float(np.mean(dls)) if dls else float("nan")),
                        ]
                    )
                    scen_rows.append(
                        [
                            sc["name"],
                            _fmt(float(np.mean(dts))),
                            _fmt(float(np.mean(dhs))),
                            _fmt(float(np.mean(dcs))),
                            _fmt(float(np.mean(dls))),
                            _fmt(float(np.mean(D_ass_vals)) if D_ass_vals else float("nan")),
                            _fmt(float(np.mean(F_tran_vals)) if F_tran_vals else float("nan")),
                            _fmt(float(np.mean(g_tran_vals)) if g_tran_vals else float("nan")),
                            _fmt(float(np.mean(H_in_1_vals)) if H_in_1_vals else float("nan")),
                            _fmt(float(np.mean(H_in_sat_minus_H_in_1_vals)) if H_in_sat_minus_H_in_1_vals else float("nan")),
                            _fmt(float(np.mean(Q_tran_vals)) if Q_tran_vals else float("nan")),
                            _fmt(float(np.mean(Q_deh_vals)) if Q_deh_vals else float("nan")),
                            _fmt(float(np.mean(Q_heat_vals)) if Q_heat_vals else float("nan")),
                            _fmt(float(np.mean(Q_vent_vals)) if Q_vent_vals else float("nan")),
                            _fmt(float(np.mean(Q_cool_vals)) if Q_cool_vals else float("nan")),
                            _fmt(float(np.mean(Q_solar_vals)) if Q_solar_vals else float("nan")),
                            _fmt(float(np.mean(Q_o_i_vals)) if Q_o_i_vals else float("nan")),
                            _fmt(float(np.mean(Q_LED_vals)) if Q_LED_vals else float("nan")),
                            _fmt(float(np.mean(R_roof_solar_vals)) if R_roof_solar_vals else float("nan")),
                            _fmt(float(np.mean(R_roof_LED_vals)) if R_roof_LED_vals else float("nan")),
                            _fmt(float(np.mean(R_cano_glo_vals)) if R_cano_glo_vals else float("nan")),
                            _fmt(float(np.mean(T_o_i_vals)) if T_o_i_vals else float("nan")),
                            _fmt(float(np.mean(T_in_1_vals)) if T_in_1_vals else float("nan")),
                            _fmt(float(np.mean(T_out_vals)) if T_out_vals else float("nan")),
                            _fmt(float(np.mean(X_cano_vals)) if X_cano_vals else float("nan")),
                        ]
                    )

        HEADERS_STATE = [
            "scenario",
            "C_in",
            "C_out",
            "D_ass",
            "D_dos",
            "D_vent",
            "F_hum",
            "F_deh",
            "F_tran",
            "F_vent",
            "F_vc",
            "g_tran",
            "g_vc",
            "H_cano",
            "H_cano - H_in",
            "H_in",
            "H_in_sat - H_in",
            "H_out",
            "L",
            "Q_cool",
            "Q_heat",
            "Q_LED",
            "Q_o_i",
            "Q_solar",
            "Q_tran",
            "Q_vent",
            "Q_deh",
            "R_cano_glo",
            "R_roof_LED",
            "R_roof_solar",
            "r_b",
            "r_s",
            "T_cover",
            "T_in",
            "T_o_i",
            "T_out",
            "X_cano",
        ]
        if scen_rows_begin:
            print("\nScenario probes: Beginning-of-step state")
            _print_table(headers=HEADERS_STATE, rows=scen_rows_begin)
        if scen_rows_end:
            print("\nScenario probes: End-of-step state")
            _print_table(headers=HEADERS_STATE, rows=scen_rows_end)
        if scen_rows_delta:
            print("\nScenario probes: Change of state")
            _print_table(
                headers=["scenario"] + [f"d{name}" for name in ("T_in", "H_in", "C_in", "L")],
                rows=scen_rows_delta,
            )


if __name__ == "__main__":
    main()
