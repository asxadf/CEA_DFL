# src/utils/info_handlers_and_plotters.py
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import gridspec, rcParams


DEFAULT_OUTDOOR_COLS = [
    "T_Outdoor(C)",
    "H_Outdoor(g/m3)",
    "CO2_Outdoor(g/m3)",
    "Radiation_Outdoor(w/m2)",
]


def load_parameters_with_x_ini(yaml_path: Path) -> dict:
    data_keeper = yaml.safe_load(Path(yaml_path).resolve().read_text(encoding="utf-8"))
    param_keeper = data_keeper["parameters"]
    param_keeper["x_ini"] = data_keeper["variables"]["x_ini"]
    return param_keeper


def load_outdoor_disturbance_csv(path: Path, *, cols: list[str] | None = None) -> tuple[pd.DatetimeIndex, np.ndarray]:
    csv_path = Path(path).resolve()
    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    ts = pd.to_datetime(df[time_col])
    use_cols = DEFAULT_OUTDOOR_COLS if cols is None else cols
    arr = df[use_cols].to_numpy(dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"disturbance shape must be (N,4), got {arr.shape}")
    return pd.DatetimeIndex(ts), arr


def compute_time_axes_from_hist(
    t_hist: list[pd.Timestamp],
    dt_default_s: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, int]:
    if len(t_hist) == 0:
        empty = pd.DatetimeIndex([])
        return empty, empty, int(dt_default_s)
    t = pd.DatetimeIndex(t_hist)
    dt_s = int((t[1] - t[0]).total_seconds()) if len(t) >= 2 else int(dt_default_s)
    t_x = t.append(pd.DatetimeIndex([t[-1] + pd.Timedelta(seconds=dt_s)]))
    return t, t_x, dt_s


def _phase_from_kappa(kappa: int | np.ndarray, pk: dict) -> np.ndarray:
    # Phase codes: 0=night, 1=transition_1 (night->day), 2=day, 3=transition_2 (day->night).
    kappa_arr = np.asarray(kappa, dtype=int)
    shape = kappa_arr.shape
    kappa_flat = np.atleast_1d(kappa_arr).reshape(-1)
    total_steps = int(pk["kappa_day_night_total_steps"])
    kappa_mod = np.mod(kappa_flat, total_steps)

    day_start = int(pk["kappa_day_start"])
    day_end = int(pk["kappa_day_end"])

    phase = np.zeros(kappa_mod.shape, dtype=int)

    has_transition_window = all(
        key in pk
        for key in (
            "kappa_transition_start_1",
            "kappa_transition_end_1",
            "kappa_transition_start_2",
            "kappa_transition_end_2",
        )
    )
    if has_transition_window:
        t1_start = int(pk["kappa_transition_start_1"])
        t1_end = int(pk["kappa_transition_end_1"])
        t2_start = int(pk["kappa_transition_start_2"])
        t2_end = int(pk["kappa_transition_end_2"])
        phase[(kappa_mod >= t1_start) & (kappa_mod < t1_end)] = 1
        phase[(kappa_mod >= t2_start) & (kappa_mod < t2_end)] = 3

    day_mask = (kappa_mod >= day_start) & (kappa_mod < day_end) & (phase == 0)
    phase[day_mask] = 2

    return phase.reshape(shape)


def _linear_ramp(kappa, start_k, end_k, v_start, v_end):
    # Linear interpolation on [start_k, end_k), with endpoint clamping outside.
    kappa_arr = np.asarray(kappa, dtype=float)
    start = float(start_k)
    end = float(end_k)
    v0 = float(v_start)
    v1 = float(v_end)
    if end <= start:
        return np.full_like(kappa_arr, v0, dtype=float)
    frac = np.clip((kappa_arr - start) / (end - start), 0.0, 1.0)
    return v0 + frac * (v1 - v0)


def _bounds_from_kappa(
    kappa: int | np.ndarray,
    pk: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kappa_arr = np.asarray(kappa, dtype=int)
    shape = kappa_arr.shape
    kappa_flat = np.atleast_1d(kappa_arr).reshape(-1)
    total_steps = int(pk["kappa_day_night_total_steps"])
    kappa_mod = np.mod(kappa_flat, total_steps)
    phase = _phase_from_kappa(kappa_mod, pk).reshape(-1)

    T_low = np.full(kappa_mod.shape, float(pk["T_in_lower_night"]), dtype=float)
    T_high = np.full(kappa_mod.shape, float(pk["T_in_upper_night"]), dtype=float)
    C_low = np.full(kappa_mod.shape, float(pk["C_in_lower_night"]), dtype=float)
    C_high = np.full(kappa_mod.shape, float(pk["C_in_upper_night"]), dtype=float)

    has_h_day_night = all(
        key in pk
        for key in (
            "H_in_lower_day",
            "H_in_upper_day",
            "H_in_lower_night",
            "H_in_upper_night",
        )
    )
    if has_h_day_night:
        H_low = np.full(kappa_mod.shape, float(pk["H_in_lower_night"]), dtype=float)
        H_high = np.full(kappa_mod.shape, float(pk["H_in_upper_night"]), dtype=float)
    else:
        H_low = np.full(kappa_mod.shape, float(pk["H_in_lower"]), dtype=float)
        H_high = np.full(kappa_mod.shape, float(pk["H_in_upper"]), dtype=float)

    day_mask = phase == 2
    T_low[day_mask] = float(pk["T_in_lower_day"])
    T_high[day_mask] = float(pk["T_in_upper_day"])
    C_low[day_mask] = float(pk["C_in_lower_day"])
    C_high[day_mask] = float(pk["C_in_upper_day"])
    if has_h_day_night:
        H_low[day_mask] = float(pk["H_in_lower_day"])
        H_high[day_mask] = float(pk["H_in_upper_day"])

    has_transition_window = all(
        key in pk
        for key in (
            "kappa_transition_start_1",
            "kappa_transition_end_1",
            "kappa_transition_start_2",
            "kappa_transition_end_2",
        )
    )
    if has_transition_window:
        t1_start = int(pk["kappa_transition_start_1"])
        t1_end = int(pk["kappa_transition_end_1"])
        t2_start = int(pk["kappa_transition_start_2"])
        t2_end = int(pk["kappa_transition_end_2"])

        t1_mask = phase == 1
        t2_mask = phase == 3

        T_low_t1_start = float(pk.get("T_in_lower_transition_start_1", pk["T_in_lower_night"]))
        T_low_t1_end = float(pk.get("T_in_lower_transition_end_1", pk["T_in_lower_day"]))
        T_high_t1_start = float(pk.get("T_in_upper_transition_start_1", pk["T_in_upper_night"]))
        T_high_t1_end = float(pk.get("T_in_upper_transition_end_1", pk["T_in_upper_day"]))
        C_low_t1_start = float(pk.get("C_in_lower_transition_start_1", pk["C_in_lower_night"]))
        C_low_t1_end = float(pk.get("C_in_lower_transition_end_1", pk["C_in_lower_day"]))
        C_high_t1_start = float(pk.get("C_in_upper_transition_start_1", pk["C_in_upper_night"]))
        C_high_t1_end = float(pk.get("C_in_upper_transition_end_1", pk["C_in_upper_day"]))

        T_low_t2_start = float(pk.get("T_in_lower_transition_start_2", pk["T_in_lower_day"]))
        T_low_t2_end = float(pk.get("T_in_lower_transition_end_2", pk["T_in_lower_night"]))
        T_high_t2_start = float(pk.get("T_in_upper_transition_start_2", pk["T_in_upper_day"]))
        T_high_t2_end = float(pk.get("T_in_upper_transition_end_2", pk["T_in_upper_night"]))
        C_low_t2_start = float(pk.get("C_in_lower_transition_start_2", pk["C_in_lower_day"]))
        C_low_t2_end = float(pk.get("C_in_lower_transition_end_2", pk["C_in_lower_night"]))
        C_high_t2_start = float(pk.get("C_in_upper_transition_start_2", pk["C_in_upper_day"]))
        C_high_t2_end = float(pk.get("C_in_upper_transition_end_2", pk["C_in_upper_night"]))
        if has_h_day_night:
            H_low_t1_start = float(pk.get("H_in_lower_transition_start_1", pk["H_in_lower_night"]))
            H_low_t1_end = float(pk.get("H_in_lower_transition_end_1", pk["H_in_lower_day"]))
            H_high_t1_start = float(pk.get("H_in_upper_transition_start_1", pk["H_in_upper_night"]))
            H_high_t1_end = float(pk.get("H_in_upper_transition_end_1", pk["H_in_upper_day"]))

            H_low_t2_start = float(pk.get("H_in_lower_transition_start_2", pk["H_in_lower_day"]))
            H_low_t2_end = float(pk.get("H_in_lower_transition_end_2", pk["H_in_lower_night"]))
            H_high_t2_start = float(pk.get("H_in_upper_transition_start_2", pk["H_in_upper_day"]))
            H_high_t2_end = float(pk.get("H_in_upper_transition_end_2", pk["H_in_upper_night"]))

        if np.any(t1_mask):
            T_low[t1_mask] = _linear_ramp(kappa_mod[t1_mask], t1_start, t1_end, T_low_t1_start, T_low_t1_end)
            T_high[t1_mask] = _linear_ramp(kappa_mod[t1_mask], t1_start, t1_end, T_high_t1_start, T_high_t1_end)
            C_low[t1_mask] = _linear_ramp(kappa_mod[t1_mask], t1_start, t1_end, C_low_t1_start, C_low_t1_end)
            C_high[t1_mask] = _linear_ramp(kappa_mod[t1_mask], t1_start, t1_end, C_high_t1_start, C_high_t1_end)
            if has_h_day_night:
                H_low[t1_mask] = _linear_ramp(kappa_mod[t1_mask], t1_start, t1_end, H_low_t1_start, H_low_t1_end)
                H_high[t1_mask] = _linear_ramp(kappa_mod[t1_mask], t1_start, t1_end, H_high_t1_start, H_high_t1_end)

        if np.any(t2_mask):
            T_low[t2_mask] = _linear_ramp(kappa_mod[t2_mask], t2_start, t2_end, T_low_t2_start, T_low_t2_end)
            T_high[t2_mask] = _linear_ramp(kappa_mod[t2_mask], t2_start, t2_end, T_high_t2_start, T_high_t2_end)
            C_low[t2_mask] = _linear_ramp(kappa_mod[t2_mask], t2_start, t2_end, C_low_t2_start, C_low_t2_end)
            C_high[t2_mask] = _linear_ramp(kappa_mod[t2_mask], t2_start, t2_end, C_high_t2_start, C_high_t2_end)
            if has_h_day_night:
                H_low[t2_mask] = _linear_ramp(kappa_mod[t2_mask], t2_start, t2_end, H_low_t2_start, H_low_t2_end)
                H_high[t2_mask] = _linear_ramp(kappa_mod[t2_mask], t2_start, t2_end, H_high_t2_start, H_high_t2_end)

    return (
        T_low.reshape(shape),
        T_high.reshape(shape),
        H_low.reshape(shape),
        H_high.reshape(shape),
        C_low.reshape(shape),
        C_high.reshape(shape),
    )


def progress_print(
    step: int,
    total: int,
    ts: pd.Timestamp | None,
    u0: np.ndarray,
    x_before: np.ndarray,
    x_after: np.ndarray,
    obj_mpc: float | None,
    cost_actual_step: float | None,
    param_keeper: dict | None = None,
    kappa_0: int | None = None,
    horizon_K: int | None = None,
    d0_pred: np.ndarray | None = None,
    d0_real: np.ndarray | None = None,
) -> None:
    ts_info = ts.isoformat(sep=" ", timespec="seconds") if ts is not None else "n/a"
    day_night_text = "unknown"
    if param_keeper is not None and kappa_0 is not None:
        phase_now = int(np.asarray(_phase_from_kappa(int(kappa_0), param_keeper), dtype=int).reshape(-1)[0])
        if phase_now == 2:
            day_night_text = "daytime"
        elif phase_now == 0:
            day_night_text = "nighttime"
        elif phase_now == 1:
            day_night_text = "transition (transition_n2d)"
        else:
            day_night_text = "transition (transition_d2n)"
    print("-" * 72)
    print(f"Simulation step {step + 1}/{total} @ {ts_info} ({day_night_text})")
    if horizon_K is not None:
        if param_keeper is not None and "delta_t" in param_keeper:
            dt_s = float(param_keeper["delta_t"])
            if ts is not None:
                win_start = pd.Timestamp(ts)
                win_end = win_start + pd.Timedelta(seconds=int(round(float(horizon_K) * dt_s)))
                start_str = win_start.isoformat(sep=" ", timespec="seconds")
                end_str = win_end.isoformat(sep=" ", timespec="seconds")
                print(f"Look-ahead window: K={int(horizon_K)} | {start_str} to {end_str} @ dt={dt_s:.0f}s")
            else:
                horizon_h = float(horizon_K) * dt_s / 3600.0
                print(f"Look-ahead window: K={int(horizon_K)} (~{horizon_h:.2f} h @ dt={dt_s:.0f}s)")
        else:
            print(f"Look-ahead window: K={int(horizon_K)}")
    if obj_mpc is not None:
        print(f"Objective: {float(obj_mpc):.6g}")
    print(
        "Groud-truth one-step actual cost "
        f"(actual energy-use cost + actual slack cost): {float(cost_actual_step):.6g}"
    )
    d_names = ["T_out", "H_out", "C_out", "R_out"]
    d_pred_vec = None if d0_pred is None else np.asarray(d0_pred, dtype=float).reshape(-1)
    d_real_vec = None if d0_real is None else np.asarray(d0_real, dtype=float).reshape(-1)
    if d_pred_vec is not None or d_real_vec is not None:
        print("Outdoor disturbance d0 (prediction vs realization):")
        for i, name in enumerate(d_names):
            pred_text = "n/a" if d_pred_vec is None or i >= d_pred_vec.size else f"{float(d_pred_vec[i]):.6g}"
            real_text = "n/a" if d_real_vec is None or i >= d_real_vec.size else f"{float(d_real_vec[i]):.6g}"
            print(f"  {name}: pred={pred_text}, real={real_text}")
    names = ["U_heat", "U_fan", "U_nat", "U_ac", "U_dos", "U_LED", "U_hum", "U_deh", "U_shad", "U_warm"]
    print("Control action U (applied u0):")
    for name, val in zip(names, np.asarray(u0, dtype=float).reshape(-1)):
        print(f"  {name}: {float(val):.6g}")
    xb = np.asarray(x_before, dtype=float).reshape(-1)
    xa = np.asarray(x_after, dtype=float).reshape(-1)
    T_low = np.nan
    T_high = np.nan
    H_low = np.nan
    H_high = np.nan
    C_low = np.nan
    C_high = np.nan
    L_target = np.nan
    if param_keeper is not None and kappa_0 is not None:
        T_low_arr, T_high_arr, H_low_arr, H_high_arr, C_low_arr, C_high_arr = _bounds_from_kappa(int(kappa_0), param_keeper)
        T_low = float(np.asarray(T_low_arr).reshape(-1)[0])
        T_high = float(np.asarray(T_high_arr).reshape(-1)[0])
        H_low = float(np.asarray(H_low_arr).reshape(-1)[0])
        H_high = float(np.asarray(H_high_arr).reshape(-1)[0])
        C_low = float(np.asarray(C_low_arr).reshape(-1)[0])
        C_high = float(np.asarray(C_high_arr).reshape(-1)[0])
        L_star_k_start = int(param_keeper["L_star_k_start"])
        L_star_k_end = int(param_keeper["L_star_k_end"])
        L_star_k_max = float(param_keeper["L_star_k_max"])
        L_star_k_slope = float(param_keeper["L_star_k_slope"])
        kk = int(kappa_0)
        if kk < L_star_k_start:
            L_target = 0.0
        elif kk >= L_star_k_end:
            L_target = L_star_k_max
        else:
            L_target = L_star_k_slope * (kk - L_star_k_start)

    print("State x (before -> after):")
    print(f"  {'T_in':<4}: {xb[0]:7.2f} -> {xa[0]:7.2f}  | range [{T_low:5.2f}, {T_high:5.2f}]")
    print(f"  {'H_in':<4}: {xb[1]:7.2f} -> {xa[1]:7.2f}  | range [{H_low:5.2f}, {H_high:5.2f}]")
    print(f"  {'C_in':<4}: {xb[2]:7.2f} -> {xa[2]:7.2f}  | range [{C_low:5.2f}, {C_high:5.2f}]")
    print(f"  {'L':<4}: {xb[3]:7.2f} -> {xa[3]:7.2f}  | step target {L_target:5.2f}")


def plot_mpc_summary_legacy(
    out_path: Path,
    *,
    x_hist,
    u_hist,
    d_pred0_hist,
    d_real_hist,
    param_keeper,
    t,
    t_x,
    dt_s,
    show: bool = True,
) -> None:
    X = np.vstack(x_hist)
    U = np.vstack(u_hist)
    Dp = np.vstack(d_pred0_hist)
    Dr = np.vstack(d_real_hist)
    N = U.shape[0]

    rcParams["font.family"] = "Helvetica"
    rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    rcParams["font.size"] = 14
    rcParams["axes.titlesize"] = 14
    rcParams["axes.labelsize"] = 14
    rcParams["xtick.labelsize"] = 14
    rcParams["ytick.labelsize"] = 14
    rcParams["legend.fontsize"] = 14

    col_out, col_u, col_x, col_pred = "#BF124D", "#E78B48", "#1C6EA4", "#777777"
    col_band = "#1C6EA4"

    total_steps = int(param_keeper["kappa_day_night_total_steps"])
    sec_midnight = (t_x.hour * 3600 + t_x.minute * 60 + t_x.second).to_numpy(dtype=int)
    kappa_x = (sec_midnight // int(dt_s)) % total_steps
    T_low_x, T_high_x, H_low_x, H_high_x, C_low_x, C_high_x = _bounds_from_kappa(kappa_x, param_keeper)
    L_star_k_max = float(param_keeper["L_star_k_max"])

    dist_defs = [
        ("", "$T^{out}$ (°C)", 0),
        ("", "$H^{out}$ (g/m³)", 1),
        ("", "$C^{out}$ (g/m³)", 2),
        ("", "$R^{out}$ (W/m²)", 3),
    ]
    ctrl_defs = [
        ("", "$U^{heat}$", 0),
        ("", "$U^{fan}$", 1),
        ("", "$U^{nat}$", 2),
        ("", "$U^{ac}$", 3),
        ("", "$U^{dos}$", 4),
        ("", "$U^{LED}$", 5),
        ("", "$U^{hum}$", 6),
        ("", "$U^{deh}$", 7),
        ("", "$U^{shad}$", 8),
        ("", "$U^{warm}$", 9),
    ]
    state_defs = [
        ("", "$T^{in}$ (°C)", 0),
        ("", "$H^{in}$ (g/m³)", 1),
        ("", "$C^{in}$ (g/m³)", 2),
        ("", "L (mol/m²)", 3),
    ]
    low_bands = [T_low_x, H_low_x, C_low_x]
    high_bands = [T_high_x, H_high_x, C_high_x]

    total_rows = 4 + 10 + 4
    fig = plt.figure(figsize=(12, max(12, 2.0 * total_rows)), constrained_layout=True)
    gs = gridspec.GridSpec(total_rows, 1, figure=fig)
    axes = []
    sharex = None
    row = 0

    for i, (title, ylabel, idx) in enumerate(dist_defs):
        ax = fig.add_subplot(gs[row, 0], sharex=sharex)
        sharex = ax if sharex is None else sharex
        ax.plot(t, Dr[:, idx], color=col_out, lw=0.7, label="")
        ax.plot(t, Dp[:, idx], color=col_pred, lw=0.7, ls="--", label="")
        ax.grid(True, ls="--", lw=0.4, alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.text(0.01, 0.92, title, transform=ax.transAxes, ha="left", va="top")
        if i == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=14)
        axes.append(ax)
        row += 1

    for title, ylabel, idx in ctrl_defs:
        ax = fig.add_subplot(gs[row, 0], sharex=sharex)
        ax.plot(t, U[:, idx], color=col_u, lw=0.75)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(True, ls="--", lw=0.4, alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.text(0.01, 0.92, title, transform=ax.transAxes, ha="left", va="top")
        axes.append(ax)
        row += 1

    for title, ylabel, idx in state_defs:
        ax = fig.add_subplot(gs[row, 0], sharex=sharex)
        if idx < 3:
            ax.fill_between(t_x, low_bands[idx], high_bands[idx], color=col_band, alpha=0.12, linewidth=0.0)
        else:
            # Show only a horizontal daily DLI target line (no shaded area).
            ax.axhline(y=L_star_k_max, color=col_pred, ls="--", lw=1.5, label="")
        ax.plot(t_x, X[:, idx], color=col_x, lw=0.8)
        ax.grid(True, ls="--", lw=0.4, alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.text(0.01, 0.92, title, transform=ax.transAxes, ha="left", va="top")
        if idx == 3:
            ax.legend(loc="upper right", frameon=False, fontsize=14)
        axes.append(ax)
        row += 1

    # --- force all y-labels to share the same x position (axes coords) ---
    for ax in axes:
        ax.yaxis.set_label_coords(-0.07, 0.5)  # tune -0.10 if you want tighter/looser left margin

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Datetime")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%m-%d"))
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show(block=True)
    plt.close(fig)
