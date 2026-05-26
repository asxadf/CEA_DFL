# experiment_runner/r4_fit_J_act_DNN_phase_wise.py
"""
Fit phase-wise DNN surrogates for one-step realized cost: COST_END ~ [U, X0, D, refs, kappa]

This trainer consumes only the base-MPC-generated dataset written by:
`experiment_runner/r3_generate_data_from_base.py`

It trains one model per phase: - night - day

Example:
python3 experiment_runner/r4_fit_J_act_DNN_phase_wise.py --season cold
python3 experiment_runner/r4_fit_J_act_DNN_phase_wise.py --season warm
"""
from __future__ import annotations

# Season/phase-specific default MLP hyperparameters. Override per run with
# --phase-hparams-json when sweeping.
COLD_DAY_MLP_HPARAMS = {
    "hidden_layer_sizes": (64, 64),
    "alpha": 1e-4,
    "learning_rate_init": 1e-5,
    "activation": "relu",
    "solver": "adam",
    "max_iter": 5000,
    "tol": 0.1,
    "n_iter_no_change": 50,
    "validation_fraction": 0.10,
    "batch_size": 128,
}

COLD_NIGHT_MLP_HPARAMS = {
    "hidden_layer_sizes": (32, 32),
    "alpha": 1e-3,
    "learning_rate_init": 1e-5,
    "activation": "softplus",
    "solver": "adam",
    "max_iter": 5000,
    "tol": 0.1,
    "n_iter_no_change": 50,
    "validation_fraction": 0.10,
    "batch_size": 32,
}

WARM_DAY_MLP_HPARAMS = {
    "hidden_layer_sizes": (64, 64),
    "alpha": 1e-3,
    "learning_rate_init": 1e-5,
    "activation": "softplus",
    "solver": "adam",
    "max_iter": 2000,
    "tol": 0.1,
    "n_iter_no_change": 50,
    "validation_fraction": 0.10,
    "batch_size": 128,
}

WARM_NIGHT_MLP_HPARAMS = {
    "hidden_layer_sizes": (32, 32),
    "alpha": 1e-3,
    "learning_rate_init": 1e-5,
    "activation": "softplus",
    "solver": "adam",
    "max_iter": 2000,
    "tol": 0.1,
    "n_iter_no_change": 50,
    "validation_fraction": 0.10,
    "batch_size": 32,
}

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

import src.builders.builder_mpc_model as builder_mpc
from src.builders.builder_mpc_model import (
    DFL_COST_TARGET_KIND,
    get_bounds_and_refs,
    get_one_step_total_cost_act_batch,
)
from src.utils.surrogate_feature_scaler import (
    feature_scaler_to_npz_payload,
    fit_surrogate_feature_scaler,
    transform_surrogate_feature_matrix,
)
from src.utils.torch_surrogate_mlp import (
    TorchMLP,
    get_default_torch_device,
    resolve_torch_device,
    save_torch_mlp_checkpoint,
)
from src.utils.vector_order import (
    get_control_names_in_order,
    get_disturbance_names_in_order,
    get_state_names_in_order,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
R3_DATA_ROOT = (REPO_ROOT / "experiment_result/r3_generate_data_from_base").resolve()
PARAM_YAML_PATH = (REPO_ROOT / "configs/var_and_param_keeper.yaml").resolve()
OUT_ROOT = (REPO_ROOT / "experiment_result/r4_fit_J_act_DNN_phase_wise").resolve()
plt.rcParams["font.family"] = "Helvetica"
R3_TRAINING_DATA_FILENAME = "training_data.csv"
R3_LEGACY_DATA_FILENAME = "data.csv"
PHASE_COL = "surrogate_phase"
REF_COLS = ["T_in_star", "H_in_star", "C_in_star", "L_star"]
TARGET_OUTPUT_COST_COLS = ("target_output_cost", "cost_actual", "cost")

TRAIN_FRAC = 0.8
MLP_VALIDATION_METRIC = "val_rmse"

USE_X0_AS_FEATURE = True
USE_D_AS_FEATURE = True
USE_REFS_AS_FEATURE = True
USE_KAPPA_AS_FEATURE = False
TORCH_OUTPUT_ACTIVATION = "softplus"

MLP_HPARAMS_BY_SEASON_AND_PHASE = {
    "cold": {
        "night": COLD_NIGHT_MLP_HPARAMS,
        "day": COLD_DAY_MLP_HPARAMS,
    },
    "warm": {
        "night": WARM_NIGHT_MLP_HPARAMS,
        "day": WARM_DAY_MLP_HPARAMS,
    },
}

SEED = 123
KAPPA_COL = "kappa_k"
PHASE_ORDER = ("night", "day")
DEFAULT_PHASE_WORKERS = min(2, len(PHASE_ORDER))
DEFAULT_PROGRESS_LOG_EVERY = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit phase-wise DNN surrogates for one-step realized cost.")
    parser.add_argument("--season", choices=("cold", "warm"), default="warm")
    parser.add_argument(
        "--phase-hparams-json",
        default=None,
        help=(
            "Optional JSON string or path to a JSON file with per-phase DNN overrides. "
            "Accepted shapes: {'day': {...}, 'night': {...}} or {'cold': {...}, 'warm': {...}}. "
            "Per-phase fields: hidden_layer_sizes, alpha, learning_rate_init, activation, solver, "
            "max_iter, tol, n_iter_no_change, validation_fraction, batch_size."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional directory for the selected season's outputs. "
            "Defaults to experiment_result/r4_fit_J_act_DNN_phase_wise/<season>."
        ),
    )
    parser.add_argument(
        "--phase-workers",
        type=int,
        default=DEFAULT_PHASE_WORKERS,
        help="Number of parallel workers used to train day/night models (default: 2).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_PROGRESS_LOG_EVERY,
        help="Print compact training progress every N epochs per phase (default: 50).",
    )
    parser.add_argument(
        "--verbose-diagnostics",
        action="store_true",
        help="Print detailed matrix health checks and full TorchMLP summaries.",
    )
    return parser.parse_args()


def _load_json_arg(raw_value: str | None, *, arg_name: str) -> dict | None:
    if raw_value is None:
        return None
    raw_text = str(raw_value).strip()
    if not raw_text:
        return None

    candidate_path = Path(raw_text).expanduser()
    if candidate_path.exists():
        text = candidate_path.read_text(encoding="utf-8")
    else:
        text = raw_text

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {arg_name} as JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{arg_name} must decode to a JSON object, got {type(payload).__name__}.")
    return payload


def _normalize_hidden_layer_sizes(raw_value: object, *, phase_name: str) -> tuple[int, ...]:
    if isinstance(raw_value, (list, tuple)):
        hidden = tuple(int(v) for v in raw_value)
    else:
        hidden = (int(raw_value),)
    if any(width < 1 for width in hidden):
        raise ValueError(
            f"All hidden layer sizes must be >= 1 for phase={phase_name!r}, got {hidden!r}."
        )
    return hidden


def _resolve_phase_hparams_for_season(
    *,
    season: str,
    phase_hparams_payload: dict | None,
) -> dict[str, dict[str, object]]:
    if season not in MLP_HPARAMS_BY_SEASON_AND_PHASE:
        raise KeyError(f"Unsupported season for MLP hyperparameters: {season!r}")

    season_hparams = {
        phase_name: dict(MLP_HPARAMS_BY_SEASON_AND_PHASE[season][phase_name])
        for phase_name in PHASE_ORDER
    }

    payload = phase_hparams_payload
    if payload is not None and season in payload and isinstance(payload[season], dict):
        payload = payload[season]

    if payload is not None:
        for phase_name, phase_overrides in payload.items():
            if phase_name not in season_hparams:
                raise KeyError(
                    f"Unsupported phase override for season={season!r}: {phase_name!r}. "
                    f"Expected one of {PHASE_ORDER}."
                )
            if not isinstance(phase_overrides, dict):
                raise TypeError(
                    f"Phase override for phase={phase_name!r} must be a JSON object, "
                    f"got {type(phase_overrides).__name__}."
                )
            merged = dict(season_hparams[phase_name])
            merged.update(phase_overrides)
            season_hparams[phase_name] = merged

    required_keys = set(COLD_DAY_MLP_HPARAMS.keys())
    supported_activations = {"relu", "tanh", "softplus", "identity", "linear", "none"}
    supported_solvers = {"adam"}
    for phase_name, hparams in season_hparams.items():
        missing = sorted(required_keys - set(hparams.keys()))
        if missing:
            raise KeyError(
                f"Resolved hyperparameters for season={season!r} phase={phase_name!r} "
                f"are missing required keys: {missing}"
            )
        hparams["hidden_layer_sizes"] = _normalize_hidden_layer_sizes(
            hparams["hidden_layer_sizes"],
            phase_name=phase_name,
        )
        hparams["alpha"] = float(hparams["alpha"])
        hparams["learning_rate_init"] = float(hparams["learning_rate_init"])
        hparams["activation"] = str(hparams["activation"]).strip().lower()
        hparams["solver"] = str(hparams["solver"]).strip().lower()
        hparams["max_iter"] = int(hparams["max_iter"])
        hparams["tol"] = float(hparams["tol"])
        hparams["n_iter_no_change"] = int(hparams["n_iter_no_change"])
        hparams["validation_fraction"] = float(hparams["validation_fraction"])
        hparams["batch_size"] = int(hparams["batch_size"])

        if hparams["alpha"] < 0.0:
            raise ValueError(f"alpha must be >= 0 for phase={phase_name!r}, got {hparams['alpha']}.")
        if hparams["learning_rate_init"] <= 0.0:
            raise ValueError(
                f"learning_rate_init must be > 0 for phase={phase_name!r}, got {hparams['learning_rate_init']}."
            )
        if hparams["activation"] not in supported_activations:
            raise ValueError(
                f"Unsupported activation for phase={phase_name!r}: {hparams['activation']!r}. "
                f"Expected one of {sorted(supported_activations)}."
            )
        if hparams["solver"] not in supported_solvers:
            raise ValueError(
                f"Unsupported solver for phase={phase_name!r}: {hparams['solver']!r}. "
                f"Expected one of {sorted(supported_solvers)}."
            )
        if hparams["max_iter"] < 1:
            raise ValueError(f"max_iter must be >= 1 for phase={phase_name!r}, got {hparams['max_iter']}.")
        if hparams["tol"] < 0.0:
            raise ValueError(f"tol must be >= 0 for phase={phase_name!r}, got {hparams['tol']}.")
        if hparams["n_iter_no_change"] < 1:
            raise ValueError(
                f"n_iter_no_change must be >= 1 for phase={phase_name!r}, got {hparams['n_iter_no_change']}."
            )
        if not (0.0 < hparams["validation_fraction"] < 1.0):
            raise ValueError(
                f"validation_fraction must be in (0, 1) for phase={phase_name!r}, "
                f"got {hparams['validation_fraction']}."
            )
        if hparams["batch_size"] < 1:
            raise ValueError(f"batch_size must be >= 1 for phase={phase_name!r}, got {hparams['batch_size']}.")

    return season_hparams


def _require_columns(df: pd.DataFrame, cols: list[str], block_name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required {block_name} columns in the training dataset: {missing}")


def _rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mape_percent(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom_eps = 1e-6 * max(1.0, float(np.median(np.abs(y_true))))
    denom = np.maximum(np.abs(y_true), denom_eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def _mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))))


def _medae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.median(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))))


def _r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    y_mean = float(np.mean(y_true_arr))
    ss_tot = float(np.sum((y_true_arr - y_mean) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    residual = y_true_arr - np.asarray(y_pred, dtype=float)
    var_y = float(np.var(y_true_arr))
    if var_y <= 0.0:
        return 1.0 if float(np.var(residual)) <= 0.0 else 0.0
    return float(1.0 - (float(np.var(residual)) / var_y))


def _max_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))))


def _phase_log(phase_name: str, message: str) -> None:
    print(f"[FIT][{phase_name}] {message}", flush=True)


def _health(name: str, A: np.ndarray) -> None:
    A = np.asarray(A, dtype=float)
    finite = np.isfinite(A)
    frac_finite = float(np.mean(finite))
    print(f"\n[health] {name}: shape={A.shape}  finite_frac={frac_finite:.6f}", flush=True)
    if frac_finite < 1.0:
        bad = np.argwhere(~finite)
        print(f"[health] {name}: non-finite count={bad.shape[0]}", flush=True)
        print(f"[health] {name}: first 10 bad idx={bad[:10].tolist()}", flush=True)
    abs_max = float(np.nanmax(np.abs(A)))
    q = np.nanquantile(np.abs(A), [0.5, 0.9, 0.99, 0.999, 1.0])
    print(f"[health] {name}: abs_max={abs_max:.3e}", flush=True)
    print(f"[health] {name}: |A| quantiles 50/90/99/99.9/100 = {[float(x) for x in q]}", flush=True)


def _build_input_features(
    U_train: np.ndarray,
    U_test: np.ndarray,
    X0_train: np.ndarray | None = None,
    X0_test: np.ndarray | None = None,
    D_train: np.ndarray | None = None,
    D_test: np.ndarray | None = None,
    REFS_train: np.ndarray | None = None,
    REFS_test: np.ndarray | None = None,
    kappa_train: np.ndarray | None = None,
    kappa_test: np.ndarray | None = None,
    u_names: list[str] | None = None,
    x0_names: list[str] | None = None,
    d_names: list[str] | None = None,
    ref_names: list[str] | None = None,
    kappa_name: str | None = None,
) -> dict:
    U_train_fit = np.asarray(U_train, dtype=float)
    U_test_fit = np.asarray(U_test, dtype=float)

    u_in_names = list(u_names) if u_names is not None else [f"U{i}" for i in range(U_train_fit.shape[1])]

    blocks_train: list[np.ndarray] = [U_train_fit]
    blocks_test: list[np.ndarray] = [U_test_fit]
    names: list[str] = list(u_in_names)

    if X0_train is not None:
        X0_train_fit = np.asarray(X0_train, float)
        X0_test_fit = np.asarray(X0_test, float)
        x0_in_names = list(x0_names) if x0_names is not None else [f"X0_{i}" for i in range(X0_train_fit.shape[1])]
        blocks_train.append(X0_train_fit)
        blocks_test.append(X0_test_fit)
        names += x0_in_names

    if D_train is not None:
        D_train_fit = np.asarray(D_train, float)
        D_test_fit = np.asarray(D_test, float)
        d_in_names = list(d_names) if d_names is not None else [f"D_{i}" for i in range(D_train_fit.shape[1])]
        blocks_train.append(D_train_fit)
        blocks_test.append(D_test_fit)
        names += d_in_names

    if REFS_train is not None:
        REFS_train_fit = np.asarray(REFS_train, float)
        REFS_test_fit = np.asarray(REFS_test, float)
        ref_in_names = list(ref_names) if ref_names is not None else [f"REF_{i}" for i in range(REFS_train_fit.shape[1])]
        blocks_train.append(REFS_train_fit)
        blocks_test.append(REFS_test_fit)
        names += ref_in_names

    if kappa_train is not None:
        k_train = np.asarray(kappa_train, dtype=float).reshape(-1, 1)
        k_test = np.asarray(kappa_test, dtype=float).reshape(-1, 1)
        nm = str(kappa_name) if kappa_name is not None else "kappa"
        blocks_train.append(k_train)
        blocks_test.append(k_test)
        names.append(nm)

    Z_train = np.hstack(blocks_train)
    Z_test = np.hstack(blocks_test)
    feature_names = [str(nm) for nm in names]

    return {
        "Z_train": Z_train,
        "Z_test": Z_test,
        "feature_names": feature_names,
        "n_features_input": int(Z_train.shape[1]),
        "n_features_u": int(U_train_fit.shape[1]),
        "n_features_x0": int(0 if X0_train is None else np.asarray(X0_train, float).shape[1]),
        "n_features_d": int(0 if D_train is None else np.asarray(D_train, float).shape[1]),
        "n_features_refs": int(0 if REFS_train is None else np.asarray(REFS_train, float).shape[1]),
        "n_features_kappa": int(0 if kappa_train is None else 1),
    }


def _fit_and_evaluate_mlp(
    phase_name: str,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    mlp_hparams: dict[str, object],
    log_every: int,
    verbose_diagnostics: bool,
    device_override: str | None = None,
) -> dict:
    if verbose_diagnostics:
        _health(f"{phase_name}:Z_train", Z_train)
        _health(f"{phase_name}:Z_test", Z_test)
        _health(f"{phase_name}:y_train", y_train.reshape(-1, 1))
        _health(f"{phase_name}:y_test", y_test.reshape(-1, 1))

    Z_train_fit = np.asarray(Z_train, dtype=float)
    Z_test_fit = np.asarray(Z_test, dtype=float)
    y_train_fit = np.asarray(y_train, dtype=float).ravel()
    y_test_fit = np.asarray(y_test, dtype=float).ravel()

    n_train_full = int(Z_train_fit.shape[0])
    if n_train_full < 2:
        raise ValueError("Need at least 2 training samples to create a validation split.")
    validation_fraction = float(mlp_hparams["validation_fraction"])
    tol = float(mlp_hparams["tol"])
    max_iter = int(mlp_hparams["max_iter"])
    n_iter_no_change = int(mlp_hparams["n_iter_no_change"])
    n_val = int(round(validation_fraction * n_train_full))
    n_val = min(max(1, n_val), n_train_full - 1)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_train_full)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]

    Z_fit = Z_train_fit[fit_idx]
    y_fit = y_train_fit[fit_idx]
    Z_val = Z_train_fit[val_idx]
    y_val = y_train_fit[val_idx]

    solver_name = str(mlp_hparams["solver"]).strip().lower()
    if solver_name != "adam":
        raise ValueError(f"Only solver='adam' is supported in the PyTorch trainer, got {solver_name!r}")

    device = get_default_torch_device() if device_override is None else resolve_torch_device(device_override)
    batch_size = min(int(mlp_hparams["batch_size"]), int(len(fit_idx)))
    model = TorchMLP(
        input_dim=int(Z_train_fit.shape[1]),
        hidden_layer_sizes=tuple(int(v) for v in mlp_hparams["hidden_layer_sizes"]),
        activation=str(mlp_hparams["activation"]),
        output_activation=TORCH_OUTPUT_ACTIVATION,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(mlp_hparams["learning_rate_init"]),
        weight_decay=float(mlp_hparams["alpha"]),
    )
    loss_fn = nn.MSELoss()

    fit_dataset = TensorDataset(
        torch.as_tensor(Z_fit, dtype=torch.float32),
        torch.as_tensor(y_fit, dtype=torch.float32),
    )
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=max(1, batch_size),
        shuffle=True,
        drop_last=False,
    )

    Z_val_tensor = torch.as_tensor(Z_val, dtype=torch.float32, device=device)

    loss_curve: list[float] = []
    val_rmse_curve: list[float] = []
    best_state: tuple[dict[str, torch.Tensor], float] | None = None
    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    log_every = max(1, int(log_every))

    _phase_log(
        phase_name,
        f"start | device={device} | fit={len(fit_idx)} | val={len(val_idx)} | test={len(y_test_fit)} | input_dim={Z_train_fit.shape[1]}",
    )

    for epoch in range(1, max_iter + 1):
        try:
            model.train()
            epoch_loss_sum = 0.0
            epoch_count = 0
            for xb_cpu, yb_cpu in fit_loader:
                xb = xb_cpu.to(device)
                yb = yb_cpu.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss encountered in TorchMLP.")
                loss.backward()
                optimizer.step()
                batch_size_now = int(xb.shape[0])
                epoch_loss_sum += float(loss.detach().cpu()) * batch_size_now
                epoch_count += batch_size_now
        except KeyboardInterrupt:
            _phase_log(phase_name, "interrupted by user; restoring best model state and stopping.")
            break

        train_loss = float(epoch_loss_sum / max(1, epoch_count))
        model.eval()
        with torch.no_grad():
            y_val_pred = model(Z_val_tensor).detach().cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(y_val_pred)):
            raise RuntimeError("Non-finite validation predictions encountered in TorchMLP.")

        val_rmse = _rmse(y_val_pred, y_val)
        loss_curve.append(train_loss)
        val_rmse_curve.append(val_rmse)

        improved = val_rmse < (best_val_rmse - tol)
        if improved:
            best_val_rmse = float(val_rmse)
            best_epoch = int(epoch)
            best_state = (
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                float(train_loss),
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= n_iter_no_change:
                _phase_log(
                    phase_name,
                    f"early stop | epoch={epoch}/{max_iter} | best_epoch={best_epoch} | best_val_rmse={best_val_rmse:.4f}",
                )
                break

        should_log_progress = epoch == 1 or epoch == max_iter or epoch % log_every == 0
        if should_log_progress:
            _phase_log(
                phase_name,
                f"epoch={epoch}/{max_iter} | loss={train_loss:.4f} | val_rmse={val_rmse:.4f} | best={best_val_rmse:.4f}",
            )

    if best_state is None:
        raise RuntimeError("Torch MLP training finished without a valid best state.")

    best_state_dict, best_loss = best_state
    model.load_state_dict(best_state_dict)
    model.eval()

    Z_train_tensor = torch.as_tensor(Z_train_fit, dtype=torch.float32, device=device)
    Z_test_tensor = torch.as_tensor(Z_test_fit, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_train_pred = model(Z_train_tensor).detach().cpu().numpy().reshape(-1)
        y_test_pred = model(Z_test_tensor).detach().cpu().numpy().reshape(-1)
    if not np.all(np.isfinite(y_train_pred)) or not np.all(np.isfinite(y_test_pred)):
        raise RuntimeError("Non-finite predictions encountered.")

    rmse_test = _rmse(y_test_pred, y_test_fit)

    params_used = {
        "hidden_layer_sizes": tuple(mlp_hparams["hidden_layer_sizes"]),
        "alpha": float(mlp_hparams["alpha"]),
        "learning_rate_init": float(mlp_hparams["learning_rate_init"]),
        "activation": str(mlp_hparams["activation"]),
        "output_activation": str(TORCH_OUTPUT_ACTIVATION),
        "solver": str(mlp_hparams["solver"]),
        "validation_metric": str(MLP_VALIDATION_METRIC),
        "validation_fraction": validation_fraction,
        "max_iter": max_iter,
        "tol": tol,
        "n_iter_no_change": n_iter_no_change,
        "batch_size": int(mlp_hparams["batch_size"]),
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": int(best_epoch),
        "train_fit_size": int(len(fit_idx)),
        "val_size": int(len(val_idx)),
        "device": device,
        "best_loss": float(best_loss),
    }

    return {
        "model": model,
        "best_params": dict(params_used),
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "best_rmse_test": float(rmse_test),
        "loss_curve": np.asarray(loss_curve, dtype=float),
        "val_rmse_curve": np.asarray(val_rmse_curve, dtype=float),
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": int(best_epoch),
        "device": device,
        "best_loss": float(best_loss),
    }


def print_mlp_summary(best_params: dict, model: TorchMLP) -> None:
    print("\n===== TorchMLP Summary =====", flush=True)
    print(f"hidden_layer_sizes: {best_params.get('hidden_layer_sizes')}", flush=True)
    print(f"alpha: {best_params.get('alpha')}", flush=True)
    print(f"learning_rate_init: {best_params.get('learning_rate_init')}", flush=True)
    print(f"activation: {best_params.get('activation')}", flush=True)
    print(f"output_activation: {best_params.get('output_activation')}", flush=True)
    print(f"solver: {best_params.get('solver')}", flush=True)
    print(
        f"validation_metric: {best_params.get('validation_metric')}  validation_fraction: {best_params.get('validation_fraction')}",
        flush=True,
    )
    print(
        f"max_iter: {best_params.get('max_iter')}  tol: {best_params.get('tol')}  n_iter_no_change: {best_params.get('n_iter_no_change')}",
        flush=True,
    )
    print(
        f"train_fit_size: {best_params.get('train_fit_size')}  val_size: {best_params.get('val_size')}  "
        f"best_val_rmse: {best_params.get('best_val_rmse')}  best_epoch: {best_params.get('best_epoch')}  device: {best_params.get('device')}",
        flush=True,
    )
    print(
        f"input_dim: {getattr(model, 'input_dim', None)}  "
        f"n_layers_: {len(tuple(getattr(model, 'hidden_layer_sizes', ()) or ())) + 2}  "
        f"best_loss_: {best_params.get('best_loss', None)}",
        flush=True,
    )


def _compute_metrics(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
) -> dict:
    rmse_train = _rmse(y_train_pred, y_train)
    rmse_test = _rmse(y_test_pred, y_test)
    r2_train = _r2(y_train_pred, y_train)
    r2_test = _r2(y_test_pred, y_test)
    mape_train = _mape_percent(y_train_pred, y_train)
    mape_test = _mape_percent(y_test_pred, y_test)
    mae_train = _mae(y_train_pred, y_train)
    mae_test = _mae(y_test_pred, y_test)
    medae_train = _medae(y_train_pred, y_train)
    medae_test = _medae(y_test_pred, y_test)
    evs_train = _explained_variance(y_train_pred, y_train)
    evs_test = _explained_variance(y_test_pred, y_test)
    maxerr_train = _max_error(y_train_pred, y_train)
    maxerr_test = _max_error(y_test_pred, y_test)

    residual_train = y_train_pred - y_train
    residual_test = y_test_pred - y_test
    residual_stats_train = {
        "mean": float(np.mean(residual_train)),
        "std": float(np.std(residual_train)),
        "p5": float(np.percentile(residual_train, 5.0)),
        "p50": float(np.percentile(residual_train, 50.0)),
        "p95": float(np.percentile(residual_train, 95.0)),
    }
    residual_stats_test = {
        "mean": float(np.mean(residual_test)),
        "std": float(np.std(residual_test)),
        "p5": float(np.percentile(residual_test, 5.0)),
        "p50": float(np.percentile(residual_test, 50.0)),
        "p95": float(np.percentile(residual_test, 95.0)),
    }

    eps = 1e-9
    rel_err_train = np.abs(residual_train) / (np.abs(y_train) + eps)
    rel_err_test = np.abs(residual_test) / (np.abs(y_test) + eps)
    relerr_bands_train = {
        "frac_le_1pct": float(np.mean(rel_err_train <= 0.01)),
        "frac_le_5pct": float(np.mean(rel_err_train <= 0.05)),
        "frac_le_10pct": float(np.mean(rel_err_train <= 0.10)),
    }
    relerr_bands_test = {
        "frac_le_1pct": float(np.mean(rel_err_test <= 0.01)),
        "frac_le_5pct": float(np.mean(rel_err_test <= 0.05)),
        "frac_le_10pct": float(np.mean(rel_err_test <= 0.10)),
    }

    return {
        "rmse_train": float(rmse_train),
        "rmse_test": float(rmse_test),
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "mape_train": float(mape_train),
        "mape_test": float(mape_test),
        "mae_train": float(mae_train),
        "mae_test": float(mae_test),
        "medae_train": float(medae_train),
        "medae_test": float(medae_test),
        "evs_train": float(evs_train),
        "evs_test": float(evs_test),
        "maxerr_train": float(maxerr_train),
        "maxerr_test": float(maxerr_test),
        "residual_stats_train": residual_stats_train,
        "residual_stats_test": residual_stats_test,
        "relerr_bands_train": relerr_bands_train,
        "relerr_bands_test": relerr_bands_test,
    }


def _plot_mlp_loss_curve(
    loss_curve: np.ndarray,
    val_rmse_curve: np.ndarray | None,
    out_dir: Path,
) -> Path | None:
    plots_dir = out_dir / "Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    loss_curve_arr = np.asarray(loss_curve, dtype=float).reshape(-1)
    if loss_curve_arr.size == 0:
        print("No loss curve found for TorchMLP; skipping loss curve plot.")
        return None

    has_val = val_rmse_curve is not None and len(val_rmse_curve) > 0

    if has_val:
        val_rmse_arr = np.asarray(val_rmse_curve, dtype=float).reshape(-1)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
        axes[0].plot(np.arange(1, loss_curve_arr.size + 1), loss_curve_arr, color="tab:blue", lw=1.2, label="training")
        axes[0].set_title("Torch MLP Training Loss Curve")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, ls="--", lw=0.4, alpha=0.6)
        axes[0].legend(loc="best")

        axes[1].plot(
            np.arange(1, val_rmse_arr.size + 1),
            val_rmse_arr,
            color="tab:orange",
            lw=1.2,
            label="validation RMSE",
        )
        axes[1].set_title("Torch MLP Validation RMSE Curve")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation RMSE")
        axes[1].grid(True, ls="--", lw=0.4, alpha=0.6)
        axes[1].legend(loc="best")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(1, loss_curve_arr.size + 1), loss_curve_arr, color="tab:blue", lw=1.2, label="training")
        ax.set_title("Torch MLP Training Loss Curve")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, ls="--", lw=0.4, alpha=0.6)
        ax.legend(loc="best")

    fig.tight_layout()
    loss_fig_path = plots_dir / "mlp_loss_curve.pdf"
    fig.savefig(loss_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return loss_fig_path


def _write_outputs_and_plots(
    out_dir: Path,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    loss_curve: np.ndarray,
    val_rmse_curve: np.ndarray,
    npz_payload: dict,
    meta_payload: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = out_dir / "Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_order = np.argsort(train_idx)
    test_order = np.argsort(test_idx)
    y_train_true_sorted = y_train[train_order]
    y_train_pred_sorted = y_train_pred[train_order]
    y_test_true_sorted = y_test[test_order]
    y_test_pred_sorted = y_test_pred[test_order]

    x_train = np.arange(len(y_train_true_sorted), dtype=int)
    x_test = np.arange(len(y_test_true_sorted), dtype=int)
    train_step = max(1, len(train_idx) // 4000)
    test_step = max(1, len(test_idx) // 4000)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    axes[0].plot(x_train[::train_step], y_train_true_sorted[::train_step], lw=0.8, color="black", label="realization")
    axes[0].plot(
        x_train[::train_step], y_train_pred_sorted[::train_step], lw=0.8, color="tab:blue", label="prediction"
    )
    axes[0].set_title("Train: Realization vs Prediction")
    axes[0].set_xlabel("Sample #")
    axes[0].set_ylabel("Cost")
    axes[0].grid(True, ls="--", lw=0.4, alpha=0.6)
    axes[0].legend(loc="best")

    axes[1].plot(x_test[::test_step], y_test_true_sorted[::test_step], lw=0.8, color="black", label="realization")
    axes[1].plot(x_test[::test_step], y_test_pred_sorted[::test_step], lw=0.8, color="tab:blue", label="prediction")
    axes[1].set_title("Test: Realization vs Prediction")
    axes[1].set_xlabel("Sample #")
    axes[1].set_ylabel("Cost")
    axes[1].grid(True, ls="--", lw=0.4, alpha=0.6)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig_path = plots_dir / "prediction_vs_realization_curves.pdf"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    loss_curve_path = _plot_mlp_loss_curve(
        loss_curve=loss_curve,
        val_rmse_curve=val_rmse_curve,
        out_dir=out_dir,
    )

    np.savez(out_dir / "model_and_data.npz", **npz_payload)

    meta = dict(meta_payload)
    meta["curve_plot_path"] = str(fig_path)
    meta["loss_curve_plot_path"] = "" if loss_curve_path is None else str(loss_curve_path)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return fig_path


def _build_phase_labels(climate_refs: dict[str, np.ndarray | float]) -> np.ndarray:
    is_day = np.asarray(climate_refs["is_day"], dtype=bool).reshape(-1)
    labels = np.full(is_day.shape, "night", dtype=object)
    labels[is_day] = "day"
    return np.asarray(labels, dtype=str)


def _resolve_r3_training_data_path(season: str) -> Path:
    preferred = (R3_DATA_ROOT / season / R3_TRAINING_DATA_FILENAME).resolve()
    if preferred.exists():
        return preferred
    legacy = (R3_DATA_ROOT / season / R3_LEGACY_DATA_FILENAME).resolve()
    if legacy.exists():
        return legacy
    return preferred


def load_phasewise_training_payload(season: str) -> dict[str, object]:
    data_csv_path = _resolve_r3_training_data_path(season)
    if not data_csv_path.exists():
        raise FileNotFoundError(
            "Required training dataset not found:\n"
            f"  {data_csv_path}\n"
            "Generate it first with:\n"
            f"  python3 {REPO_ROOT / 'experiment_runner/r3_generate_data_from_base.py'} --season {season}"
        )

    data_df = pd.read_csv(data_csv_path)
    param_keeper = builder_mpc.load_parameters(PARAM_YAML_PATH)
    u_names = list(get_control_names_in_order(param_keeper))
    x_names = list(get_state_names_in_order(param_keeper))
    d_names = list(get_disturbance_names_in_order(param_keeper))

    u_cols = list(u_names)
    x0_cols = [f"X_ini_{name}" for name in x_names]
    d_cols = [f"D_{name}" for name in d_names]
    dx_cols = [f"DX_d{name}" for name in x_names]
    x1_cols = [f"X1_{name}" for name in x_names]

    _require_columns(data_df, u_cols, "U")
    _require_columns(data_df, x0_cols, "X_ini")
    _require_columns(data_df, d_cols, "D")
    _require_columns(data_df, dx_cols, "DX")
    _require_columns(data_df, [KAPPA_COL], "kappa")

    U = data_df[u_cols].to_numpy(dtype=float)
    X0 = data_df[x0_cols].to_numpy(dtype=float)
    D = data_df[d_cols].to_numpy(dtype=float)
    DX = data_df[dx_cols].to_numpy(dtype=float)
    if all(col in data_df.columns for col in x1_cols):
        X1 = data_df[x1_cols].to_numpy(dtype=float)
    else:
        X1 = X0 + DX
        print("[INFO] X1_* columns missing; reconstructing X1 = X_ini + DX.")
    kappa_k = data_df[KAPPA_COL].to_numpy(dtype=int)

    ref_cols = list(REF_COLS)
    climate_refs: dict[str, np.ndarray | float] | None = None
    if PHASE_COL in data_df.columns:
        phase_labels = data_df[PHASE_COL].astype(str).to_numpy()
    else:
        climate_refs = get_bounds_and_refs(param_keeper, kappa_k)
        print(
            "[INFO] surrogate_phase column missing in training data; recomputing phase labels from kappa_k.",
            flush=True,
        )
        phase_labels = _build_phase_labels(climate_refs)

    if all(col in data_df.columns for col in ref_cols):
        REFS = data_df[ref_cols].to_numpy(dtype=float)
    else:
        if climate_refs is None:
            climate_refs = get_bounds_and_refs(param_keeper, kappa_k)
        print(
            "[INFO] Reference columns missing in training data; recomputing refs from kappa_k.",
            flush=True,
        )
        REFS = np.column_stack(
            [
                np.asarray(climate_refs["T_in_star"], dtype=float),
                np.asarray(climate_refs["H_in_star"], dtype=float),
                np.asarray(climate_refs["C_in_star"], dtype=float),
                np.asarray(climate_refs["L_star"], dtype=float),
            ]
        )

    target_col = next((col for col in TARGET_OUTPUT_COST_COLS if col in data_df.columns), None)
    if target_col is not None:
        y = data_df[target_col].to_numpy(dtype=float)
    else:
        print(
            "[INFO] Target output cost column missing in training data; recomputing labels from X1/U/kappa.",
            flush=True,
        )
        y = get_one_step_total_cost_act_batch(
            x_act=X1,
            u0=U,
            kappa_act=kappa_k,
            keeper_path=PARAM_YAML_PATH,
        )

    return {
        "season": season,
        "data_csv_path": data_csv_path,
        "param_keeper": param_keeper,
        "u_cols": u_cols,
        "x_names": x_names,
        "d_names": d_names,
        "ref_cols": ref_cols,
        "U": U,
        "X0": X0,
        "D": D,
        "REFS": REFS,
        "kappa_k": kappa_k,
        "y": y,
        "phase_labels": phase_labels,
    }


def _fit_all_phases(
    *,
    season: str,
    phase_hparams_by_phase: dict[str, dict[str, object]],
    phase_workers: int,
    log_every: int,
    verbose_diagnostics: bool,
    data_csv_path: Path,
    out_base_dir: Path,
    param_keeper: dict,
    u_cols: list[str],
    x_names: list[str],
    d_names: list[str],
    ref_cols: list[str],
    U: np.ndarray,
    X0: np.ndarray,
    D: np.ndarray,
    REFS: np.ndarray,
    kappa_k: np.ndarray,
    y: np.ndarray,
    phase_labels: np.ndarray,
) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for phase_offset, phase_name in enumerate(PHASE_ORDER):
        jobs.append(
            {
                "season": season,
                "phase_name": phase_name,
                "phase_mask": np.asarray(phase_labels == phase_name, dtype=bool),
                "phase_seed": int(SEED + phase_offset),
                "mlp_hparams": dict(phase_hparams_by_phase[phase_name]),
                "out_dir": out_base_dir / phase_name,
                "log_every": int(log_every),
                "verbose_diagnostics": bool(verbose_diagnostics),
                "data_csv_path": data_csv_path,
                "param_keeper": param_keeper,
                "u_cols": u_cols,
                "x_names": x_names,
                "d_names": d_names,
                "ref_cols": ref_cols,
                "U": U,
                "X0": X0,
                "D": D,
                "REFS": REFS,
                "kappa_k": kappa_k,
                "y": y,
            }
        )

    requested_workers = max(1, min(int(phase_workers), len(jobs)))
    max_workers = requested_workers
    if torch.backends.mps.is_available() and requested_workers > 1:
        print(
            "[PHASE_TRAIN] MPS detected; forcing sequential phase training "
            "because concurrent MPS workers can hang or produce unstable results.",
            flush=True,
        )
        max_workers = 1
    if max_workers == 1:
        print("[PHASE_TRAIN] running day/night fits sequentially.", flush=True)
        return [_fit_one_phase(**job) for job in jobs]

    print(f"[PHASE_TRAIN] running day/night fits in parallel with max_workers={max_workers}.", flush=True)
    results_by_phase: dict[str, dict[str, object]] = {}
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        future_to_phase = {
            executor.submit(_fit_one_phase, **job): str(job["phase_name"])
            for job in jobs
        }
        for future in as_completed(future_to_phase):
            phase_name = future_to_phase[future]
            results_by_phase[phase_name] = future.result()
            print(f"[PHASE_TRAIN] completed phase={phase_name}.", flush=True)

    return [results_by_phase[str(job["phase_name"])] for job in jobs]


def _fit_one_phase(
    *,
    season: str,
    phase_name: str,
    phase_mask: np.ndarray,
    phase_seed: int,
    mlp_hparams: dict[str, object],
    out_dir: Path,
    log_every: int,
    verbose_diagnostics: bool,
    device_override: str | None = None,
    data_csv_path: Path,
    param_keeper: dict,
    u_cols: list[str],
    x_names: list[str],
    d_names: list[str],
    ref_cols: list[str],
    U: np.ndarray,
    X0: np.ndarray,
    D: np.ndarray,
    REFS: np.ndarray,
    kappa_k: np.ndarray,
    y: np.ndarray,
) -> dict[str, object]:
    mlp_hparams = dict(mlp_hparams)

    source_indices = np.flatnonzero(phase_mask)
    n_samples = int(source_indices.size)
    if n_samples < 3:
        raise ValueError(
            f"Phase '{phase_name}' has only {n_samples} samples; need at least 3 "
            "to make non-empty train/test splits and a validation split inside training."
        )

    U_phase = U[phase_mask]
    X0_phase = X0[phase_mask]
    D_phase = D[phase_mask]
    REFS_phase = REFS[phase_mask]
    kappa_phase = kappa_k[phase_mask]
    y_phase = y[phase_mask]

    rng = np.random.default_rng(phase_seed)
    idx = rng.permutation(n_samples)
    n_train = int(round(TRAIN_FRAC * n_samples))
    n_train = max(2, min(n_train, n_samples - 1))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    U_train, U_test = U_phase[train_idx], U_phase[test_idx]
    y_train, y_test = y_phase[train_idx], y_phase[test_idx]

    X0_train = X0_phase[train_idx] if USE_X0_AS_FEATURE else None
    X0_test = X0_phase[test_idx] if USE_X0_AS_FEATURE else None
    D_train = D_phase[train_idx] if USE_D_AS_FEATURE else None
    D_test = D_phase[test_idx] if USE_D_AS_FEATURE else None
    REFS_train = REFS_phase[train_idx] if USE_REFS_AS_FEATURE else None
    REFS_test = REFS_phase[test_idx] if USE_REFS_AS_FEATURE else None
    kappa_train = kappa_phase[train_idx] if USE_KAPPA_AS_FEATURE else None
    kappa_test = kappa_phase[test_idx] if USE_KAPPA_AS_FEATURE else None

    features = _build_input_features(
        U_train=U_train,
        U_test=U_test,
        X0_train=X0_train,
        X0_test=X0_test,
        D_train=D_train,
        D_test=D_test,
        REFS_train=REFS_train,
        REFS_test=REFS_test,
        kappa_train=kappa_train,
        kappa_test=kappa_test,
        u_names=u_cols,
        x0_names=[f"X_ini_{name}" for name in x_names] if USE_X0_AS_FEATURE else None,
        d_names=[f"D_{name}" for name in d_names] if USE_D_AS_FEATURE else None,
        ref_names=ref_cols if USE_REFS_AS_FEATURE else None,
        kappa_name=KAPPA_COL if USE_KAPPA_AS_FEATURE else None,
    )

    feature_scaler = fit_surrogate_feature_scaler(
        features["Z_train"],
        features["feature_names"],
        kappa_total_steps=int(param_keeper["kappa_day_night_total_steps"]),
    )
    Z_train_scaled = transform_surrogate_feature_matrix(
        features["Z_train"],
        feature_scaler,
        feature_names=features["feature_names"],
    )
    Z_test_scaled = transform_surrogate_feature_matrix(
        features["Z_test"],
        feature_scaler,
        feature_names=features["feature_names"],
    )

    fit = _fit_and_evaluate_mlp(
        phase_name=phase_name,
        Z_train=Z_train_scaled,
        y_train=y_train,
        Z_test=Z_test_scaled,
        y_test=y_test,
        mlp_hparams=mlp_hparams,
        log_every=log_every,
        verbose_diagnostics=verbose_diagnostics,
        device_override=device_override,
    )

    if verbose_diagnostics:
        print_mlp_summary(best_params=fit["best_params"], model=fit["model"])

    metrics = _compute_metrics(
        y_train=y_train,
        y_train_pred=fit["y_train_pred"],
        y_test=y_test,
        y_test_pred=fit["y_test_pred"],
    )

    baseline_mean = float(np.mean(y_train))
    baseline_rmse_test = _rmse(np.full_like(y_test, baseline_mean, dtype=float), y_test)

    corr_u_vs_cost_end = {}
    y_std = float(np.std(y_phase))
    for j, label in enumerate(u_cols):
        u_j = U_phase[:, j]
        if float(np.std(u_j)) <= 0.0 or y_std <= 0.0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(u_j, y_phase)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        corr_u_vs_cost_end[label] = corr

    feature_names = list(features["feature_names"])

    model_dir = out_dir / "Model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "torch_mlp_model.pt"
    save_torch_mlp_checkpoint(
        model_path,
        fit["model"],
        input_dim=int(features["n_features_input"]),
        hidden_layer_sizes=tuple(int(v) for v in mlp_hparams["hidden_layer_sizes"]),
        activation=str(mlp_hparams["activation"]),
        output_activation=str(TORCH_OUTPUT_ACTIVATION),
        extra_metadata={
            "season": season,
            "phase_name": phase_name,
            "feature_names": feature_names,
            "cost_target_kind": DFL_COST_TARGET_KIND,
        },
    )

    npz_payload = {
        "season": np.asarray(season, dtype=str),
        "phase_name": np.asarray(phase_name, dtype=str),
        "source_indices": source_indices.astype(int),
        "n_features_input": int(features["n_features_input"]),
        "n_features_u": int(features["n_features_u"]),
        "n_features_x0": int(features["n_features_x0"]),
        "n_features_d": int(features["n_features_d"]),
        "n_features_refs": int(features["n_features_refs"]),
        "n_features_kappa": int(features["n_features_kappa"]),
        "cost_target_kind": np.asarray(DFL_COST_TARGET_KIND, dtype=str),
        "feature_names": np.asarray(feature_names, dtype=str),
        "train_idx": train_idx.astype(int),
        "test_idx": test_idx.astype(int),
        "y_train_pred": np.asarray(fit["y_train_pred"], dtype=float),
        "y_test_pred": np.asarray(fit["y_test_pred"], dtype=float),
        "y_train_true": y_train.astype(float),
        "y_test_true": y_test.astype(float),
        "loss_curve": np.asarray(fit["loss_curve"], dtype=float),
        "val_rmse_curve": np.asarray(fit["val_rmse_curve"], dtype=float),
        "validation_metric": np.asarray(MLP_VALIDATION_METRIC, dtype=str),
        "model_torch_path": np.asarray(str(model_path), dtype=str),
        "torch_output_activation": np.asarray(TORCH_OUTPUT_ACTIVATION, dtype=str),
    }
    npz_payload.update(feature_scaler_to_npz_payload(feature_scaler))

    meta_payload = {
        "season": season,
        "phase_name": phase_name,
        "data_csv_path": str(data_csv_path),
        "param_yaml_path": str(PARAM_YAML_PATH),
        "model_type": "TorchMLP",
        "torch_output_activation": str(TORCH_OUTPUT_ACTIVATION),
        "validation_metric": MLP_VALIDATION_METRIC,
        "mlp_best_params": fit["best_params"],
        "mlp_best_rmse_test": float(fit["best_rmse_test"]),
        "mlp_best_val_rmse": float(fit["best_val_rmse"]),
        "model_torch_path": str(model_path),
        "cost_target_kind": DFL_COST_TARGET_KIND,
        "torch_device": str(fit["device"]),
        "n_features_input": int(features["n_features_input"]),
        "n_features_u": int(features["n_features_u"]),
        "n_features_x0": int(features["n_features_x0"]),
        "n_features_d": int(features["n_features_d"]),
        "n_features_refs": int(features["n_features_refs"]),
        "n_features_kappa": int(features["n_features_kappa"]),
        "feature_scaler_kind": "deterministic_affine_0_1",
        "use_x0_as_feature": bool(USE_X0_AS_FEATURE),
        "use_d_as_feature": bool(USE_D_AS_FEATURE),
        "use_refs_as_feature": bool(USE_REFS_AS_FEATURE),
        "use_kappa_as_feature": bool(USE_KAPPA_AS_FEATURE),
        "feature_dim_u": int(U_phase.shape[1]),
        "u_names": u_cols,
        "x_names": x_names,
        "d_names": d_names,
        "ref_names": ref_cols if USE_REFS_AS_FEATURE else [],
        "N": int(n_samples),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "source_indices_count": int(source_indices.size),
        "source_kappa_min": int(np.min(kappa_phase)),
        "source_kappa_max": int(np.max(kappa_phase)),
        "rmse_train": float(metrics["rmse_train"]),
        "rmse_test": float(metrics["rmse_test"]),
        "r2_train": float(metrics["r2_train"]),
        "r2_test": float(metrics["r2_test"]),
        "mape_train_percent": float(metrics["mape_train"]),
        "mape_test_percent": float(metrics["mape_test"]),
        "mae_train": float(metrics["mae_train"]),
        "mae_test": float(metrics["mae_test"]),
        "medae_train": float(metrics["medae_train"]),
        "medae_test": float(metrics["medae_test"]),
        "evs_train": float(metrics["evs_train"]),
        "evs_test": float(metrics["evs_test"]),
        "maxerr_train": float(metrics["maxerr_train"]),
        "maxerr_test": float(metrics["maxerr_test"]),
        "residual_stats_train": metrics["residual_stats_train"],
        "residual_stats_test": metrics["residual_stats_test"],
        "relerr_bands_train": metrics["relerr_bands_train"],
        "relerr_bands_test": metrics["relerr_bands_test"],
        "baseline_rmse_test": float(baseline_rmse_test),
        "corr_u_vs_cost_end": corr_u_vs_cost_end,
    }

    fig_path = _write_outputs_and_plots(
        out_dir=out_dir,
        train_idx=train_idx,
        test_idx=test_idx,
        y_train=y_train,
        y_train_pred=np.asarray(fit["y_train_pred"], dtype=float),
        y_test=y_test,
        y_test_pred=np.asarray(fit["y_test_pred"], dtype=float),
        loss_curve=np.asarray(fit["loss_curve"], dtype=float),
        val_rmse_curve=np.asarray(fit["val_rmse_curve"], dtype=float),
        npz_payload=npz_payload,
        meta_payload=meta_payload,
    )

    _phase_log(
        phase_name,
        f"done | N={n_samples} | train={len(train_idx)} | test={len(test_idx)} | rmse_test={metrics['rmse_test']:.4f} | r2_test={metrics['r2_test']:.4f} | best_val_rmse={fit['best_val_rmse']:.4f}",
    )

    return {
        "season": season,
        "phase_name": phase_name,
        "out_dir": str(out_dir),
        "model_torch_path": str(model_path),
        "curve_plot_path": str(fig_path),
        "n_samples": int(n_samples),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "rmse_train": float(metrics["rmse_train"]),
        "rmse_test": float(metrics["rmse_test"]),
        "mae_train": float(metrics["mae_train"]),
        "mae_test": float(metrics["mae_test"]),
        "r2_train": float(metrics["r2_train"]),
        "r2_test": float(metrics["r2_test"]),
        "best_val_rmse": float(fit["best_val_rmse"]),
        "baseline_rmse_test": float(baseline_rmse_test),
    }


def main() -> None:
    args = parse_args()
    season = str(args.season)
    phase_hparams_payload = _load_json_arg(args.phase_hparams_json, arg_name="--phase-hparams-json")
    phase_hparams_by_phase = _resolve_phase_hparams_for_season(
        season=season,
        phase_hparams_payload=phase_hparams_payload,
    )
    phase_workers = int(args.phase_workers)
    log_every = int(args.log_every)
    verbose_diagnostics = bool(args.verbose_diagnostics)
    if phase_workers < 1:
        raise ValueError(f"--phase-workers must be >= 1, got {phase_workers}")
    if log_every < 1:
        raise ValueError(f"--log-every must be >= 1, got {log_every}")
    out_base_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (OUT_ROOT / season).resolve()
    )
    training_payload = load_phasewise_training_payload(season)
    data_csv_path = Path(training_payload["data_csv_path"])

    if out_base_dir.exists():
        shutil.rmtree(out_base_dir)
    out_base_dir.mkdir(parents=True, exist_ok=True)
    param_keeper = dict(training_payload["param_keeper"])
    u_cols = list(training_payload["u_cols"])
    x_names = list(training_payload["x_names"])
    d_names = list(training_payload["d_names"])
    ref_cols = list(training_payload["ref_cols"])
    U = np.asarray(training_payload["U"], dtype=float)
    X0 = np.asarray(training_payload["X0"], dtype=float)
    D = np.asarray(training_payload["D"], dtype=float)
    REFS = np.asarray(training_payload["REFS"], dtype=float)
    kappa_k = np.asarray(training_payload["kappa_k"], dtype=int)
    y = np.asarray(training_payload["y"], dtype=float)
    phase_labels = np.asarray(training_payload["phase_labels"], dtype=str)

    phase_counts = {phase: int(np.sum(phase_labels == phase)) for phase in PHASE_ORDER}
    print(f"[PHASE_COUNTS] season={season} counts={phase_counts}", flush=True)

    summary_records = _fit_all_phases(
        season=season,
        phase_hparams_by_phase=phase_hparams_by_phase,
        phase_workers=phase_workers,
        log_every=log_every,
        verbose_diagnostics=verbose_diagnostics,
        data_csv_path=data_csv_path,
        out_base_dir=out_base_dir,
        param_keeper=param_keeper,
        u_cols=u_cols,
        x_names=x_names,
        d_names=d_names,
        ref_cols=ref_cols,
        U=U,
        X0=X0,
        D=D,
        REFS=REFS,
        kappa_k=kappa_k,
        y=y,
        phase_labels=phase_labels,
    )

    season_summary = {
        "season": season,
        "data_csv_path": str(data_csv_path),
        "param_yaml_path": str(PARAM_YAML_PATH),
        "cost_target_kind": DFL_COST_TARGET_KIND,
        "mlp_hparams_by_phase": {
            phase_name: dict(phase_hparams_by_phase[phase_name]) for phase_name in PHASE_ORDER
        },
        "phase_order": list(PHASE_ORDER),
        "phase_counts": phase_counts,
        "phase_models": summary_records,
    }
    (out_base_dir / "meta.json").write_text(json.dumps(season_summary, indent=2), encoding="utf-8")

    print(
        f"\n[SUMMARY] season={season} | phase_workers={phase_workers} | log_every={log_every} | counts={phase_counts}",
        flush=True,
    )
    for record in summary_records:
        print(
            f"[SUMMARY][{record['phase_name']}] N={record['n_samples']} | train={record['train_size']} | "
            f"test={record['test_size']} | rmse_test={record['rmse_test']:.4f} | "
            f"r2_test={record['r2_test']:.4f} | best_val_rmse={record['best_val_rmse']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
