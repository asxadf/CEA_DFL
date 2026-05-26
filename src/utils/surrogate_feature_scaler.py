from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


MODE_IDENTITY = "identity"
MODE_AFFINE_0_1 = "affine_0_1"
MODE_CONSTANT_ZERO = "constant_zero"
NPZ_PREFIX = "feature_scaler_"


def _as_feature_names(feature_names: Sequence[str]) -> list[str]:
    names = [str(name) for name in feature_names]
    if len(names) == 0:
        raise ValueError("feature_names must not be empty.")
    return names


def _validate_scaler_payload(
    feature_names: Sequence[str],
    methods: np.ndarray,
    offsets: np.ndarray,
    denominators: np.ndarray,
) -> dict[str, object]:
    names = _as_feature_names(feature_names)
    methods_arr = np.asarray(methods, dtype=str).reshape(-1)
    offsets_arr = np.asarray(offsets, dtype=float).reshape(-1)
    denominators_arr = np.asarray(denominators, dtype=float).reshape(-1)

    n = len(names)
    if methods_arr.size != n or offsets_arr.size != n or denominators_arr.size != n:
        raise ValueError(
            "Feature scaler size mismatch: "
            f"feature_names={n}, methods={methods_arr.size}, "
            f"offsets={offsets_arr.size}, denominators={denominators_arr.size}"
        )
    if not np.all(np.isfinite(offsets_arr)):
        raise FloatingPointError("Feature scaler offsets contain non-finite values.")
    if not np.all(np.isfinite(denominators_arr)):
        raise FloatingPointError("Feature scaler denominators contain non-finite values.")

    allowed = {MODE_IDENTITY, MODE_AFFINE_0_1, MODE_CONSTANT_ZERO}
    bad_methods = sorted(set(methods_arr.tolist()) - allowed)
    if bad_methods:
        raise ValueError(f"Unsupported feature scaler methods: {bad_methods}")

    affine_mask = methods_arr == MODE_AFFINE_0_1
    if np.any(denominators_arr[affine_mask] <= 0.0):
        raise ValueError("Affine feature scaler denominators must be positive.")

    return {
        "feature_names": names,
        "methods": methods_arr,
        "offsets": offsets_arr,
        "denominators": denominators_arr,
    }


def fit_surrogate_feature_scaler(
    Z_train: np.ndarray,
    feature_names: Sequence[str],
    *,
    kappa_total_steps: int | None = None,
) -> dict[str, object]:
    Z = np.asarray(Z_train, dtype=float)
    names = _as_feature_names(feature_names)
    if Z.ndim != 2:
        raise ValueError(f"Z_train must be 2D, got shape {Z.shape}")
    if Z.shape[1] != len(names):
        raise ValueError(
            f"Z_train width {Z.shape[1]} must match feature_names length {len(names)}."
        )
    if not np.all(np.isfinite(Z)):
        raise FloatingPointError("Z_train contains non-finite values; cannot fit scaler.")

    methods: list[str] = []
    offsets: list[float] = []
    denominators: list[float] = []

    total_steps = None if kappa_total_steps is None else int(kappa_total_steps)
    if total_steps is not None and total_steps <= 0:
        raise ValueError(f"kappa_total_steps must be positive, got {total_steps}")

    for j, name in enumerate(names):
        col = Z[:, j]
        if name.startswith("U_"):
            methods.append(MODE_IDENTITY)
            offsets.append(0.0)
            denominators.append(1.0)
            continue

        if name == "kappa_k":
            if total_steps is None:
                raise ValueError("kappa_total_steps is required to scale kappa_k.")
            methods.append(MODE_AFFINE_0_1)
            offsets.append(1.0)
            denominators.append(float(total_steps))
            continue

        c_min = float(np.min(col))
        c_max = float(np.max(col))
        span = c_max - c_min
        if span <= 0.0:
            methods.append(MODE_CONSTANT_ZERO)
            offsets.append(c_min)
            denominators.append(1.0)
            continue

        methods.append(MODE_AFFINE_0_1)
        offsets.append(-c_min)
        denominators.append(span)

    return _validate_scaler_payload(
        feature_names=names,
        methods=np.asarray(methods, dtype=str),
        offsets=np.asarray(offsets, dtype=float),
        denominators=np.asarray(denominators, dtype=float),
    )


def transform_surrogate_feature_matrix(
    Z: np.ndarray,
    scaler: Mapping[str, object],
    *,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    X = np.asarray(Z, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape {X.shape}")
    scaler_names = _as_feature_names(scaler["feature_names"])
    if feature_names is not None:
        provided_names = _as_feature_names(feature_names)
        if provided_names != scaler_names:
            raise ValueError(
                "Provided feature_names do not match scaler feature_names: "
                f"provided={provided_names[:10]} scaler={scaler_names[:10]}"
            )
    names = scaler_names
    payload = _validate_scaler_payload(
        feature_names=names,
        methods=np.asarray(scaler["methods"], dtype=str),
        offsets=np.asarray(scaler["offsets"], dtype=float),
        denominators=np.asarray(scaler["denominators"], dtype=float),
    )
    if X.shape[1] != len(payload["feature_names"]):
        raise ValueError(
            f"Z width {X.shape[1]} must match scaler width {len(payload['feature_names'])}."
        )

    Y = np.asarray(X, dtype=float).copy()
    methods = np.asarray(payload["methods"], dtype=str)
    offsets = np.asarray(payload["offsets"], dtype=float)
    denominators = np.asarray(payload["denominators"], dtype=float)

    affine_mask = methods == MODE_AFFINE_0_1
    constant_mask = methods == MODE_CONSTANT_ZERO

    if np.any(affine_mask):
        Y[:, affine_mask] = (Y[:, affine_mask] + offsets[affine_mask]) / denominators[affine_mask]
        Y[:, affine_mask] = np.clip(Y[:, affine_mask], 0.0, 1.0)
    if np.any(constant_mask):
        Y[:, constant_mask] = 0.0

    if not np.all(np.isfinite(Y)):
        raise FloatingPointError("Scaled feature matrix contains non-finite values.")
    return Y


def transform_surrogate_feature_vector(
    z: np.ndarray,
    scaler: Mapping[str, object],
    *,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    vec = np.asarray(z, dtype=float).reshape(-1)
    mat = transform_surrogate_feature_matrix(
        vec.reshape(1, -1),
        scaler,
        feature_names=feature_names,
    )
    return mat.reshape(-1)


def feature_scaler_to_npz_payload(
    scaler: Mapping[str, object],
    *,
    prefix: str = NPZ_PREFIX,
) -> dict[str, np.ndarray]:
    payload = _validate_scaler_payload(
        feature_names=scaler["feature_names"],
        methods=np.asarray(scaler["methods"], dtype=str),
        offsets=np.asarray(scaler["offsets"], dtype=float),
        denominators=np.asarray(scaler["denominators"], dtype=float),
    )
    return {
        f"{prefix}feature_names": np.asarray(payload["feature_names"], dtype=str),
        f"{prefix}methods": np.asarray(payload["methods"], dtype=str),
        f"{prefix}offsets": np.asarray(payload["offsets"], dtype=float),
        f"{prefix}denominators": np.asarray(payload["denominators"], dtype=float),
    }


def feature_scaler_from_npz(
    npz: Mapping[str, object],
    *,
    prefix: str = NPZ_PREFIX,
    expected_feature_names: Sequence[str] | None = None,
) -> dict[str, object] | None:
    keys = (
        f"{prefix}feature_names",
        f"{prefix}methods",
        f"{prefix}offsets",
        f"{prefix}denominators",
    )
    present = [key in npz for key in keys]
    if not any(present):
        return None
    if not all(present):
        missing = [key for key, ok in zip(keys, present) if not ok]
        raise KeyError(f"Incomplete feature scaler payload in surrogate artifact: missing {missing}")

    names = [str(x) for x in np.asarray(npz[keys[0]], dtype=str).reshape(-1).tolist()]
    if expected_feature_names is not None:
        expected = _as_feature_names(expected_feature_names)
        if names != expected:
            raise ValueError(
                "Feature scaler names do not match surrogate feature_names: "
                f"scaler={names[:10]} expected={expected[:10]}"
            )

    return _validate_scaler_payload(
        feature_names=names,
        methods=np.asarray(npz[keys[1]], dtype=str),
        offsets=np.asarray(npz[keys[2]], dtype=float),
        denominators=np.asarray(npz[keys[3]], dtype=float),
    )
