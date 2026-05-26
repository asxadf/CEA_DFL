from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


DEFAULT_PARAMS_PATH = (Path(__file__).resolve().parents[2] / "configs/var_and_param_keeper.yaml").resolve()
_U_IDX_PREFIX = "u_idx_"
_X_IDX_PREFIX = "x_idx_"
_D_IDX_PREFIX = "d_idx_"


@lru_cache(maxsize=1)
def _load_default_params() -> dict[str, Any]:
    text = DEFAULT_PARAMS_PATH.read_text(encoding="utf-8")
    param_keeper: dict[str, Any] = {}
    in_parameters = False

    for raw_line in text.splitlines():
        line_no_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_no_comment.strip():
            continue

        if not in_parameters:
            if line_no_comment.strip() == "parameters:":
                in_parameters = True
            continue

        if raw_line and not raw_line.startswith("  "):
            break

        stripped = line_no_comment.strip()
        if ":" not in stripped:
            continue

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not raw_value:
            continue

        if key in {"num_u", "num_x", "num_d"} or key.startswith(_U_IDX_PREFIX) or key.startswith(_X_IDX_PREFIX) or key.startswith(_D_IDX_PREFIX):
            param_keeper[key] = int(float(raw_value))

    if "num_u" not in param_keeper:
        raise ValueError(f"Missing 'num_u' in parameters section of {DEFAULT_PARAMS_PATH}")
    if "num_x" not in param_keeper:
        raise ValueError(f"Missing 'num_x' in parameters section of {DEFAULT_PARAMS_PATH}")
    if "num_d" not in param_keeper:
        raise ValueError(f"Missing 'num_d' in parameters section of {DEFAULT_PARAMS_PATH}")
    if not any(key.startswith(_U_IDX_PREFIX) for key in param_keeper):
        raise ValueError(f"Missing '{_U_IDX_PREFIX}*' entries in parameters section of {DEFAULT_PARAMS_PATH}")
    if not any(key.startswith(_X_IDX_PREFIX) for key in param_keeper):
        raise ValueError(f"Missing '{_X_IDX_PREFIX}*' entries in parameters section of {DEFAULT_PARAMS_PATH}")
    if not any(key.startswith(_D_IDX_PREFIX) for key in param_keeper):
        raise ValueError(f"Missing '{_D_IDX_PREFIX}*' entries in parameters section of {DEFAULT_PARAMS_PATH}")

    return param_keeper


def _resolve_params(params: Mapping[str, object] | None = None) -> dict[str, Any]:
    resolved = dict(_load_default_params())
    if params is None:
        return resolved

    params_mapping: Mapping[str, object] = params
    nested = params_mapping.get("parameters")
    if isinstance(nested, Mapping):
        params_mapping = nested

    resolved.update(dict(params_mapping))
    return resolved


def _extract_indices(
    params: Mapping[str, object],
    *,
    idx_prefix: str,
    name_prefix: str = "",
) -> dict[str, int]:
    idx = {
        f"{name_prefix}{key[len(idx_prefix):]}": int(value)
        for key, value in params.items()
        if key.startswith(idx_prefix)
    }
    if not idx:
        raise ValueError(f"No '{idx_prefix}*' entries found in vector-order parameters")
    return idx


def get_num_controls(params: Mapping[str, object] | None = None) -> int:
    resolved = _resolve_params(params)
    return int(resolved["num_u"])


def get_num_states(params: Mapping[str, object] | None = None) -> int:
    resolved = _resolve_params(params)
    return int(resolved["num_x"])


def get_num_disturbances(params: Mapping[str, object] | None = None) -> int:
    resolved = _resolve_params(params)
    return int(resolved["num_d"])


def get_control_indices(params: Mapping[str, object] | None = None) -> dict[str, int]:
    resolved = _resolve_params(params)
    idx = _extract_indices(resolved, idx_prefix=_U_IDX_PREFIX, name_prefix="U_")

    num_u = get_num_controls(resolved)
    values = list(idx.values())
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate control indices detected: {idx}")
    if min(values) < 0:
        raise ValueError(f"Negative control index detected: {idx}")
    if max(values) >= num_u:
        raise ValueError(f"Control index out of range for num_u={num_u}: {idx}")
    return idx


def get_state_indices(params: Mapping[str, object] | None = None) -> dict[str, int]:
    resolved = _resolve_params(params)
    idx = _extract_indices(resolved, idx_prefix=_X_IDX_PREFIX)

    num_x = get_num_states(resolved)
    values = list(idx.values())
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate state indices detected: {idx}")
    if min(values) < 0:
        raise ValueError(f"Negative state index detected: {idx}")
    if max(values) >= num_x:
        raise ValueError(f"State index out of range for num_x={num_x}: {idx}")
    return idx


def get_disturbance_indices(params: Mapping[str, object] | None = None) -> dict[str, int]:
    resolved = _resolve_params(params)
    idx = _extract_indices(resolved, idx_prefix=_D_IDX_PREFIX)

    num_d = get_num_disturbances(resolved)
    values = list(idx.values())
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate disturbance indices detected: {idx}")
    if min(values) < 0:
        raise ValueError(f"Negative disturbance index detected: {idx}")
    if max(values) >= num_d:
        raise ValueError(f"Disturbance index out of range for num_d={num_d}: {idx}")
    return idx


def _names_in_order(idx: Mapping[str, int], size: int, vector_label: str) -> list[str]:
    names: list[str | None] = [None] * size
    for name, position in idx.items():
        if names[position] is not None:
            raise ValueError(f"Duplicate {vector_label} position {position} for {name} and {names[position]}")
        names[position] = name

    missing = [i for i, name in enumerate(names) if name is None]
    if missing:
        raise ValueError(f"Missing {vector_label} names for positions {missing}")
    return [str(name) for name in names]


def get_control_names_in_order(params: Mapping[str, object] | None = None) -> list[str]:
    resolved = _resolve_params(params)
    idx = get_control_indices(resolved)
    num_u = get_num_controls(resolved)
    return _names_in_order(idx, num_u, "control")


def get_state_names_in_order(params: Mapping[str, object] | None = None) -> list[str]:
    resolved = _resolve_params(params)
    idx = get_state_indices(resolved)
    num_x = get_num_states(resolved)
    return _names_in_order(idx, num_x, "state")


def get_disturbance_names_in_order(params: Mapping[str, object] | None = None) -> list[str]:
    resolved = _resolve_params(params)
    idx = get_disturbance_indices(resolved)
    num_d = get_num_disturbances(resolved)
    return _names_in_order(idx, num_d, "disturbance")
