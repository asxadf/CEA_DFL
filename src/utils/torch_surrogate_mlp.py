from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import torch
from torch import nn


def require_torch(context: str = "PyTorch surrogate") -> None:
    if torch is None or nn is None:
        raise ImportError(
            f"{context} requires PyTorch, but torch is not available in the active Python environment."
        )


def resolve_torch_device(preferred: str | None = None) -> str:
    require_torch()
    raw = "auto" if preferred is None else str(preferred).strip().lower()
    if raw in {"", "auto", "default"}:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if raw == "cpu":
        return "cpu"
    if raw == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested device='mps', but MPS is not available in this environment.")
        return "mps"
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device='cuda', but CUDA is not available in this environment.")
        return "cuda"
    if raw.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device={preferred!r}, but CUDA is not available in this environment.")
        try:
            index = int(raw.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device specifier: {preferred!r}") from exc
        if index < 0 or index >= int(torch.cuda.device_count()):
            raise ValueError(
                f"CUDA device index out of range for {preferred!r}; available device count={torch.cuda.device_count()}."
            )
        return f"cuda:{index}"
    raise ValueError(
        f"Unsupported torch device override: {preferred!r}. Expected one of 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N'."
    )


def get_default_torch_device() -> str:
    return resolve_torch_device("auto")


def _activation_module(name: str):
    require_torch()
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "softplus":
        return nn.Softplus()
    if key in ("identity", "linear", "none"):
        return nn.Identity()
    raise ValueError(f"Unsupported Torch MLP activation: {name!r}")


class TorchMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: str = "relu",
        output_activation: str = "linear",
    ) -> None:
        require_torch()
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        hidden = [int(h) for h in hidden_layer_sizes]
        if any(h < 1 for h in hidden):
            raise ValueError(f"All hidden layer sizes must be >= 1, got {hidden_layer_sizes}")

        layers: list[nn.Module] = []
        prev = int(input_dim)
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            act = _activation_module(activation)
            if not isinstance(act, nn.Identity):
                layers.append(act)
            prev = width
        layers.append(nn.Linear(prev, 1))
        out_act = _activation_module(output_activation)
        if not isinstance(out_act, nn.Identity):
            layers.append(out_act)
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_layer_sizes = tuple(hidden)
        self.activation = str(activation)
        self.output_activation = str(output_activation)

    def forward(self, x):
        y = self.net(x)
        return y.squeeze(-1)


def save_torch_mlp_checkpoint(
    path: str | Path,
    model: TorchMLP,
    *,
    input_dim: int,
    hidden_layer_sizes: Sequence[int],
    activation: str,
    output_activation: str,
    extra_metadata: dict | None = None,
) -> Path:
    require_torch()
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_dim": int(input_dim),
        "hidden_layer_sizes": tuple(int(h) for h in hidden_layer_sizes),
        "activation": str(activation),
        "output_activation": str(output_activation),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra_metadata:
        payload["extra_metadata"] = dict(extra_metadata)
    torch.save(payload, out_path)
    return out_path


def load_torch_mlp_checkpoint(
    path: str | Path,
    *,
    map_location: str = "cpu",
):
    require_torch()
    ckpt_path = Path(path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Torch surrogate checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=map_location)
    input_dim = int(payload["input_dim"])
    hidden_layer_sizes = tuple(int(h) for h in payload["hidden_layer_sizes"])
    activation = str(payload["activation"])
    output_activation = str(payload.get("output_activation", "linear"))
    model = TorchMLP(
        input_dim=input_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        output_activation=output_activation,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "model": model,
        "input_dim": input_dim,
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "output_activation": output_activation,
        "extra_metadata": dict(payload.get("extra_metadata", {})),
        "checkpoint_path": str(ckpt_path),
    }


def predict_torch_mlp(
    model: TorchMLP,
    x: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    require_torch()
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    model = model.to(device)
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_arr, dtype=torch.float32, device=device)
        y = model(x_tensor).detach().cpu().numpy().reshape(-1)
    return np.asarray(y, dtype=float)
