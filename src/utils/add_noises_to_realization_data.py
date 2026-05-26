"""
Smoke test and analyze one-step digital twin behavior.

CLI:
python3 src/utils/add_noises_to_realization_data.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    (REPO_ROOT / "data" / "processed" / "testing_outdoor_realization_warm_robust_2022.csv").resolve(),
    (REPO_ROOT / "data" / "processed" / "testing_outdoor_realization_warm_robust_2023.csv").resolve(),
)


@dataclass(frozen=True)
class NoiseSpec:
    col: str
    low: float
    high: float
    clamp_nonneg: bool
    decimals: int | None  # None => integer output


SPECS: tuple[NoiseSpec, ...] = (
    NoiseSpec("T_Outdoor(C)", low=-0.10, high=0.10, clamp_nonneg=False, decimals=2),
    NoiseSpec("H_Outdoor(g/m3)", low=-0.05, high=0.05, clamp_nonneg=True, decimals=2),
    NoiseSpec("CO2_Outdoor(g/m3)", low=-0.05, high=0.05, clamp_nonneg=True, decimals=2),
    NoiseSpec("Radiation_Outdoor(w/m2)", low=-0.10, high=0.10, clamp_nonneg=True, decimals=None),
)


def add_multiplicative_noise(
    df: pd.DataFrame,
    specs: Iterable[NoiseSpec],
    *,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Apply per-row multiplicative noise: new = old * (1 + u), u ~ Uniform(low, high)."""
    rng = np.random.default_rng(seed)
    out = df.copy()

    for spec in specs:
        if spec.col not in out.columns:
            raise KeyError(f"Missing required column: {spec.col}")

        s = pd.to_numeric(out[spec.col], errors="coerce")
        noise = rng.uniform(spec.low, spec.high, size=len(out))
        s_noisy = s * (1.0 + noise)

        if spec.clamp_nonneg:
            s_noisy = s_noisy.clip(lower=0)

        if spec.decimals is None:
            out[spec.col] = np.rint(s_noisy).astype("Int64").fillna(0).astype(int)
        else:
            out[spec.col] = s_noisy.round(spec.decimals)

    return out


def process_file(
    path: Path,
    *,
    seed: int | None = 42,
    output_dir: Path | None = None,
    suffix: str = "_noisy",
) -> Path:
    df = pd.read_csv(path)
    df_noisy = add_multiplicative_noise(df, SPECS, seed=seed)
    target_dir = path.parent if output_dir is None else output_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = (target_dir / f"{path.stem}{suffix}{path.suffix}").resolve()
    df_noisy.to_csv(out_path, index=False)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add multiplicative noise to outdoor realization CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "CSV files to process. Default: repo data/processed/"
            "outdoor_realization_cold.csv and outdoor_realization_warm.csv."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible noise generation (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for generated noisy CSVs. Default: same directory as each input.",
    )
    parser.add_argument(
        "--suffix",
        default="_noisy",
        help="Suffix appended to each output filename before the extension (default: _noisy).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inputs = tuple(path.resolve() for path in args.inputs) if args.inputs else DEFAULT_INPUTS

    for p in inputs:
        out_path = process_file(
            p,
            seed=args.seed,
            output_dir=args.output_dir,
            suffix=args.suffix,
        )
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
