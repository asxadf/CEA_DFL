#!/usr/bin/env python3
"""
Enrich outdoor weather CSVs with relative humidity (%) and CO2 (ppm) columns.

Usage:
    python3 src/utils/add_relative_humidity_and_co2_ppm_columns.py
    python3 src/utils/add_relative_humidity_and_co2_ppm_columns.py data/processed/s1_winter.csv
    python3 src/utils/add_relative_humidity_and_co2_ppm_columns.py \
        data/processed/s1_winter.csv data/processed/s2_summer.csv
    python3 src/utils/add_relative_humidity_and_co2_ppm_columns.py --output-dir data/processed_enriched
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from tempfile import NamedTemporaryFile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILES = [
    (REPO_ROOT / "data" / "processed" / "s1_winter.csv").resolve(),
    (REPO_ROOT / "data" / "processed" / "s2_summer.csv").resolve(),
]

TEMP_COL = "T_Outdoor(C)"
ABS_HUMIDITY_COL = "H_Outdoor(g/m3)"
RH_COL = "RH_Outdoor(%)"
CO2_GM3_COL = "CO2_Outdoor(g/m3)"
CO2_PPM_COL = "CO2_Outdoor(ppm)"

P_ATM_PA = 101_325.0
R_UNIV = 8.314462618
R_V = 461.5
M_CO2 = 0.04401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add RH_Outdoor(%) and CO2_Outdoor(ppm) columns to outdoor CSV files."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUT_FILES,
        help=(
            "CSV files to enrich. Defaults to "
            "`data/processed/s1_winter.csv` and `data/processed/s2_summer.csv`."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, files are updated in place.",
    )
    return parser.parse_args()


def _saturation_vapor_pressure_pa(temp_c: float) -> float:
    return 610.94 * math.exp((17.625 * temp_c) / (temp_c + 243.04))


def absolute_humidity_gm3_to_relative_humidity_percent(abs_humidity_gm3: float, temp_c: float) -> float:
    temp_k = temp_c + 273.15
    rho_v = abs_humidity_gm3 / 1000.0
    partial_pressure_pa = rho_v * R_V * temp_k
    saturation_pressure_pa = _saturation_vapor_pressure_pa(temp_c)
    if saturation_pressure_pa <= 0.0:
        return 0.0
    rh = 100.0 * partial_pressure_pa / saturation_pressure_pa
    return min(max(rh, 0.0), 100.0)


def co2_gm3_to_ppm(co2_gm3: float, temp_c: float, p_pa: float = P_ATM_PA) -> float:
    temp_k = temp_c + 273.15
    rho_co2 = co2_gm3 / 1000.0
    concentration_mol_m3 = rho_co2 / M_CO2
    mole_fraction = concentration_mol_m3 * R_UNIV * temp_k / p_pa
    return max(mole_fraction * 1e6, 0.0)


def _build_output_fieldnames(fieldnames: list[str]) -> list[str]:
    required = {TEMP_COL, ABS_HUMIDITY_COL, CO2_GM3_COL}
    missing = [col for col in required if col not in fieldnames]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output_fieldnames: list[str] = []
    for fieldname in fieldnames:
        if fieldname in {RH_COL, CO2_PPM_COL}:
            continue
        output_fieldnames.append(fieldname)
        if fieldname == ABS_HUMIDITY_COL:
            output_fieldnames.append(RH_COL)
        if fieldname == CO2_GM3_COL:
            output_fieldnames.append(CO2_PPM_COL)
    return output_fieldnames


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def _enrich_row(row: dict[str, str]) -> dict[str, str]:
    temp_c = float(row[TEMP_COL])
    abs_humidity_gm3 = float(row[ABS_HUMIDITY_COL])
    co2_gm3 = float(row[CO2_GM3_COL])

    row[RH_COL] = _format_float(
        absolute_humidity_gm3_to_relative_humidity_percent(abs_humidity_gm3, temp_c)
    )
    row[CO2_PPM_COL] = _format_float(co2_gm3_to_ppm(co2_gm3, temp_c))
    return row


def enrich_csv_file(input_file: Path, output_file: Path) -> None:
    with input_file.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError(f"No header found: {input_file}")
        output_fieldnames = _build_output_fieldnames(list(reader.fieldnames))

        if output_file == input_file:
            with NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8",
                delete=False,
                dir=str(input_file.parent),
                prefix=f"{input_file.name}.tmp.",
            ) as tmp:
                tmp_path = Path(tmp.name)
                writer = csv.DictWriter(tmp, fieldnames=output_fieldnames)
                writer.writeheader()
                for row in reader:
                    writer.writerow(_enrich_row(row))
            tmp_path.replace(input_file)
            return

    with input_file.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=output_fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow(_enrich_row(row))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None

    for csv_path in args.csv_files:
        input_file = csv_path.expanduser().resolve()
        output_file = input_file if output_dir is None else output_dir / input_file.name
        enrich_csv_file(input_file, output_file)
        print(f"Enriched: {input_file} -> {output_file}")


if __name__ == "__main__":
    main()
