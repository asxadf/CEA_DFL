#!/usr/bin/env python3
"""
Convert CSV Datetime column values from:
    2025-01-09 00:00:00
to:
    2025-01-09T00:00:00

Usage:
    python3 src/utils/convert_datetime_column_to_ISO.py
    python3 src/utils/convert_datetime_column_to_ISO.py data/processed
    python3 src/utils/convert_datetime_column_to_ISO.py data/processed/outdoor_realization_cold.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile


INPUT_FORMAT = "%Y-%m-%d %H:%M:%S"
OUTPUT_FORMAT = "%Y-%m-%dT%H:%M:%S"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = (REPO_ROOT / "data" / "processed").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Datetime column in CSV file(s) to ISO-like format."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATH,
        help=(
            "Path to a CSV file or a directory containing CSV files "
            f"(default: {DEFAULT_PATH})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, files are updated in place.",
    )
    return parser.parse_args()


def find_datetime_column(fieldnames: list[str]) -> str:
    if "Datetime " in fieldnames:
        return "Datetime "
    if "Datetime" in fieldnames:
        return "Datetime"
    raise ValueError("Missing 'Datetime ' or 'Datetime' column.")


def convert_cell(value: str) -> str:
    raw = value.strip()
    if not raw:
        return value
    try:
        dt = datetime.strptime(raw, INPUT_FORMAT)
    except ValueError:
        return value
    return dt.strftime(OUTPUT_FORMAT)


def convert_csv_file(input_file: Path, output_file: Path) -> None:
    with input_file.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError(f"No header found: {input_file}")
        fieldnames = list(reader.fieldnames)
        datetime_col = find_datetime_column(fieldnames)

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
                writer = csv.DictWriter(tmp, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    row[datetime_col] = convert_cell(row.get(datetime_col, ""))
                    writer.writerow(row)
            tmp_path.replace(input_file)
            return

    with input_file.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        datetime_col = find_datetime_column(fieldnames)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                row[datetime_col] = convert_cell(row.get(datetime_col, ""))
                writer.writerow(row)


def iter_csv_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Not a CSV file: {path}")
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.path).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None

    files = iter_csv_files(input_path)
    if not files:
        print(f"No CSV files found under: {input_path}")
        return

    for csv_file in files:
        out_file = csv_file if output_dir is None else output_dir / csv_file.name
        convert_csv_file(csv_file, out_file)
        print(f"Converted: {csv_file} -> {out_file}")


if __name__ == "__main__":
    main()
