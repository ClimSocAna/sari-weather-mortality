#!/usr/bin/env python
"""Process old/new raw Destatis workbooks into the native mortality NPZ."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MORTALITY_ARRAY_NAMES = (
    "de_daily", "men_de_age_week", "women_de_age_week",
    "men_land_age_week", "women_land_age_week",
)
MISSING_VALUES = ("X", "X ", "...", ".")


def process_deaths(
    old_workbook: Path,
    new_workbook: Path,
    output_path: Path,
    *,
    last_year: int = 2026,
) -> Path:
    """Combine 2000–2020 and 2021–last_year sources into the five-array NPZ.

    Each workbook contains descending annual row blocks, as in the supplied
    original script. Preserve 366 daily/53 weekly slots per year and -1 markers.
    Never delete raw workbooks or write temporary annual NPY exports.
    """
    old = _read_deaths_workbook(old_workbook, 2000, 2020, old_format=True)
    new = _read_deaths_workbook(new_workbook, 2021, last_year, old_format=False)
    arrays = {}
    for name in MORTALITY_ARRAY_NAMES:
        blocks = {**old[name], **new[name]}
        annual = [blocks[year] for year in range(2000, last_year + 1)]
        if name in ("de_daily", "men_de_age_week", "women_de_age_week"):
            annual = [values[np.newaxis, ...] for values in annual]
        arrays[name] = np.concatenate(annual, axis=-1)
    return _save_archive(output_path, arrays)


def _read_deaths_workbook(
    path: Path, first_year: int, last_year: int, *, old_format: bool,
) -> dict[str, dict[int, np.ndarray]]:
    if last_year < first_year:
        raise ValueError("Mortality workbook's last year precedes its first year")
    # name, sheet, rows/year, first value column, final column, temporal slots
    sheets = (
        (
            "de_daily", "12613-08" if old_format else "12613-07", 16,
            2 if old_format else 3, -1 if old_format else None, 366,
        ),
        ("men_de_age_week", "12613-03" if old_format else "12613-02", 16, 2, None, 53),
        ("women_de_age_week", "12613-04" if old_format else "12613-03", 16, 2, None, 53),
        ("men_land_age_week", "12613-10" if old_format else "12613-09", 80, 3, None, 53),
        ("women_land_age_week", "12613-11" if old_format else "12613-10", 80, 3, None, 53),
    )
    arrays = {}
    with pd.ExcelFile(path.expanduser(), engine="openpyxl") as workbook:
        for name, sheet, rows, first, stop, slots in sheets:
            frame = pd.read_excel(workbook, sheet_name=sheet, skiprows=3)
            required_rows = (last_year - first_year + 1) * rows
            if frame.shape[0] < required_rows:
                raise ValueError(f"Sheet {sheet} has fewer rows than required for {first_year}–{last_year}")
            blocks = {}
            for index, year in enumerate(range(last_year, first_year - 1, -1)):
                source_year = pd.to_numeric(frame.iloc[index * rows, 0], errors="coerce")
                if pd.notna(source_year) and source_year != year:
                    raise ValueError(f"Sheet {sheet}: expected year block {year}, found {source_year}")
                block = frame.iloc[index * rows:(index + 1) * rows, first:stop]
                if block.shape != (rows, slots):
                    raise ValueError(f"Sheet {sheet}, year {year}: expected {(rows, slots)}, got {block.shape}")
                values = block.replace(list(MISSING_VALUES), np.nan).apply(pd.to_numeric, errors="raise")
                values = values.fillna(-1).to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values < -1).any() or (values != np.floor(values)).any():
                    raise ValueError(f"Sheet {sheet}, year {year} contains invalid death counts")
                values = values.astype(np.int64)
                if rows == 80:
                    values = values.reshape(16, 5, slots)
                    # Preserve the original Bremen/Saarland age-band imputation.
                    missing = (values[:, 0] != -1) & (values[:, 1] == -1)
                    remainder = values[:, 0] - values[:, 3] - values[:, 4]
                    values[:, 1][missing] = (remainder[missing] * 0.43).astype(np.int64)
                    values[:, 2][missing] = (remainder[missing] * 0.57).astype(np.int64)
                blocks[year] = values
            arrays[name] = blocks
    return arrays


def _save_archive(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    path = path.expanduser().resolve()
    if path.suffix != ".npz":
        raise ValueError("Prepared data output must have the .npz extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            np.savez_compressed(output, **arrays)
            output.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--old-workbook", type=Path, required=True, help="Workbook covering 2000–2020")
    parser.add_argument("--new-workbook", type=Path, required=True, help="Workbook covering 2021–last-year")
    parser.add_argument("--last-year", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "death" / "mortality_20260714.npz")
    args = parser.parse_args()
    output = process_deaths(args.old_workbook, args.new_workbook, args.output, last_year=args.last_year)
    print(f"Saved {output}; mortality_first_year is 2000", flush=True)


if __name__ == "__main__":
    main()
