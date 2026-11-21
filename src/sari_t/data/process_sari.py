#!/usr/bin/env python
"""Process a raw RKI SARI TSV into daily incidence for native model training."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SARI_AGE_BUCKETS = ("15-34", "35-59", "60-79", "80+")
SARI_AGE_REPEATS = (2, 5, 4, 4)


def process_sari(
    source_path: Path,
    output_path: Path,
    *,
    start_date: date = date(2014, 9, 29),
) -> Path:
    """Write daily source-unit SARI/COVID arrays with shape (15, day).

    Sort by actual ISO calendar weeks. Zero-pad COVID only before each required
    series' first observation; missing later observations raise. Interpolate at
    day 3 of each week with PCHIP, clip negative daily values, then select dates.
    The effect model owns the /100 scale; do not scale these prepared values.
    """
    from scipy.interpolate import PchipInterpolator

    frame = pd.read_csv(source_path.expanduser(), sep="\t")
    frame = frame.loc[
        frame["SARI"].isin(("Gesamt", "COVID-19"))
        & frame["Altersgruppe"].isin(("00+", *SARI_AGE_BUCKETS))
    ].copy()
    if frame.duplicated(["Kalenderwoche", "Altersgruppe", "SARI"]).any():
        raise ValueError("SARI source contains duplicate week/age/category rows")
    frame["week_start"] = frame["Kalenderwoche"].map(_week_start)
    frame["Hospitalisierungsinzidenz"] = pd.to_numeric(frame["Hospitalisierungsinzidenz"], errors="raise")
    incidence = frame["Hospitalisierungsinzidenz"].to_numpy(dtype=float)
    if not np.isfinite(incidence).all() or (incidence < 0).any():
        raise ValueError("SARI source contains nonfinite or negative incidence")
    weeks = sorted(frame.loc[frame["SARI"].eq("Gesamt") & frame["Altersgruppe"].eq("00+"), "week_start"])
    if len(weeks) < 2 or any((current - previous).days != 7 for previous, current in zip(weeks, weeks[1:])):
        raise ValueError("SARI source requires at least two contiguous observed weeks")
    coverage_end = weeks[-1] + timedelta(days=6)
    if not weeks[0] <= start_date <= coverage_end:
        raise ValueError(f"SARI start date must be within {weeks[0]}–{coverage_end}")

    total_all = _weekly_incidence(frame, "Gesamt", "00+", weeks)
    covid_all = _weekly_incidence(frame, "COVID-19", "00+", weeks)
    if (total_all <= 0).any():
        raise ValueError("SARI total incidence for age 00+ must be positive")
    covid_ratio = covid_all / total_all
    anchor_days = np.arange(len(weeks)) * 7 + 3
    days = np.arange(len(weeks) * 7)
    start = (start_date - weeks[0]).days
    sari_daily = []
    covid_daily = []
    for age, repeats in zip(SARI_AGE_BUCKETS, SARI_AGE_REPEATS):
        total = _weekly_incidence(frame, "Gesamt", age, weeks)
        covid = (
            _weekly_incidence(frame, "COVID-19", age, weeks)
            if age in ("60-79", "80+") else total * covid_ratio
        )
        sari = total - covid if age in ("60-79", "80+") else total * (1 - covid_ratio)
        for values, prepared in ((sari, sari_daily), (covid, covid_daily)):
            daily = PchipInterpolator(anchor_days, values)(days).clip(0)[start:]
            if not np.isfinite(daily).all():
                raise ValueError(f"SARI interpolation produced nonfinite values for age {age}")
            prepared.append(np.repeat(daily[np.newaxis, :], repeats, axis=0))
    arrays = {
        "sari": np.concatenate(sari_daily).astype(np.float32),
        "covid": np.concatenate(covid_daily).astype(np.float32),
    }
    return _save_archive(output_path, arrays)


def _week_start(value: str) -> date:
    # RKI calendar-week strings use YYYY-Www, independent of TSV row order.
    if not isinstance(value, str) or len(value) != 8 or value[4:6] != "-W":
        raise ValueError(f"Invalid SARI calendar week: {value}")
    return date.fromisocalendar(int(value[:4]), int(value[6:]), 1)


def _weekly_incidence(
    frame: pd.DataFrame, category: str, age: str, weeks: list[date],
) -> np.ndarray:
    series = frame.loc[frame["SARI"].eq(category) & frame["Altersgruppe"].eq(age)]
    series = series.set_index("week_start")["Hospitalisierungsinzidenz"]
    if series.empty:
        raise ValueError(f"SARI source has no {category} observations for age {age}")
    ordered = series.reindex(weeks)
    if category == "COVID-19":
        ordered.loc[ordered.index < series.index.min()] = 0.0
    if ordered.isna().any():
        raise ValueError(f"SARI source has missing {category} weeks for age {age}")
    return ordered.to_numpy(dtype=float)


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
    parser.add_argument("--input", type=Path, required=True, help="Raw SARI-Hospitalisierungsinzidenz TSV")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2014, 9, 29))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "sari" / "sari_20260709.npz")
    args = parser.parse_args()
    output = process_sari(args.input, args.output, start_date=args.start_date)
    print(f"Saved {output}; surveillance_start_date is {args.start_date.isoformat()}", flush=True)


if __name__ == "__main__":
    main()
