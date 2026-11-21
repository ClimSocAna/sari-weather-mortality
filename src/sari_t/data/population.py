"""Read pipeline population tables and prepare dated daily model populations."""

from __future__ import annotations

from calendar import isleap
from datetime import date

import numpy as np
import pandas as pd
import torch

from ..models.grouping import KREIS_COUNT
from .config import DataConfig
from .interfaces import PopulationData


SOURCE_AGE_GROUP_COUNT = 20
# The first six five-year source bands cover ages 0–29.
UNDER_30_SOURCE_GROUP_COUNT = 6


def load_population(config: DataConfig) -> PopulationData:
    """Return daily population from the configured start year through a fixed tail.

    The normalized pipeline CSV supplies ordered source ages within each Kreis.
    Preserve that age order while ordering years and district codes. Male/female
    values and interpolation use float32, matching the model's numeric policy.
    """
    frame = pd.read_csv(
        config.data_root / config.population_path,
        dtype={"krs_code": str},
    )
    frame = frame.sort_values(["year", "krs_code"], kind="mergesort")
    annual = _annual_population(frame)
    daily = _interpolate_population(annual, config)
    return PopulationData(
        values=torch.from_numpy(daily),
        start_date=date(config.population_start_year, 1, 1),
    )


def _annual_population(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    annual = {}
    for year, yearly in frame.groupby("year", sort=True):
        values = yearly[["männlich", "weiblich"]].to_numpy(dtype=np.float32)
        values = values.reshape(KREIS_COUNT, SOURCE_AGE_GROUP_COUNT, 2)
        annual[int(year)] = np.concatenate(
            (
                values[:, :UNDER_30_SOURCE_GROUP_COUNT].sum(axis=1, keepdims=True),
                values[:, UNDER_30_SOURCE_GROUP_COUNT:],
            ),
            axis=1,
        )
    return annual


def _interpolate_population(
    annual: dict[int, np.ndarray],
    config: DataConfig,
) -> np.ndarray:
    first_year = min(annual)
    last_year = max(annual)
    daily = []
    for year in range(config.population_start_year, last_year + 1):
        days = 365 + isleap(year)
        if config.population_interpolation and year > first_year:
            values = np.linspace(
                annual[year - 1], annual[year], days, axis=0, dtype=np.float32,
            )
        else:
            # No previous-year anchor exists for the first stored population year.
            values = np.repeat(annual[year][None], days, axis=0)
        daily.append(values)
    tail_offset = max(
        0,
        (date(config.population_start_year, 1, 1) - date(last_year + 1, 1, 1)).days,
    )
    tail_days = max(0, config.population_tail_padding_days - tail_offset)
    daily.append(np.repeat(annual[last_year][None], tail_days, axis=0))
    return np.concatenate(daily, axis=0)
