"""Read dated stage-1 labels from pipeline mortality NPZ archives."""

from __future__ import annotations

from datetime import date

import numpy as np
import torch

from .config import DataConfig
from .interfaces import MortalityLabels


DAILY_POSITIONS_PER_YEAR = 366
WEEKLY_POSITIONS_PER_YEAR = 53


def load_mortality(
    config: DataConfig,
    *,
    start_year: int | None = None,
) -> MortalityLabels:
    """Remove source -1 slots using the original stage-1 transformation.

    Annual arrays reserve 366 daily and 53 weekly slots, starting in 2000 by
    default. The missing marker identifies unused/unpublished positions; do not
    assume that padding always occupies the final slot of an annual block.
    start_year optionally skips complete stored annual blocks before filtering.
    Source series must be contiguous after removing these positions, as assumed
    by the original reader. Exact prediction-period selection remains separate.
    """
    selected_year = config.mortality_first_year if start_year is None else start_year
    year_offset = selected_year - config.mortality_first_year
    if year_offset < 0:
        raise ValueError("Selected mortality start year precedes the first stored year")
    day_offset = year_offset * DAILY_POSITIONS_PER_YEAR
    week_offset = year_offset * WEEKLY_POSITIONS_PER_YEAR
    with np.load(config.data_root / config.mortality_path) as archive:
        daily = archive["de_daily"][0, :, day_offset:]
        men_de = archive["men_de_age_week"][0, 1:, week_offset:]
        women_de = archive["women_de_age_week"][0, 1:, week_offset:]
        men_land = archive["men_land_age_week"][:, 1:, week_offset:]
        women_land = archive["women_land_age_week"][:, 1:, week_offset:]

    missing = config.mortality_missing_value
    daily_land = _remove_missing(daily, missing).T
    weekly_de = np.stack(
        (_remove_missing(men_de, missing).T, _remove_missing(women_de, missing).T),
        axis=-1,
    )
    weekly_land = np.stack(
        (
            _remove_missing(men_land, missing).transpose(2, 0, 1),
            _remove_missing(women_land, missing).transpose(2, 0, 1),
        ),
        axis=-1,
    )
    return MortalityLabels(
        daily_land=torch.from_numpy(daily_land.astype(np.float32)),
        weekly_de_age_sex=torch.from_numpy(weekly_de.astype(np.float32)),
        weekly_land_coarse_age_sex=torch.from_numpy(weekly_land.astype(np.float32)),
        daily_start_date=date(selected_year, 1, 1),
        weekly_start_date=date.fromisocalendar(selected_year, 1, 1),
    )


def _remove_missing(values: np.ndarray, missing_value: int) -> np.ndarray:
    """Preserve source series axes while removing the marked temporal slots."""
    return values[values != missing_value].reshape(*values.shape[:-1], -1)
