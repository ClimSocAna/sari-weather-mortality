import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config
from .config import DEATH_DATA_START_DAY, DEATH_DATA_START_WEEK
from .grouping import age_grouping

POPULATION_SHAPE = (-1, 400, 20, 2)
AGE_GROUPS_15 = (tuple(range(6)), *((age_index,) for age_index in range(6, 20)))
POPULATION_START_YEAR = 2011
POPULATION_TAIL_PADDING_DAYS = 1000


def _assert_file_exists(path: str | Path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required data file not found: {path}")


def _load_numpy_tensor(path: str | Path) -> torch.Tensor:
    """Load numpy array and convert to torch tensor."""
    _assert_file_exists(path)
    try:
        return torch.from_numpy(np.load(path))
    except Exception as exc:
        raise ValueError(f"Failed to load numpy data from {path}") from exc


def _interpolate_population(
    arr_pop: np.ndarray,
    interpolation: bool,
    end_year: int | None = None,
) -> np.ndarray:
    """Interpolate population data across years (daily values)."""
    if end_year is None:
        end_year = POPULATION_START_YEAR + arr_pop.shape[0] - 1

    pop_interpolation = [arr_pop[0]] * 365
    for year in range(POPULATION_START_YEAR + 1, end_year + 1):
        days = 365 + calendar.isleap(year)
        if interpolation:
            daily_interpolation = np.linspace(
                arr_pop[year - POPULATION_START_YEAR - 1],
                arr_pop[year - POPULATION_START_YEAR],
                days,
                axis=0,
            )
            pop_interpolation.extend(daily_interpolation)
        else:
            pop_interpolation.extend([arr_pop[year - POPULATION_START_YEAR]] * days)
    pop_interpolation.extend([arr_pop[-1]] * POPULATION_TAIL_PADDING_DAYS)
    return np.stack(pop_interpolation, axis=0)


def _load_population_tensor(interpolation: bool) -> torch.Tensor:
    """Load population CSV and return grouped daily population tensor."""
    population_csv_path = config.POPULATION_DIR / "population_2011-2024.csv"
    _assert_file_exists(population_csv_path)
    try:
        df_pop = pd.read_csv(population_csv_path)
    except Exception as exc:
        raise ValueError(f"Failed to load population CSV from {population_csv_path}") from exc

    if df_pop.shape[1] < 6:
        raise ValueError("Population CSV has unexpected format: expected at least 6 columns")

    arr_pop = df_pop.iloc[:, 5:].to_numpy().reshape(*POPULATION_SHAPE)
    arr_pop = _interpolate_population(arr_pop, interpolation)
    return age_grouping(torch.from_numpy(arr_pop), 2, AGE_GROUPS_15)


def _load_stage1_deaths() -> list[torch.Tensor]:
    """Load and preprocess stage-1 death labels."""
    death_daily = _load_numpy_tensor(config.DEATH_DIR / "de_daily.npy")
    death_de_men = _load_numpy_tensor(config.DEATH_DIR / "men_de_age_week.npy")
    death_de_women = _load_numpy_tensor(config.DEATH_DIR / "women_de_age_week.npy")
    death_land_men = _load_numpy_tensor(config.DEATH_DIR / "men_land_age_week.npy")
    death_land_women = _load_numpy_tensor(config.DEATH_DIR / "women_land_age_week.npy")

    death_daily = death_daily[0, :, DEATH_DATA_START_DAY:]
    death_daily = death_daily.masked_select(death_daily != -1).reshape(16, -1).swapaxes(0, 1)
    death_de_men = death_de_men[0, 1:, DEATH_DATA_START_WEEK:]
    death_de_women = death_de_women[0, 1:, DEATH_DATA_START_WEEK:]
    death_de_men = death_de_men.masked_select(death_de_men != -1).reshape(15, -1)
    death_de_women = death_de_women.masked_select(death_de_women != -1).reshape(15, -1)
    death_de = torch.stack([death_de_men, death_de_women], dim=-1).swapaxes(0, 1)

    death_land_men = death_land_men[:, 1:, DEATH_DATA_START_WEEK:]
    death_land_women = death_land_women[:, 1:, DEATH_DATA_START_WEEK:]
    death_land_men = death_land_men.masked_select(death_land_men != -1).reshape(16, 4, -1)
    death_land_women = death_land_women.masked_select(death_land_women != -1).reshape(16, 4, -1)
    death_land = torch.stack([death_land_men, death_land_women], dim=-1).moveaxis(2, 0)

    if death_daily.ndim != 2:
        raise ValueError(f"Unexpected daily death shape: {tuple(death_daily.shape)}")
    if death_de.ndim != 3:
        raise ValueError(f"Unexpected DE weekly death shape: {tuple(death_de.shape)}")
    if death_land.ndim != 4:
        raise ValueError(f"Unexpected land weekly death shape: {tuple(death_land.shape)}")

    return [death_daily, death_de, death_land]


def _load_stage2_landkreis_deaths() -> list[torch.Tensor]:
    """Load stage-2 Landkreis labels from config.LANDKREIS_DATA_PATH.

    Expected to return a single-item list:
    [daily_kreis_age_sex].
    The final mapping from .pt content to model label tensors is not implemented yet.
    """
    if config.LANDKREIS_DATA_PATH is None:
        raise ValueError("LANDKREIS_DATA_PATH must be set in model.config for stage 2")
    raise NotImplementedError(
        "Stage 2 Landkreis label loading is not implemented yet. "
        "Expected source: model.config.LANDKREIS_DATA_PATH (.pt file)."
    )


def population_preprocessing(interpolation: bool = True, stage: int = 1) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Load and preprocess population and death statistics data.

    Args:
        interpolation: Whether to interpolate population values across days

    Returns:
        Tuple of (population_tensor, death_statistics_list) where:
        - population_tensor: Shape (days, age_groups, sex)
        - death_statistics_list (stage 1): [daily_land, weekly_de_age_sex, weekly_land_coarse_age_sex]
        - death_statistics_list (stage 2): [daily_kreis_age_sex]
    """
    population = _load_population_tensor(interpolation)

    if stage == 1:
        death_statistic = _load_stage1_deaths()
    elif stage == 2:
        death_statistic = _load_stage2_landkreis_deaths()
    else:
        raise ValueError(f"Unsupported training stage: {stage}")

    return population, death_statistic
