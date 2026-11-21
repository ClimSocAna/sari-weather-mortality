"""Shared run preparation before training or evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import torch

from ..baseline import BaselineConfig, calculate_baseline_death_rate
from ..data.interfaces import PreparedData, SourceData
from ..data.preparation import load_sources, prepare_data
from ..models.config import PredictionPeriod
from .config import RunConfig

if TYPE_CHECKING:
    from torch import Tensor


DAYS_PER_WEEK = 7


@dataclass
class PreparedRun:
    """Prepared data and baseline state, without models or prediction caches.

    prepare_run produces CPU tensors; training initialization can place them on
    another device in a new container.

    Baseline rates cover data.window with axes (day, 1, age, sex); fitting can
    replace them after an explicit baseline update. Labels and covariates remain
    on data, and weather_variables follows the selected model channel order.
    A split rounded to zero validation weeks has validation_period=None.
    """

    config: RunConfig
    data: PreparedData
    baseline_rate: Tensor
    weather_variables: tuple[str, ...]
    training_period: PredictionPeriod
    validation_period: PredictionPeriod | None

    def to(self, device: torch.device | str) -> PreparedRun:
        """Return a new run container with all tensors on device as float32.

        Dates, configuration and channel order are preserved. Tensor storage may
        be shared when the device/dtype already match, as with Tensor.to.
        """
        return PreparedRun(
            config=self.config,
            data=self.data.to(device),
            baseline_rate=self.baseline_rate.to(device=device, dtype=torch.float32),
            weather_variables=self.weather_variables,
            training_period=self.training_period,
            validation_period=self.validation_period,
        )


def prepare_run(
    config: RunConfig,
    *,
    sources: SourceData | None = None,
) -> PreparedRun:
    """Prepare a full CPU period and chronological split without model execution.

    Optional sources must correspond to config.data_config and selected_weather in the
    specified channel order. Loading otherwise retains the available mortality
    archive for baseline history. No alignment, augmentation or adaptation runs.
    """
    if sources is None:
        sources = load_sources(config.data_config, config.selected_weather)
    data = prepare_data(sources, config.period, config.model_config)
    full_baseline, baseline_start = initialize_baseline(sources, config.initial_baseline_config)
    day_count = (config.period.end_date - config.period.start_date).days + 1
    baseline_offset = (config.period.start_date - baseline_start).days
    baseline_rate = full_baseline[baseline_offset:baseline_offset + day_count].clone()

    total_weeks = day_count // DAYS_PER_WEEK
    validation_weeks = int(total_weeks * config.training_config.validation_fraction)
    training_weeks = total_weeks - validation_weeks
    validation_start = config.period.start_date + timedelta(
        days=training_weeks * DAYS_PER_WEEK,
    )
    return PreparedRun(
        config=config,
        data=data,
        baseline_rate=baseline_rate,
        weather_variables=sources.weather_variables,
        training_period=PredictionPeriod(
            start_date=config.period.start_date,
            end_date=validation_start - timedelta(days=1),
        ),
        validation_period=(
            PredictionPeriod(validation_start, config.period.end_date)
            if validation_weeks else None
        ),
    )


def initialize_baseline(
    sources: SourceData,
    config: BaselineConfig,
) -> tuple[Tensor, date]:
    """Return initial daily rates and their origin using shared source history.

    Calculate before selecting prediction dates, as in the original training
    initialization. Use complete national mortality weeks covered by population,
    aligned from each source's actual origin. Weather and surveillance coverage
    do not limit this history. Sources must provide sufficient shared weeks for
    the selected baseline method. Returned axes are (day, 1, age group, sex).
    """
    weekly_start = sources.labels.weekly_start_date
    population_end = sources.population_start_date + timedelta(
        days=sources.population.shape[0],
    )
    first_week = max(
        0,
        ((sources.population_start_date - weekly_start).days + DAYS_PER_WEEK - 1)
        // DAYS_PER_WEEK,
    )
    stop_week = min(
        sources.labels.weekly_de_age_sex.shape[0],
        (population_end - weekly_start).days // DAYS_PER_WEEK,
    )
    baseline_start = weekly_start + timedelta(days=first_week * DAYS_PER_WEEK)
    population_start = (baseline_start - sources.population_start_date).days
    population_stop = population_start + (stop_week - first_week) * DAYS_PER_WEEK
    national_population = sources.population[population_start:population_stop].sum(dim=1)
    weekly_deaths = sources.labels.weekly_de_age_sex[first_week:stop_week]
    return (
        calculate_baseline_death_rate(weekly_deaths, national_population, config),
        baseline_start,
    )
