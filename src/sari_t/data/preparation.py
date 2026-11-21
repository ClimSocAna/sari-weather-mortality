"""Assemble explicitly selected source data for training and evaluation."""

from __future__ import annotations

from datetime import date, timedelta

from ..models.config import MortalityModelConfig, PredictionPeriod
from .config import DataConfig
from .interfaces import PreparedData, SourceData
from .mortality import load_mortality
from .population import load_population
from .surveillance import load_surveillance
from .training_inputs import prepare_model_inputs, select_mortality_labels
from .weather import load_weather


def load_sources(
    config: DataConfig,
    selected_weather: tuple[str, ...],
    *,
    mortality_start_year: int | None = None,
) -> SourceData:
    """Combine readers without shifting their independent source calendars.

    Weather is aligned by its reader in the explicitly requested channel order.
    Mortality can omit earlier annual source blocks. No prediction window,
    baseline, calibration, device transfer or model construction occurs here.
    """
    weather = load_weather(config, selected_weather)
    population = load_population(config)
    labels = load_mortality(config, start_year=mortality_start_year)
    surveillance = load_surveillance(config)
    return SourceData(
        weather=weather.values,
        weather_variables=weather.variables,
        weather_start_date=weather.start_date,
        population=population.values,
        population_start_date=population.start_date,
        sari=surveillance.sari,
        covid=surveillance.covid,
        surveillance_start_date=surveillance.start_date,
        labels=labels,
    )


def prepare_data(
    sources: SourceData,
    period: PredictionPeriod,
    model_config: MortalityModelConfig,
) -> PreparedData:
    """Align sources to the selected mortality period and its covariate histories.

    Population and labels cover prediction dates only. Weather and surveillance
    additionally retain their own leading kernel histories. Every source uses its
    actual date origin; the first included week is not a fixed indexing offset.
    """
    weather_start = period.start_date - timedelta(days=model_config.kernel_T_days - 1)
    surveillance_start = period.start_date - timedelta(days=model_config.kernel_sari_days - 1)
    for name, origin, day_count, requested_start in (
        ("weather", sources.weather_start_date, sources.weather.shape[-1], weather_start),
        ("SARI", sources.surveillance_start_date, sources.sari.shape[-1], surveillance_start),
        ("COVID", sources.surveillance_start_date, sources.covid.shape[-1], surveillance_start),
        ("population", sources.population_start_date, sources.population.shape[0], period.start_date),
        ("daily mortality", sources.labels.daily_start_date, sources.labels.daily_land.shape[0], period.start_date),
        ("national weekly mortality", sources.labels.weekly_start_date, sources.labels.weekly_de_age_sex.shape[0] * 7, period.start_date),
        ("state weekly mortality", sources.labels.weekly_start_date, sources.labels.weekly_land_coarse_age_sex.shape[0] * 7, period.start_date),
    ):
        _require_coverage(name, origin, day_count, requested_start, period.end_date)

    labels = select_mortality_labels(sources.labels, period)
    population_start = (period.start_date - sources.population_start_date).days
    population_days = (period.end_date - period.start_date).days + 1
    return PreparedData(
        window=period,
        model_inputs=prepare_model_inputs(sources, period, model_config),
        population=sources.population[population_start:population_start + population_days],
        labels=labels,
    )


def _require_coverage(
    name: str,
    source_start: date,
    day_count: int,
    requested_start: date,
    requested_end: date,
) -> None:
    source_end = source_start + timedelta(days=day_count - 1)
    if requested_start < source_start or requested_end > source_end:
        raise ValueError(
            f"{name} coverage {source_start}–{source_end} does not include "
            f"the requested range {requested_start}–{requested_end}"
        )
