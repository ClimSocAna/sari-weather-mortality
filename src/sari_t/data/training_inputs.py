"""CPU window selection shared by training and evaluation preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models.config import MortalityModelConfig, PredictionPeriod
from .interfaces import MortalityLabels, SourceData

if TYPE_CHECKING:
    from ..models.interfaces import MortalityModelInputs


def prepare_model_inputs(
    sources: SourceData,
    period: PredictionPeriod,
    model_config: MortalityModelConfig,
) -> MortalityModelInputs:
    """Select lagged covariates from covering sources without device transfer.

    Source tensors must cover the requested period and both kernel histories.
    Surveillance retains source incidence units; the effect model owns scaling.
    Channel order is already resolved in SourceData by reader/run preparation.
    """
    from ..models.interfaces import MortalityModelInputs

    t_start = (period.start_date - sources.weather_start_date).days + 1 - model_config.kernel_T_days
    t_end = (period.end_date - sources.weather_start_date).days + 1
    sari_start = (period.start_date - sources.surveillance_start_date).days + 1 - model_config.kernel_sari_days
    sari_end = (period.end_date - sources.surveillance_start_date).days + 1
    return MortalityModelInputs(
        temperature=sources.weather[..., t_start:t_end],
        sari=sources.sari[:, sari_start:sari_end],
        covid=sources.covid[:, sari_start:sari_end],
    )


def select_mortality_labels(
    labels: MortalityLabels,
    period: PredictionPeriod,
) -> MortalityLabels:
    """Select covering stage-1 labels for inclusive complete mortality weeks.

    The reader preserves each label stream's calendar and missing positions.
    Coverage and availability are checked by run/window preparation before this
    selection; slices must not be interpreted as repairing absent source data.
    """
    if period.start_date.weekday() != 0 or period.end_date.weekday() != 6:
        raise ValueError("Stage-1 labels require a Monday–Sunday prediction period")
    if period.end_date < period.start_date:
        raise ValueError("Prediction period end precedes its start")
    day_start = (period.start_date - labels.daily_start_date).days
    day_end = (period.end_date - labels.daily_start_date).days + 1
    week_start = (period.start_date - labels.weekly_start_date).days // 7
    week_count = ((period.end_date - period.start_date).days + 1) // 7
    return MortalityLabels(
        daily_land=labels.daily_land[day_start:day_end],
        weekly_de_age_sex=labels.weekly_de_age_sex[week_start:week_start + week_count],
        weekly_land_coarse_age_sex=labels.weekly_land_coarse_age_sex[
            week_start:week_start + week_count
        ],
        daily_start_date=period.start_date,
        weekly_start_date=period.start_date,
    )
