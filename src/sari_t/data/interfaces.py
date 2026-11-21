"""Source and prepared stage-1 data, without training or baseline state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from ..models.config import PredictionPeriod

if TYPE_CHECKING:
    from torch import Tensor, device as Device

    from ..models.interfaces import MortalityModelInputs


@dataclass(frozen=True)
class WeatherData:
    """Selected float32 CPU weather with axes (Kreis, channel, day).

    Variables follow the requested source-key order, including repeated physical
    variables from different datasets. Coverage is the selected sources' common
    interval, not their union.
    """

    values: Tensor
    variables: tuple[str, ...]
    start_date: date

    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.values.shape[-1] - 1)


@dataclass(frozen=True)
class PopulationData:
    """Daily CPU population with axes (day, Kreis, age group, sex)."""

    values: Tensor
    start_date: date

    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.values.shape[0] - 1)


@dataclass(frozen=True)
class SurveillanceData:
    """Paired daily CPU incidence tensors with axes (age group, day).

    Pipeline-prepared values retain source units; the effect model owns scaling.
    """

    sari: Tensor
    covid: Tensor
    start_date: date

    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.sari.shape[-1] - 1)


@dataclass(frozen=True)
class MortalityLabels:
    """Stage-1 observed counts with independent daily and weekly origins.

    Axes are (day, state), (week, age, sex), and (week, state, coarse age, sex).
    Source missing/padding markers have been removed, following the original
    reader. Each series is assumed contiguous from its origin after filtering;
    its length determines the available daily or weekly coverage.
    """

    daily_land: Tensor
    weekly_de_age_sex: Tensor
    weekly_land_coarse_age_sex: Tensor
    daily_start_date: date
    weekly_start_date: date

    def to(self, device: Device | str) -> MortalityLabels:
        """Return observed counts on device as float32, preserving dates."""
        import torch

        return MortalityLabels(
            daily_land=self.daily_land.to(device=device, dtype=torch.float32),
            weekly_de_age_sex=self.weekly_de_age_sex.to(device=device, dtype=torch.float32),
            weekly_land_coarse_age_sex=self.weekly_land_coarse_age_sex.to(
                device=device, dtype=torch.float32,
            ),
            daily_start_date=self.daily_start_date,
            weekly_start_date=self.weekly_start_date,
        )


@dataclass(frozen=True)
class SourceData:
    """CPU source tensors before mortality-window selection.

    Weather axes are (Kreis, channel, day), population (day, Kreis, age, sex),
    and SARI/COVID (age, day). Surveillance retains its source incidence units.
    Each origin describes the actual tensor calendar; no implicit population
    date shift or baseline alignment is encoded in this container.
    """

    weather: Tensor
    weather_variables: tuple[str, ...]
    weather_start_date: date
    population: Tensor
    population_start_date: date
    sari: Tensor
    covid: Tensor
    surveillance_start_date: date
    labels: MortalityLabels


@dataclass(frozen=True)
class PreparedData:
    """Complete inputs, population and labels for one mortality window.

    Data preparation produces CPU tensors; execution can transfer them together.

    Covariates include kernel history and retain source surveillance units;
    scaling belongs to the effect model. Population and labels cover mortality dates only. Labels carry their
    selected origins. These containers may share tensor storage with sources;
    frozen fields do not make tensor contents immutable.
    """

    window: PredictionPeriod
    model_inputs: MortalityModelInputs
    population: Tensor
    labels: MortalityLabels

    def to(self, device: Device | str) -> PreparedData:
        """Return aligned inputs, population and labels on device as float32."""
        import torch

        return PreparedData(
            window=self.window,
            model_inputs=self.model_inputs.to(device),
            population=self.population.to(device=device, dtype=torch.float32),
            labels=self.labels.to(device),
        )
