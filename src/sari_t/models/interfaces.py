from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MortalityModelInputs:
    """Model covariates with explicit channel and time dimensions.

    Temperature has shape (Kreis, weather channel, lagged day), including a
    singleton channel axis for one weather variable. SARI and COVID have shape
    (age group, lagged day), in source incidence units. The effect model applies
    its configured surveillance scale. Missing SARI disables both disease effects, even
    when a COVID tensor is supplied.
    """

    temperature: torch.Tensor
    sari: torch.Tensor | None
    covid: torch.Tensor | None

    def to(self, device: torch.device | str) -> MortalityModelInputs:
        """Return covariates on device as float32, preserving absent inputs."""
        return MortalityModelInputs(
            temperature=self.temperature.to(device=device, dtype=torch.float32),
            sari=self.sari.to(device=device, dtype=torch.float32) if self.sari is not None else None,
            covid=self.covid.to(device=device, dtype=torch.float32) if self.covid is not None else None,
        )

    def detach(self) -> MortalityModelInputs:
        """Return covariates detached from autograd, sharing tensor storage."""
        return MortalityModelInputs(
            temperature=self.temperature.detach(),
            sari=self.sari.detach() if self.sari is not None else None,
            covid=self.covid.detach() if self.covid is not None else None,
        )


@dataclass(frozen=True)
class MortalityModelOutputs:
    """Raw model effects before conversion to mortality counts."""

    temperature_effect: torch.Tensor
    sari_effect: torch.Tensor
    covid_effect: torch.Tensor

    def detach(self) -> MortalityModelOutputs:
        """Return detached effects without copying tensor storage."""
        return MortalityModelOutputs(
            temperature_effect=self.temperature_effect.detach(),
            sari_effect=self.sari_effect.detach(),
            covid_effect=self.covid_effect.detach(),
        )


@dataclass(frozen=True)
class MortalityPredictionInputs:
    """Covariates and aligned daily context for mortality prediction.

    Mortality days cover complete Monday–Sunday weeks, excluding input lag history.
    Population has axes (mortality day, Kreis, age group, sex). Baseline rates are
    daily per-person rates with the same axes or a singleton Kreis axis for
    broadcasting. Both tensors cover mortality days only.
    """

    model_inputs: MortalityModelInputs
    baseline_rate: torch.Tensor
    population: torch.Tensor

    def detach(self) -> MortalityPredictionInputs:
        """Return detached covariates and context."""
        return MortalityPredictionInputs(
            model_inputs=self.model_inputs.detach(),
            baseline_rate=self.baseline_rate.detach(),
            population=self.population.detach(),
        )


@dataclass(frozen=True)
class MortalityPredictionOutputs:
    """Daily component and total counts with axes (day, Kreis, age group, sex).

    Temperature deaths include reference/baseline mortality; they are not
    temperature-attributable excess deaths. Components remain differentiable.
    Raw predictions also carry the detached minimum absolute temperature effect
    over days and districts, with shape (1, 1, age group, sex). Derived aligned
    predictions may omit this reference metadata.
    """

    temperature_deaths: torch.Tensor
    sari_deaths: torch.Tensor
    covid_deaths: torch.Tensor
    total_deaths: torch.Tensor
    minimum_temperature_effect: torch.Tensor | None = None

    def select_days(self, start: int, days: int) -> MortalityPredictionOutputs:
        """Return count slices, omitting the minimum effect from the full period.

        Slices share tensor storage and preserve gradients. The minimum effect
        cannot be recalculated for the selected days from mortality counts.
        """
        selected = slice(start, start + days)
        return MortalityPredictionOutputs(
            temperature_deaths=self.temperature_deaths[selected],
            sari_deaths=self.sari_deaths[selected],
            covid_deaths=self.covid_deaths[selected],
            total_deaths=self.total_deaths[selected],
        )

    def to(self, device: torch.device | str) -> MortalityPredictionOutputs:
        """Return count/reference tensors on device as float32."""
        return MortalityPredictionOutputs(
            temperature_deaths=self.temperature_deaths.to(device=device, dtype=torch.float32),
            sari_deaths=self.sari_deaths.to(device=device, dtype=torch.float32),
            covid_deaths=self.covid_deaths.to(device=device, dtype=torch.float32),
            total_deaths=self.total_deaths.to(device=device, dtype=torch.float32),
            minimum_temperature_effect=(
                self.minimum_temperature_effect.to(device=device, dtype=torch.float32)
                if self.minimum_temperature_effect is not None else None
            ),
        )

    def detach(self) -> MortalityPredictionOutputs:
        """Return detached counts/reference metadata, sharing tensor storage."""
        return MortalityPredictionOutputs(
            temperature_deaths=self.temperature_deaths.detach(),
            sari_deaths=self.sari_deaths.detach(),
            covid_deaths=self.covid_deaths.detach(),
            total_deaths=self.total_deaths.detach(),
            minimum_temperature_effect=(
                self.minimum_temperature_effect.detach()
                if self.minimum_temperature_effect is not None else None
            ),
        )
