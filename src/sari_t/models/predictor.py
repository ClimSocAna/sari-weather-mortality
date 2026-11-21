"""Differentiable mortality counts from model effects and explicit context."""

import torch
from torch import nn

from .config import MortalityPredictorConfig
from .exponential import ExponentialMortalityModel
from .grouping import LAND_COUNT, land_extension
from .interfaces import MortalityPredictionInputs, MortalityPredictionOutputs


class MortalityPredictor(nn.Module):
    """Compose an effect model with population, baseline, and calendar factors.

    Callers supply aligned inputs and choose training/evaluation mode. Baseline
    adaptation, attribution, and fitting schedules remain outside this module.
    Prediction windows start on Monday; preparation owns calendar alignment.
    """

    def __init__(
        self,
        effect_model: ExponentialMortalityModel,
        config: MortalityPredictorConfig,
    ) -> None:
        super().__init__()
        self.effect_model = effect_model
        self.config = config
        reference = next(effect_model.parameters())
        for name, size, learned in (
            ("weekday_correction", 6, config.weekday_correction),
            ("state_correction", LAND_COUNT, config.state_correction == "default"),
        ):
            values = torch.ones(size, device=reference.device, dtype=reference.dtype)
            if learned:
                self.register_parameter(name, nn.Parameter(values))
            else:
                self.register_buffer(name, values, persistent=False)

    def forward(self, inputs: MortalityPredictionInputs) -> MortalityPredictionOutputs:
        effects = self.effect_model(inputs.model_inputs)
        correction_factors = self.correction_factors(effects.temperature_effect.shape[0])
        temperature_deaths = (
            effects.temperature_effect.abs()
            * inputs.baseline_rate
            * inputs.population
            * correction_factors
        )
        sari_deaths = (
            effects.sari_effect
            * inputs.baseline_rate
            * inputs.population
            * correction_factors
        )
        covid_deaths = (
            effects.covid_effect
            * inputs.baseline_rate
            * inputs.population
            * correction_factors
        )
        return MortalityPredictionOutputs(
            temperature_deaths=temperature_deaths,
            sari_deaths=sari_deaths,
            covid_deaths=covid_deaths,
            total_deaths=temperature_deaths + sari_deaths + covid_deaths,
            minimum_temperature_effect=effects.temperature_effect.detach().abs().amin(
                dim=(0, 1), keepdim=True,
            ),
        )

    def correction_factors(self, day_count: int) -> torch.Tensor:
        """Return weekday/state factors for mortality windows starting on Monday."""
        # Monday is the fixed reference; learned values represent Tuesday–Sunday.
        weekday_values = torch.cat(
            (self.weekday_correction.new_ones(1), self.weekday_correction),
        )
        weekday_indices = torch.arange(day_count, device=weekday_values.device) % 7
        weekday_factors = weekday_values[weekday_indices].reshape(-1, 1, 1, 1)
        return weekday_factors * self._state_factors()

    def _state_factors(self) -> torch.Tensor:
        """Expand state corrections onto the trained Kreis order."""
        if self.config.state_correction in ("default", None):
            return land_extension(self.state_correction, land_axis=0).reshape(
                1, -1, 1, 1,
            )
        raise NotImplementedError(
            f"State correction method is not implemented: {self.config.state_correction}"
        )
