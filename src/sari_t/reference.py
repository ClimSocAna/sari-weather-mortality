"""Reference-temperature operations shared by baseline adaptation and interpretation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .models.exponential import ExponentialMortalityModel
from .models.interfaces import MortalityModelInputs, MortalityPredictionInputs


# Alduchov–Eskridge Magnus coefficients, with temperature in Celsius.
MAGNUS_A = 17.625
MAGNUS_B = 243.04


@dataclass(frozen=True)
class ReferenceConfig:
    """None selects the empirical factor minimum; otherwise use fixed weather."""

    temperature_c: float | None = None
    relative_humidity_percent: float = 50.0


def build_reference_weather(
    weather: torch.Tensor,
    weather_variables: tuple[str, ...],
    temperature_c: float,
    relative_humidity_percent: float,
    *,
    kernel_days: int,
) -> torch.Tensor:
    """Build constant reference channels in the supplied physical-variable order.

    Weather supplies channel metadata, dtype and device. Return just one district
    and one kernel window; its effect broadcasts over prediction dates/districts.
    """
    if weather.shape[1] != len(weather_variables):
        raise ValueError("Reference weather variable names do not match the input channel count")
    values = [
        _reference_value(variable, temperature_c, relative_humidity_percent)
        for variable in weather_variables
    ]
    reference = torch.as_tensor(values, device=weather.device, dtype=weather.dtype)
    return reference.view(1, -1, 1).expand(1, -1, kernel_days).clone()


@torch.no_grad()
def reference_temperature_effect(
    effect_model: ExponentialMortalityModel,
    model_inputs: MortalityModelInputs,
    weather_variables: tuple[str, ...],
    config: ReferenceConfig,
    *,
    minimum_temperature_effect: torch.Tensor,
) -> torch.Tensor:
    """Return an absolute reference factor, before population/baseline scaling.

    The empirical mode uses the detached minimum returned by the predictor,
    avoiding retention of full effects or another actual model pass. Fixed
    reference evaluation uses eval/no_grad and restores the prior model mode.
    The caller materializes lazy layers before performing reference evaluation.
    """
    if config.temperature_c is None:
        return minimum_temperature_effect

    reference_weather = build_reference_weather(
        model_inputs.temperature,
        weather_variables,
        config.temperature_c,
        config.relative_humidity_percent,
        kernel_days=effect_model.config.kernel_T_days,
    )
    was_training = effect_model.training
    try:
        effect_model.eval()
        effects = effect_model(MortalityModelInputs(
            temperature=reference_weather,
            sari=None,
            covid=None,
        ))
        return effects.temperature_effect.abs()
    finally:
        effect_model.train(was_training)


def reference_temperature_deaths(
    reference_effect: torch.Tensor,
    inputs: MortalityPredictionInputs,
    correction_factors: torch.Tensor,
) -> torch.Tensor:
    """Apply the same daily context used for actual temperature mortality.

    Baseline, population and calendar factors must correspond to the actual
    prediction being compared. This is reference mortality, not attributable
    excess; it contains no mean-alignment or label-dependent update.
    """
    return (
        reference_effect.abs()
        * inputs.baseline_rate
        * inputs.population
        * correction_factors
    )


def _reference_value(variable: str, temperature_c: float, humidity_percent: float) -> float:
    if variable in {"tas", "tas_min"}:
        return temperature_c
    if variable == "tdew":
        return dew_point_c(temperature_c, humidity_percent)
    if variable == "twb":
        return wet_bulb_temperature_c(temperature_c, humidity_percent)
    raise ValueError(f"Unsupported weather variable for attribution reference: {variable}")


def dew_point_c(
    temperature_c: float | torch.Tensor,
    relative_humidity_percent: float | torch.Tensor,
) -> float | torch.Tensor:
    """Magnus dew point in Celsius for scalars or broadcasting tensors.

    Tensor calculations retain their device/dtype and autograd relationships.
    Reference weather and counterfactuals use the same Alduchov–Eskridge constants.
    """
    temperature_c, relative_humidity_percent = _weather_pair(temperature_c, relative_humidity_percent)
    log = torch.log if isinstance(temperature_c, torch.Tensor) else math.log
    gamma = log(relative_humidity_percent / 100.0) + (
        MAGNUS_A * temperature_c / (MAGNUS_B + temperature_c)
    )
    return MAGNUS_B * gamma / (MAGNUS_A - gamma)


def relative_humidity_percent(
    temperature_c: float | torch.Tensor,
    dew_point_temperature_c: float | torch.Tensor,
) -> float | torch.Tensor:
    """Return relative humidity in percent using the matching Magnus constants."""
    temperature_c, dew_point_temperature_c = _weather_pair(temperature_c, dew_point_temperature_c)
    exp = torch.exp if isinstance(temperature_c, torch.Tensor) else math.exp
    return 100 * exp(
        MAGNUS_A * dew_point_temperature_c / (MAGNUS_B + dew_point_temperature_c)
        - MAGNUS_A * temperature_c / (MAGNUS_B + temperature_c)
    )


def wet_bulb_temperature_c(
    temperature_c: float | torch.Tensor,
    relative_humidity_percent: float | torch.Tensor,
) -> float | torch.Tensor:
    """Original wet-bulb approximation for scalars or broadcasting tensors."""
    temperature_c, relative_humidity_percent = _weather_pair(temperature_c, relative_humidity_percent)
    atan = torch.atan if isinstance(temperature_c, torch.Tensor) else math.atan
    sqrt = torch.sqrt if isinstance(temperature_c, torch.Tensor) else math.sqrt
    humidity_power_term = relative_humidity_percent ** 1.5
    return (
        temperature_c * atan(0.151977 * sqrt(relative_humidity_percent + 8.313659))
        + atan(temperature_c + relative_humidity_percent)
        - atan(relative_humidity_percent - 1.676331)
        + 0.00391838 * humidity_power_term * atan(0.023101 * relative_humidity_percent)
        - 4.686035
    )


def _weather_pair(
    first: float | torch.Tensor, second: float | torch.Tensor,
) -> tuple[float | torch.Tensor, float | torch.Tensor]:
    """Place mixed scalar/tensor operands on the supplied tensor's device/dtype."""
    if isinstance(first, torch.Tensor):
        return first, torch.as_tensor(second, device=first.device, dtype=first.dtype)
    if isinstance(second, torch.Tensor):
        return torch.as_tensor(first, device=second.device, dtype=second.dtype), second
    return first, second
