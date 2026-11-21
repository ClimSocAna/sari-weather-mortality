"""Baseline-rate calculations shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .models.interfaces import MortalityPredictionOutputs


DAYS_PER_WEEK = 7
SUPPORTED_BASELINE_MODES = ("bottom_10", "median_ewma", "median_ewma_d", "trimmed_avg")


@dataclass(frozen=True)
class BaselineConfig:
    """Serializable method and numeric settings; no update schedule or run state."""

    method: str = "bottom_10"
    window_weeks: int = 52
    trim_size: int = 10
    bottom_k: int = 10
    median_filter_kernel: int = 3
    ewma_halflife: float = 4.0


def calculate_baseline_death_rate(
    death: torch.Tensor,
    population: torch.Tensor,
    config: BaselineConfig,
) -> torch.Tensor:
    """Calculate daily per-person rates using the original baseline equations.

    death contains weekly national counts (week, age, sex); population contains
    aligned daily national counts (day, age, sex). Callers provide float32 tensors
    covering complete mortality weeks. Output has axes (day, 1, age, sex) for
    broadcasting over districts. For median_ewma, the first smoothed rate is used
    for weeks one and two; subsequent weeks use the preceding week's smoothed
    rate. No source selection or update schedule occurs. The centered
    median_ewma_d is an offline comparison. Rolling methods require at least
    max(1, window_weeks - 1) observed weeks for their original prefix padding.
    """
    baseline_mode = config.method
    weeks = config.window_weeks
    if baseline_mode not in SUPPORTED_BASELINE_MODES:
        raise ValueError(f"Unsupported baseline_mode: {baseline_mode}")

    if baseline_mode in {"median_ewma", "median_ewma_d"}:
        weekly_death_rate = smooth_median_ewma(
            death.float() / population[::DAYS_PER_WEEK],
            k=config.median_filter_kernel,
            halflife=config.ewma_halflife,
            double_sided=baseline_mode == "median_ewma_d",
        ) / DAYS_PER_WEEK
        if baseline_mode == "median_ewma":
            weekly_death_rate = torch.cat(
                (weekly_death_rate[:1], weekly_death_rate[:-1]),
                dim=0,
            )
        rate = weekly_death_rate.repeat_interleave(DAYS_PER_WEEK, dim=0)
    else:
        minimum_weeks = max(1, weeks - 1)
        if death.shape[0] < minimum_weeks:
            raise ValueError(
                f"Baseline method {baseline_mode!r} with window_weeks={weeks} requires "
                f"at least {minimum_weeks} observed weeks; got {death.shape[0]}"
            )
        padded_deaths = torch.cat([death[: weeks - 1], death], dim=0)
        smoothed_deaths: list[torch.Tensor] = []
        for start_index in range(padded_deaths.shape[0] - weeks + 1):
            window = padded_deaths[start_index : start_index + weeks]
            if baseline_mode == "trimmed_avg":
                sorted_deaths, _ = torch.sort(window, dim=0)
                trimmed_deaths = sorted_deaths[
                    config.trim_size:-config.trim_size,
                    :,
                    :,
                ]
                smoothed_deaths.append(trimmed_deaths.mean(dim=0) / DAYS_PER_WEEK)
            else:
                bottom_k_deaths = window.topk(config.bottom_k, dim=0, largest=False).values
                smoothed_deaths.append(bottom_k_deaths.sum(dim=0) / (config.bottom_k * DAYS_PER_WEEK))
        daily_baseline_deaths = torch.stack(smoothed_deaths).repeat_interleave(DAYS_PER_WEEK, 0)
        rate = daily_baseline_deaths / population[: daily_baseline_deaths.shape[0]]
    return rate.unsqueeze(1)


@torch.no_grad()
def calculate_weekly_residual_deaths(
    observed_weekly_deaths: torch.Tensor,
    predictions: MortalityPredictionOutputs,
    reference_temperature_deaths: torch.Tensor,
) -> torch.Tensor:
    """Subtract raw attributable components from observed national weekly counts.

    Component predictions and reference counts cover the same complete
    Monday–Sunday weeks with axes (day, Kreis, age, sex). Observations have axes
    (week, age, sex). Reference counts must use the same baseline, population
    and calendar factors as predictions; neither input is mean-aligned.
    """
    temperature_excess = predictions.temperature_deaths - reference_temperature_deaths
    additional_deaths = temperature_excess + predictions.sari_deaths + predictions.covid_deaths
    daily_national = additional_deaths.sum(dim=1)
    weekly_additional = daily_national.reshape(
        -1, DAYS_PER_WEEK, *daily_national.shape[1:],
    ).sum(dim=1)
    observed = observed_weekly_deaths.to(
        device=weekly_additional.device,
        dtype=weekly_additional.dtype,
    )
    return observed - weekly_additional


@torch.no_grad()
def calculate_updated_baseline_rate(
    observed_weekly_deaths: torch.Tensor,
    predictions: MortalityPredictionOutputs,
    reference_temperature_deaths: torch.Tensor,
    population: torch.Tensor,
    config: BaselineConfig,
) -> torch.Tensor:
    """Return one baseline update without mutating predictor or run state.

    Population has axes (day, Kreis, age, sex) for the same complete weeks as
    the raw predictions, reference counts and weekly observations. The caller
    chooses the method, when to run the update and whether to adopt its result.
    """
    residual = calculate_weekly_residual_deaths(
        observed_weekly_deaths,
        predictions,
        reference_temperature_deaths,
    )
    national_population = population.to(
        device=residual.device,
        dtype=residual.dtype,
    ).sum(dim=1)
    return calculate_baseline_death_rate(residual, national_population, config)


def smooth_median_ewma(
    values: torch.Tensor,
    k: int,
    halflife: float,
    dim: int = 0,
    double_sided: bool = False,
) -> torch.Tensor:
    """
    Apply median filtering followed by exponential weighted moving average smoothing.

    This two-stage smoothing approach first applies a median filter to reduce noise,
    then applies EWMA to create smooth trends.

    Args:
        values: Input tensor to smooth
        k: Explicit median-filter kernel size
        halflife: Halflife parameter for EWMA (controls smoothing strength)
        dim: Dimension along which to apply smoothing
        double_sided: If True, apply symmetric padding; else asymmetric padding

    Returns:
        Smoothed tensor with same shape as input
    """
    median_filtered = _median_filter1d_along(values, k, dim, double_sided)
    return _ewma_along(median_filtered, halflife, dim)


def _median_filter1d_along(
    series: torch.Tensor,
    k: int,
    dim: int = -1,
    double_sided: bool = False,
) -> torch.Tensor:
    """Apply 1D median filter along specified dimension."""
    series = torch.movedim(series, dim, -1)
    if not double_sided:
        padded_values = F.pad(series, (k - 1, 0), mode="replicate")
    else:
        pad_left = (k - 1) // 2
        pad_right = (k - 1) - pad_left
        padded_values = F.pad(series, (pad_left, pad_right), mode="replicate")
    patches = padded_values.unfold(-1, k, 1)
    median_values = patches.median(dim=-1).values
    return torch.movedim(median_values, -1, dim)


def _ewma_along(
    series: torch.Tensor,
    halflife: float,
    dim: int = -1,
) -> torch.Tensor:
    """Apply exponential weighted moving average along specified dimension."""
    alpha = 1 - 2.0 ** (-1.0 / halflife)
    series = torch.movedim(series, dim, -1)
    smoothed_values = torch.empty_like(series)
    smoothed_values[..., 0] = series[..., 0]
    for index in range(1, series.size(-1)):
        smoothed_values[..., index] = (
            alpha * series[..., index]
            + (1 - alpha) * smoothed_values[..., index - 1]
        )
    return torch.movedim(smoothed_values, -1, dim)
