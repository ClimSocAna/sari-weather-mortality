import torch
import torch.nn.functional as F

from .config import (
    BASELINE_WINDOW_WEEKS,
    TRIMMED_AVERAGE_TRIM_SIZE,
    BOTTOM_K_SIZE,
    MEDIAN_FILTER_KERNEL,
    EWMA_HALFLIFE,
    DAYS_PER_WEEK,
    STATIC_BASELINE_METHOD,
    SUPPORTED_BASELINE_MODES,
)


def calculate_baseline_death_rate(
    death: torch.Tensor,
    population: torch.Tensor,
    baseline_mode: str = STATIC_BASELINE_METHOD,
    weeks: int = BASELINE_WINDOW_WEEKS,
) -> torch.Tensor:
    """
    Calculate baseline death rates using the specified baseline/adaptation method.

    Args:
        death: Tensor of death counts
        population: Tensor of population values
        baseline_mode: Baseline calculation method (median_ewma, median_ewma_d, trimmed_avg, bottom_10)
        weeks: Number of weeks for windowing calculations

    Returns:
        Tensor of baseline death rates normalized by population
    """
    if baseline_mode not in SUPPORTED_BASELINE_MODES:
        raise ValueError(f"Unsupported baseline_mode: {baseline_mode}")

    if baseline_mode in {"median_ewma", "median_ewma_d"}:
        weekly_death_rate = smooth_median_ewma(
            death.float() / population[::DAYS_PER_WEEK],
            double_sided=baseline_mode == "median_ewma_d",
        ) / DAYS_PER_WEEK
        rate_by_channel = weekly_death_rate.swapaxes(0, 2)
        rate = F.interpolate(
            rate_by_channel,
            scale_factor=DAYS_PER_WEEK,
            mode="linear",
            align_corners=False,
        ).swapaxes(0, 2)
    else:
        padded_deaths = torch.cat([death[: weeks - 1], death], dim=0)
        smoothed_deaths: list[torch.Tensor] = []
        for start_index in range(padded_deaths.shape[0] - weeks + 1):
            window = padded_deaths[start_index : start_index + weeks]
            if baseline_mode == "trimmed_avg":
                sorted_deaths, _ = torch.sort(window, dim=0)
                trimmed_deaths = sorted_deaths[
                    TRIMMED_AVERAGE_TRIM_SIZE:-TRIMMED_AVERAGE_TRIM_SIZE,
                    :,
                    :,
                ]
                smoothed_deaths.append(trimmed_deaths.mean(dim=0) / DAYS_PER_WEEK)
            else:
                bottom_k_deaths = window.topk(BOTTOM_K_SIZE, dim=0, largest=False).values
                smoothed_deaths.append(bottom_k_deaths.sum(dim=0) / (BOTTOM_K_SIZE * DAYS_PER_WEEK))
        daily_baseline_deaths = torch.stack(smoothed_deaths).repeat_interleave(DAYS_PER_WEEK, 0)
        rate = daily_baseline_deaths / population[: daily_baseline_deaths.shape[0]]
    return rate.unsqueeze(1)


def smooth_median_ewma(
    values: torch.Tensor,
    k: int = MEDIAN_FILTER_KERNEL,
    halflife: float = EWMA_HALFLIFE,
    dim: int = 0,
    double_sided: bool = False,
) -> torch.Tensor:
    """
    Apply median filtering followed by exponential weighted moving average smoothing.

    This two-stage smoothing approach first applies a median filter to reduce noise,
    then applies EWMA to create smooth trends.

    Args:
        values: Input tensor to smooth
        k: Kernel size for median filtering (default: MEDIAN_FILTER_KERNEL)
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
