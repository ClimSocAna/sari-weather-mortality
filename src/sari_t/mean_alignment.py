"""Mean-alignment operations shared by training and inference."""

from typing import TypeVar

import torch

from .models.interfaces import MortalityPredictionOutputs


AlignmentValues = TypeVar("AlignmentValues", torch.Tensor, MortalityPredictionOutputs)


@torch.no_grad()
def estimate_mean_alignment(
    predictions: MortalityPredictionOutputs,
    labels: torch.Tensor,
    *,
    max_weeks: int | None = None,
) -> torch.Tensor:
    """Estimate age/sex factors against weekly national observed death counts.

    Predictions and labels must start on the same Monday in the explicitly
    selected calibration period. Labels have shape (week, age group, sex).
    Use matching complete weeks only; max_weeks=260 reproduces the legacy cap
    when supplied with the original training calibration period.
    """
    daily = predictions.total_deaths
    weeks = min(daily.shape[0] // 7, labels.shape[0])
    if max_weeks is not None:
        weeks = min(weeks, max_weeks)
    if weeks <= 0:
        raise ValueError("Mean alignment requires at least one matching complete calibration week")
    predicted = daily[:weeks * 7].sum(dim=(0, 1))
    observed = labels[:weeks].to(device=predicted.device, dtype=predicted.dtype).sum(dim=0)
    if not torch.isfinite(predicted).all() or not torch.isfinite(observed).all():
        raise ValueError("Mean alignment requires finite predicted and observed calibration totals")
    if (predicted == 0).any():
        raise ValueError("Mean alignment has a zero predicted calibration total for an age/sex group")
    factor = observed / predicted
    if not torch.isfinite(factor).all():
        raise ValueError("Mean alignment produced a nonfinite age/sex factor")
    return factor


def apply_mean_alignment(
    predictions: AlignmentValues,
    factor: torch.Tensor | None,
) -> AlignmentValues:
    """Scale counts or prediction components; return the input unchanged for None.

    Application preserves gradients; estimation controls whether factors are
    detached. The age/sex factor broadcasts over mortality days and districts.
    """
    if factor is None:
        return predictions
    if isinstance(predictions, torch.Tensor):
        return predictions * factor.to(device=predictions.device, dtype=predictions.dtype)
    factor = factor.to(
        device=predictions.total_deaths.device,
        dtype=predictions.total_deaths.dtype,
    )
    temperature = predictions.temperature_deaths * factor
    sari = predictions.sari_deaths * factor
    covid = predictions.covid_deaths * factor
    return MortalityPredictionOutputs(
        temperature_deaths=temperature,
        sari_deaths=sari,
        covid_deaths=covid,
        total_deaths=temperature + sari + covid,
    )
