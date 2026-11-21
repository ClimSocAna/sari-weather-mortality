"""Label-based mortality metrics shared by training and evaluation."""

import torch
import torch.nn.functional as F

from .aggregation import MortalityAggregates
from .data.interfaces import MortalityLabels


@torch.no_grad()
def calculate_national_mse(
    predictions: MortalityAggregates,
    labels: MortalityLabels,
) -> dict[str, torch.Tensor]:
    """Return daily and weekly national MSE for one aligned complete period.

    The caller chooses raw or aligned predictions and supplies matching dates.
    Daily counts sum over states; weekly counts sum over age and sex. Means are
    over days/weeks, not individual districts or age/sex cells. There is no epoch
    accumulation, distributed reduction, calibration or implicit label selection.
    """
    daily_prediction = predictions.daily_land.sum(dim=1)
    weekly_prediction = predictions.weekly_de_age_sex.sum(dim=(1, 2))
    daily_target = labels.daily_land.to(
        device=daily_prediction.device,
        dtype=daily_prediction.dtype,
    ).sum(dim=1)
    weekly_target = labels.weekly_de_age_sex.to(
        device=weekly_prediction.device,
        dtype=weekly_prediction.dtype,
    ).sum(dim=(1, 2))
    return {
        "daily_mse": F.mse_loss(daily_prediction, daily_target, reduction="mean"),
        "weekly_mse": F.mse_loss(weekly_prediction, weekly_target, reduction="mean"),
    }
