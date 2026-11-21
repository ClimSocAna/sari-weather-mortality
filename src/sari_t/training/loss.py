"""Differentiable stage-1 mortality objectives on shared aggregates."""

import torch
import torch.nn.functional as F

from ..aggregation import MortalityAggregates
from ..data.interfaces import MortalityLabels


def calculate_loss_components(
    predictions: MortalityAggregates,
    labels: MortalityLabels,
    *,
    loss_fn: str = "poisson",
    daily_limit: int | None = None,
    weekly_limit: int | None = None,
) -> dict[str, torch.Tensor]:
    """Return the original five summed loss terms and their total.

    Predictions and labels must cover matching dates. Optional limits select
    prefixes of the supplied period, reproducing the original state-update time
    masks; the training caller chooses these limits and when to apply them.
    weekly_total sums age only and retains sex. No calibration, aggregation,
    epoch-dependent weighting or prediction detachment occurs here.
    """
    daily = predictions.daily_land[:daily_limit]
    weekly_de = predictions.weekly_de_age_sex[:weekly_limit]
    weekly_land = predictions.weekly_land_coarse_age_sex[:weekly_limit]
    daily_target = labels.daily_land[:daily_limit].to(
        device=daily.device, dtype=daily.dtype,
    )
    weekly_de_target = labels.weekly_de_age_sex[:weekly_limit].to(
        device=weekly_de.device, dtype=weekly_de.dtype,
    )
    weekly_land_target = labels.weekly_land_coarse_age_sex[:weekly_limit].to(
        device=weekly_land.device, dtype=weekly_land.dtype,
    )

    loss_daily = _sum_loss(daily, daily_target, loss_fn)
    loss_daily_total = _sum_loss(daily.sum(1), daily_target.sum(1), loss_fn)
    loss_weekly_de = _sum_loss(weekly_de, weekly_de_target, loss_fn)
    loss_weekly_land = _sum_loss(weekly_land, weekly_land_target, loss_fn)
    loss_weekly_total = _sum_loss(weekly_de.sum(1), weekly_de_target.sum(1), loss_fn)
    return {
        "daily": loss_daily,
        "daily_total": loss_daily_total,
        "weekly_de": loss_weekly_de,
        "weekly_land": loss_weekly_land,
        "weekly_total": loss_weekly_total,
        "total": (
            loss_daily_total + loss_weekly_total + loss_daily
            + loss_weekly_de + loss_weekly_land
        ),
    }


def _sum_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_fn: str,
) -> torch.Tensor:
    if loss_fn == "poisson":
        return F.poisson_nll_loss(
            prediction, target, log_input=False, full=True, reduction="none",
        ).sum()
    if loss_fn == "mse":
        return F.mse_loss(prediction, target, reduction="none").sum()
    raise ValueError(f"Unsupported loss function: {loss_fn}")
