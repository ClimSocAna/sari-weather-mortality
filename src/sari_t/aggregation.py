"""Differentiable mortality aggregation shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models.grouping import age_grouping, land_grouping


DAYS_PER_WEEK = 7


@dataclass(frozen=True)
class MortalityAggregates:
    """Predicted counts at the three observed stage-1 label resolutions."""

    daily_land: torch.Tensor
    weekly_de_age_sex: torch.Tensor
    weekly_land_coarse_age_sex: torch.Tensor

    def to(self, device: torch.device | str) -> MortalityAggregates:
        """Return stored aggregates on device as float32."""
        return MortalityAggregates(
            daily_land=self.daily_land.to(device=device, dtype=torch.float32),
            weekly_de_age_sex=self.weekly_de_age_sex.to(device=device, dtype=torch.float32),
            weekly_land_coarse_age_sex=self.weekly_land_coarse_age_sex.to(device=device, dtype=torch.float32),
        )

    def detach(self) -> MortalityAggregates:
        """Return reporting aggregates detached from autograd, without copying."""
        return MortalityAggregates(
            daily_land=self.daily_land.detach(),
            weekly_de_age_sex=self.weekly_de_age_sex.detach(),
            weekly_land_coarse_age_sex=self.weekly_land_coarse_age_sex.detach(),
        )


def aggregate_mortality(daily_counts: torch.Tensor) -> MortalityAggregates:
    """Aggregate (day, Kreis, age, sex) counts for complete Monday–Sunday weeks.

    The caller selects the period and raw or aligned total/component counts.
    Gradients are retained. This function performs no source/date selection,
    calibration, label comparison, or partial-week padding.
    """
    n_days, n_kreis, n_age, n_sex = daily_counts.shape
    weekly_counts = daily_counts.reshape(
        n_days // DAYS_PER_WEEK,
        DAYS_PER_WEEK,
        n_kreis,
        n_age,
        n_sex,
    ).sum(dim=1)
    daily_land_age_sex = land_grouping(daily_counts)
    weekly_land_age_sex = land_grouping(weekly_counts)
    return MortalityAggregates(
        daily_land=daily_land_age_sex.sum(dim=(-1, -2)),
        weekly_de_age_sex=weekly_land_age_sex.sum(dim=1),
        weekly_land_coarse_age_sex=age_grouping(weekly_land_age_sex, axis=2),
    )
