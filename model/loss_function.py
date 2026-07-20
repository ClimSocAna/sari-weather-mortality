import torch

from .grouping import age_grouping, land_grouping
from .containers import MortalityLabels, MortalityLabelsStage1, MortalityLabelsStage2, MortalityPredictions
from .config import COVID_FREE_DAYS, COVID_FREE_WEEKS

COARSE_AGE_GROUPS = (tuple(range(8)), (8, 9), (10, 11), (12, 13, 14))


class ElementwiseLoss(torch.nn.Module):
    """Unreduced Poisson/MSE loss wrapper with optional masking."""

    def __init__(self, loss_fn: str = "poisson") -> None:
        super().__init__()

        if loss_fn == "poisson":
            self.loss = torch.nn.PoissonNLLLoss(log_input=False, full=True, reduction="none")
        elif loss_fn == "mse":
            self.loss = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculate loss with optional masking."""
        if mask is None:
            return self.loss(prediction, target).sum()
        return self.loss(prediction[mask], target[mask]).sum()


class HeatMortalityLoss(torch.nn.Module):
    """
    Aggregate mortality loss across daily state totals and weekly grouped targets.

    This loss function combines multiple loss components at different aggregation levels
    to provide strong supervisions signals across spatial and temporal dimensions.
    """

    def __init__(self, loss_fn: str = "poisson") -> None:
        super().__init__()
        self.loss = ElementwiseLoss(loss_fn)

    @staticmethod
    def _full_mask(length: int, device: torch.device) -> torch.Tensor:
        """Create a full mask of ones for the given length and device."""
        return torch.ones([length], device=device, dtype=torch.bool)

    def _build_time_masks(self, labels: MortalityLabelsStage1, with_covid: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build temporal masks for loss calculation.

        Masks exclude COVID-era data if with_covid=False to maintain consistency
        with pre-COVID baseline periods.
        """
        daily_mask = self._full_mask(labels.daily_land.shape[0], labels.daily_land.device)
        weekly_mask = self._full_mask(labels.weekly_de_age_sex.shape[0], labels.weekly_de_age_sex.device)

        if not with_covid:
            daily_mask[COVID_FREE_DAYS:] = False
            weekly_mask[COVID_FREE_WEEKS:] = False

        return daily_mask, weekly_mask

    @staticmethod
    def _group_predictions(predictions: MortalityPredictions) -> dict[str, torch.Tensor]:
        """Aggregate predictions to appropriate geographic and demographic levels."""
        daily_land_age_sex = land_grouping(predictions.daily_kreis_age_sex)
        weekly_land_age_sex = land_grouping(predictions.weekly_kreis_age_sex)
        weekly_de_age_sex = weekly_land_age_sex.sum(dim=1)
        weekly_land_coarse_age_sex = age_grouping(weekly_land_age_sex, 2, COARSE_AGE_GROUPS)

        return {
            "daily_land": daily_land_age_sex.sum(dim=(-1, -2)),
            "weekly_de_age_sex": weekly_de_age_sex,
            "weekly_land_coarse_age_sex": weekly_land_coarse_age_sex,
        }

    def compute_components(
        self,
        predictions: MortalityPredictions,
        labels: MortalityLabels,
        with_covid: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss components across all aggregation levels.

        Combines loss from daily state totals, weekly Germany-wide aggregates,
        and weekly state-level age/sex breakdowns.

        Args:
            predictions: Model predictions at kreis (district) level
            labels: Ground truth labels at various aggregation levels
            with_covid: Include COVID-era data in loss calculation
        Returns:
            Dictionary of scalar loss components and the total loss
        """
        if isinstance(labels, MortalityLabelsStage1):
            daily_mask, weekly_mask = self._build_time_masks(labels, with_covid)
            grouped_predictions = self._group_predictions(predictions)

            daily_land_prediction = grouped_predictions["daily_land"]
            weekly_de_prediction = grouped_predictions["weekly_de_age_sex"]
            weekly_land_prediction = grouped_predictions["weekly_land_coarse_age_sex"]

            loss_daily = self.loss(daily_land_prediction, labels.daily_land, mask=daily_mask)
            loss_daily_total = self.loss(daily_land_prediction.sum(1), labels.daily_land.sum(1), mask=daily_mask)
            loss_weekly_de = self.loss(weekly_de_prediction, labels.weekly_de_age_sex, mask=weekly_mask)
            loss_weekly_land = self.loss(weekly_land_prediction, labels.weekly_land_coarse_age_sex, mask=weekly_mask)
            loss_weekly_total = self.loss(weekly_de_prediction.sum(1), labels.weekly_de_age_sex.sum(1), mask=weekly_mask)
            return {
                "daily": loss_daily,
                "daily_total": loss_daily_total,
                "weekly_de": loss_weekly_de,
                "weekly_land": loss_weekly_land,
                "weekly_total": loss_weekly_total,
                "total": loss_daily_total + loss_weekly_total + loss_daily + loss_weekly_de + loss_weekly_land,
            }

        if isinstance(labels, MortalityLabelsStage2):
            raise NotImplementedError("Stage 2 loss computation is not implemented yet")

        raise TypeError(f"Unsupported label type for loss: {type(labels)}")

    def forward(
        self,
        predictions: MortalityPredictions,
        labels: MortalityLabels,
        with_covid: bool = True,
    ) -> torch.Tensor:
        """Compute total loss across all aggregation levels."""
        return self.compute_components(predictions, labels, with_covid=with_covid)["total"]
