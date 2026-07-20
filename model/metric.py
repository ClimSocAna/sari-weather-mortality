import torch
from torch import nn
from torchmetrics import Metric

from .containers import MortalityLabels, MortalityLabelsStage1, MortalityLabelsStage2, MortalityPredictions


class HeatMortalityAccuracy(Metric):
    """Metric for averaged daily and weekly MSE in Germany."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state("mse_daily_sum", default=torch.tensor(0.0, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("mse_weekly_sum", default=torch.tensor(0.0, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("num_updates", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.mse = nn.MSELoss(reduction="mean")

    def update(self, predictions: MortalityPredictions, labels: MortalityLabels) -> None:
        if isinstance(labels, MortalityLabelsStage1):
            daily_prediction = predictions.daily_kreis_age_sex.sum((1, 2, 3))
            weekly_prediction = predictions.weekly_kreis_age_sex.sum((1, 2, 3))
            daily_target = labels.daily_land.sum(1)
            weekly_target = labels.weekly_de_age_sex.sum((1, 2))
            daily_mse = self.mse(daily_prediction.detach(), daily_target).double()
            weekly_mse = self.mse(weekly_prediction.detach(), weekly_target).double()
        elif isinstance(labels, MortalityLabelsStage2):
            raise NotImplementedError("Stage 2 metric computation is not implemented yet")
        else:
            raise TypeError(f"Unsupported label type for metric: {type(labels)}")

        self.mse_daily_sum += daily_mse
        self.mse_weekly_sum += weekly_mse
        self.num_updates += 1

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = self.num_updates.clamp_min(1).double()
        return self.mse_daily_sum / denominator, self.mse_weekly_sum / denominator
