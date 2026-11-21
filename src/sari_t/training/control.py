"""Optimizer setup and update decisions for mortality training."""

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import TrainingConfig


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
) -> tuple[AdamW, ReduceLROnPlateau]:
    """Configure the original optimizer and scheduler over all model parameters.

    The fitting loop owns gradient clipping, temporary correction freezing and
    optimizer/scheduler stepping, including the scheduler's delayed start.
    """
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=config.scheduler_patience,
        threshold=config.scheduler_threshold,
        factor=config.scheduler_factor,
    )
    return optimizer, scheduler


def corrections_trainable(epoch: int, config: TrainingConfig) -> bool:
    """Release weekday/state corrections when warmup ends."""
    return epoch >= config.correction_warmup_epochs


def should_update_state(epoch: int, config: TrainingConfig) -> bool:
    """Select state-update epochs; the start epoch itself is excluded."""
    return (
        epoch > config.state_update_start_after_epoch
        and epoch % config.state_update_interval == 0
    )


def should_update_baseline(epoch: int, config: TrainingConfig) -> bool:
    """Select enabled baseline updates, including the configured start epoch."""
    return (
        config.baseline_adaptation
        and epoch >= config.baseline_update_start_epoch
        and epoch % config.baseline_update_interval == 0
    )


def should_step_scheduler(
    epoch: int,
    step_index: int,
    config: TrainingConfig,
) -> bool:
    """Step only on the first training step after scheduler warmup."""
    return step_index == 0 and epoch > config.scheduler_start_after_epoch


def raw_loss_weight(epoch: int, config: TrainingConfig) -> float:
    """Weight raw loss when mixing raw and mean-aligned objectives."""
    weight = 1 - (
        (epoch - config.raw_loss_weight_start_epoch)
        / config.raw_loss_weight_decay_epochs
    )
    return min(1.0, max(config.raw_loss_weight_minimum, weight))
