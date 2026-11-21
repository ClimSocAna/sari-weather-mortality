"""Stage-1 fitting settings, independent of data access and trainer imports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Explicit settings for the original full-window training procedure.

    Each training step uses the full prepared period, then selects the training
    weeks for its objective. Validation uses the cached remaining weeks. Steps
    per epoch are optimizer repetitions, not data chunks. Epochs are zero-based.
    Model dimensions/corrections, source selection, baseline mathematics and
    reference weather are configured separately.
    """

    validation_fraction: float = 0.25
    train_steps_per_epoch: int = 10
    validation_steps_per_epoch: int = 1
    max_epochs: int = 5000

    # AdamW and ReduceLROnPlateau, monitoring the training loss.
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    scheduler_patience: int = 100
    scheduler_threshold: float = 1e-4
    scheduler_factor: float = 0.3
    scheduler_start_after_epoch: int = 400
    gradient_clip_norm: float = 1.0

    temperature_noise_std: float = 0.5
    exclude_temperature: bool = False

    # Corrections freeze while epoch < warmup_epochs. State updates require
    # epoch > state_update_start_after_epoch and a multiple of the interval.
    correction_warmup_epochs: int = 300
    state_update_start_after_epoch: int = 300
    state_update_interval: int = 10

    # Adaptation runs when epoch >= start and epoch is a multiple of interval.
    # Initial and adaptive BaselineConfig values belong to run preparation.
    baseline_adaptation: bool = True
    baseline_update_start_epoch: int = 200
    baseline_update_interval: int = 50

    align_training_mean: bool = True
    alignment_max_weeks: int = 260
    # Raw loss weight: clip(1 - (epoch - start) / decay_epochs, minimum, 1).
    raw_loss_weight_start_epoch: int = 200
    raw_loss_weight_decay_epochs: int = 600
    raw_loss_weight_minimum: float = 0.3
    # Preserve the distinct original daily and weekly state-update loss masks.
    state_loss_days: int = 5 * 365
    state_loss_weeks: int = 5 * 52

    early_stopping_patience: int = 500
    early_stopping_min_delta: float = 0.0
    # Original step returns this proxy through scheduler warmup (inclusive);
    # logged train_loss remains the calculated objective.
    warmup_proxy_loss: float = 1e8
