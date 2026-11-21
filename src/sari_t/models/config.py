"""Network architecture and mortality prediction settings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


AGE_GROUP_COUNT = 15
GENDER_COUNT = 2
AGE_GENDER_CHANNEL_COUNT = AGE_GROUP_COUNT * GENDER_COUNT


@dataclass(frozen=True)
class MortalityModelConfig:
    """Network dimensions and disease branches, without data or training settings.

    Weather channels are inferred from the first input tensor. Source selection
    and channel ordering belong to data/run preparation.
    The COVID branch requires SARI, matching the existing training configuration.
    """

    out_channel: int = AGE_GENDER_CHANNEL_COUNT
    hidden_dim_init: int = 128
    kernel_T_days: int = 6
    kernel_sari_days: int = 10
    include_sari: bool = True
    include_covid: bool = True
    surveillance_scale: float = 0.01

    def __post_init__(self) -> None:
        if self.include_covid:
            object.__setattr__(self, "include_sari", True)


@dataclass(frozen=True)
class MortalityPredictorConfig:
    """Select calendar corrections; None disables state correction."""

    weekday_correction: bool = True
    state_correction: str | None = "default"


@dataclass(frozen=True)
class PredictionPeriod:
    """Requested inclusive prediction dates, excluding leading input history.

    Source coverage is checked separately when preparing data for this period.
    """

    start_date: date
    end_date: date
