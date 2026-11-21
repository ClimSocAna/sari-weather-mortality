"""Compose run settings without reading sources or constructing models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..data.config import DataConfig
from ..models.config import MortalityModelConfig, MortalityPredictorConfig, PredictionPeriod
from ..training.config import TrainingConfig

if TYPE_CHECKING:
    from ..baseline import BaselineConfig
    from ..reference import ReferenceConfig


@dataclass(frozen=True)
class EvaluationConfig:
    """Iteration limit and convergence tolerance for evaluation baseline fitting."""

    maximum_iterations: int = 100
    convergence_tolerance: float = 0.001


@dataclass(frozen=True)
class RunConfig:
    """Explicit stage-1 inputs and policies needed to prepare a training run.

    selected_weather contains DataConfig.weather keys in model channel order,
    allowing different datasets for the same physical variable. period covers
    both training and validation; TrainingConfig determines their weekly split.
    initial_baseline_config and adaptive_baseline_config are independent: the
    original training initializes with bottom_10 even when adaptation uses median_ewma.

    This is configuration, not a checkpoint or a result cache. Evaluation changes
    must be supplied explicitly without mutating the recorded training settings.
    Device selection and checkpoint paths belong to the execution/loading layer.
    """

    data_config: DataConfig
    selected_weather: tuple[str, ...]
    period: PredictionPeriod
    model_config: MortalityModelConfig
    predictor_config: MortalityPredictorConfig
    training_config: TrainingConfig
    initial_baseline_config: BaselineConfig
    adaptive_baseline_config: BaselineConfig
    reference_config: ReferenceConfig

    @property
    def run_name(self) -> str:
        """Describe the resolved stage-1 setup using the original run-name order."""
        terms = ["HeatMortality_EXP"]
        if not self.training_config.exclude_temperature:
            terms.append(f"t_{self.model_config.kernel_T_days}")
        if self.model_config.include_sari:
            terms.append(f"sari_{self.model_config.kernel_sari_days}")
        if self.model_config.include_covid:
            terms.append("covid")
        if self.training_config.baseline_adaptation:
            terms.append(self.adaptive_baseline_config.method)
        else:
            terms.append(f"static_{self.initial_baseline_config.method}")
        temperature = self.reference_config.temperature_c
        terms.append("temp_ref_min" if temperature is None else f"temp_ref_{temperature:g}c")
        terms.append("corr_on" if self.training_config.align_training_mean else "corr_off")
        terms.append("-".join(self.selected_weather))
        terms.append("stage1")
        return "_".join(terms)
