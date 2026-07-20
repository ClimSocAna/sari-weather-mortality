from .base_model import HeatMortalityBase
from .baseline import calculate_baseline_death_rate, smooth_median_ewma
from .config import DataWindowConfig, HeatMortalityConfig
from .exp_model import HeatMortality_EXP
from .exp_model_test import HeatMortality_TEST
from .grouping import KREIS_COUNT_BY_LAND, age_grouping, land_extension, land_grouping
from .layers import Conv1dPos, LinearPos
from .population_data import population_preprocessing
from .sari_data import sari_preprocessing
from .weather_data import weather_preprocessing


ModelClass = type[HeatMortalityBase]

MODEL_REGISTRY: dict[str, ModelClass] = {
    "HeatMortality_EXP": HeatMortality_EXP,
    "HeatMortality_TEST": HeatMortality_TEST,
}


def get_model_class_by_name(model_name: str) -> ModelClass:
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_name '{model_name}'. Available models: {available}")
    return model_class

__all__ = [
    "Conv1dPos",
    "DataWindowConfig",
    "HeatMortalityBase",
    "HeatMortality_EXP",
    "HeatMortality_TEST",
    "HeatMortalityConfig",
    "KREIS_COUNT_BY_LAND",
    "LinearPos",
    "MODEL_REGISTRY",
    "age_grouping",
    "calculate_baseline_death_rate",
    "get_model_class_by_name",
    "land_extension",
    "land_grouping",
    "population_preprocessing",
    "sari_preprocessing",
    "smooth_median_ewma",
    "weather_preprocessing",
]
