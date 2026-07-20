from __future__ import annotations

import torch

from . import config
from .config import HeatMortalityConfig


def weather_preprocessing(model_config: HeatMortalityConfig) -> torch.Tensor:
    if not model_config.weather_files:
        raise ValueError("weather_files must contain at least one filename")

    weather_paths = [config.WEATHER_DIR / weather_filename for weather_filename in model_config.weather_files]
    loaded = [torch.load(path).float() for path in weather_paths]
    if len(loaded) == 1:
        return loaded[0].unsqueeze(1)
    return torch.stack(loaded, dim=1)
