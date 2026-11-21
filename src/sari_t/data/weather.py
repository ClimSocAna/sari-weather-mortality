"""Read explicitly selected daily weather artifacts on a common calendar."""

from __future__ import annotations

from datetime import timedelta

import torch

from .config import DataConfig
from .interfaces import WeatherData


def load_weather(
    config: DataConfig,
    selected_sources: tuple[str, ...],
) -> WeatherData:
    """Load only selected source keys and preserve their requested channel order.

    Pipeline tensors use (Kreis, day); paired observation/MOSMIX artifacts select
    temperature with variable='tas' or dew point with variable='tdew'. Each start_date is
    the first daily value of the selected variable. No source discovery, gap
    filling, extrapolation, model mutation, or population weighting occurs here.
    """
    if not selected_sources:
        raise ValueError("Weather preparation requires at least one selected source")

    loaded_files = {}
    channels = []
    starts = []
    ends = []
    variables = []
    for key in selected_sources:
        source = config.weather[key]
        path = config.data_root / source.path
        if path not in loaded_files:
            loaded_files[path] = torch.load(path, map_location="cpu", weights_only=True)
        values = loaded_files[path]
        if isinstance(values, (tuple, list)):
            values = values[{"tas": 0, "tdew": 1}[source.variable]]
        if values.ndim != 2:
            raise ValueError(f"Weather source {key!r} must select a two-dimensional daily tensor")
        channels.append(values.float())
        starts.append(source.start_date)
        ends.append(source.start_date + timedelta(days=values.shape[-1] - 1))
        variables.append(source.variable)

    start_date = max(starts)
    end_date = min(ends)
    if end_date < start_date:
        raise ValueError("Selected weather sources have no common daily coverage")
    day_count = (end_date - start_date).days + 1
    aligned = []
    for values, source_start in zip(channels, starts):
        start_index = (start_date - source_start).days
        aligned.append(values[:, start_index:start_index + day_count])
    return WeatherData(
        values=torch.stack(aligned, dim=1),
        variables=tuple(variables),
        start_date=start_date,
    )
