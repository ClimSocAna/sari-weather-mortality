"""Explicit source settings without file access or model construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class WeatherSource:
    """One available daily weather dataset, addressed by its inventory key.

    variable selects tas or tdew within paired pipeline artifacts. Stored tensors
    use the pipeline's (day, Kreis) layout.
    """

    variable: str
    path: Path
    start_date: date


@dataclass(frozen=True)
class DataConfig:
    """Available sources and preprocessing settings, without feature selection.

    Source paths are absolute or relative to data_root. Weather keys identify
    datasets, so variants of the same physical variable can coexist. Run preparation
    selects source keys and channel order; dictionary order is irrelevant.
    Constructing this configuration performs no I/O.
    """

    data_root: Path
    weather: dict[str, WeatherSource]
    population_path: Path
    mortality_path: Path
    surveillance_path: Path
    surveillance_start_date: date
    mortality_first_year: int = 2000
    population_start_year: int = 2011
    population_interpolation: bool = True
    population_tail_padding_days: int = 1000
    mortality_missing_value: int = -1
    preprocessing_version: str = "stage1-v1"
