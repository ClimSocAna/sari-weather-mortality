from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

BASE_DIR = Path(__file__).parent.parent

WEATHER_DIR = BASE_DIR / 'data' / 'weather'
POPULATION_DIR = BASE_DIR / 'data' / 'population'
DEATH_DIR = BASE_DIR / 'data' / 'death'
DEFAULT_SARI_FILENAME = "SARI-Hospitalisierungsinzidenz_20260702.tsv"
SARI_FILE = BASE_DIR / 'data' / 'sari' / DEFAULT_SARI_FILENAME
WEATHER_FILE_CATALOG_PATH = BASE_DIR / 'model' / 'weather_file_catalog.yaml'
LANDKREIS_DATA_PATH: Path | None = None  # Optional .pt source used by population_data stage 2 pipeline


# Model Architecture Constants
AGE_GROUP_COUNT = 15
AGE_GENDER_CHANNEL_COUNT = 30
WEEKDAY_CORRECTION_SIZE = 6
STATE_CORRECTION_SIZE = 16
GENDER_COUNT = 2

# COVID Exclusion Periods (days and weeks)
COVID_FREE_DAYS = 5 * 365
COVID_FREE_WEEKS = 5 * 52

# Training Hyperparameters
LEARNING_RATE = 0.003
SCHEDULER_PATIENCE = 100
GRADIENT_CLIP_VALUE = 1.0
EARLY_STOPPING_PATIENCE = 500
EARLY_STOPPING_MIN_DELTA = 0.0
TRAINER_MAX_EPOCHS = 5000
NUM_SANITY_VAL_STEPS = 0
TEMPERATURE_NOISE_STD = 0.5
SCHEDULER_WARMUP_PROXY_LOSS = 1e8
BASELINE_EVAL_ITERATIONS = 5
WEIGHT_DECAY = 0.01

# Training Schedule (epochs)
WARMUP_EPOCHS = 300
BASELINE_ADAPTATION_START_EPOCH = 200
BASELINE_ADAPTATION_UPDATE_INTERVAL = 50
STATE_CORRECTION_START_EPOCH = 300
STATE_CORRECTION_UPDATE_INTERVAL = 10
EARLY_STOPPING_START_EPOCH = 400

# Baseline Calculation Constants
BASELINE_WINDOW_WEEKS = 52
TRIMMED_AVERAGE_TRIM_SIZE = 10
BOTTOM_K_SIZE = 10
MEDIAN_FILTER_KERNEL = 3
EWMA_HALFLIFE = 4.0
DAYS_PER_WEEK = 7

# Population Data Constants
DEATH_DATA_START_DAY = 11 * 366
DEATH_DATA_START_WEEK = 11 * 53

# Training loop step defaults
TRAIN_STEPS_PER_EPOCH = 10
VAL_STEPS_PER_EPOCH = 1
DEFAULT_DATALOADER_WORKERS = 0

# Baseline/adaptation method names
STATIC_BASELINE_METHOD = "bottom_10"
ADAPTATION_METHODS = ("median_ewma", "median_ewma_d", "trimmed_avg")
SUPPORTED_BASELINE_MODES = (STATIC_BASELINE_METHOD, *ADAPTATION_METHODS)


def get_global_config_constants() -> dict[str, int | float | str | bool | None]:
    """Return serializable module-level constants for experiment logging."""
    constants: dict[str, int | float | str | bool | None] = {}
    for key, value in globals().items():
        if not key.isupper():
            continue
        if isinstance(value, (bool, int, float, str)):
            constants[key] = value
        elif isinstance(value, Path):
            constants[key] = str(value)
        elif value is None:
            constants[key] = None
    return constants

BaselineMethod = Literal[
    "bottom_10",
    "median_ewma",
    "median_ewma_d",
    "trimmed_avg",
]

WeatherVariable = Literal["tas", "tdew", "twb", "tas_min"]
WeatherDataKind = Literal["pop_average", "area_mean"]
ModelName = str
ConfigValue = bool | int | float | str | list[str] | dict[str, str] | None
WeatherCatalog = dict[str, dict[str, str | list[str]]]


def _load_catalog(file_path: Path) -> WeatherCatalog:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required weather YAML file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()
        data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {file_path}: expected mapping at root")
    return data


def _normalize_weather_variables(values: list[str] | tuple[str, ...]) -> list[str]:
    canonical_order = ["tas", "tdew", "twb", "tas_min"]
    valid = set(canonical_order)
    normalized: list[str] = []

    for value in values:
        if value not in valid:
            raise ValueError(f"Unsupported weather variable: {value}")
        if value in normalized:
            raise ValueError(f"Duplicate weather variable: {value}")
        normalized.append(value)

    normalized.sort(key=canonical_order.index)
    if not normalized:
        raise ValueError("weather_variables must contain at least one variable")
    return normalized


@dataclass
class DataWindowConfig:
    start: str = "2014-10-13"
    end: str = "2024-12-29"

    def as_list(self) -> list[str]:
        return [self.start, self.end]

    def to_dict(self) -> dict[str, str]:
        return {
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "DataWindowConfig":
        return cls(
            start=data["start"],
            end=data["end"],
        )


@dataclass
class HeatMortalityConfig:
    """Configuration for the heat mortality model and data pipeline.

    Hyperparameters:
        exclude_t: If True, removes temperature input effect during training/inference.
        include_sari: If True, includes SARI signal branch.
        include_covid: If True, includes COVID signal branch.
        baseline_adaptation: If True, periodically updates baseline mortality rate.
        adaptation_method: Baseline adaptation method (see BaselineMethod).
        date_range_stage1: Date window for stage-1 training.
        date_range_stage2: Date window for stage-2 training.
        in_channel: Number of weather input channels consumed by the temperature branch.
        out_channel: Output channel count before reshaping to age/sex dimensions.
        kernel_T_days: Temporal kernel size (days) for weather convolution.
        kernel_sari_days: Temporal kernel size (days) for SARI/COVID convolutions.
        val_split: Fraction of weeks reserved for validation.
        hidden_dim_init: Hidden channel width of the temperature branch.
        population_interpolation: If True, interpolate yearly population to daily values.
        weekday_correction: If True, learn weekday correction parameters.
        state_correction: If True, learn state-level correction parameters.
        correct_training_mean: If True, re-scale train-step predictions to match observed global mean counts.
        temperature_attribution_reference_c: Reference temperature in Celsius for temperature-attributable mortality.
            If None, use the empirical minimum fitted temperature factor as the attribution reference.
        sari_filename: SARI source filename under data/sari.
        weather_variables: Weather variables to include (subset of tas, tdew, twb).
        weather_data_kind: Aggregation variant of weather tensors (pop_average or area_mean).
        weather_files: Resolved weather tensor filenames under data/weather.
        stage: Training stage (1 or 2); 1=baseline training, 2=fine-tune on Landkreis data.
        death_daily_start_stage1: First date represented by stage-1 daily death labels.
        death_weekly_start_stage1: First date represented by stage-1 weekly death labels.
        death_daily_start_stage2: First date represented by stage-2 daily death labels.
        checkpoint_path: Path to Stage 1 checkpoint for loading weights during Stage 2 fine-tuning.
        model_name: Model class name used for training and YAML-based reconstruction.
    """
    exclude_t: bool = False
    include_sari: bool = True
    include_covid: bool = True
    baseline_adaptation: bool = True
    adaptation_method: BaselineMethod = "median_ewma"
    date_range_stage1: DataWindowConfig = field(default_factory=DataWindowConfig)
    date_range_stage2: DataWindowConfig = field(default_factory=lambda: DataWindowConfig(start="2022-01-03", end="2024-12-29"))
    in_channel: int = 1
    out_channel: int = 30
    kernel_T_days: int = 6
    kernel_sari_days: int = 10
    val_split: float = 0.25
    hidden_dim_init: int = 128
    population_interpolation: bool = True
    weekday_correction: bool = False
    state_correction: bool = False
    correct_training_mean: bool = True
    temperature_attribution_reference_c: float | None = None
    weather_variables: list[WeatherVariable] = field(default_factory=lambda: ["tas", "tdew"])
    weather_data_kind: WeatherDataKind = "pop_average"
    weather_files: list[str] = field(default_factory=list)
    sari_filename: str = DEFAULT_SARI_FILENAME
    stage: int = 1
    death_daily_start_stage1: str = "2011-01-01"
    death_weekly_start_stage1: str = "2011-01-03"
    death_daily_start_stage2: str = "2022-01-01"
    checkpoint_path: Path | None = None
    model_name: ModelName = "HeatMortality_EXP"

    def get_active_date_range(self) -> DataWindowConfig:
        if self.stage == 1:
            return self.date_range_stage1
        if self.stage == 2:
            return self.date_range_stage2
        raise ValueError(f"Unsupported training stage: {self.stage}")

    def get_death_label_start_dates(self) -> tuple[str, str | None]:
        if self.stage == 1:
            return self.death_daily_start_stage1, self.death_weekly_start_stage1
        if self.stage == 2:
            return self.death_daily_start_stage2, None
        raise ValueError(f"Unsupported training stage: {self.stage}")

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        if self.include_covid:
            self.include_sari = True

        if self.weather_data_kind not in ("pop_average", "area_mean"):
            raise ValueError(f"Unsupported weather_data_kind: {self.weather_data_kind}")

        self.weather_variables = _normalize_weather_variables(self.weather_variables)

        if self.weather_files:
            self.weather_files = [str(name) for name in self.weather_files]
            return

        catalog_data = _load_catalog(WEATHER_FILE_CATALOG_PATH)
        kind_catalog = catalog_data.get(self.weather_data_kind)
        if not isinstance(kind_catalog, dict):
            raise ValueError(f"Missing weather catalog entry for kind: {self.weather_data_kind}")

        selected_files: list[str] = []
        for variable in self.weather_variables:
            filename = kind_catalog.get(variable)
            if filename is None:
                raise ValueError(
                    f"Weather variable '{variable}' is not available for weather_data_kind '{self.weather_data_kind}'"
                )
            if isinstance(filename, list):
                filename = filename[0]
            selected_files.append(filename)

        self.weather_files = [str(name) for name in selected_files]

    def to_dict(self) -> dict[str, ConfigValue]:
        return {
            "exclude_t": self.exclude_t,
            "include_sari": self.include_sari,
            "include_covid": self.include_covid,
            "baseline_adaptation": self.baseline_adaptation,
            "adaptation_method": self.adaptation_method,
            "date_range_stage1": self.date_range_stage1.to_dict(),
            "date_range_stage2": self.date_range_stage2.to_dict(),
            "in_channel": self.in_channel,
            "out_channel": self.out_channel,
            "kernel_T_days": self.kernel_T_days,
            "kernel_sari_days": self.kernel_sari_days,
            "val_split": self.val_split,
            "hidden_dim_init": self.hidden_dim_init,
            "population_interpolation": self.population_interpolation,
            "weekday_correction": self.weekday_correction,
            "state_correction": self.state_correction,
            "correct_training_mean": self.correct_training_mean,
            "temperature_attribution_reference_c": self.temperature_attribution_reference_c,
            "weather_variables": self.weather_variables,
            "weather_data_kind": self.weather_data_kind,
            "weather_files": self.weather_files,
            "sari_filename": self.sari_filename,
            "stage": self.stage,
            "death_daily_start_stage1": self.death_daily_start_stage1,
            "death_weekly_start_stage1": self.death_weekly_start_stage1,
            "death_daily_start_stage2": self.death_daily_start_stage2,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, ConfigValue]) -> "HeatMortalityConfig":
        data = dict(data)

        if "date_range_stage1" in data and data["date_range_stage1"] is not None:
            data["date_range_stage1"] = DataWindowConfig.from_dict(data["date_range_stage1"])
        else:
            data["date_range_stage1"] = DataWindowConfig()

        if "date_range_stage2" in data and data["date_range_stage2"] is not None:
            data["date_range_stage2"] = DataWindowConfig.from_dict(data["date_range_stage2"])
        else:
            data["date_range_stage2"] = DataWindowConfig(start="2022-01-03", end=data["date_range_stage1"].end)

        if data.get("checkpoint_path"):
            data["checkpoint_path"] = Path(data["checkpoint_path"])

        data.setdefault("model_name", "HeatMortality_EXP")
        return cls(**data)
