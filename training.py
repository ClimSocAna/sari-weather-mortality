#!/usr/bin/env python3
"""Command-line entry point for training mortality models."""

from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from model import MODEL_REGISTRY, get_model_class_by_name
from model.config import (
    ADAPTATION_METHODS,
    DEFAULT_DATALOADER_WORKERS,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    NUM_SANITY_VAL_STEPS,
    STATIC_BASELINE_METHOD,
    TRAINER_MAX_EPOCHS,
    TRAIN_STEPS_PER_EPOCH,
    VAL_STEPS_PER_EPOCH,
    DataWindowConfig,
    HeatMortalityConfig,
)
from model.dataset import DummyDataset

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training_outputs"
WEATHER_VARIABLES = ("tas", "tdew", "twb", "tas_min")


def parse_weather_variables(value: str) -> list[str]:
    """Parse a comma-separated list of weather variables for argparse."""
    variables = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in variables if item not in WEATHER_VARIABLES]
    if not variables:
        raise ArgumentTypeError("at least one weather variable is required")
    if unknown:
        raise ArgumentTypeError(f"unsupported weather variable(s): {', '.join(unknown)}")
    if len(set(variables)) != len(variables):
        raise ArgumentTypeError("weather variables must not be repeated")
    return variables


def build_model_config(
    kernel_t_days: int = 6,
    kernel_sari_days: int = 10,
    exclude_t: bool = False,
    include_sari: bool = True,
    include_covid: bool = True,
    baseline_adaptation: bool = True,
    adaptation_method: str = "median_ewma",
    val_split: float = 0.25,
    weather_data_kind: str = "pop_average",
    weather_variables: list[str] | None = None,
    model_name: str = "HeatMortality_EXP",
    stage: int = 1,
    checkpoint_path: Path | None = None,
    correct_training_mean: bool = True,
    temperature_attribution_reference_c: float | None = None,
) -> HeatMortalityConfig:
    """Build the model configuration represented by the CLI options."""
    baseline_method = adaptation_method if baseline_adaptation else STATIC_BASELINE_METHOD

    return HeatMortalityConfig(
        model_name=model_name,
        population_interpolation=True,
        kernel_T_days=kernel_t_days,
        kernel_sari_days=kernel_sari_days,
        exclude_t=exclude_t,
        include_sari=include_sari,
        include_covid=include_covid,
        val_split=val_split,
        weekday_correction=True,
        state_correction=True,
        correct_training_mean=correct_training_mean,
        temperature_attribution_reference_c=temperature_attribution_reference_c,
        hidden_dim_init=128,
        baseline_adaptation=baseline_adaptation,
        adaptation_method=baseline_method,
        date_range_stage1=DataWindowConfig(start="2014-10-13", end="2024-12-29"),
        date_range_stage2=DataWindowConfig(start="2022-01-03", end="2024-12-29"),
        weather_data_kind=weather_data_kind,
        weather_variables=weather_variables or ["tas", "tdew"],
        stage=stage,
        checkpoint_path=checkpoint_path,
    )


def build_run_name(model_config: HeatMortalityConfig) -> str:
    """Return a descriptive directory name for a training configuration."""
    terms = [model_config.model_name]
    if not model_config.exclude_t:
        terms.append(f"t_{model_config.kernel_T_days}")
    if model_config.include_sari:
        terms.append(f"sari_{model_config.kernel_sari_days}")
    if model_config.include_covid:
        terms.append("covid")
    if model_config.baseline_adaptation:
        terms.append(model_config.adaptation_method)
    if model_config.temperature_attribution_reference_c is None:
        terms.append("temp_ref_min")
    else:
        terms.append(f"temp_ref_{model_config.temperature_attribution_reference_c:g}c")
    terms.append("corr_on" if model_config.correct_training_mean else "corr_off")
    terms.append(model_config.weather_data_kind)
    terms.append("-".join(model_config.weather_variables))
    terms.append(f"stage{model_config.stage}")
    return "_".join(terms)


def train(
    *,
    model_config: HeatMortalityConfig,
    version: int,
    seed: int | None,
    output_dir: Path,
    accelerator: str,
    devices: int,
    max_epochs: int,
    enable_progress_bar: bool,
) -> None:
    """Construct the model and run one Lightning training replicate."""
    if version < 0:
        raise ValueError("version must be non-negative")
    if devices < 1:
        raise ValueError("devices must be at least 1")
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1")
    if model_config.stage == 2 and model_config.checkpoint_path is None:
        raise ValueError("--checkpoint-path is required with --stage 2")

    pl.seed_everything(version if seed is None else seed, workers=True)

    train_loader = torch.utils.data.DataLoader(
        DummyDataset(TRAIN_STEPS_PER_EPOCH),
        batch_size=None,
        num_workers=DEFAULT_DATALOADER_WORKERS,
    )
    val_loader = torch.utils.data.DataLoader(
        DummyDataset(VAL_STEPS_PER_EPOCH),
        batch_size=None,
        num_workers=DEFAULT_DATALOADER_WORKERS,
    )

    run_name = build_run_name(model_config)
    model_class = get_model_class_by_name(model_config.model_name)
    model = model_class(model_config)

    checkpoint_dir = output_dir / "checkpoints" / run_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=(
            f"v{version}-epoch={{epoch}}-"
            "train_daily_mse={train_daily_mse:.2E}-"
            "val_daily_mse={val_daily_mse:.2E}"
        ),
        auto_insert_metric_name=False,
    )
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        checkpoint_callback,
        EarlyStopping(
            monitor="train_loss",
            min_delta=EARLY_STOPPING_MIN_DELTA,
            patience=EARLY_STOPPING_PATIENCE,
            mode="min",
        ),
    ]
    logger = CSVLogger(
        save_dir=output_dir / "logs",
        name=run_name,
        version=f"v{version}",
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=output_dir,
        log_every_n_steps=1,
        num_sanity_val_steps=NUM_SANITY_VAL_STEPS,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train one replicate of the mortality attribution model.",
    )
    parser.add_argument("--exclude-t", action=BooleanOptionalAction, default=False)
    parser.add_argument("--include-sari", action=BooleanOptionalAction, default=True)
    parser.add_argument("--include-covid", action=BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-adaptation", action=BooleanOptionalAction, default=True)
    parser.add_argument("--adaptation-method", default="median_ewma", choices=ADAPTATION_METHODS)
    parser.add_argument("--kernel-t-days", type=int, default=6)
    parser.add_argument("--kernel-sari-days", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.25)
    parser.add_argument("--weather-data-kind", default="pop_average", choices=("pop_average", "area_mean"))
    parser.add_argument(
        "--weather-variables",
        type=parse_weather_variables,
        default=["tas", "tdew"],
        metavar="VARIABLES",
        help="comma-separated variables chosen from tas,tdew,twb,tas_min",
    )
    parser.add_argument("--version", type=int, default=0, help="replicate identifier; also used as the default seed")
    parser.add_argument("--seed", type=int, default=None, help="random seed; defaults to the replicate version")
    parser.add_argument("--model-name", default="HeatMortality_EXP", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2))
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--correct-training-mean", action=BooleanOptionalAction, default=True)
    parser.add_argument(
        "--temperature-attribution-reference-c",
        type=float,
        default=None,
        metavar="CELSIUS",
        help="omit to use the empirical minimum-risk temperature",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "mps"), default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=TRAINER_MAX_EPOCHS)
    parser.add_argument("--progress-bar", action=BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model_config = build_model_config(
        kernel_t_days=args.kernel_t_days,
        kernel_sari_days=args.kernel_sari_days,
        exclude_t=args.exclude_t,
        include_sari=args.include_sari,
        include_covid=args.include_covid,
        baseline_adaptation=args.baseline_adaptation,
        adaptation_method=args.adaptation_method,
        val_split=args.val_split,
        weather_data_kind=args.weather_data_kind,
        weather_variables=args.weather_variables,
        model_name=args.model_name,
        stage=args.stage,
        checkpoint_path=args.checkpoint_path,
        correct_training_mean=args.correct_training_mean,
        temperature_attribution_reference_c=args.temperature_attribution_reference_c,
    )
    train(
        model_config=model_config,
        version=args.version,
        seed=args.seed,
        output_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        enable_progress_bar=args.progress_bar,
    )


if __name__ == "__main__":
    main()
