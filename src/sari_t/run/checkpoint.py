"""Resolved run configuration and epoch-boundary training checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
import yaml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..baseline import BaselineConfig
from ..data.config import DataConfig, WeatherSource
from ..models.config import MortalityModelConfig, MortalityPredictorConfig, PredictionPeriod
from ..models.exponential import ExponentialMortalityModel
from ..models.predictor import MortalityPredictor
from ..reference import ReferenceConfig
from ..training.config import TrainingConfig
from .config import RunConfig
from .preparation import PreparedRun


CHECKPOINT_FORMAT_VERSION = 1


def encode_run_config(config: RunConfig) -> dict[str, Any]:
    """Encode all settings with absolute source paths and plain serialized values."""
    values = _plain_values(asdict(config))
    _resolve_data_paths(values["data_config"])
    return values


def decode_run_config(values: dict[str, Any]) -> RunConfig:
    """Reconstruct existing configuration dataclasses from saved settings."""
    data = dict(values["data_config"])
    for name in ("data_root", "population_path", "mortality_path", "surveillance_path"):
        data[name] = Path(data[name])
    data["weather"] = {
        key: WeatherSource(
            variable=source["variable"],
            path=Path(source["path"]),
            start_date=date.fromisoformat(source["start_date"]),
        )
        for key, source in data["weather"].items()
    }
    data["surveillance_start_date"] = date.fromisoformat(data["surveillance_start_date"])
    period = values["period"]
    return RunConfig(
        data_config=DataConfig(**data),
        selected_weather=tuple(values["selected_weather"]),
        period=PredictionPeriod(
            date.fromisoformat(period["start_date"]),
            date.fromisoformat(period["end_date"]),
        ),
        model_config=MortalityModelConfig(**values["model_config"]),
        predictor_config=MortalityPredictorConfig(**values["predictor_config"]),
        training_config=TrainingConfig(**values["training_config"]),
        initial_baseline_config=BaselineConfig(**values["initial_baseline_config"]),
        adaptive_baseline_config=BaselineConfig(**values["adaptive_baseline_config"]),
        reference_config=ReferenceConfig(**values["reference_config"]),
    )


def load_run_config(
    path: Path, *, setup: str | None = None, reference: str | None = None,
) -> RunConfig:
    """Resolve a named setup/reference onto shared YAML training settings.

    Setup overrides preserve unspecified shared fields. An explicit reference
    profile is applied afterward. Profile definitions are not saved as RunConfig.
    """
    path = path.expanduser().resolve()
    values = _plain_values(yaml.safe_load(path.read_text(encoding="utf-8")))
    setups = values.pop("setups", {})
    references = values.pop("references", {})
    if setup is not None:
        overrides = setups[setup]
        unknown = overrides.keys() - RunConfig.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown run settings in setup {setup!r}: {sorted(unknown)}")
        for section, override in overrides.items():
            if isinstance(override, dict):
                values[section].update(override)
            else:
                values[section] = override
    if reference is not None:
        values["reference_config"].update(references[reference])
    _resolve_data_paths(values["data_config"], base_directory=path.parent)
    return decode_run_config(values)


def _resolve_data_paths(data: dict[str, Any], base_directory: Path | None = None) -> None:
    """Normalize paths in a fresh YAML/serialization mapping before decoding/saving."""
    root = Path(data["data_root"]).expanduser()
    if base_directory is not None and not root.is_absolute():
        root = base_directory / root
    root = root.resolve()
    data["data_root"] = str(root)
    for name in ("population_path", "mortality_path", "surveillance_path"):
        data[name] = str((root / Path(data[name]).expanduser()).resolve())
    for source in data["weather"].values():
        source["path"] = str((root / Path(source["path"]).expanduser()).resolve())


def save_run_config(run_directory: Path, config: RunConfig) -> Path:
    """Publish the shared readable configuration once, including parallel sbatch jobs."""
    values = encode_run_config(config)
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / "config.yaml"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=run_directory, delete=False) as stream:
        yaml.safe_dump(values, stream, sort_keys=False, allow_unicode=True)
        temporary_path = Path(stream.name)
    try:
        os.link(temporary_path, path)
    except FileExistsError:
        if yaml.safe_load(path.read_text(encoding="utf-8")) != values:
            raise ValueError("Run directory already belongs to a different configuration")
    finally:
        temporary_path.unlink()
    return path


def save_training_checkpoint(
    run_directory: Path,
    member_id: int,
    predictor: MortalityPredictor,
    prepared_run: PreparedRun,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    *,
    epoch: int,
    global_step: int,
    best_loss: float,
    epochs_without_improvement: int,
) -> Path:
    """Save post-update state after a completed epoch, without source arrays.

    The checkpoint embeds authoritative settings; readable YAML is saved separately. Cached
    pre-update alignment/predictions are deliberately not saved as evaluated state.
    """
    config = encode_run_config(prepared_run.config)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "member_id": member_id,
        "run_config": config,
        "model_state_dict": _cpu_copy(predictor.state_dict()),
        "baseline_rate": _cpu_copy(prepared_run.baseline_rate),
        "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
        "scheduler_state_dict": _cpu_copy(scheduler.state_dict()),
        "progress": {
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = run_directory / "checkpoints" / f"last_{member_id:03d}.pt"
    _write_checkpoint(path, checkpoint)
    return path


def final_checkpoint_path(run_directory: Path, member_id: int) -> Path:
    return run_directory / "checkpoints" / f"final_{member_id:03d}.pt"


def save_final_model(
    run_directory: Path,
    member_id: int,
    predictor: MortalityPredictor,
    config: RunConfig,
) -> Path:
    """Save a completed model for inference, without training continuation state.

    Call only after successful fitting. Restore with restore_predictor; optimizer,
    baseline and RNG state remain exclusive to the full training checkpoint.
    """
    path = final_checkpoint_path(run_directory, member_id)
    _write_checkpoint(path, {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "member_id": member_id,
        "run_config": encode_run_config(config),
        "model_state_dict": _cpu_copy(predictor.state_dict()),
    }, overwrite=False)
    return path


def _write_checkpoint(path: Path, checkpoint: dict[str, Any], *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            torch.save(checkpoint, stream)
            stream.flush()
            if overwrite:
                temporary_path.replace(path)
            else:
                os.link(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read a portable snapshot; source data and models are not constructed."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise NotImplementedError(f"Unsupported checkpoint format: {checkpoint['format_version']}")
    return checkpoint


def restore_predictor(
    checkpoint: dict[str, Any],
    device: torch.device | str = "cpu",
) -> MortalityPredictor:
    """Restore learned parameters/BatchNorm without sources or a dummy forward.

    The caller chooses training/evaluation mode. This does not prepare baseline
    iterations, mean alignment, labels or evaluated predictions.
    """
    config = decode_run_config(checkpoint["run_config"])
    predictor = MortalityPredictor(
        ExponentialMortalityModel(config.model_config), config.predictor_config,
    )
    predictor.load_state_dict(checkpoint["model_state_dict"])
    return predictor.to(device=device, dtype=torch.float32)


def restore_training_state(
    checkpoint: dict[str, Any],
    prepared_run: PreparedRun,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
) -> dict[str, int | float]:
    """Restore continuation state after predictor/optimizer construction.

    Prepared data must come from the checkpoint's config and already be placed
    on the model device. Resume at progress['epoch'] + 1. Restore RNG last, after
    construction, so setup does not consume the continuation's random sequence.
    """
    prepared_run.baseline_rate = checkpoint["baseline_rate"].to(
        device=prepared_run.data.population.device, dtype=torch.float32,
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if checkpoint["cuda_rng_states"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    return dict(checkpoint["progress"])


def _plain_values(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _plain_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_values(item) for item in value]
    return value


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value
