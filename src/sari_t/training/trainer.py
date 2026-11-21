"""Compose mortality training steps from independent models and shared operations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..aggregation import MortalityAggregates, aggregate_mortality
from ..baseline import calculate_updated_baseline_rate
from ..data.interfaces import MortalityLabels, SourceData
from ..data.training_inputs import select_mortality_labels
from ..mean_alignment import apply_mean_alignment, estimate_mean_alignment
from ..metrics import calculate_national_mse
from ..models.exponential import ExponentialMortalityModel
from ..models.interfaces import (
    MortalityModelInputs,
    MortalityPredictionInputs,
    MortalityPredictionOutputs,
)
from ..models.predictor import MortalityPredictor
from ..reference import reference_temperature_deaths, reference_temperature_effect
from ..run.checkpoint import (
    decode_run_config,
    final_checkpoint_path,
    load_checkpoint,
    restore_predictor,
    restore_training_state,
    save_run_config,
    save_training_checkpoint,
    save_final_model,
)
from ..run.config import RunConfig
from ..run.preparation import PreparedRun, prepare_run
from .control import (
    build_optimizer_and_scheduler,
    corrections_trainable,
    raw_loss_weight,
    should_step_scheduler,
    should_update_baseline,
    should_update_state,
)
from .loss import calculate_loss_components


@dataclass
class TrainingStep:
    """One actual forward and its objectives, before any optimizer update.

    predictions cover the full period for later baseline updates.
    training_aggregates use mean alignment when enabled; validation stays raw.
    loss_components contain raw terms with the mixed total. optimization_loss
    instead uses the raw prefix-limited loss on state-update epochs.
    """

    predictions: MortalityPredictionOutputs
    alignment_factor: torch.Tensor | None
    loss_components: dict[str, torch.Tensor]
    optimization_loss: torch.Tensor
    training_aggregates: MortalityAggregates
    validation_aggregates: MortalityAggregates | None

    def detach(self) -> TrainingStep:
        """Return a reporting result without autograd links or tensor copies.

        Use after backward when retaining results, not for optimization.
        """
        return TrainingStep(
            predictions=self.predictions.detach(),
            alignment_factor=(
                self.alignment_factor.detach() if self.alignment_factor is not None else None
            ),
            loss_components={name: value.detach() for name, value in self.loss_components.items()},
            optimization_loss=self.optimization_loss.detach(),
            training_aggregates=self.training_aggregates.detach(),
            validation_aggregates=(
                self.validation_aggregates.detach()
                if self.validation_aggregates is not None else None
            ),
        )


@dataclass
class TrainingResult:
    """Final training state; checkpoint_path is the lightweight completed model.

    resume_checkpoint_path retains full epoch-boundary continuation state.
    Evaluation data and baselines are prepared separately.
    """

    predictor: MortalityPredictor
    prepared_run: PreparedRun
    checkpoint_path: Path
    resume_checkpoint_path: Path | None
    metrics_path: Path


def fit(
    config: RunConfig,
    run_directory: Path,
    *,
    member_id: int = 1,
    device: torch.device | str = "cpu",
    sources: SourceData | None = None,
) -> TrainingResult | None:
    """Fit a numbered member, saving final_NNN.pt after normal completion.

    Members share run_directory/config.yaml; select a distinct member_id per member.
    last_NNN.pt retains full resume state each epoch. Training retains the shared
    full period and cached validation policy. An existing final checkpoint
    returns None before data preparation.
    """
    final_path = final_checkpoint_path(run_directory, member_id)
    if final_path.is_file():
        return None
    predictor, prepared_run = initialize_training(prepare_run(config, sources=sources), device)
    optimizer, scheduler = build_optimizer_and_scheduler(predictor, config.training_config)
    return _fit_epochs(
        predictor, prepared_run, optimizer, scheduler, run_directory,
        member_id=member_id,
        epoch=-1, global_step=0, best_loss=float("inf"), epochs_without_improvement=0,
    )


def resume_training(
    checkpoint_path: Path,
    *,
    run_directory: Path | None = None,
    device: torch.device | str = "cpu",
    sources: SourceData | None = None,
) -> TrainingResult | None:
    """Continue at the next epoch using the checkpoint's configuration and state.

    Supply the full last_NNN.pt checkpoint, not the lightweight final_NNN.pt model.
    By default continue in the same directory. When branching from an older
    snapshot, choose a new directory to keep metric histories separate.
    Returns None if the destination already has a final checkpoint.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    member_id = checkpoint.get("member_id", 1)
    default_directory = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    run_directory = default_directory if run_directory is None else run_directory
    final_path = final_checkpoint_path(run_directory, member_id)
    if final_path.is_file():
        return None
    config = decode_run_config(checkpoint["run_config"])
    prepared_run = prepare_run(config, sources=sources).to(device)
    predictor = restore_predictor(checkpoint, device)
    optimizer, scheduler = build_optimizer_and_scheduler(predictor, config.training_config)
    progress = restore_training_state(checkpoint, prepared_run, optimizer, scheduler)
    del checkpoint
    return _fit_epochs(
        predictor, prepared_run, optimizer, scheduler,
        run_directory,
        member_id=member_id,
        epoch=int(progress["epoch"]),
        global_step=int(progress["global_step"]),
        best_loss=float(progress["best_loss"]),
        epochs_without_improvement=int(progress["epochs_without_improvement"]),
    )


def _fit_epochs(
    predictor: MortalityPredictor,
    prepared_run: PreparedRun,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    run_directory: Path,
    *,
    member_id: int,
    epoch: int,
    global_step: int,
    best_loss: float,
    epochs_without_improvement: int,
) -> TrainingResult:
    config = prepared_run.config.training_config
    number = f"{member_id:03d}"
    save_run_config(run_directory, prepared_run.config)
    checkpoint_path = None
    metrics_path = run_directory / "metrics" / f"metrics_{number}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    training_labels = select_mortality_labels(prepared_run.data.labels, prepared_run.training_period)
    validation_labels = (
        select_mortality_labels(prepared_run.data.labels, prepared_run.validation_period)
        if prepared_run.validation_period is not None else None
    )
    for epoch in range(epoch + 1, config.max_epochs):
        training_reports = []
        validation_aggregates = None
        for step_index in range(config.train_steps_per_epoch):
            step = train_step(predictor, prepared_run, optimizer, scheduler, epoch, step_index)
            report = _scalar_report(
                step.loss_components, step.training_aggregates, training_labels, "train",
            )
            report["train_optimization_loss"] = step.optimization_loss.detach().item()
            training_reports.append(report)
            if step.validation_aggregates is not None:
                validation_aggregates = step.validation_aggregates.detach()
            global_step += 1
            del step

        metrics = _mean_reports(training_reports)
        if validation_aggregates is not None and validation_labels is not None:
            validation_reports = []
            with torch.no_grad():
                for _ in range(config.validation_steps_per_epoch):
                    losses = calculate_loss_components(validation_aggregates, validation_labels)
                    validation_reports.append(_scalar_report(
                        losses, validation_aggregates, validation_labels, "val",
                    ))
            if validation_reports:
                metrics.update(_mean_reports(validation_reports))
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]

        improved = metrics["train_loss"] < best_loss - config.early_stopping_min_delta
        if improved:
            best_loss = metrics["train_loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint_path = save_training_checkpoint(
            run_directory, member_id, predictor, prepared_run, optimizer, scheduler,
            epoch=epoch, global_step=global_step, best_loss=best_loss,
            epochs_without_improvement=epochs_without_improvement,
        )
        record = json.dumps({"epoch": epoch, "global_step": global_step, **metrics})
        with metrics_path.open("a") as stream:
            stream.write(record + "\n")
        print(f"member={member_id:03d} {record}", flush=True)
        if not improved and epochs_without_improvement >= config.early_stopping_patience:
            break
    final_path = save_final_model(run_directory, member_id, predictor, prepared_run.config)
    return TrainingResult(
        predictor=predictor,
        prepared_run=prepared_run,
        checkpoint_path=final_path,
        resume_checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )


@torch.no_grad()
def _scalar_report(
    losses: dict[str, torch.Tensor],
    aggregates: MortalityAggregates,
    labels: MortalityLabels,
    prefix: str,
) -> dict[str, float]:
    values = {
        (f"{prefix}_loss" if name == "total" else f"{prefix}_loss_{name}"): value.detach().item()
        for name, value in losses.items()
    }
    values.update({
        f"{prefix}_{name}": value.item()
        for name, value in calculate_national_mse(aggregates, labels).items()
    })
    return values


def _mean_reports(reports: list[dict[str, float]]) -> dict[str, float]:
    return {name: sum(report[name] for report in reports) / len(reports) for name in reports[0]}


def initialize_training(
    prepared_run: PreparedRun,
    device: torch.device | str = "cpu",
) -> tuple[MortalityPredictor, PreparedRun]:
    """Construct the model and place prepared tensors before optimizer setup.

    Return a new prepared container without modifying the CPU preparation.
    Lazy layers initialize on the first actual forward; no dummy pass updates
    BatchNorm or consumes an extra weather perturbation during setup.
    """
    predictor = MortalityPredictor(
        ExponentialMortalityModel(prepared_run.config.model_config),
        prepared_run.config.predictor_config,
    ).to(device=device, dtype=torch.float32)
    return predictor, prepared_run.to(device)


def train_step(
    predictor: MortalityPredictor,
    prepared_run: PreparedRun,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    step_index: int,
) -> TrainingStep:
    """Optimize once and apply scheduled updates using the same actual forward.

    Model and prepared tensors must already share a device. Returned predictions
    and reporting losses describe the pre-update forward, as in the original
    cached training/validation procedure. Baseline updates cover the full period.
    """
    config = prepared_run.config.training_config
    predictor.train()
    for parameter in predictor.parameters():
        parameter.requires_grad = True
    train_corrections = corrections_trainable(epoch, config)
    predictor.weekday_correction.requires_grad = (
        predictor.config.weekday_correction and train_corrections
    )
    predictor.state_correction.requires_grad = (
        isinstance(predictor.state_correction, torch.nn.Parameter) and train_corrections
    )
    step = calculate_training_step(predictor, prepared_run, epoch)

    updated_baseline = None
    if should_update_baseline(epoch, config):
        with torch.no_grad():
            reference_effect = reference_temperature_effect(
                predictor.effect_model,
                prepared_run.data.model_inputs,
                prepared_run.weather_variables,
                prepared_run.config.reference_config,
                minimum_temperature_effect=cast(
                    torch.Tensor, step.predictions.minimum_temperature_effect,
                ),
            )
            factors = predictor.correction_factors(
                step.predictions.total_deaths.shape[0],
            )
            reference_deaths = reference_temperature_deaths(
                reference_effect,
                MortalityPredictionInputs(
                    model_inputs=prepared_run.data.model_inputs,
                    baseline_rate=prepared_run.baseline_rate,
                    population=prepared_run.data.population,
                ),
                factors,
            )
            updated_baseline = calculate_updated_baseline_rate(
                prepared_run.data.labels.weekly_de_age_sex,
                step.predictions,
                reference_deaths,
                prepared_run.data.population,
                prepared_run.config.adaptive_baseline_config,
            )
            if not (torch.isfinite(updated_baseline) & (updated_baseline > 0)).all():
                print(
                    f"Rejected adaptive baseline at epoch {epoch}, step {step_index}: "
                    "candidate contains non-finite or non-positive rates. Keeping previous baseline.",
                    flush=True,
                )
                updated_baseline = None

    optimizer.zero_grad()
    step.optimization_loss.backward()
    if not should_update_state(epoch, config):
        predictor.state_correction.grad = None
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), config.gradient_clip_norm)
    optimizer.step()
    with torch.no_grad():
        predictor.state_correction.div_(predictor.state_correction[0].clone())

    if updated_baseline is not None:
        prepared_run.baseline_rate = updated_baseline
    if should_step_scheduler(epoch, step_index, config):
        scheduler.step(step.loss_components["total"].detach())
    return step


def calculate_training_step(
    predictor: MortalityPredictor,
    prepared_run: PreparedRun,
    epoch: int,
) -> TrainingStep:
    """Calculate one full-window forward, training loss and cached validation.

    The caller sets training mode and correction gradient flags and supplies
    tensors on the predictor's device. This performs no optimization, baseline
    adaptation or mode changes. Excluding temperature replaces weather by zeros,
    retaining the temperature branch's learned baseline effect. Noise covers
    training dates and their lag history only; cached validation may share that
    noisy history at the boundary, but later weather dates are unperturbed.
    """
    config = prepared_run.config.training_config
    covariates = prepared_run.data.model_inputs
    training_days = (
        prepared_run.training_period.end_date - prepared_run.training_period.start_date
    ).days + 1
    if config.exclude_temperature:
        temperature = torch.zeros_like(covariates.temperature)
    else:
        temperature = covariates.temperature.clone()
        history_days = temperature.shape[-1] - prepared_run.data.population.shape[0]
        training_weather = temperature[..., :history_days + training_days]
        training_weather.add_(torch.randn_like(training_weather) * config.temperature_noise_std)
    inputs = MortalityPredictionInputs(
        model_inputs=MortalityModelInputs(temperature, covariates.sari, covariates.covid),
        baseline_rate=prepared_run.baseline_rate,
        population=prepared_run.data.population,
    )
    predictions = predictor(inputs)
    raw_aggregates = aggregate_mortality(predictions.total_deaths)
    training_weeks = training_days // 7
    raw_training = _slice_aggregates(raw_aggregates, slice(0, training_days), slice(0, training_weeks))
    labels = select_mortality_labels(prepared_run.data.labels, prepared_run.training_period)
    loss_components = calculate_loss_components(raw_training, labels)

    alignment_factor = None
    training_aggregates = raw_training
    if config.align_training_mean:
        alignment_factor = estimate_mean_alignment(
            predictions, labels.weekly_de_age_sex, max_weeks=config.alignment_max_weeks,
        )
        aligned_total = apply_mean_alignment(predictions.total_deaths[:training_days], alignment_factor)
        training_aggregates = aggregate_mortality(aligned_total)
        aligned_loss = calculate_loss_components(training_aggregates, labels)
        weight = raw_loss_weight(epoch, config)
        loss_components["total"] = (
            weight * loss_components["total"] + (1 - weight) * aligned_loss["total"]
        )

    optimization_loss = loss_components["total"]
    if should_update_state(epoch, config):
        optimization_loss = calculate_loss_components(
            raw_training, labels,
            daily_limit=config.state_loss_days, weekly_limit=config.state_loss_weeks,
        )["total"]

    validation_aggregates = None
    if prepared_run.validation_period is not None:
        validation_aggregates = _slice_aggregates(
            raw_aggregates, slice(training_days, None), slice(training_weeks, None),
        )
    return TrainingStep(
        predictions=predictions,
        alignment_factor=alignment_factor,
        loss_components=loss_components,
        optimization_loss=optimization_loss,
        training_aggregates=training_aggregates,
        validation_aggregates=validation_aggregates,
    )


def _slice_aggregates(
    values: MortalityAggregates,
    days: slice,
    weeks: slice,
) -> MortalityAggregates:
    return MortalityAggregates(
        daily_land=values.daily_land[days],
        weekly_de_age_sex=values.weekly_de_age_sex[weeks],
        weekly_land_coarse_age_sex=values.weekly_land_coarse_age_sex[weeks],
    )
