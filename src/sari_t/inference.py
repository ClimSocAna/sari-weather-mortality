"""Saved-run evaluation with rebuilt baselines and inspectable mortality results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast
import warnings

import torch

from .aggregation import aggregate_mortality
from .baseline import calculate_updated_baseline_rate
from .data.config import DataConfig
from .data.interfaces import MortalityLabels, PreparedData, SourceData
from .data.training_inputs import select_mortality_labels
from .mean_alignment import apply_mean_alignment, estimate_mean_alignment
from .metrics import calculate_national_mse
from .models.config import MortalityModelConfig, PredictionPeriod
from .models.interfaces import MortalityModelInputs, MortalityPredictionInputs, MortalityPredictionOutputs
from .models.exponential import ExponentialMortalityModel
from .models.predictor import MortalityPredictor
from .reference import (
    build_reference_weather,
    dew_point_c,
    reference_temperature_deaths,
    reference_temperature_effect,
    relative_humidity_percent,
)
from .run.checkpoint import decode_run_config, encode_run_config, load_checkpoint
from .run.config import EvaluationConfig, RunConfig
from .run.preparation import prepare_run


@dataclass
class MemberResult:
    """Processed outputs for one member, with no model, baseline or alignment state."""

    model_config: MortalityModelConfig
    labels: MortalityLabels
    period: PredictionPeriod
    predictions: MortalityPredictionOutputs
    scenarios: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]

    def select_dates(self, start_date: date | str, end_date: date | str) -> MemberResult:
        """Return a complete-week subset with metrics for its existing predictions."""
        period = _select_period(self.period, start_date, end_date)
        start = (period.start_date - self.period.start_date).days
        days = (period.end_date - period.start_date).days + 1
        predictions = self.predictions.select_days(start, days)
        labels = select_mortality_labels(self.labels, period)
        return MemberResult(
            model_config=self.model_config, labels=labels, period=period,
            predictions=predictions,
            scenarios={name: value[start:start + days] for name, value in self.scenarios.items()},
            metrics=calculate_national_mse(aggregate_mortality(predictions.total_deaths), labels),
        )

    def to(self, device: torch.device | str) -> MemberResult:
        return MemberResult(
            model_config=self.model_config, labels=self.labels.to(device), period=self.period,
            predictions=self.predictions.to(device),
            scenarios={name: value.to(device) for name, value in self.scenarios.items()},
            metrics={name: value.to(device) for name, value in self.metrics.items()},
        )


class InferenceModel:
    """Reusable model/data workspace; loading weights clears member-specific state.

    Supply all scenarios to evaluate after load_weights. Output objects contain
    processed counts only; changing weights never changes them.
    """

    def __init__(
        self, config: RunConfig, *, data_config: DataConfig | None = None,
        period: PredictionPeriod | None = None, evaluation_config: EvaluationConfig | None = None,
        device: torch.device | str = "cpu", sources: SourceData | None = None,
    ) -> None:
        self._evaluation_config = evaluation_config or EvaluationConfig()
        self._config = config
        self._calibration_period = self._get_calibration_period(config)
        self._device = torch.device(device)
        prepared = prepare_run(
            replace(
                config, data_config=data_config or config.data_config,
                period=period or config.period,
            ),
            sources=sources,
        )
        self.data = prepared.data
        self.weather_variables = prepared.weather_variables
        self.model_config = config.model_config
        self._initial_baseline = prepared.baseline_rate
        self.predictor = MortalityPredictor(
            ExponentialMortalityModel(config.model_config), config.predictor_config,
        )
        self._baseline_rate: torch.Tensor | None = None
        self._alignment_factor: torch.Tensor | None = None
        self.result: MemberResult | None = None
        self.to(self._device)

    @staticmethod
    def _get_calibration_period(config: RunConfig) -> PredictionPeriod:
        total_weeks = ((config.period.end_date - config.period.start_date).days + 1) // 7
        training_weeks = total_weeks - int(total_weeks * config.training_config.validation_fraction)
        calibration_weeks = min(training_weeks, config.training_config.alignment_max_weeks)
        return PredictionPeriod(
            config.period.start_date,
            config.period.start_date + timedelta(days=calibration_weeks * 7 - 1),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, **options: Any) -> InferenceModel:
        checkpoint = load_checkpoint(Path(checkpoint_path).expanduser())
        model = cls(decode_run_config(checkpoint["run_config"]), **options)
        model._load_state(checkpoint)
        return model

    def load_weights(self, checkpoint_path: str | Path) -> None:
        """Load compatible weights into this model and invalidate all prior fitted state."""
        self._load_state(load_checkpoint(Path(checkpoint_path).expanduser()))

    def _load_state(self, checkpoint: dict[str, Any]) -> None:
        expected = encode_run_config(self._config)
        loaded = encode_run_config(decode_run_config(checkpoint["run_config"]))
        differences = [name for name in expected if expected[name] != loaded[name]]
        if differences:
            raise ValueError(
                "Checkpoint configuration differs from this inference model: "
                + ", ".join(differences)
            )
        self.clear_member()
        self.to(self._device)
        self.predictor.load_state_dict(checkpoint["model_state_dict"])
        self.predictor.eval()

    def clear_member(self) -> None:
        """Release fitted state/output references before processing another member."""
        self._baseline_rate = None
        self._alignment_factor = None
        self.result = None

    @property
    def period(self) -> PredictionPeriod:
        return self.data.window

    @property
    def labels(self) -> MortalityLabels:
        return self.data.labels

    @property
    def baseline_rate(self) -> torch.Tensor:
        """Return daily fitted baseline rates for the evaluated member."""
        if self._baseline_rate is None:
            raise RuntimeError("Evaluate the member before reading its fitted baseline")
        return self._baseline_rate

    @property
    def alignment_factor(self) -> torch.Tensor | None:
        """Return the age/sex calibration factors used by the evaluated member."""
        if self.result is None:
            raise RuntimeError("Evaluate the member before reading its alignment factors")
        return self._alignment_factor

    def to(self, device: torch.device | str) -> InferenceModel:
        """Move model, prepared inputs and current results to the selected device."""
        self._device = torch.device(device)
        self.predictor.to(device=device, dtype=torch.float32)
        self.data = self.data.to(device)
        self._initial_baseline = self._initial_baseline.to(device=device, dtype=torch.float32)
        if self._baseline_rate is not None:
            self._baseline_rate = self._baseline_rate.to(device=device, dtype=torch.float32)
        if self._alignment_factor is not None:
            self._alignment_factor = self._alignment_factor.to(device=device, dtype=torch.float32)
        if self.result is not None:
            self.result = self.result.to(device)
        return self

    @torch.no_grad()
    def _estimate_alignment_factor(
        self, predictions: MortalityPredictionOutputs,
    ) -> torch.Tensor | None:
        """Estimate alignment using saved training dates, or return None if disabled."""
        config = self._config.training_config
        if not config.align_training_mean:
            return None
        period = self._calibration_period
        if period.start_date < self.period.start_date or period.end_date > self.period.end_date:
            raise ValueError("The analysis period must include the saved training mean-alignment dates")
        labels = select_mortality_labels(self.labels, period)
        start = (period.start_date - self.period.start_date).days
        days = (period.end_date - period.start_date).days + 1
        return estimate_mean_alignment(
            predictions.select_days(start, days), labels.weekly_de_age_sex,
            max_weeks=config.alignment_max_weeks,
        )

    @torch.no_grad()
    def _evaluate_baseline(
        self, covariates: MortalityModelInputs,
    ) -> tuple[MortalityPredictionInputs, MortalityPredictionOutputs, torch.Tensor]:
        """Predict with the initial or adapted baseline and its reference mortality."""
        config = self._config
        inputs = MortalityPredictionInputs(
            covariates, self._initial_baseline.clone(), self.data.population,
        )
        predictions = self.predictor(inputs)
        reference_effect = reference_temperature_effect(
            self.predictor.effect_model, covariates, self.weather_variables,
            config.reference_config,
            minimum_temperature_effect=cast(torch.Tensor, predictions.minimum_temperature_effect),
        )
        factors = self.predictor.correction_factors(self.data.population.shape[0])
        if config.training_config.baseline_adaptation:
            settings = self._evaluation_config
            relative_change = None
            for iteration in range(1, settings.maximum_iterations + 1):
                reference_deaths = reference_temperature_deaths(reference_effect, inputs, factors)
                updated = calculate_updated_baseline_rate(
                    self.labels.weekly_de_age_sex, predictions, reference_deaths,
                    inputs.population, config.adaptive_baseline_config,
                )
                if not (torch.isfinite(updated) & (updated > 0)).all():
                    warnings.warn(
                        f"Rejected adaptive baseline at inference iteration {iteration}: "
                        "candidate contains non-finite or non-positive rates. Keeping previous baseline.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break
                relative_change = (
                    (updated - inputs.baseline_rate).abs() / inputs.baseline_rate.abs()
                ).amax().item()
                inputs = MortalityPredictionInputs(covariates, updated, inputs.population)
                predictions = self.predictor(inputs)
                if relative_change <= settings.convergence_tolerance:
                    break
            else:
                raise RuntimeError(
                    f"Evaluation baseline did not converge after {settings.maximum_iterations} iterations; "
                    f"relative change={relative_change}"
                )
        reference_deaths = reference_temperature_deaths(reference_effect, inputs, factors)
        return inputs, predictions, reference_deaths

    @torch.no_grad()
    def evaluate(
        self, *, scenarios: dict[str, dict[str, Any]] | None = None,
    ) -> MemberResult:
        """Evaluate factual counts and supplied scenarios with the loaded weights."""
        if scenarios is not None and "reference" in scenarios:
            raise ValueError("Scenario name 'reference' is reserved for the configured model reference")
        self.clear_member()
        self.predictor.eval()
        config = self._config
        covariates = self.data.model_inputs
        if config.training_config.exclude_temperature:
            covariates = MortalityModelInputs(
                torch.zeros_like(covariates.temperature), covariates.sari, covariates.covid,
            )
        inputs, predictions, reference_deaths = self._evaluate_baseline(covariates)
        self._baseline_rate = inputs.baseline_rate
        self._alignment_factor = self._estimate_alignment_factor(predictions)
        reference = apply_mean_alignment(
            reference_deaths + predictions.sari_deaths + predictions.covid_deaths,
            self._alignment_factor,
        )
        predictions = apply_mean_alignment(predictions, self._alignment_factor)
        self.result = MemberResult(
            model_config=self.model_config, labels=self.labels, period=self.period,
            predictions=predictions, scenarios={"reference": reference},
            metrics=calculate_national_mse(aggregate_mortality(predictions.total_deaths), self.labels),
        )
        for name, specification in (scenarios or {}).items():
            self._run_scenario(name, specification)
        return self.result

    def _run_scenario(self, name: str, specification: dict[str, Any]) -> None:
        if "temperature" in specification:
            self._predict_scenario(name, specification["temperature"])
        elif "temperature_c" in specification:
            self._reference_scenario(name, **specification)
        else:
            self._temperature_scenario(name, **specification)

    @torch.no_grad()
    def _predict_scenario(self, name: str, temperature: torch.Tensor) -> torch.Tensor:
        """Store scenario total mortality, reusing factual disease components.

        Weather follows the prepared channel order and includes lag history.
        A constant-weather kernel may broadcast over all mortality dates.
        """
        self.predictor.eval()
        temperature = temperature.to(device=self.data.population.device, dtype=torch.float32)
        if self._config.training_config.exclude_temperature:
            temperature = torch.zeros_like(temperature)
        model_inputs = MortalityModelInputs(temperature, None, None)
        effects = self.predictor.effect_model(model_inputs)
        factors = self.predictor.correction_factors(self.data.population.shape[0])
        temperature_deaths = reference_temperature_deaths(
            effects.temperature_effect,
            MortalityPredictionInputs(
                model_inputs=model_inputs,
                baseline_rate=cast(torch.Tensor, self._baseline_rate),
                population=self.data.population,
            ),
            factors,
        )
        temperature_deaths = apply_mean_alignment(temperature_deaths, self._alignment_factor)
        total = temperature_deaths + self.result.predictions.sari_deaths + self.result.predictions.covid_deaths
        self.result.scenarios[name] = total
        return total

    def _scenario_channels(self) -> tuple[int, int | None]:
        variables = self.weather_variables
        if variables.count("tas") != 1 or variables.count("tdew") > 1 or set(variables) - {"tas", "tdew"}:
            raise NotImplementedError(
                "Notebook temperature transformations require one tas and optional tdew channel; "
                "supply a scenario temperature tensor with transformed weather for other layouts."
            )
        return variables.index("tas"), variables.index("tdew") if "tdew" in variables else None

    @torch.no_grad()
    def _temperature_scenario(
        self, name: str, *, minimum_c: float | None = None, maximum_c: float | None = None,
    ) -> torch.Tensor:
        """Cap/floor temperature, adjusting dew point to preserve humidity."""
        temperature_index, dew_point_index = self._scenario_channels()
        weather = self.data.model_inputs.temperature.clone()
        actual = weather[:, temperature_index, :].clone()
        changed = actual
        if minimum_c is not None:
            changed = changed.clamp_min(minimum_c)
        if maximum_c is not None:
            changed = changed.clamp_max(maximum_c)
        if dew_point_index is not None:
            dew_point = weather[:, dew_point_index, :]
            humidity = relative_humidity_percent(actual, dew_point)
            weather[:, dew_point_index, :] = torch.where(
                changed != actual,
                dew_point_c(changed, humidity), dew_point,
            )
        weather[:, temperature_index, :] = changed
        return self._predict_scenario(name, weather)

    @torch.no_grad()
    def _reference_scenario(
        self, name: str, *, temperature_c: float, relative_humidity_percent: float = 50.0,
    ) -> torch.Tensor:
        """Evaluate one constant-weather kernel and expand its effect to all dates."""
        self.predictor.eval()
        weather = build_reference_weather(
            self.data.model_inputs.temperature, self.weather_variables,
            temperature_c, relative_humidity_percent,
            kernel_days=self.model_config.kernel_T_days,
        )
        return self._predict_scenario(name, weather)


@dataclass
class EnsembleResult:
    """Processed means and optional processed member outputs; no fitted member state."""

    predictions: MortalityPredictionOutputs
    model_config: MortalityModelConfig
    data: PreparedData
    metrics: dict[str, torch.Tensor]
    scenarios: dict[str, torch.Tensor]
    members: dict[Path, MemberResult]

    @property
    def labels(self) -> MortalityLabels:
        return self.data.labels

    @property
    def period(self) -> PredictionPeriod:
        return self.data.window

    def select_dates(self, start_date: date | str, end_date: date | str) -> EnsembleResult:
        """Return an inclusive Monday–Sunday subset without refitting any member.

        Tensor slices share storage; the original result and its metrics remain
        unchanged. Prepared covariates retain their respective kernel histories.
        """
        period = _select_period(self.period, start_date, end_date)
        start = (period.start_date - self.period.start_date).days
        days = (period.end_date - period.start_date).days + 1
        stop = start + days
        covariates = self.data.model_inputs
        model_inputs = MortalityModelInputs(
            temperature=covariates.temperature[..., start:stop + self.model_config.kernel_T_days - 1],
            sari=(
                covariates.sari[..., start:stop + self.model_config.kernel_sari_days - 1]
                if covariates.sari is not None else None
            ),
            covid=(
                covariates.covid[..., start:stop + self.model_config.kernel_sari_days - 1]
                if covariates.covid is not None else None
            ),
        )
        labels = select_mortality_labels(self.labels, period)
        data = PreparedData(
            window=period, model_inputs=model_inputs,
            population=self.data.population[start:stop], labels=labels,
        )
        predictions = self.predictions.select_days(start, days)
        return EnsembleResult(
            predictions=predictions, model_config=self.model_config, data=data,
            metrics=calculate_national_mse(aggregate_mortality(predictions.total_deaths), labels),
            scenarios={name: value[start:stop] for name, value in self.scenarios.items()},
            members={path: member.select_dates(period.start_date, period.end_date) for path, member in self.members.items()},
        )

    def to(self, device: torch.device | str) -> EnsembleResult:
        self.data = self.data.to(device)
        self.predictions = self.predictions.to(device)
        self.members = {path: member.to(device) for path, member in self.members.items()}
        self.metrics = {name: value.to(device) for name, value in self.metrics.items()}
        self.scenarios = {name: value.to(device) for name, value in self.scenarios.items()}
        return self


@torch.no_grad()
def evaluate_runs(
    checkpoint_paths: str | Path | Sequence[str | Path], *, keep_members: bool = False,
    scenarios: dict[str, dict[str, Any]] | None = None,
    data_config: DataConfig | None = None, period: PredictionPeriod | None = None,
    evaluation_config: EvaluationConfig | None = None,
    device: torch.device | str = "cpu", sources: SourceData | None = None,
) -> EnsembleResult:
    """Reuse one model/data workspace, process weights sequentially, and retain CPU means."""
    if isinstance(checkpoint_paths, (str, Path)):
        path = Path(checkpoint_paths).expanduser()
        paths = tuple(sorted((path / "checkpoints").glob("final_*.pt"))) if path.is_dir() else (path,)
    else:
        paths = tuple(Path(path) for path in checkpoint_paths)
    paths = tuple(path.expanduser().resolve() for path in paths)
    if not paths:
        raise FileNotFoundError("No completed final_*.pt checkpoints found")
    model = InferenceModel.from_checkpoint(
        paths[0], data_config=data_config, period=period, evaluation_config=evaluation_config,
        device=device, sources=sources,
    )
    total = None
    scenario_totals: dict[str, torch.Tensor] = {}
    retained = {}
    for index, path in enumerate(paths):
        if index:
            model.load_weights(path)
        member = model.evaluate(scenarios=scenarios)
        total = _accumulate_predictions(total, member.predictions)
        for name, output in member.scenarios.items():
            if name in scenario_totals:
                scenario_totals[name].add_(output.to("cpu"))
            else:
                scenario_totals[name] = output.to("cpu").clone()
        if keep_members:
            retained[path] = member.to("cpu")
        del member, output
        model.clear_member()
    model.to("cpu")
    predictions = _finish_mean(cast(MortalityPredictionOutputs, total), len(paths))
    return EnsembleResult(
        predictions=predictions, model_config=model.model_config, data=model.data,
        metrics=calculate_national_mse(aggregate_mortality(predictions.total_deaths), model.labels),
        scenarios={name: value.div_(len(paths)) for name, value in scenario_totals.items()},
        members=retained,
    )


@torch.no_grad()
def evaluate_run(
    checkpoint_path: str | Path, *, scenarios: dict[str, dict[str, Any]] | None = None,
    data_config: DataConfig | None = None, period: PredictionPeriod | None = None,
    evaluation_config: EvaluationConfig | None = None,
    device: torch.device | str = "cpu", sources: SourceData | None = None,
) -> MemberResult:
    """One-shot processed results for all supplied scenarios."""
    model = InferenceModel.from_checkpoint(
        checkpoint_path, data_config=data_config, period=period, evaluation_config=evaluation_config,
        device=device, sources=sources,
    )
    return model.evaluate(scenarios=scenarios)


def _select_period(
    available: PredictionPeriod, start_date: date | str, end_date: date | str,
) -> PredictionPeriod:
    start = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    end = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
    if end < start:
        raise ValueError("Selection end precedes its start")
    if start < available.start_date or end > available.end_date:
        raise ValueError("Selection dates must be within the evaluated period")
    if start.weekday() != 0 or end.weekday() != 6:
        raise ValueError("Result selection requires complete Monday–Sunday weeks")
    return PredictionPeriod(start, end)


def _accumulate_predictions(
    total: MortalityPredictionOutputs | None, value: MortalityPredictionOutputs,
) -> MortalityPredictionOutputs:
    fields = ("temperature_deaths", "sari_deaths", "covid_deaths", "total_deaths")
    if total is None:
        return MortalityPredictionOutputs(**{
            name: getattr(value, name).detach().to("cpu").clone() for name in fields
        })
    for name in fields:
        getattr(total, name).add_(getattr(value, name).detach().to("cpu"))
    return total


def _finish_mean(total: MortalityPredictionOutputs, count: int) -> MortalityPredictionOutputs:
    for name in ("temperature_deaths", "sari_deaths", "covid_deaths", "total_deaths"):
        getattr(total, name).div_(count)
    return total
