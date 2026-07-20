from datetime import date, timedelta
import math
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
import yaml

from .config import (
    HeatMortalityConfig,
    get_global_config_constants,
    DAYS_PER_WEEK,
    COVID_FREE_WEEKS,
    WEEKDAY_CORRECTION_SIZE,
    STATE_CORRECTION_SIZE,
    WARMUP_EPOCHS,
    BASELINE_ADAPTATION_START_EPOCH,
    BASELINE_ADAPTATION_UPDATE_INTERVAL,
    STATE_CORRECTION_START_EPOCH,
    STATE_CORRECTION_UPDATE_INTERVAL,
    EARLY_STOPPING_START_EPOCH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SCHEDULER_PATIENCE,
    GRADIENT_CLIP_VALUE,
    TEMPERATURE_NOISE_STD,
    SCHEDULER_WARMUP_PROXY_LOSS,
    BASELINE_EVAL_ITERATIONS,
)
from .containers import (
    MortalityLabels,
    MortalityLabelsStage1,
    MortalityLabelsStage2,
    MortalityModelInputs,
    MortalityModelOutputs,
    MortalityPredictions,
)
from .baseline import calculate_baseline_death_rate
from .grouping import land_extension
from .loss_function import HeatMortalityLoss
from .metric import HeatMortalityAccuracy
from .population_data import population_preprocessing
from .sari_data import sari_preprocessing
from .weather_data import weather_preprocessing

TEMPERATURE_DATA_START = date(2011, 1, 1)
WEEKLY_DATA_START = date(2011, 1, 3)
SARI_DATA_START = date(2014, 9, 29)
MAX_SUPPORTED_END_YEAR = 2026
REFERENCE_RELATIVE_HUMIDITY_PERCENT = 50.0


class HeatMortalityBase(pl.LightningModule):
    """
    Base class for heat mortality modelling.

    Subclasses define the network architecture, while this class handles
    data preparation, training-time aggregation, and baseline adaptation.
    """

    def __init__(self, config: HeatMortalityConfig | dict) -> None:
        super().__init__()

        if isinstance(config, dict):
            config = HeatMortalityConfig.from_dict(config)

        self.config = config
        self._validate_config(self.config)

        self.daily_predictions = None
        self.weekly_predictions = None
        self.daily_predictions_corrected = None
        self.weekly_predictions_corrected = None
        self.training_mean_correction_factor: torch.Tensor | None = None
        self.sari_data_start = SARI_DATA_START
        active_date_range = self.config.get_active_date_range()
        self.date_window = [date.fromisoformat(active_date_range.start), date.fromisoformat(active_date_range.end)]

        self._get_weather_data()
        self._init_data()
        self._init_model()

        self.weekday_correction = self._init_optional_parameter(WEEKDAY_CORRECTION_SIZE, self.config.weekday_correction)
        self.state_correction = self._init_optional_parameter(STATE_CORRECTION_SIZE, self.config.state_correction)

        self.total_weeks = ((self.date_window[1] - self.date_window[0]).days + 1) // DAYS_PER_WEEK
        self.validation_weeks = int(self.total_weeks * self.config.val_split)
        self.training_weeks = self.total_weeks - self.validation_weeks

        self._data_slicing()

        self.loss_module = HeatMortalityLoss()
        self.accuracy_metric = HeatMortalityAccuracy()

        self.automatic_optimization = False

        self.save_hyperparameters({
            "config": self.config.to_dict(),
            "global_constants": get_global_config_constants(),
        })

        if self.config.stage == 2 and self.config.checkpoint_path:
            self._load_stage1_checkpoint(self.config.checkpoint_path)

    @classmethod
    def from_lightning_hparams_yaml(cls, hparams_yaml_path: str | Path) -> "HeatMortalityBase":
        """Construct a model from Lightning-generated hparams YAML.

        Expects a YAML file containing a `config` section compatible with
        HeatMortalityConfig.to_dict().
        """
        with open(hparams_yaml_path, "r", encoding="utf-8") as file:
            hparams = yaml.safe_load(file)

        if not isinstance(hparams, dict):
            raise ValueError("Invalid hparams YAML content: expected a mapping at root")

        if "config" not in hparams:
            raise KeyError("Missing 'config' section in hparams YAML")

        config_data = hparams["config"]
        if isinstance(config_data, HeatMortalityConfig):
            config = config_data
        else:
            config = HeatMortalityConfig.from_dict(dict(config_data))

        if cls is HeatMortalityBase:
            from . import get_model_class_by_name

            model_cls = get_model_class_by_name(config.model_name)
            return model_cls(config=config)

        if config.model_name != cls.__name__:
            raise ValueError(
                f"Model mismatch: YAML requests '{config.model_name}', but called on '{cls.__name__}'"
            )
        return cls(config=config)

    def forward(self, model_inputs: MortalityModelInputs) -> MortalityModelOutputs:
        raise NotImplementedError

    def training_step(self, _batch: object, batch_index: int) -> torch.Tensor:
        return self._step(batch_index, mode="train")

    def validation_step(self, _batch: object, batch_index: int) -> torch.Tensor:
        return self._step(batch_index, mode="val")

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE, threshold=1e-4, factor=0.3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _init_model(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _validate_config(config: HeatMortalityConfig) -> None:
        active_date_range = config.get_active_date_range()
        start_date = date.fromisoformat(active_date_range.start)
        end_date = date.fromisoformat(active_date_range.end)
        death_daily_start = date.fromisoformat(config.get_death_label_start_dates()[0])
        death_weekly_start_str = config.get_death_label_start_dates()[1]

        if start_date.isoweekday() != 1:
            raise ValueError("Starting date should be Monday")
        if end_date.isoweekday() != 7:
            raise ValueError("Ending date should be Sunday")
        if config.include_sari and start_date - timedelta(config.kernel_sari_days - 1) < SARI_DATA_START:
            raise ValueError("Starting date outside training data")
        if (start_date - TEMPERATURE_DATA_START).days < 0:
            raise ValueError("Starting date outside training data for temperature")
        if start_date < death_daily_start:
            raise ValueError("Starting date outside death label range for selected stage")
        if death_weekly_start_str is not None and start_date < date.fromisoformat(death_weekly_start_str):
            raise ValueError("Starting date outside weekly death label range for selected stage")
        if end_date.year > MAX_SUPPORTED_END_YEAR or end_date <= start_date:
            raise ValueError("End date should be larger than start date and before 2026")
        if not 0 < config.val_split < 1:
            raise ValueError("val_split should be between 0 and 1")
        if config.kernel_T_days <= 0 or config.kernel_sari_days <= 0:
            raise ValueError("Kernel sizes should be positive")
        if config.hidden_dim_init <= 0 or config.out_channel <= 0:
            raise ValueError("Model dimensions should be positive")

    @staticmethod
    def _day_index(current_date: date, base_date: date) -> int:
        return (current_date - base_date).days

    @staticmethod
    def _week_index(current_date: date, base_date: date = WEEKLY_DATA_START) -> int:
        return (current_date - base_date).days // 7

    @staticmethod
    def _init_optional_parameter(size: int, trainable: bool) -> torch.Tensor | nn.Parameter:
        values = torch.ones([size], dtype=torch.float32)
        if trainable:
            return nn.Parameter(values)
        return values

    def _move_stage1_labels_to_device(self, labels: MortalityLabelsStage1) -> MortalityLabelsStage1:
        return MortalityLabelsStage1(
            daily_land=labels.daily_land.to(self.device),
            weekly_de_age_sex=labels.weekly_de_age_sex.to(self.device),
            weekly_land_coarse_age_sex=labels.weekly_land_coarse_age_sex.to(self.device),
        )

    def _move_stage2_labels_to_device(self, labels: MortalityLabelsStage2) -> MortalityLabelsStage2:
        return MortalityLabelsStage2(
            daily_kreis_age_sex=labels.daily_kreis_age_sex.to(self.device),
        )

    def _prepare_model_inputs(self) -> None:
        """Prepare aligned model inputs for the configured time window."""
        t_start = self._day_index(self.date_window[0], TEMPERATURE_DATA_START) + 1 - self.config.kernel_T_days
        t_end = self._day_index(self.date_window[1], TEMPERATURE_DATA_START) + 1
        sari_start = self._day_index(self.date_window[0], self.sari_data_start) + 1 - self.config.kernel_sari_days
        sari_end = self._day_index(self.date_window[1], self.sari_data_start) + 1

        self.temperature_input = self.tensor_t[..., t_start:t_end].to(device=self.device)
        self.sari_input = (self.sari_data[:, sari_start:sari_end] / 10**2).to(device=self.device)
        self.covid_input = (self.covid_data[:, sari_start:sari_end] / 10**2).to(device=self.device)

    def _move_training_state_to_device(self) -> None:
        """Move all tensors used in training/validation to the current device."""
        self.population_range = self.population_range.to(self.device)
        self.basic_death_rate_range = self.basic_death_rate_range.to(self.device)
        if self.config.stage == 1:
            self.train_labels = self._move_stage1_labels_to_device(self.train_labels)
            self.val_labels = self._move_stage1_labels_to_device(self.val_labels)
        elif self.config.stage == 2:
            self.train_labels = self._move_stage2_labels_to_device(self.train_labels)
            self.val_labels = self._move_stage2_labels_to_device(self.val_labels)
        self.weekday_correction = self.weekday_correction.to(self.device)
        self.state_correction = self.state_correction.to(self.device)

    def _slice_labels_for_stage(self, death_day_start: int, death_day_end: int, death_week_start: int | None) -> None:
        """Route label slicing by configured training stage."""
        if self.config.stage == 1:
            if death_week_start is None:
                raise ValueError("Stage 1 requires weekly death-label start date")
            self._slice_labels_stage1(death_day_start, death_day_end, death_week_start)
            return
        if self.config.stage == 2:
            self._slice_labels_stage2(death_day_start, death_day_end)
            return
        raise ValueError(f"Unsupported training stage: {self.config.stage}")

    def _slice_labels_stage1(self, death_day_start: int, death_day_end: int, death_week_start: int) -> None:
        """Slice daily and weekly labels into train and validation windows."""
        train_daily = self.deaths[0][death_day_start:death_day_start + self.training_weeks * DAYS_PER_WEEK]
        val_daily = self.deaths[0][death_day_start + self.training_weeks * DAYS_PER_WEEK:death_day_end]
        train_weekly_de = self.deaths[1][death_week_start:death_week_start + self.training_weeks]
        val_weekly_de = self.deaths[1][death_week_start + self.training_weeks:death_week_start + self.total_weeks]
        train_weekly_land = self.deaths[2][death_week_start:death_week_start + self.training_weeks]
        val_weekly_land = self.deaths[2][death_week_start + self.training_weeks:death_week_start + self.total_weeks]

        self.train_labels = MortalityLabelsStage1(
            daily_land=train_daily,
            weekly_de_age_sex=train_weekly_de,
            weekly_land_coarse_age_sex=train_weekly_land,
        )
        self.val_labels = MortalityLabelsStage1(
            daily_land=val_daily,
            weekly_de_age_sex=val_weekly_de,
            weekly_land_coarse_age_sex=val_weekly_land,
        )

    def _slice_labels_stage2(self, death_day_start: int, death_day_end: int) -> None:
        """Slice stage-2 daily kreis-age-sex labels into train and validation windows."""
        if len(self.deaths) < 1:
            raise ValueError("Stage 2 expects deaths[0] with daily_kreis_age_sex labels")

        daily_labels = self.deaths[0]
        train_daily = daily_labels[death_day_start:death_day_start + self.training_weeks * DAYS_PER_WEEK]
        val_daily = daily_labels[death_day_start + self.training_weeks * DAYS_PER_WEEK:death_day_end]

        self.train_labels = MortalityLabelsStage2(
            daily_kreis_age_sex=train_daily,
        )
        self.val_labels = MortalityLabelsStage2(
            daily_kreis_age_sex=val_daily,
        )

    def _compute_loss_components(
        self,
        predictions: MortalityPredictions,
        labels: MortalityLabels,
        corrected_predictions: MortalityPredictions | None = None,
    ) -> dict[str, torch.Tensor]:
        """Route loss computation by configured training stage and handle optional dual loss."""
        loss_components = self.loss_module.compute_components(predictions, labels, with_covid=True)

        if corrected_predictions is not None:
            loss_components_corrected = self.loss_module.compute_components(corrected_predictions, labels, with_covid=True)

            # Sum original and corrected losses for gradient computation.
            factor = max(0.3, (1 - (self.current_epoch - BASELINE_ADAPTATION_START_EPOCH) / 600))
            factor = min(factor, 1.0)
            loss_components["total"] = factor * loss_components["total"] + \
                                       (1 - factor) * loss_components_corrected["total"]

        return loss_components

    def _freeze_correction_parameters_for_warmup(self) -> None:
        """Freeze correction parameters during warmup phase."""
        if self.current_epoch < WARMUP_EPOCHS:
            self.weekday_correction.requires_grad = False
            self.state_correction.requires_grad = False

    def _compute_predictions(
        self,
        random_t_sigma: float = TEMPERATURE_NOISE_STD,
        calculate_additional_deaths: bool = True,
        update_training_mean_correction_factor: bool = False,
    ) -> None:
        """Run forward inference and cache daily/weekly predictions."""
        self.daily_predictions = self.calculate_mortality(
            self.temperature_input,
            self.sari_input,
            self.covid_input,
            random_t_sigma=random_t_sigma,
            calculate_additional_deaths=calculate_additional_deaths,
        )
        n_days, n_kreis, n_age_group, n_sex = self.daily_predictions.shape
        self.weekly_predictions = self.daily_predictions.reshape(
            n_days // DAYS_PER_WEEK,
            DAYS_PER_WEEK,
            n_kreis,
            n_age_group,
            n_sex,
        ).sum(1)

        self.daily_predictions_corrected = None
        self.weekly_predictions_corrected = None

        if not self.config.correct_training_mean:
            return

        correction_factor = self.training_mean_correction_factor
        if update_training_mean_correction_factor:
            correction_factor = self._compute_training_mean_correction_factor()
            self.training_mean_correction_factor = correction_factor

        if correction_factor is None:
            return

        self.daily_predictions_corrected = self.apply_training_mean_correction_tensor(self.daily_predictions)
        self.weekly_predictions_corrected = self.apply_training_mean_correction_tensor(self.weekly_predictions)

    def _select_step_tensors(self, mode: str) -> tuple[MortalityPredictions, MortalityLabels, MortalityPredictions | None]:
        """Select train or validation slices of predictions and labels."""
        if mode == "train":
            predictions = MortalityPredictions(
                daily_kreis_age_sex=self.daily_predictions[:DAYS_PER_WEEK * self.training_weeks],
                weekly_kreis_age_sex=self.weekly_predictions[:self.training_weeks],
            )
            labels = self.train_labels
            corrected_predictions = None
            if self.daily_predictions_corrected is not None and self.weekly_predictions_corrected is not None:
                corrected_predictions = MortalityPredictions(
                    daily_kreis_age_sex=self.daily_predictions_corrected[:DAYS_PER_WEEK * self.training_weeks],
                    weekly_kreis_age_sex=self.weekly_predictions_corrected[:self.training_weeks],
                )
        else:
            predictions = MortalityPredictions(
                daily_kreis_age_sex=self.daily_predictions[DAYS_PER_WEEK * self.training_weeks:],
                weekly_kreis_age_sex=self.weekly_predictions[self.training_weeks:],
            )
            labels = self.val_labels
            corrected_predictions = None
        return predictions, labels, corrected_predictions

    def _should_apply_training_mean_correction(
        self,
        mode: str,
    ) -> bool:
        """Return True when training mean correction should be applied."""
        return mode == "train" and self.config.correct_training_mean

    def apply_training_mean_correction_tensor(
        self,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the train-window mean correction factor to a prediction tensor."""
        correction_factor = self.training_mean_correction_factor
        if correction_factor is None:
            if self.config.correct_training_mean:
                raise RuntimeError(
                    "Training-mean correction was requested, but no prepared correction factor is available. "
                )
            return prediction

        channel_scale = correction_factor
        while channel_scale.ndim < prediction.ndim:
            channel_scale = channel_scale.unsqueeze(0)
        return prediction * channel_scale

    def apply_training_mean_correction(
        self,
        predictions: MortalityPredictions,
    ) -> MortalityPredictions:
        """Apply the train-window mean correction factor to daily and weekly predictions."""
        correction_factor = self.training_mean_correction_factor
        if correction_factor is None:
            if self.config.correct_training_mean:
                raise RuntimeError(
                    "Training-mean correction was requested, but no prepared correction factor is available. "
                )
            return predictions

        return MortalityPredictions(
            daily_kreis_age_sex=self.apply_training_mean_correction_tensor(predictions.daily_kreis_age_sex),
            weekly_kreis_age_sex=self.apply_training_mean_correction_tensor(predictions.weekly_kreis_age_sex),
        )

    def _compute_training_mean_correction_factor(
        self,
    ) -> torch.Tensor | None:
        """Compute training-mean correction factor on-the-fly for the current step."""
        if not self._should_apply_training_mean_correction("train"):
            return None

        if self.config.stage == 1:
            return self._correct_training_mean_stage1()
        if self.config.stage == 2:
            return self._correct_training_mean_stage2()
        return None

    def _correct_training_mean_stage1(
        self,
    ) -> torch.Tensor | None:
        """Compute demographic correction factor for stage-1 training."""

        predictions = MortalityPredictions(
            daily_kreis_age_sex=self.daily_predictions[:DAYS_PER_WEEK * self.training_weeks],
            weekly_kreis_age_sex=self.weekly_predictions[:self.training_weeks],
        )
        labels = self.train_labels

        if not isinstance(labels, MortalityLabelsStage1):
            return None

        with torch.no_grad():
            covid_free_weeks = min(COVID_FREE_WEEKS, predictions.weekly_kreis_age_sex.shape[0], labels.weekly_de_age_sex.shape[0])

            if covid_free_weeks <= 0:
                return None

            pred_weekly_de = predictions.weekly_kreis_age_sex[:covid_free_weeks].sum(dim=(0, 1))
            obs_weekly_de = labels.weekly_de_age_sex[:covid_free_weeks].sum(dim=0)
            age_sex_scale = obs_weekly_de / pred_weekly_de
            return age_sex_scale

    def _correct_training_mean_stage2(
        self,
    ) -> torch.Tensor | None:
        """Stage-2 correction placeholder; intentionally no-op until finalized."""
        return None

    def _is_scheduler_warmup_phase(self) -> bool:
        """Return True while LR scheduler stepping is intentionally delayed."""
        return self.current_epoch <= EARLY_STOPPING_START_EPOCH

    def _should_step_scheduler(self, batch_index: int) -> bool:
        """Step LR scheduler once per epoch after burn-in."""
        return batch_index == 0 and not self._is_scheduler_warmup_phase()

    def _should_update_baseline_death_rate(self) -> bool:
        """Return True when baseline death rates should be updated this epoch."""
        return (
            self.config.stage == 1
            and self.current_epoch >= BASELINE_ADAPTATION_START_EPOCH
            and self.config.baseline_adaptation
            and self.current_epoch % BASELINE_ADAPTATION_UPDATE_INTERVAL == 0
        )

    def _update_baseline_death_rate(self) -> None:
        """Update baseline death rate estimates during training."""
        with torch.no_grad():
            est_base_deaths = self._get_weekly_baseline_est()
            updated_rate = calculate_baseline_death_rate(
                est_base_deaths,
                self.population_range.sum(1),
                self.config.adaptation_method,
            )
            self.basic_death_rate_range.copy_(updated_rate)

    def _should_update_state_correction(self) -> bool:
        """Check if state correction parameters should be updated."""
        return self.current_epoch > STATE_CORRECTION_START_EPOCH and self.current_epoch % STATE_CORRECTION_UPDATE_INTERVAL == 0


    def _log_step_metrics(
        self,
        mode: str,
        loss: torch.Tensor,
        accuracy: tuple[torch.Tensor, torch.Tensor],
        loss_components: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Log training and validation metrics."""
        component_logs = {
            f"{mode}_loss_daily": loss_components["daily"],
            f"{mode}_loss_daily_total": loss_components["daily_total"],
            f"{mode}_loss_weekly_de": loss_components["weekly_de"],
            f"{mode}_loss_weekly_land": loss_components["weekly_land"],
            f"{mode}_loss_weekly_total": loss_components["weekly_total"],
        }

        if mode == "train":
            self.log_dict(
                {f"{mode}_loss": loss, f"{mode}_daily_mse": accuracy[0], **component_logs},
                on_epoch=True,
                prog_bar=True,
            )
            if self._is_scheduler_warmup_phase():
                return torch.tensor(SCHEDULER_WARMUP_PROXY_LOSS, device=loss.device)
            return loss

        self.log_dict(
            {
                f"{mode}_loss": loss,
                f"{mode}_daily_mse": accuracy[0],
                f"{mode}_weekly_mse": accuracy[1],
                **component_logs,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def on_train_start(self) -> None:
        """Prepare cached inputs and labels once trainer/device state is ready."""
        self._prepare_model_inputs()
        self._move_training_state_to_device()

    def _step(self, batch_index: int, mode: str = "train") -> torch.Tensor:
        """Shared train/validation step implementation."""
        if batch_index == 0:
            self.accuracy_metric.reset()

        if mode == "train":
            for parameter in self.parameters():
                parameter.requires_grad = True
            self._freeze_correction_parameters_for_warmup()
            update_baseline_death_rate = self._should_update_baseline_death_rate()
            self._compute_predictions(
                calculate_additional_deaths=update_baseline_death_rate,
                update_training_mean_correction_factor=True,
            )

        predictions, labels, corrected_predictions = self._select_step_tensors(mode)

        loss_components = self._compute_loss_components(predictions, labels, corrected_predictions=corrected_predictions)
        loss = loss_components["total"]

        if mode == "train":
            # Manual optimization over the full cached training window.
            training_loss = loss
            update_state_correction = self._should_update_state_correction()

            if update_state_correction:
                training_loss = self.loss_module(predictions, labels, with_covid=False)

            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.zero_grad()

            self.manual_backward(training_loss)

            if not update_state_correction:
                self.state_correction.grad = None

            self.clip_gradients(optimizer, gradient_clip_val=GRADIENT_CLIP_VALUE, gradient_clip_algorithm="norm")
            optimizer.step()

            with torch.no_grad():
                self.state_correction.div_(self.state_correction[0].clone())

            if update_baseline_death_rate:
                self._update_baseline_death_rate()

        metric_predictions = corrected_predictions if corrected_predictions is not None else predictions
        accuracy = self.accuracy_metric(metric_predictions, labels)

        loss = self._log_step_metrics(mode, loss, accuracy, loss_components)
        if mode == "train" and self._should_step_scheduler(batch_index):
            scheduler.step(loss)
        return loss

    def calculate_mortality(
        self,
        temperature_input: torch.Tensor,
        sari_input: torch.Tensor | None,
        covid_input: torch.Tensor | None,
        random_t_sigma: float = 0,
        basic_death_rate: torch.Tensor | None = None,
        population: torch.Tensor | None = None,
        calculate_additional_deaths: bool = True,
    ) -> torch.Tensor:
        """Calculate mortality predictions by combining temperature, SARI, and COVID effects."""
        rand_t = torch.randn_like(temperature_input, device=self.device, dtype=torch.float) * random_t_sigma
        if self.config.exclude_t:
            temperature_input = torch.zeros_like(temperature_input, device=self.device, dtype=torch.float)
        else:
            temperature_input = temperature_input + rand_t

        model_inputs = MortalityModelInputs(
            temperature=temperature_input,
            sari=sari_input,
            covid=covid_input,
        )
        model_outputs = self.forward(model_inputs)
        correction_factors = self.get_correction_factors(model_outputs.temperature_effect.shape)

        death_temp_model = self.get_death_temp_model(model_outputs.temperature_effect, correction_factors, basic_death_rate, population)
        death_sari = self.get_additional_deaths_disease(model_outputs.sari_effect, correction_factors, basic_death_rate, population)
        death_covid = self.get_additional_deaths_disease(model_outputs.covid_effect, correction_factors, basic_death_rate, population)
        prediction = death_temp_model + death_sari + death_covid

        if calculate_additional_deaths:
            reference_model_outputs = None
            if self.config.temperature_attribution_reference_c is not None:
                reference_temperature_input = self._build_temperature_attribution_reference_input(temperature_input)
                reference_model_outputs = self._calculate_reference_model_outputs(
                    reference_temperature_input,
                    sari_input,
                    covid_input,
                )
            death_temp_additional = self.get_additional_death_temp(
                model_outputs.temperature_effect,
                None if reference_model_outputs is None else reference_model_outputs.temperature_effect,
                correction_factors,
                basic_death_rate,
                population,
            )
            self.additional_deaths = (death_temp_additional + death_sari + death_covid).detach()

        return prediction

    def _calculate_reference_model_outputs(
        self,
        reference_temperature_input: torch.Tensor,
        sari_input: torch.Tensor | None,
        covid_input: torch.Tensor | None,
    ) -> MortalityModelOutputs:
        """Run the reference-temperature forward pass without updating training state."""
        restore_training_mode = self.training
        try:
            self.eval()
            with torch.no_grad():
                return self.forward(
                    MortalityModelInputs(
                        temperature=reference_temperature_input,
                        sari=sari_input,
                        covid=covid_input,
                    )
                )
        finally:
            if restore_training_mode:
                self.train()

    def _build_temperature_attribution_reference_input(self, temperature_input: torch.Tensor) -> torch.Tensor:
        """Build counterfactual weather input at the configured reference temperature."""
        reference_temperature = self.config.temperature_attribution_reference_c
        if self.config.exclude_t:
            return torch.zeros_like(temperature_input)

        if temperature_input.ndim < 3:
            return torch.full_like(temperature_input, reference_temperature)

        channel_count = temperature_input.shape[1]
        if channel_count != len(self.config.weather_variables):
            raise ValueError(
                "Cannot build temperature-attribution reference input: "
                f"{channel_count} weather channels but {len(self.config.weather_variables)} weather variable names"
            )

        reference_values = [
            self._reference_value_for_weather_variable(variable)
            for variable in self.config.weather_variables
        ]
        reference_tensor = torch.as_tensor(
            reference_values,
            device=temperature_input.device,
            dtype=temperature_input.dtype,
        )
        return reference_tensor.view(1, -1, 1).expand_as(temperature_input).clone()

    def _reference_value_for_weather_variable(self, variable: str) -> float:
        reference_temperature = self.config.temperature_attribution_reference_c
        if variable in {"tas", "tas_min"}:
            return reference_temperature
        if variable == "tdew":
            return self._dew_point_c(reference_temperature, REFERENCE_RELATIVE_HUMIDITY_PERCENT)
        if variable == "twb":
            return self._wet_bulb_temperature_c(reference_temperature, REFERENCE_RELATIVE_HUMIDITY_PERCENT)
        raise ValueError(f"Unsupported weather variable for attribution reference: {variable}")

    @staticmethod
    def _dew_point_c(temperature_c: float, relative_humidity_percent: float) -> float:
        """Approximate dew point from air temperature and relative humidity."""
        magnus_a = 17.625
        magnus_b = 243.04
        gamma = math.log(relative_humidity_percent / 100.0) + (
            magnus_a * temperature_c / (magnus_b + temperature_c)
        )
        return magnus_b * gamma / (magnus_a - gamma)

    @staticmethod
    def _wet_bulb_temperature_c(temperature_c: float, relative_humidity_percent: float) -> float:
        """Approximate wet-bulb temperature from air temperature and relative humidity."""
        humidity_power_term = relative_humidity_percent ** 1.5
        return (
            temperature_c * math.atan(0.151977 * math.sqrt(relative_humidity_percent + 8.313659))
            + math.atan(temperature_c + relative_humidity_percent)
            - math.atan(relative_humidity_percent - 1.676331)
            + 0.00391838 * humidity_power_term * math.atan(0.023101 * relative_humidity_percent)
            - 4.686035
        )

    def get_correction_factors(self, prediction_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        """Build multiplicative weekday and state correction factors."""
        weekday_corr = torch.cat([torch.ones([1], dtype=torch.float32, device=self.device), self.weekday_correction.to(self.device)])
        weekday_corr = weekday_corr.repeat(prediction_shape[0] // DAYS_PER_WEEK + 1).view(-1, 1, 1, 1)[:prediction_shape[0]]
        state_corr = land_extension(self.state_correction, land_axis=0).view(-1, 1, 1)
        return weekday_corr * state_corr

    def get_death_temp_model(
        self,
        model_signal: torch.Tensor,
        correction_factors: torch.Tensor,
        basic_death_rate: torch.Tensor | None = None,
        population: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert multiplicative temperature signal into total death estimates."""
        temperature_factors = model_signal.abs()
        if basic_death_rate is None:
            basic_death_rate = self.basic_death_rate_range
        if population is None:
            population = self.population_range
        return temperature_factors * basic_death_rate * population * correction_factors

    def get_additional_death_temp(
        self,
        model_signal: torch.Tensor,
        reference_model_signal: torch.Tensor | None,
        correction_factors: torch.Tensor,
        basic_death_rate: torch.Tensor | None = None,
        population: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate temperature-attributable deaths relative to fixed reference or empirical minimum."""
        temperature_factors = model_signal.abs()
        if reference_model_signal is None:
            reference_temperature_factors = temperature_factors.amin(dim=(0, 1), keepdim=True)
        else:
            reference_temperature_factors = reference_model_signal.abs()
        if basic_death_rate is None:
            basic_death_rate = self.basic_death_rate_range
        if population is None:
            population = self.population_range
        return (
            (temperature_factors - reference_temperature_factors)
            * basic_death_rate
            * population
            * correction_factors
        ).detach()

    def get_additional_deaths_disease(
        self,
        model_signal: torch.Tensor,
        correction_factors: torch.Tensor,
        basic_death_rate: torch.Tensor | None = None,
        population: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert additive disease signal into attributable deaths."""
        if basic_death_rate is None:
            basic_death_rate = self.basic_death_rate_range
        if population is None:
            population = self.population_range
        return model_signal * basic_death_rate * population * correction_factors

    def _init_data(self) -> None:
        """Load and preprocess population, death, and disease surveillance data."""
        self.population, self.deaths = population_preprocessing(
            interpolation=self.config.population_interpolation,
            stage=self.config.stage,
        )
        if self.config.stage == 1:
            self.basic_death_rate = calculate_baseline_death_rate(self.deaths[1], self.population.sum(1))
        elif self.config.stage == 2:
            self.basic_death_rate = torch.ones_like(self.population, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported training stage: {self.config.stage}")
        self.sari_data, self.covid_data = sari_preprocessing(self.config)

    def _get_weather_data(self) -> None:
        """Load weather covariates and configure model input channel count."""
        self.tensor_t = weather_preprocessing(self.config)
        self.config.in_channel = self.tensor_t.shape[1]

    def _get_weekly_baseline_est(self) -> torch.Tensor:
        """Estimate baseline weekly deaths by removing predicted additional deaths."""
        total_additional_deaths = self.additional_deaths.sum(1)
        total_additional_deaths_week = total_additional_deaths.reshape(
            -1,
            DAYS_PER_WEEK,
            total_additional_deaths.shape[1],
            total_additional_deaths.shape[2],
        ).sum(1)
        registered_deaths = torch.cat([self.train_labels.weekly_de_age_sex, self.val_labels.weekly_de_age_sex])
        return registered_deaths - total_additional_deaths_week

    def _data_slicing(self) -> None:
        """Slice preprocessed population and death tensors to configured date window."""
        pop_start = self._day_index(self.date_window[0], WEEKLY_DATA_START)
        pop_end = self._day_index(self.date_window[1], WEEKLY_DATA_START) + 1
        self.population_range = self.population[pop_start:pop_end]
        self.basic_death_rate_range = self.basic_death_rate[pop_start:pop_end]

        death_daily_start_str, death_weekly_start_str = self.config.get_death_label_start_dates()
        death_daily_start = date.fromisoformat(death_daily_start_str)
        death_day_start = self._day_index(self.date_window[0], death_daily_start)
        death_day_end = self._day_index(self.date_window[1], death_daily_start) + 1

        death_week_start = None
        if death_weekly_start_str is not None:
            death_week_start = self._week_index(self.date_window[0], date.fromisoformat(death_weekly_start_str))
        self._slice_labels_for_stage(death_day_start, death_day_end, death_week_start)

    def prepare_for_evaluation(self) -> None:
        """Prepare baseline, predictions, and post-processing calibration for evaluation."""
        self.training_mean_correction_factor = None
        with torch.no_grad():
            if self.config.stage == 1 and self.config.baseline_adaptation:
                for _ in range(BASELINE_EVAL_ITERATIONS):
                    self.daily_predictions = self.calculate_mortality(
                        self.temperature_input,
                        self.sari_input,
                        self.covid_input,
                    )
                    est_base_deaths = self._get_weekly_baseline_est()
                    updated_rate = calculate_baseline_death_rate(
                        est_base_deaths,
                        self.population_range.sum(1),
                        self.config.adaptation_method,
                    )
                    self.basic_death_rate_range.copy_(updated_rate)

            self._compute_predictions(
                random_t_sigma=0,
                calculate_additional_deaths=True,
                update_training_mean_correction_factor=True,
            )

    def evaluate_baseline(self) -> None:
        """Backward-compatible alias for evaluation preparation."""
        self.prepare_for_evaluation()

    def _load_stage1_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load Stage-1 weights before Stage-2 fine-tuning."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.load_state_dict(state_dict, strict=False)
