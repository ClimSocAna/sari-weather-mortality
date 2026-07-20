from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MortalityPredictions:
    """Model mortality predictions at daily and weekly district resolution."""

    daily_kreis_age_sex: torch.Tensor
    weekly_kreis_age_sex: torch.Tensor


@dataclass(frozen=True)
class MortalityModelInputs:
    """Weather and disease covariates passed into model forward methods."""

    temperature: torch.Tensor
    sari: torch.Tensor | None
    covid: torch.Tensor | None


@dataclass(frozen=True)
class MortalityModelOutputs:
    """Raw model effects before conversion to mortality counts."""

    temperature_effect: torch.Tensor
    sari_effect: torch.Tensor
    covid_effect: torch.Tensor


@dataclass(frozen=True)
class MortalityLabelsStage1:
    """Stage-1 observed mortality labels at available reporting resolutions."""

    daily_land: torch.Tensor
    weekly_de_age_sex: torch.Tensor
    weekly_land_coarse_age_sex: torch.Tensor


@dataclass(frozen=True)
class MortalityLabelsStage2:
    """Stage-2 observed daily district-age-sex mortality labels."""

    daily_kreis_age_sex: torch.Tensor


MortalityLabels = MortalityLabelsStage1 | MortalityLabelsStage2
