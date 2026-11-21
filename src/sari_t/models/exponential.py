"""Reusable exponential mortality network."""

import torch
import torch.nn as nn

from .interfaces import MortalityModelInputs, MortalityModelOutputs
from .config import AGE_GENDER_CHANNEL_COUNT, AGE_GROUP_COUNT, GENDER_COUNT, MortalityModelConfig
from .layers import Conv1dPos, LinearPos


class ExponentialMortalityModel(nn.Module):
    """Temperature, SARI, and COVID effects, independent of data and training."""

    def __init__(self, config: MortalityModelConfig) -> None:
        super().__init__()
        self.config = config
        self.batch_norm = nn.LazyBatchNorm1d(eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1 = nn.LazyConv1d(
            out_channels=self.config.hidden_dim_init,
            kernel_size=self.config.kernel_T_days,
            stride=1,
        )
        self.fc = LinearPos(self.config.hidden_dim_init, self.config.out_channel)
        self.sari_conv = Conv1dPos(
            in_channels=AGE_GROUP_COUNT,
            out_channels=AGE_GENDER_CHANNEL_COUNT,
            groups=AGE_GROUP_COUNT,
            kernel_size=self.config.kernel_sari_days,
            stride=1,
            bias=False,
        )
        self.covid_conv = Conv1dPos(
            in_channels=AGE_GROUP_COUNT,
            out_channels=AGE_GENDER_CHANNEL_COUNT,
            groups=AGE_GROUP_COUNT,
            kernel_size=self.config.kernel_sari_days,
            stride=1,
            bias=False,
        )
        nn.init.constant_(self.sari_conv.weight, 0.01)
        nn.init.constant_(self.covid_conv.weight, 0.01)

    def forward(self, model_inputs: MortalityModelInputs) -> MortalityModelOutputs:
        temperature = model_inputs.temperature
        sari = model_inputs.sari
        covid = model_inputs.covid
        temperature = self.batch_norm(temperature)
        temperature_effect = self.conv1(temperature).permute(2, 0, 1)
        temperature_effect.exp_()
        temperature_effect = self.fc(temperature_effect)
        temperature_effect = temperature_effect.reshape(*temperature_effect.shape[:-1], -1, GENDER_COUNT)

        if self.config.include_sari and sari is not None:
            sari = sari * self.config.surveillance_scale
            if covid is not None:
                covid = covid * self.config.surveillance_scale
            if not self.config.include_covid and covid is not None:
                sari = sari + covid
            sari_effect = self.sari_conv(sari.unsqueeze(0)).permute(2, 0, 1)
            sari_effect = sari_effect.reshape(*sari_effect.shape[:-1], -1, GENDER_COUNT)
        else:
            return MortalityModelOutputs(
                temperature_effect=temperature_effect,
                sari_effect=torch.zeros_like(temperature_effect),
                covid_effect=torch.zeros_like(temperature_effect),
            )

        if self.config.include_covid and covid is not None:
            covid_effect = self.covid_conv(covid.unsqueeze(0)).permute(2, 0, 1)
            covid_effect = covid_effect.reshape(*covid_effect.shape[:-1], -1, GENDER_COUNT)
        else:
            covid_effect = torch.zeros_like(temperature_effect)
        return MortalityModelOutputs(
            temperature_effect=temperature_effect,
            sari_effect=sari_effect,
            covid_effect=covid_effect,
        )
