import torch
import torch.nn as nn

from .base_model import HeatMortalityBase
from .containers import MortalityModelInputs, MortalityModelOutputs
from .config import AGE_GENDER_CHANNEL_COUNT, AGE_GROUP_COUNT, GENDER_COUNT
from .layers import Conv1dPos, LinearPos


class HeatMortality_TEST(HeatMortalityBase):
    """Exponential model for death case prediction."""

    def _init_model(self) -> None:
        self.batch_norm = nn.BatchNorm1d(self.config.in_channel, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.temp_fc1 = nn.LazyLinear(16)
        self.temp_fc2 = nn.LazyLinear(16)
        self.temp_fc3 = nn.LazyLinear(2)

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
        if temperature.ndim == 2:
            temperature = temperature.unsqueeze(1)
        temperature = self.batch_norm(temperature)

        temperature = temperature.permute(0, 2, 1)
        temperature = self.temp_fc1(temperature)
        temperature = torch.relu(temperature)
        temperature = self.temp_fc2(temperature)
        temperature = torch.relu(temperature)
        temperature = self.temp_fc3(temperature)
        temperature = temperature.permute(0, 2, 1)

        temperature_effect = self.conv1(temperature).permute(2, 0, 1)
        temperature_effect = temperature_effect.exp()
        temperature_effect = self.fc(temperature_effect)
        temperature_effect = temperature_effect.reshape(*temperature_effect.shape[:-1], -1, GENDER_COUNT)

        if not self.config.include_covid and covid is not None:
            sari = sari + covid

        if self.config.include_sari and sari is not None:
            sari_effect = self.sari_conv(sari.unsqueeze(0)).permute(2, 0, 1)
            sari_effect = sari_effect.reshape(*sari_effect.shape[:-1], -1, GENDER_COUNT)
        else:
            sari_effect = torch.zeros_like(temperature_effect)

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
