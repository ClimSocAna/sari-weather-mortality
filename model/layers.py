import torch
import torch.nn as nn


def _positive_weight(weight: torch.Tensor) -> torch.Tensor:
    """Use a non-negative view of layer weights during forward passes."""
    return weight.abs()


class Conv1dPos(nn.Conv1d):
    """1D convolution layer with non-negative weights."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(inputs, _positive_weight(self.weight), self.bias)


class LinearPos(nn.Linear):
    """Linear layer with non-negative weights."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(inputs, _positive_weight(self.weight), self.bias)