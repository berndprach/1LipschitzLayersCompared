
import torch
from torch import nn


class FirstChannels(nn.Module):
    def __init__(self, nrof_channels: int):
        super().__init__()
        self.nrof_channels = nrof_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :self.nrof_channels]

    def __repr__(self):
        return f"FirstChannels({self.nrof_channels})"

