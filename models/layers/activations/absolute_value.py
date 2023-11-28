
import torch
import torch.nn as nn


class AbsoluteValue(nn.Module):
    """Absolute value activation function."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)
