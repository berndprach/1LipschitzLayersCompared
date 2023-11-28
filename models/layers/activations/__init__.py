
from torch import nn

from .absolute_value import AbsoluteValue as Abs
from .max_min import MaxMin


class Identity(nn.Module):
    @staticmethod
    def forward(x):
        return x
