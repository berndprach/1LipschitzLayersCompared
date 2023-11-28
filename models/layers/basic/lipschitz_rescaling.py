
import torch
import torch.nn as nn


class LipschitzRescaling(nn.Module):
    def __init__(self, width):
        # self.arcsins = nn.Parameter(torch.Tensor([width]))
        super().__init__()
        # self.arc_sin_factors = nn.Parameter(torch.zeros(width))
        self.arc_sin_factors = nn.Parameter(torch.normal(
            torch.zeros(width),
            0.01 * torch.ones(width)
        ))

    def forward(self, x):
        factors = torch.sin(self.arc_sin_factors)
        if len(x.shape) == 2:
            return x * factors
        elif len(x.shape) == 4:
            return x * factors[:, None, None]
        else:
            raise NotImplementedError

