
import torch
import torch.nn as nn
from torch.nn.functional import lp_pool2d

class L2Pooling(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim
        self.kwargs = kwargs  # e.g. keepdims=True
        self.eps = 1e-8

    def forward(self, x):
        l2_squared = torch.sum(x**2, dim=self.dim, **self.kwargs)
        return torch.sqrt(l2_squared + self.eps)

class AdaptiveL2Pooling2d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return lp_pool2d(x,2,kernel_size=x.shape[-2:],**self.kwargs)