from typing import Optional, Callable

import torch
from torch import nn, Tensor

from .spectral_normal_control import bjorck_orthonormalize as bjorck_bowie_orthonormalize


class BnBLinearRescale(nn.Module):
    def __init__(self,
                 train_iterations,
                 val_iterations,
                 **kwargs):
        super().__init__()

        self.train_iterations = train_iterations
        self.val_iterations = val_iterations
        self.kwargs = kwargs

    def forward(self, weight: Tensor) -> Tensor:
        if self.training:
            return bjorck_bowie_orthonormalize(
                weight,
                iters=self.train_iterations,
                **self.kwargs,
            )
        else:
            return bjorck_bowie_orthonormalize(
                weight,
                iters=self.val_iterations,
                **self.kwargs,
            )


class BnBLinear(nn.Linear):
    def __init__(self,
                 *args,
                 initializer: Optional[Callable] = None,
                 train_iterations=3,
                 val_iterations=15,
                 **kwargs) -> None:

        super().__init__(*args, **kwargs)

        if initializer is None:
            initializer = nn.init.eye_
        initializer(self.weight)

        self.rescale = BnBLinearRescale(train_iterations, val_iterations)
        self.val_cache = None

    def forward(self, x):
        if self.training:
            self.val_cache = None
            weight = self.rescale(self.weight)
            return torch.nn.functional.linear(x, weight, self.bias)

        if self.val_cache is None:
            self.val_cache = self.rescale(self.weight)

        return torch.nn.functional.linear(x, self.val_cache, self.bias)


class BnBLinearBCOP(BnBLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         train_iterations=20,
                         val_iterations=20,
                         **kwargs)
