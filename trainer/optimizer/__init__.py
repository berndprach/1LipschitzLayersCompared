from functools import partial
from typing import Iterable, Protocol

from torch import nn, optim
from torch.optim import Optimizer


class PartialOptimizer(Protocol):
    r'''
    Protocol for the partially initialized optimizer. Only parameters are
    allowed
    '''
    def __call__(self, params: Iterable[nn.Parameter]) -> Optimizer:
        ...


def Adams(**kwargs) -> PartialOptimizer:
    r"""
        Wrapper of the original Adam optimizer for partial initialization
    """
    return partial(optim.Adam, **kwargs)


def SGD(**kwargs) -> PartialOptimizer:
    r"""
        Wrapper of the original Adam optimizer for partial initialization
    """
    return partial(optim.SGD, **kwargs)
