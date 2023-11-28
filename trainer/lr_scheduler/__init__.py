from functools import partial
from typing import Protocol

from torch.optim import lr_scheduler
# from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from ..optimizer import Optimizer


class PartialLRScheduler(Protocol):
    r'''
    Protocol for partial lr scheduler. Only the optimizer is allowed during
    the initialization.
    '''

    def __call__(self, optimizer: Optimizer) -> LRScheduler:
        ...


def MultiStepLR(**kwargs) -> PartialLRScheduler:
    r"""
        Wrapper of the original MultiStep scheduler for partial initialization
    """
    return partial(lr_scheduler.MultiStepLR, **kwargs)


class OneCycleLR_(lr_scheduler.OneCycleLR):
    r"""
    Since the original scheduler requires the maximum learning rate to be specified,
    for a better generalization we deduce this value from the first learning rate
    of the optimizer. The number of steps per epoch is fixed to 1.
    """

    def __init__(self, optimizer, epochs, **kwargs):
        max_lr = optimizer.param_groups[0]['lr']
        max_lr = [pg['lr'] for pg in optimizer.param_groups]

        super().__init__(optimizer, max_lr=max_lr,
                         epochs=epochs, steps_per_epoch=1, **kwargs,)


def OneCycleLR(**kwargs) -> PartialLRScheduler:
    r"""
    Wrapper of the original OneCycleLR scheduler for partial initialization
    """
    return partial(OneCycleLR_, **kwargs)
