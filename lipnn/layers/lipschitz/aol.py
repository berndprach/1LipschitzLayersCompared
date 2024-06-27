from typing import Optional, Callable, Union

from torch import nn, Tensor
import torch
from torch.nn.common_types import _size_2_t
from torch.nn.utils.parametrize import register_parametrization


class AOLConv2DRescaling(nn.Module):
    @staticmethod
    def forward(weight: Tensor) -> Tensor:
        """ Expected weight shape: out_channels x in_channels x ks1 x ks_2 """
        _, _, k1, k2 = weight.shape
        weight_tp = weight.transpose(0, 1)
        v = torch.nn.functional.conv2d(
            weight_tp, weight_tp, padding=(k1 - 1, k2 - 1))
        v_scaled = v.abs().sum(dim=(1, 2, 3), keepdim=True).transpose(0, 1)
        return weight / (v_scaled + 1e-6).sqrt()


class AOLConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 initializer: Optional[Callable] = None,
                 padding: Union[_size_2_t, str] = 'same',
                 padding_mode: str = 'circular',
                 **kwargs) -> None:

        super().__init__(in_channels, out_channels, kernel_size,
                         padding=padding, padding_mode=padding_mode, **kwargs)

        if initializer is None:
            initializer = nn.init.dirac_
        initializer(self.weight)

        torch.nn.init.zeros_(self.bias)

        register_parametrization(self, 'weight', AOLConv2DRescaling())


class AOLConv2dOrth(AOLConv2d):
    """ This class is an alias for AOLConv2d with orthogonal init. """
    def __init__(self, *args, **kwargs):
        initializer = torch.nn.init.orthogonal_
        super().__init__(*args, initializer=initializer, **kwargs)


class AOLConv2dDirac(AOLConv2d):
    """ This class is an alias for AOLConv2d with Dirac initialization. """
    def __init__(self, *args, **kwargs):
        initializer = torch.nn.init.dirac_
        super().__init__(*args, initializer=initializer, **kwargs)
