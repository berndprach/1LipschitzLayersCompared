from typing import Union, Optional, Callable
from torch import nn, Tensor
from torch.nn.common_types import _size_2_t
from .spectral_normal_control import bjorck_orthonormalize
from torch.nn.utils.parametrize import register_parametrization


class Bjorck_and_Bowie(nn.Module):
    def __init__(self,
                 bjorck_iters_train,
                 bjorck_iters_eval,
                 bjorck_beta=0.5,
                 bjorck_order=1,
                 bjorck_iters_scheduler=None,
                 power_iteration_scaling=True):
        super().__init__()
        self.bjorck_beta = bjorck_beta
        self.bjorck_iters_train = bjorck_iters_train
        self.bjorck_iters_eval = bjorck_iters_eval
        self.bjorck_order = bjorck_order
        self.bjorck_iters_scheduler = bjorck_iters_scheduler
        self.power_iteration_scaling = power_iteration_scaling

    def forward(self, weight: Tensor) -> Tensor:
        if weight.shape == 4:
            assert weight.shape[-1] == weight.shape[-2] == 1
            # Here was a missing return.
            return self.forward(weight[:, :, 0, 0]).reshape(weight.shape)
        bjorck_iters = self.bjorck_iters_train if self.training else self.bjorck_iters_eval
        return bjorck_orthonormalize(
            weight,
            beta=self.bjorck_beta,
            iters=bjorck_iters,
            order=self.bjorck_order,
            power_iteration_scaling=self.power_iteration_scaling,
            default_scaling=not self.power_iteration_scaling,
        )

    def right_inverse(self, weight: Tensor) -> Tensor:
        return bjorck_orthonormalize(
            weight,
            beta=self.bjorck_beta,
            iters=self.bjorck_iters_eval,
            order=self.bjorck_order,
            power_iteration_scaling=self.power_iteration_scaling,
            default_scaling=not self.power_iteration_scaling,
        )


class BjorckLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, initializer: Optional[Callable] = None,
                 bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        if initializer is not None:
            initializer(self.weight)
        else:
            nn.init.orthogonal_(self.weight)
        register_parametrization(self, "weight", Bjorck_and_Bowie(
            bjorck_iters_train=3, bjorck_iters_eval=15))


class BjorckConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: _size_2_t, initializer: Optional[Callable] = None,
                 stride: _size_2_t = 1, padding: Union[_size_2_t, str] = 0, dilation: _size_2_t = 1,
                 groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)
        assert self.kernel_size[0] == self.kernel_size[1] == 1, 'Only 1x1 convolutions are supported'
        if initializer is not None:
            initializer(self.weight)
        else:
            self.weight.data[:, :, 0, 0] = nn.init.orthogonal(
                self.weight.data[:, :, 0, 0])
        register_parametrization(self, "weight", Bjorck_and_Bowie(
            bjorck_iters_train=3, bjorck_iters_eval=15))


if __name__ == '__main__':
    import torch
    fc = BjorckLinear(10, 10)
    x = torch.randn(1, 10)
    print(fc(x))
    conv = BjorckConv2d(3, 3, 1)
    x = torch.randn(1, 3, 10, 10)
    print(conv(x))
    print()
