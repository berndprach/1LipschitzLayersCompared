"""
Convex Potential Layer:
https://arxiv.org/pdf/2110.12690.pdf
Based on
https://github.com/araujoalexandre/Lipschitz-SLL-Networks/blob/main/core/models/layers.py
Implementation by the original authors (using a different initialization):
https://github.com/MILES-PSL/Convex-Potential-Layer/blob/main/layers.py
"""

from typing import Optional, Callable, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.common_types import _size_2_t


class CPLConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 initializer: Optional[Callable] = None,
                 padding: Union[_size_2_t, str] = 'same',
                 padding_mode: str = 'circular',
                 val_niter: int = 100,
                 **kwargs) -> None:
        super().__init__()

        self.activation = nn.ReLU(inplace=False)
        self.val_niter = val_niter
        self.kernel = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.u_initialized = False

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.padding = padding

        self.epsilon = 1e-6

        self.val_rescaling_cached = None

    @torch.no_grad()
    def update_uv(self, num_iteration, epsilon):
        u, v = self.u, None
        tp_padding = (self.kernel.shape[-1] - 1) // 2
        for _ in range(num_iteration):
            v0 = F.conv2d(u, self.kernel, padding="same")
            v = v0 / (torch.norm(v0, p=2) + epsilon)
            u0 = F.conv_transpose2d(v, self.kernel, padding=tp_padding)
            u = u0 / (torch.norm(u0, p=2) + epsilon)
        return u, v

    def calculate_rescaling(self, u, v):
        conv_u = F.conv2d(u, self.kernel, padding="same")
        # rescaling = 2 / ((conv_u * v).sum((1, 2, 3)) ** 2)
        rescaling = 2 / ((conv_u * v).sum() ** 2 + self.epsilon)
        return rescaling

    def forward(self, x):
        res = F.conv2d(x, self.kernel, bias=self.bias, padding=self.padding)
        res = self.activation(res)
        tp_padding = (self.kernel.shape[-1] - 1) // 2
        res = F.conv_transpose2d(res, self.kernel, padding=tp_padding)

        if not self.u_initialized:
            u_size = (1, self.kernel.shape[0], x.shape[-2], x.shape[-1])
            u = torch.randn(u_size, device=x.device)
            self.register_buffer("u", u)
            self.u_initialized = True

        if self.training:
            self.val_rescaling_cached = None
            u, v = self.update_uv(1, self.epsilon)
            self.u = u
            rescaling = self.calculate_rescaling(u, v)
        elif self.val_rescaling_cached is None:
            # u, v = self.update_uv(self.val_niter, 0.)
            u, v = self.update_uv(self.val_niter, 1e-12)
            rescaling = self.calculate_rescaling(u, v)
            self.val_rescaling_cached = rescaling
        else:
            rescaling = self.val_rescaling_cached

        return x - rescaling * res


# CPL only for testing
class CPLConv2d10k(CPLConv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 initializer: Optional[Callable[..., Any]] = None,
                 padding: Union[_size_2_t, str] = 'same',
                 padding_mode: str = 'circular', **kwargs) -> None:
        super().__init__(in_channels, out_channels, kernel_size,
                         initializer, padding, padding_mode, 10000, **kwargs)
        self.tolerance = 1e-6

    @torch.no_grad()
    def update_uv(self, num_iteration, epsilon):
        u, v = self.u, None
        tp_padding = (self.kernel.shape[-1] - 1) // 2
        rescaling = 0.
        for _ in range(num_iteration):
            v0 = F.conv2d(u, self.kernel, padding="same")
            v = v0 / (torch.norm(v0, p=2) + epsilon)
            u0 = F.conv_transpose2d(v, self.kernel, padding=tp_padding)
            u = u0 / (torch.norm(u0, p=2) + epsilon)
            new_rescaling = self.calculate_rescaling(u, v)
            if torch.abs(new_rescaling - rescaling) < self.tolerance:
                break
            rescaling = new_rescaling
        return u, v
