"""
SDP-based Lipschitz Layers,
introduced in paper https://openreview.net/pdf?id=k71IGLC8cfc.
Code (adapted) from
https://github.com/araujoalexandre/Lipschitz-SLL-Networks/blob/main/core/models/layers.py
"""

from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class SLLConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 initializer: Optional[Callable] = None,
                 padding: Union[_size_2_t, str] = 'same',
                 padding_mode: str = 'circular',
                 **kwargs) -> None:
        super().__init__()

        self.activation = nn.ReLU(inplace=False)

        self.kernel = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.q = nn.Parameter(torch.randn(out_channels))

        self.val_rescaling_cached = None

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.padding = padding

        self.epsilon = 1e-6

    def forward(self, x):
        res = F.conv2d(x, self.kernel, bias=self.bias, padding=self.padding)
        res = self.activation(res)

        if self.training:
            self.val_rescaling_cached = None
            rescaling = self.get_rescaling()
        elif self.val_rescaling_cached is None:
            rescaling = self.get_rescaling()
            self.val_rescaling_cached = rescaling
        else:
            rescaling = self.val_rescaling_cached

        res = rescaling[None, :, None, None] * res
        if self.padding == 'same':
            padding = (
                (self.kernel.shape[-2]-1) // 2,
                (self.kernel.shape[-1]-1) // 2
            )
        else:
            padding = self.padding
        res = F.conv_transpose2d(res, self.kernel, padding=padding)
        out = x - res
        return out

    def get_rescaling(self):
        kkt = F.conv2d(self.kernel, self.kernel,
                       padding=self.kernel.shape[-1] - 1)
        q_abs = torch.abs(self.q) + self.epsilon
        kkt_rs = kkt * q_abs[None, :, None, None]
        bound = kkt_rs.abs().sum((1, 2, 3)) / q_abs
        rescaling = 2 / (bound + self.epsilon)
        return rescaling


class SLLLinear(nn.Module):

    def __init__(self, config, cin, cout, epsilon=1e-6):
        super().__init__()

        self.activation = nn.ReLU(inplace=False)
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.rand(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        q_abs = torch.abs(self.q)
        q = q_abs[None, :]
        q_inv = (1 / (q_abs + self.epsilon))[:, None]
        T = 2 / (torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1)
                 + self.epsilon)
        res = T * res
        res = F.linear(res, self.weights.t())
        out = x - res
        return out
