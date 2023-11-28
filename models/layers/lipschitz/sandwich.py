"""
Adapted from https://github.com/acfr/LBDN/blob/main/layer.py.
"""
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.common_types import _size_2_t

# from https://github.com/locuslab/orthogonal-convolutions


def cayley(W):  # Typically, W.shape = (s*s//2+s, c, 2c), S := s*s//2+s
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)

    # print(W.shape)
    # print(W)

    U, V = W[:, :cin], W[:, cin:]  # shapes: (., c, c), (., c, c)
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V

    # print(A.shape)
    # print(A)

    # Complexity: S * c^3
    iIpA = torch.inverse(I + A)  # Complexity: S * c^3
    result = torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)
    # Complexity: S * 2 * c^3
    return result


def fft_shift_matrix(n, s):
    shift = torch.arange(0, n).repeat((n, 1))
    shift = shift + shift.T
    return torch.exp(1j * 2 * np.pi * s * shift / n)


class SandwichConv(nn.Conv2d):
    # def __init__(self, *args, **kwargs):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 # initializer: Optional[Callable] = None,
                 padding: Union[_size_2_t, str] = 'same',
                 # padding_mode: str = 'circular',
                 numerical_stable: bool = False,
                 **kwargs) -> None:
        # args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            raise NotImplementedError()
            """
            args = list(args)
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2) # //2 kernel_size; optional
                kwargs['padding'] = args[2] // 2 # TODO: added maxes recently
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
            """
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        # args[0] += args[1]
        new_in = in_channels + out_channels

        # args = tuple(args)
        super().__init__(new_in, out_channels, kernel_size,
                         padding=padding, **kwargs)
        # Rename in_channels with the proper number of channels
        self.in_channels = in_channels
        # Potential place for producing NaN? (e.g. exp(30) ~ NaN)
        # self.psi = nn.Parameter(torch.zeros(args[1]))
        self.psi = nn.Parameter(torch.zeros(out_channels))
        # self.psi = nn.Parameter(torch.zeros(out_channels), requires_grad=False)

        self.scale = scale
        self.register_parameter('alpha', None)
        self.Qfft = None

        self.numerical_stable = numerical_stable
        # self.alpha: Optional[nn.Parameter] = None
        # self.shift_matrix: Optional[torch.Tensor] = None

    def unstable_forward(self, x):
        x = self.scale * x
        cout, chn, _, _ = self.weight.shape
        cin = chn - cout
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            # s = (self.weight.shape[2] - 1) // 2
            # self.shift_matrix = fft_shift_matrix(n, -s)[:, :( n/ /2 + 1)]
            # .reshape(n * (n // 2 + 1), 1, 1).to(x.device)
            self.set_shift_matrix(n, x.device)

        if self.training or self.Qfft is None or self.alpha is None:
            # weight.shape = (co, ci+co, k, k)
            w_fft = torch.fft.rfft2(self.weight, s=(n, n))
            w_fft_rh = w_fft.reshape(cout, chn, n * (n // 2 + 1))  # (c, 2c, .)
            w_fft_perm = w_fft_rh.permute(2, 0, 1).conj()  # (n*n/2+n, c, 2c)
            wfft = self.shift_matrix * w_fft_perm  # (n*n/2+n, c, 2c)
            # wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n))
            # .reshape(cout, chn, n * (n // 2 + 1)).permute(2, 0, 1).conj()

            if self.alpha is None:
                # self.alpha = nn.Parameter(
                #     torch.tensor(wfft.norm().item(), requires_grad=True).to(
                #         x.device))
                self.set_alpha(wfft, x.device)

            # Potential place producing NaN?:
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())

        Qfft = self.Qfft if self.training else self.Qfft.detach()
        # Afft, Bfft = Qfft[:,:,:cout], Qfft[:,:,cout:]
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1),
                                                              cin, batches)
        xfft = 2 ** 0.5 * torch.exp(-self.psi).diag().type(xfft.dtype) @ Qfft[
            :, :,
            cout:] @ xfft
        x = torch.fft.irfft2(
            xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))
        if self.bias is not None:
            x += self.bias[:, None, None]
        xfft = torch.fft.rfft2(F.relu(x)).permute(2, 3, 1, 0).reshape(
            n * (n // 2 + 1), cout, batches)
        xfft = 2 ** 0.5 * Qfft[:, :, :cout].conj().transpose(1, 2) @ torch.exp(
            self.psi).diag().type(xfft.dtype) @ xfft

        # Maybe a problem with 1x1?
        x = torch.fft.irfft2(
            xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1)
        )

        return x

    def stable_forward(self, x):
        x = self.scale * x
        cout, chn, _, _ = self.weight.shape
        cin = chn - cout
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            # s = (self.weight.shape[2] - 1) // 2
            # self.shift_matrix = fft_shift_matrix(n, -s)[:, :( n/ /2 + 1)]
            # .reshape(n * (n // 2 + 1), 1, 1).to(x.device)
            self.set_shift_matrix(n, x.device)

        if self.training or self.Qfft is None or self.alpha is None:
            # weight.shape = (co, ci+co, k, k)
            w_fft = torch.fft.rfft2(self.weight, s=(n, n))
            w_fft_rh = w_fft.reshape(cout, chn, n * (n // 2 + 1))  # (c, 2c, .)
            w_fft_perm = w_fft_rh.permute(2, 0, 1).conj()  # (n*n/2+n, c, 2c)
            wfft = self.shift_matrix * w_fft_perm  # (n*n/2+n, c, 2c)
            # wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n))
            # .reshape(cout, chn, n * (n // 2 + 1)).permute(2, 0, 1).conj()

            if self.alpha is None:
                # self.alpha = nn.Parameter(
                #     torch.tensor(wfft.norm().item(), requires_grad=True).to(
                #         x.device))
                self.set_alpha(wfft, x.device)

            # Potential place producing NaN?:
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())

        Qfft = self.Qfft if self.training else self.Qfft.detach()
        # Afft, Bfft = Qfft[:,:,:cout], Qfft[:,:,cout:]
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1),
                                                              cin, batches)
        xfft = 2 ** 0.5 * Qfft[:, :, cout:] @ xfft
        x = torch.fft.irfft2(
            xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))
        if self.bias is not None:
            x += torch.exp(self.psi)[:, None, None] * self.bias[:, None, None]
        xfft = torch.fft.rfft2(F.relu(x)).permute(2, 3, 1, 0).reshape(
            n * (n // 2 + 1), cout, batches)
        xfft = 2 ** 0.5 * Qfft[:, :, :cout].conj().transpose(1, 2) @ xfft

        # Maybe a problem with 1x1?
        x = torch.fft.irfft2(
            xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1)
        )

        return x

    def forward(self, x: torch.Tensor):
        if self.numerical_stable:
            return self.stable_forward(x)
        else:
            return self.unstable_forward(x)

    def set_shift_matrix(self, n, device):  # -> shape (n*n//2+n, 1, 1)
        s = (self.weight.shape[2] - 1) // 2
        shift_matrix_values = fft_shift_matrix(n, -s)[:, :(n // 2 + 1)]
        shift_matrix = shift_matrix_values.reshape(n * (n // 2 + 1), 1, 1)
        self.shift_matrix = shift_matrix.to(device)

    def set_alpha(self, wfft, device):
        # alpha = nn.Parameter(torch.tensor(wfft.norm().item(),
        #                                   requires_grad=True))
        # self.alpha = alpha.to(device)

        alpha_tensor = torch.tensor(wfft.norm().item(), requires_grad=True)
        alpha_tensor.to(device)
        self.alpha = nn.Parameter(alpha_tensor)


class SandwichFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, numerical_stable: bool = False):
        super().__init__(in_features+out_features, out_features, bias)
        # Rename in_features with the proper number of channels
        self.in_features = in_features
        self.alpha = nn.Parameter(torch.ones(1,
                                             dtype=torch.float32,
                                             requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.psi = nn.Parameter(torch.zeros(out_features,
                                            dtype=torch.float32,
                                            requires_grad=True))
        self.numerial_stable = numerical_stable
        self.Q = None

    def unstable_forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B*h
        if self.psi is not None:
            # sqrt(2) \Psi^{-1} B * h
            x = x * torch.exp(-self.psi) * (2 ** 0.5)
        if self.bias is not None:
            x += self.bias
        x = F.relu(x) * torch.exp(self.psi)  # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T)  # sqrt(2) A^top \Psi z
        return x

    def stable_forward(self, x):
        # Psi is only applied to the bias
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B*h
        if self.psi is not None:
            # sqrt(2) \Psi^{-1} B * h
            x = x * (2 ** 0.5)
        if self.bias is not None:
            x += self.bias*torch.exp(self.psi)
        x = F.relu(x)
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T)  # sqrt(2) A^top \Psi z
        return x

    def forward(self, x: torch.Tensor):
        if self.numerial_stable:
            return self.stable_forward(x)
        else:
            return self.unstable_forward(x)


if __name__ == "__main__":
    print('checking linear')
    sw = SandwichFc(16, 16)
    x = torch.randn(2, 16)
    y_0 = sw.unstable_forward(x)
    y_1 = sw.stable_forward(x)
    check = torch.allclose(y_0, y_1, atol=0.001)
    print(f'Outputs are the same: {check}')
    print('checking conv')
    sw = SandwichConv(16, 16, 3)
    x = torch.randn(2, 16, 8, 8)
    y_0 = sw.unstable_forward(x)
    y_1 = sw.stable_forward(x)
    check = torch.allclose(y_0, y_1, atol=0.001)
    print(f'Outputs are the same: {check}')
