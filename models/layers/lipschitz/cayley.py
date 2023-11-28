"""
Cayley Implementation is adapted from
https://github.com/locuslab/orthogonal-convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)

    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)


class CayleyConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_parameter('alpha', None)

        # self.wfft_cache = None
        self.val_cache = None  # caching fft. and mat. inv. computations.

    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * torch.pi * s * shift / n)

    def get_weight_fft(self, cout, cin, n):  # -> torch.Tensor[n^2/2, c, c]:
        wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(
            cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()

        if self.alpha is None:
            # self.alpha = nn.Parameter(torch.tensor(
            #     wfft.norm().item(), requires_grad=True).to(x.device))
            self.alpha = nn.Parameter(torch.tensor(
                # wfft.norm().item(), requires_grad=True).to(wfft.device))
                wfft.norm().item(), requires_grad=True).to(self.weight.device))

        return cayley(self.alpha * wfft / wfft.norm())

    def forward(self, x):
        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(
                n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)

        if self.training:
            self.val_cache = None
            weight_fft = self.get_weight_fft(cout, cin, n)
        else:
            if self.val_cache is None:
                self.val_cache = self.get_weight_fft(cout, cin, n)
            weight_fft = self.val_cache

        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(
            n * (n // 2 + 1), cin, batches)

        yfft = (weight_fft @ xfft).reshape(n, n // 2 + 1, cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y


class CayleyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.empty(1).fill_(
            self.weight.norm().item()), requires_grad=True)

        self.Q_cached = None

    def reset_parameters(self):
        std = 1 / self.weight.shape[1] ** 0.5
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

        # self.Q = None
        self.Q_cached = None

    def forward(self, X):
        if self.training:
            self.Q_cached = None
            # self.Q = cayley(self.alpha * self.weight / self.weight.norm())
            Q = cayley(self.alpha * self.weight / self.weight.norm())
        else:
            if self.Q_cached is None:
                with torch.no_grad():
                    self.Q_cached = cayley(
                        self.alpha * self.weight / self.weight.norm())
            Q = self.Q_cached
            # with torch.no_grad():
            #     self.Q = cayley(self.alpha * self.weight / self.weight.norm())

        # return F.linear(X, self.Q, self.bias)
        return F.linear(X, Q, self.bias)


if __name__ == '__main__':
    fc = CayleyLinear(7, 5)
    print(fc.weight)
    x = torch.randn(1, 7)
    y = fc(x)
    print()
