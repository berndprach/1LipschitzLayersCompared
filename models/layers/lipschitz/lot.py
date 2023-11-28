"""
Based on https://github.com/AI-secure/Layerwise-Orthogonal-Training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class LOT(nn.Conv2d):
    def __init__(self, *args, iter_T: int = 10, eval_iter_T: int = 10, **kwargs):
        if 'stride' in kwargs and kwargs['stride'] == 2:
            self.strd = 2
            args = tuple((args[0]*self.strd*self.strd,)) + args[1:]
            kwargs['stride'] = 1
        else:
            self.strd = 1
        self.iter_T = iter_T
        self.eval_iter_T = eval_iter_T

        # self.use_cached_w = False
        self.w_cached = None

        super().__init__(*args, **kwargs)

        if self.weight.shape[0] == self.weight.shape[1]:
            # Identity Init
            N, N, K, K2 = self.weight.shape
            assert K == K2
            self.weight.data.zero_()
            self.weight.data[np.arange(N), np.arange(N), K//2, K//2] = 1.0
            if self.bias is not None:
                self.bias.data.zero_()

    def fft_shift_matrix(self, n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)

    def forward(self, x):
        if self.strd > 1:
            x = einops.rearrange(
                x, "b c (w k1) (h k2) -> b (c k1 k2) w h", k1=self.strd, k2=self.strd)

        padded_n = 0
        assert len(
            self.kernel_size) == 2 and self.kernel_size[0] == self.kernel_size[1]
        if self.kernel_size[0] > 1:  # zero-pad
            x = F.pad(x, (self.kernel_size[0]//2,)*4)
            padded_n = padded_n + self.kernel_size[0]//2

        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape

        xfft = torch.fft.fft2(x).permute(
            2, 3, 1, 0).reshape(n * n, cin, batches)

        # if self.use_cached_w:
        #     wfft_ortho = self.cached_w
        # else:

        # if self.training or self.w_cached is None:
        #     wfft_ortho = self.calculate_wfft(x.device, n)
        # else:
        #     wfft_ortho = self.w_cached
        if self.training:
            self.w_cached = None
            wfft_ortho = self.calculate_wfft(x.device, n)
        elif self.w_cached is None:
            wfft_ortho = self.calculate_wfft(x.device, n)
            self.w_cached = wfft_ortho
        else:
            wfft_ortho = self.w_cached

        # self.w_cached = None if self.training else wfft_ortho

        zfft = wfft_ortho @ xfft
        zfft = zfft.reshape(n, n, cout, batches).permute(3, 2, 0, 1)
        z = torch.fft.ifft2(zfft).real

        if padded_n > 0:
            z = z[:, :, padded_n:-padded_n, padded_n:-padded_n]
        if self.bias is not None:
            z += self.bias[:, None, None]

        return z

    def calculate_wfft(self, device, n):
        cout, cin, _, _ = self.weight.shape

        shift_matrix = self.fft_shift_matrix(
            n, -(self.weight.shape[2] - 1) // 2).to(device)
        wfft = (shift_matrix * torch.fft.fft2(self.weight, (n, n)
                                              ).conj()).reshape(cout, cin,
                                                                n * n).permute(
            2, 0, 1)
        wfft_normed = wfft
        # conj - unitary, not orthogonal
        sfft = wfft_normed @ wfft_normed.transpose(1, 2).conj()
        sfft = sfft + 1e-4 * \
               torch.eye(sfft.shape[-1]).to(sfft).unsqueeze(0)
        norm_sfft = sfft.norm(p=None, dim=(1, 2), keepdim=True) + 1e-4
        sfft = sfft.div(norm_sfft)

        I = torch.eye(cout, dtype=sfft.dtype).to(
            sfft.device).expand(sfft.shape)
        Y, Z = sfft, I
        if self.training:
            iter_T = self.iter_T
        else:
            iter_T = self.eval_iter_T
        for t in range(iter_T):
            T = (0.5 + 0j) * ((3 + 0j) * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z
        bfft = Z
        wfft_ortho = (bfft @ wfft_normed) / (norm_sfft.sqrt())
        return wfft_ortho

    def frozen_w_ortho(self, n):
        if self.strd > 1:
            n = n // self.strd
        n = n + self.kernel_size[0]//2 * 2
        cout, cin, _, _ = self.weight.shape
        shift_matrix = self.fft_shift_matrix(
            n, -(self.weight.shape[2]-1)//2).to(self.weight.device)
        wfft = (shift_matrix * torch.fft.fft2(self.weight, (n, n)).conj()
                ).reshape(cout, cin, n * n).permute(2, 0, 1)
        wfft_normed = wfft
        wfft_normed = wfft_normed.cdouble()  # double
        # conj - unitary, not orthogonal
        sfft = wfft_normed @ wfft_normed.transpose(1, 2).conj()
        sfft = sfft + 1e-4 * torch.eye(sfft.shape[-1]).to(sfft).unsqueeze(0)
        norm_sfft = sfft.norm(p=None, dim=(1, 2), keepdim=True) + 1e-4
        sfft = sfft.div(norm_sfft)

        I = torch.eye(cout, dtype=sfft.dtype).to(
            sfft.device).expand(sfft.shape)
        Y, Z = sfft, I
        if self.training:
            iter_T = self.iter_T
        else:
            iter_T = self.eval_iter_T
        for t in range(iter_T):
            T = (0.5+0j) * ((3+0j) * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z
        bfft = Z
        wfft_ortho = (bfft @ wfft_normed) / (norm_sfft.sqrt())
        wfft_ortho = wfft_ortho.cfloat()

        self.use_cached_w = True
        self.cached_w = wfft_ortho.detach()


class LOT2t(LOT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, iter_T=2, eval_iter_T=10, **kwargs)
