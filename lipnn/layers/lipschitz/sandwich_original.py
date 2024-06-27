import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

# from https://github.com/locuslab/orthogonal-convolutions


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


def fft_shift_matrix(n, s):
    shift = torch.arange(0, n).repeat((n, 1))
    shift = shift + shift.T
    return torch.exp(1j * 2 * np.pi * s * shift / n)


class StridedConv(nn.Module):
    def __init__(self, *args, **kwargs):
        striding = False
        if 'stride' in kwargs and kwargs['stride'] == 2:
            kwargs['stride'] = 1
            striding = True
        super().__init__(*args, **kwargs)
        downsample = "b c (w k1) (h k2) -> b (c k1 k2) w h"
        if striding:
            self.register_forward_pre_hook(lambda _, x:
                                           einops.rearrange(x[0], downsample, k1=2, k2=2))


class PaddingChannels(nn.Module):
    def __init__(self, ncin, ncout, scale=1.0):
        super().__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.scale = scale

    def forward(self, x):
        bs, _, size1, size2 = x.shape
        out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
        out[:, :self.ncin] = self.scale * x
        return out


class PaddingFeatures(nn.Module):
    def __init__(self, fin, n_features, scale=1.0):
        super().__init__()
        self.n_features = n_features
        self.fin = fin
        self.scale = scale

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.n_features, device=x.device)
        out[:, :self.fin] = self.scale * x
        return out


class PlainConv(nn.Conv2d):
    def forward(self, x):
        return super().forward(F.pad(x, (1, 1, 1, 1)))


class LinearNormalized(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super(LinearNormalized, self).__init__(in_features, out_features, bias)
        self.scale = scale

    def forward(self, x):
        self.Q = F.normalize(self.weight, p=2, dim=1)
        return F.linear(self.scale * x, self.Q, self.bias)


class FirstChannel(nn.Module):
    def __init__(self, cout, scale=1.0):
        super().__init__()
        self.cout = cout
        self.scale = scale

    def forward(self, x):
        xdim = len(x.shape)
        if xdim == 4:
            return self.scale * x[:, :self.cout, :, :]
        elif xdim == 2:
            return self.scale * x[:, :self.cout]


class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(
            1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.AB = AB
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B @ x
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T)  # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x


class SandwichFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(
            1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.psi = nn.Parameter(torch.zeros(
            out_features, dtype=torch.float32, requires_grad=True))
        self.Q = None

    def forward(self, x):
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


class SandwichConv1(nn.Module):
    def __init__(self, cin, cout, scale=1.0) -> None:
        super().__init__()
        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin+cout))
        self.bias = nn.Parameter(torch.empty(cout))
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.alpha = nn.Parameter(torch.ones(
            1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.kernel.norm()
        self.Q = None

    def forward(self, x):
        cout = self.kernel.shape[0]
        if self.training or self.Q is None:
            P = cayley(self.alpha * self.kernel / self.kernel.norm())
            self.Q = 2 * P[:, :cout].T @ P[:, cout:]
        Q = self.Q if self.training else self.Q.detach()
        x = F.conv2d(self.scale * x, Q[:, :, None, None])
        x += self.bias[:, None, None]
        return F.relu(x)


class SandwichConv1Lin(nn.Module):
    def __init__(self, cin, cout, scale=1.0) -> None:
        super().__init__()
        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin+cout))
        self.bias = nn.Parameter(torch.empty(cout))
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.alpha = nn.Parameter(torch.ones(
            1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.kernel.norm()
        self.Q = None

    def forward(self, x):
        cout = self.kernel.shape[0]
        if self.training or self.Q is None:
            P = cayley(self.alpha * self.kernel / self.kernel.norm())
            self.Q = 2 * P[:, :cout].T @ P[:, cout:]
        Q = self.Q if self.training else self.Q.detach()
        x = F.conv2d(self.scale * x, Q[:, :, None, None])
        x += self.bias[:, None, None]
        return x


class SandwichConvLin(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0]  # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2)  # //2 kernel_size; optional
                kwargs['padding'] = args[2] // 2  # TODO: added maxes recently
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        args[0] += args[1]
        args = tuple(args)
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.register_parameter('alpha', None)
        self.Qfft = None

    def forward(self, x):
        x = self.scale * x
        cout, chn, _, _ = self.weight.shape
        cin = chn - cout
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(
                n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)

        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(
                cout, chn, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(
                    wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())

        Qfft = self.Qfft if self.training else self.Qfft.detach()
        # Afft, Bfft = Qfft[:,:,:cout], Qfft[:,:,cout:]
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(
            n * (n // 2 + 1), cin, batches)
        xfft = 2 * Qfft[:, :, :cout].conj().transpose(1,
                                                      2) @ Qfft[:, :, cout:] @ xfft
        x = torch.fft.irfft2(xfft.reshape(
            n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))
        if self.bias is not None:
            x += self.bias[:, None, None]

        return x


class SandwichConv(StridedConv, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0]  # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2)  # //2 kernel_size; optional
                kwargs['padding'] = args[2] // 2  # TODO: added maxes recently
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        #args[0] += args[1]
        #args = tuple(args)
        new_in_channels = in_channels + out_channels
        super().__init__(new_in_channels, out_channels, kernel_size, **kwargs)
        self.psi = nn.Parameter(torch.zeros_like(self.bias))
        self.scale = scale
        self.register_parameter('alpha', None)
        self.Qfft = None

    def forward(self, x):
        x = self.scale * x
        cout, chn, _, _ = self.weight.shape
        cin = chn - cout
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(
                n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)

        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(
                cout, chn, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(
                    wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())

        Qfft = self.Qfft if self.training else self.Qfft.detach()
        # Afft, Bfft = Qfft[:,:,:cout], Qfft[:,:,cout:]
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(
            n * (n // 2 + 1), cin, batches)
        xfft = 2 ** 0.5 * \
            torch.exp(-self.psi).diag().type(
                xfft.dtype) @ Qfft[:, :, cout:] @ xfft
        x = torch.fft.irfft2(xfft.reshape(
            n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))
        if self.bias is not None:
            x += self.bias[:, None, None]
        xfft = torch.fft.rfft2(F.relu(x)).permute(
            2, 3, 1, 0).reshape(n * (n // 2 + 1), cout, batches)
        xfft = 2 ** 0.5 * Qfft[:, :, :cout].conj().transpose(1,
                                                             2) @ torch.exp(self.psi).diag().type(xfft.dtype) @ xfft
        x = torch.fft.irfft2(xfft.reshape(
            n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))

        return x
