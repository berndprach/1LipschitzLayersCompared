r"""
Utility functions taken from https://github.com/ColinQiyangLi/LConvNet.git
"""
from typing import Tuple

import torch
import numpy as np


@torch.no_grad()
def module_symmetric_power_iteration_all_iterations(module: torch.nn.Module,
                                                    input_size: Tuple,
                                                    nrof_iterations: int,
                                                    tolerance: float = 1e-6,
                                                    ):
    """
    Calculated power iteration on JTJ, for J the Jacobian of the module.
    The Jacobian is evaluated at the zero vector.
    returns the square root of the current estimates of the largest eigenvalue
    of JTJ.
    """
    all_values = torch.empty(nrof_iterations)

    weight_device = next(module.parameters()).device
    u = torch.randn(1, *input_size, device=weight_device)
    zeros = torch.zeros_like(u)

    for i in range(nrof_iterations):
        u = u / torch.linalg.vector_norm(u)
        # Multiplication with J:
        u = torch.autograd.functional.jvp(lambda x: module(x), zeros, u)[1]
        # Multiplication with JT:
        u = torch.autograd.functional.vjp(lambda x: module(x), zeros, u)[1]
        all_values[i] = torch.linalg.vector_norm(u) ** 0.5
        if i > 0 and torch.abs(all_values[i] - all_values[i - 1]) < tolerance:
            print(f"Converged after {i+1} iterations "
                  f"on module {module.__class__.__name__}.")
            print(f"All values: {all_values[:i+1]}")
            all_values[i:] = all_values[i]
            break

    return all_values


def module_symmetric_power_iteration(*args, **kwargs):
    return module_symmetric_power_iteration_all_iterations(*args, **kwargs)[-1]


# The following is directly taken from https://arxiv.org/pdf/1805.10408.pdf
def conv_singular_values_numpy(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)


def conv_singular_values(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    transforms = torch.fft.fft2(kernel, s=2 * [input_shape], dim=(-2, -1))
    return torch.linalg.svd(transforms.permute(-2, -1, 0, 1))[1]


def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 0, 1)
    conv_filter_T = torch.flip(conv_filter_T, [2, 3])
    return conv_filter_T


def l2_normalize_batch(tensor, dim=-1, eps=1e-12):
    norm = (torch.sqrt(torch.sum(tensor.abs().float() *
            (tensor).abs().float(), dim, keepdim=True)))
    norm = norm+eps  # max(norm, eps)
    ans = tensor / norm
    return ans


def fantastic_one_batch(conv_filter, num_iters=50, return_vectors=False):
    b, out_ch, in_ch = conv_filter.shape
    device = conv_filter.device

    u1 = torch.randn((b, 1, in_ch), device=device, requires_grad=False)
    u1.data = l2_normalize_batch(u1.data, dim=2)

    v1 = torch.randn((b, out_ch, 1), device=device, requires_grad=False)
    v1.data = l2_normalize_batch(v1.data, dim=1)

    for i in range(num_iters):
        v1.data = l2_normalize_batch(
            (conv_filter.data*u1.data).sum(2, keepdim=True).data, dim=1)
        u1.data = l2_normalize_batch(
            (torch.conj(conv_filter).data*(v1.data)).sum(1, keepdim=True).data, dim=2)

    sigma1 = torch.sum(conv_filter.data*u1.data *
                       torch.conj(v1.data), (1, 2), keepdim=True).abs()

    if return_vectors:
        return v1, u1
    else:
        return sigma1.abs()


if __name__ == '__main__':
    from lipschitz.aol import AOLConv2d
    conv = AOLConv2d(13, 17, 3)
    x = torch.randn(1, 13, 32, 32)
    _ = conv(x)
    conv.eval()
    print(conv)
    print(module_symmetric_power_iteration_all_iterations(conv, (13, 32, 32), 1000))
    print()
