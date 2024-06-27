import torch

from .power_method import power_iteration


def bjorck_orthonormalize(
        w, beta=0.5, iters=20, order=1, power_iteration_scaling=False,
        default_scaling=False):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the
    best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """

    if w.shape[-2] < w.shape[-1]:
        return bjorck_orthonormalize(
            w.transpose(-1, -2),
            beta=beta, iters=iters, order=order,
            power_iteration_scaling=power_iteration_scaling,
            default_scaling=default_scaling).transpose(
            -1, -2)

    if power_iteration_scaling:
        with torch.no_grad():
            s = power_iteration(w, return_uv=False)
        w = w / s.unsqueeze(-1).unsqueeze(-1)
    elif default_scaling:
        w = w / ((w.shape[0] * w.shape[1]) ** 0.5)
    assert order == 1, "only first order Bjorck is supported"
    for _ in range(iters):
        w_t_w = w.transpose(-1, -2) @ w
        w = (1 + beta) * w - beta * w @ w_t_w
    return w
