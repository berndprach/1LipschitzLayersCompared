import torch


def power_iteration(A, init_u=None, n_iters=10, return_uv=True):
    """
    Power iteration for matrix
    """
    shape = list(A.shape)
    # shape[-2] = shape[-1]
    shape[-1] = 1
    shape = tuple(shape)
    if init_u is None:
        u = torch.randn(*shape, dtype=A.dtype, device=A.device)
    else:
        assert tuple(init_u.shape) == shape, (init_u.shape, shape)
        u = init_u
    for _ in range(n_iters):
        v = A.transpose(-1, -2) @ u
        v /= v.norm(dim=-2, keepdim=True)
        u = A @ v
        u /= u.norm(dim=-2, keepdim=True)
    s = (u.transpose(-1, -2) @ A @ v).squeeze(-1).squeeze(-1)
    if return_uv:
        return u, s, v
    return s
