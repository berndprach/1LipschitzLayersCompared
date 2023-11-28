import torch


def fantastic_four(conv_filter, num_iters=50):
    out_ch, in_ch, h, w = conv_filter.shape

    u1 = torch.randn((1, in_ch, 1, w), requires_grad=False)
    u1.data = l2_normalize(u1.data)

    u2 = torch.randn((1, in_ch, h, 1), requires_grad=False)
    u2.data = l2_normalize(u2.data)

    u3 = torch.randn((1, in_ch, h, w), requires_grad=False)
    u3.data = l2_normalize(u3.data)

    u4 = torch.randn((out_ch, 1, h, w), requires_grad=False)
    u4.data = l2_normalize(u4.data)

    v1 = torch.randn((out_ch, 1, h, 1), requires_grad=False)
    v1.data = l2_normalize(v1.data)

    v2 = torch.randn((out_ch, 1, 1, w), requires_grad=False)
    v2.data = l2_normalize(v2.data)

    v3 = torch.randn((out_ch, 1, 1, 1), requires_grad=False)
    v3.data = l2_normalize(v3.data)

    v4 = torch.randn((1, in_ch, 1, 1), requires_grad=False)
    v4.data = l2_normalize(v4.data)

    for i in range(num_iters):
        v1.data = l2_normalize(
            (conv_filter.data*u1.data).sum((1, 3), keepdim=True).data)
        u1.data = l2_normalize(
            (conv_filter.data*v1.data).sum((0, 2), keepdim=True).data)

        v2.data = l2_normalize(
            (conv_filter.data*u2.data).sum((1, 2), keepdim=True).data)
        u2.data = l2_normalize(
            (conv_filter.data*v2.data).sum((0, 3), keepdim=True).data)

        v3.data = l2_normalize(
            (conv_filter.data*u3.data).sum((1, 2, 3), keepdim=True).data)
        u3.data = l2_normalize(
            (conv_filter.data*v3.data).sum(0, keepdim=True).data)

        v4.data = l2_normalize(
            (conv_filter.data*u4.data).sum((0, 2, 3), keepdim=True).data)
        u4.data = l2_normalize(
            (conv_filter.data*v4.data).sum(1, keepdim=True).data)

    return u1, v1, u2, v2, u3, v3, u4, v4


def l2_normalize(tensor, eps=1e-12):
    norm = float(torch.sqrt(torch.sum(tensor.float() * tensor.float())))
    norm = max(norm, eps)
    ans = tensor / norm
    return ans

