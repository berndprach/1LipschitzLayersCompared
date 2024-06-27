import torch
import torch.nn as nn


class MaxMin(nn.Module):
    @staticmethod
    def forward(x):
        # reshape:
        in_size = x.size()
        if in_size[1] % 2 != 0:
            # Odd size: Do not apply max-min to the last channel.
            x_padded = nn.functional.pad(x, (0, 0, 0, 0, 0, 1), mode = 'replicate')
            return MaxMin.forward(x_padded)[:, :-1, :, :]
        x_rs = x.view(in_size[0], in_size[1]//2, 2, *in_size[2:])
        x_max = torch.max(x_rs, dim=2, keepdim=True)[0]
        x_min = torch.min(x_rs, dim=2, keepdim=True)[0]
        x_max_min = torch.cat((x_max, x_min), dim=2)
        return x_max_min.view(*in_size)


if __name__ == '__main__':
    x = torch.randn(1, 5, 1, 1)
    print(x)
    print(MaxMin.forward(x))
    print()
