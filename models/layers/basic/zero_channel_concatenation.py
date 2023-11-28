
import torch
from torch import nn


class ZeroChannelConcatenation(nn.Module):
    def __init__(self, concatenate_to: int):
        super().__init__()
        self.concatenate_to = concatenate_to

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Append zeros to the channel dimension of the input tensor.
        pad = self.concatenate_to - x.shape[1]
        pad = max(pad, 0)
        padding = torch.zeros(x.shape[0], pad, *x.shape[2:], device=x.device)
        return torch.cat((x, padding), 1)
        # return torch.cat((padding, x), 1)

    def __repr__(self):
        return f"ZeroChannelConcatenation({self.concatenate_to})"


