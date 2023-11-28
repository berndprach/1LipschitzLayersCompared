from .base_metric_class import Metric
import torch
from torch import Tensor, ones
from torch.nn.functional import multi_margin_loss
import warnings


class MultiMarginLoss(Metric):
    def __init__(self, p: int = 1, margin: int = 1.0):
        super().__init__('mean')
        self.margin = margin
        assert p in {1, 2}, 'p must be 1 or 2'
        self.p = p

    def forward(self, out: Tensor, labels: Tensor) -> Tensor:
        margin = torch.empty_like(
            labels, dtype=out.dtype).fill_(self.margin).unsqueeze(1)
        out = out.scatter_add(
            dim=-1, index=labels.unsqueeze(1), src=-margin)
        out_y = out - \
            out.gather(dim=-1, index=labels.unsqueeze(1))
        if self.p == 2:
            return out_y.relu().pow(self.p).mean()
        return out_y.relu().mean()


if __name__ == '__main__':
    x = 100*torch.rand(3, 10)
    labels = torch.randint(0, 10, (3,))
    y_custom = MultiMarginLoss(margin=1)(x, labels)
    y = multi_margin_loss(x, labels, margin=1)
    print(torch.allclose(y, y_custom))
    print(y)
    print(y_custom)
    print()
