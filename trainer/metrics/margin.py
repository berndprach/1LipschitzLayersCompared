import torch
from torch.nn import Module
from .base_metric_class import Metric


class Margin(Metric):
    def __init__(self, aggregation='mean'):
        r'''
        The margin function that, if applied to a signed distance function,
        estimate the distance from the classification boundary
        '''
        super().__init__(aggregation)

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out, sort_idx = torch.topk(y, 2, dim=-1)
        out = out[:, 0] - out[:, 1]
        out[sort_idx[:, 0] != labels] = 0.  # apply the relu
        return out


class SignedMargin(Metric):
    def __init__(self, aggregation='mean'):
        """ Margin but can also be negative. """
        super().__init__(aggregation)

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_oh = torch.nn.functional.one_hot(labels, y.shape[-1])
        true_score = (y * label_oh).sum(dim=-1)
        best_other = (y - label_oh * 1e6).max(dim=-1)[0]
        return true_score - best_other


if __name__ == '__main__':
    margin = Margin(reduction='none')
    import torch
    x = torch.randn(8, 5)
    labels = torch.argmax(x, dim=-1)
    labels[0] = -1
    out = margin(x, labels)
    print(x)
    print(labels)
    print(out)
