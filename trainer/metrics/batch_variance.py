
import torch
from .base_metric_class import Metric


class BatchVariance(Metric):
    def __init__(self):
        super().__init__(aggregation='mean')

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return y.var(dim=0).mean()
