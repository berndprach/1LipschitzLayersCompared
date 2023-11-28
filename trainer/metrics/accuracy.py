
import torch

from torch import Tensor, nn
from .base_metric_class import Metric


class Accuracy(Metric):
    def __init__(self):
        super().__init__('mean')

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        accuracy = (logits.argmax(dim=1) == labels).float()
        # if self.aggregation != 'none':
        #     accuracy = getattr(torch, self.aggregation)(accuracy)
        return accuracy


class Corrects(Metric):
    def __init__(self, aggregation='sum'):
        super().__init__(aggregation)

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        corrects = (logits.argmax(dim=1) == labels).float()
        return corrects
