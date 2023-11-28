from .margin import Margin
import torch
from torch import Tensor
from .base_metric_class import Metric


class RobustAccuracy(Metric):
    def __init__(self, eps: float, p: int = 2):
        r'''
        Robust Accuracy provides the percentage of samples that are correctly classified and for which
        the Minimal Adversarial Perturbation (MAP) is larger than a given threshold eps.
        Args:
            eps: Desired threshold for robust accuracy
            p: Norm to use for the MAP Esistimation
        '''
        super().__init__('mean')
        assert eps > 0.
        if p != 2:
            raise NotImplementedError
        self.eps = eps
        self.margin = Margin()

        self.name = f"CRA{eps:.2f}"

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: extend to other norms
        out = self.margin(logits, labels)/(2**0.5)
        count = (out > self.eps)
        return count
