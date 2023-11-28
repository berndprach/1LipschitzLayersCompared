from typing import Callable, Union
from .base_metric_class import Metric
from time import time
from torch import Tensor


class Throughput(Metric):
    r'''
    Metric to calculate the throughput of the model in samples per second.
    '''

    def __init__(self, aggregation='mean') -> None:
        super().__init__(aggregation)
        self._last_call = None

    def forward(self, out: Tensor, labels: Tensor) -> None:
        if self._last_call is None:
            self._last_call = time()
            return Tensor([0.])
        actual_time = time()
        thoughput = out.size(0) / (actual_time - self._last_call)
        self._last_call = actual_time
        return Tensor([thoughput])
