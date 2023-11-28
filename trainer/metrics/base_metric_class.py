
from torch import nn, Tensor
from typing import Union, Callable
import torch


class Metric(nn.Module):
    def __init__(self, aggregation: Union[str, Callable] = 'mean') -> None:
        r'''
        Base class for metrics:
        Args:
            aggregation: Aggregation function to use for the metric
            Note that the aggregations is only a property of the metrics
            the result of the loss is not aggregated through this function

        This metric can be used as a base class for other metrics that can be read from
        the setting file setting.yaml.
        Example:
            eval_metrics:
                margin_avg:
                    !metric
                    name: Margin
                    aggregation: mean

                margin_median:
                    !metric
                    name: Margin
                    aggregation: 'median'
        '''
        self.aggregation = aggregation

        super().__init__()

    def forward(self, out: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError

    def get_name(self):
        if hasattr(self, "name"):
            return self.name
        return self.__class__.__name__

    def aggregated(self, *args, **kwargs):
        """ Batch aggreated. (pytorch would call it reduced) """
        pre_aggregated = self(*args, **kwargs)
        if self.aggregation == 'none':
            return pre_aggregated

        return getattr(torch, self.aggregation)(pre_aggregated.float())
