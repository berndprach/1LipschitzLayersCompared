from .base_metric_class import Metric
from .margin import Margin, SignedMargin
from .accuracy import Accuracy, Corrects
from .robust_accuracy import RobustAccuracy
from .lipschit_ce_loss import LipCrossEntropyLoss
from .batch_variance import BatchVariance
from .troughput import Throughput
from .multi_margin_loss import MultiMarginLoss
from torch.nn import CrossEntropyLoss as CrossEntropyLoss_
from torch.nn import MultiMarginLoss as MultiMarginLoss_


CRA = RobustAccuracy


def CrossEntropyLoss(**kwargs):
    loss = CrossEntropyLoss_(**kwargs, reduction='mean')
    setattr(loss, 'aggregation', 'mean')
    return loss


def MultiMarginLossBUGGED(**kwargs):  # It seems some bug in this function
    from torch import ones
    from warnings import warn
    warn('''torch.nn.MultiMarginLoss is bugged for version 2.0.0.
          Use the self implemented version instead.''')
    loss = MultiMarginLoss_(**kwargs, reduction='mean')
    setattr(loss, 'aggregation', 'mean')
    return loss
