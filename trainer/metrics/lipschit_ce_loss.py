from torch import Tensor
import torch
from .base_metric_class import Metric
# from base_metric_class import Metric
from torch.nn.functional import cross_entropy


class LipCrossEntropyLossOLD(Metric):
    def __init__(self, margin: float = 0., temperature: float = 1) -> None:
        super().__init__('mean')
        self.margin = margin
        self.temperature = temperature

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        margin = torch.empty_like(
            labels, dtype=inputs.dtype).fill_(self.margin).unsqueeze(1)
        out = inputs.scatter_add(
            dim=-1, index=labels.unsqueeze(1), src=-margin)
        return cross_entropy(out / self.temperature, labels)


class LipCrossEntropyLoss(Metric):
    def __init__(self, margin: float = 0., temperature: float = 1) -> None:
        super().__init__('mean')
        self.margin = margin
        self.temperature = temperature

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        margin = torch.empty_like(
            labels, dtype=inputs.dtype).fill_(self.margin).unsqueeze(1)
        out = inputs.scatter_add(
            dim=-1, index=labels.unsqueeze(1), src=-margin)
        return cross_entropy(out / self.temperature, labels) * self.temperature


if __name__ == '__main__':
    inputs = torch.zeros((7, 10))
    labels = torch.randint(0, 10, (7,))

    margins = [m-10 for m in range(20)]
    loss_fns = [LipCrossEntropyLoss(margin=m, temperature=1)
                for m in margins]
    losses = [loss(inputs, labels) for loss in loss_fns]

    import matplotlib.pyplot as plt
    plt.plot(margins, losses)
    plt.show()

    for loss in losses:
        print(loss(inputs, labels))

    x = 100 * torch.rand(7, 10)
    labels = torch.argmax(x, dim=-1)
    y = LipCrossEntropyLoss(reduction='none', margin=11,
                            temperature=1 / 64)(x, labels)
    print(x)
    print(labels)
    print(y)
    print()
