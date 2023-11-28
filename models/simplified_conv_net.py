from dataclasses import dataclass

import torch
from torch import nn
from typing import Type, Optional, Callable
from collections import OrderedDict

from .layers.basic.first_channels import FirstChannels
from .layers.basic.zero_channel_concatenation import \
    ZeroChannelConcatenation
from typing import Union


@dataclass
class SimplifiedConvNetHyperparameters:
    get_activation: Type[nn.Module]
    get_conv: Type[nn.Module]
    get_conv_first: Type[nn.Module] = None
    get_conv_head: Type[nn.Module] = None
    get_linear: Type[nn.Linear] = None

    # Size:
    base_width: int = 16
    nrof_blocks: int = 5
    nrof_layers_per_block: int = 3
    kernel_size: int = 1

    # Classification head:
    nrof_classes: Optional[int] = 10

    def __post_init__(self):
        if self.get_conv_first is None:
            self.get_conv_first = self.get_conv
        if self.get_conv_head is None:
            self.get_conv_head = self.get_conv


class ConvBlock(nn.Sequential):
    def __init__(self,
                 get_conv: Type[nn.Module],
                 get_activation: Callable,
                 in_channels: int,
                 length: int,
                 kernel_size: int):
        layers = []
        for _ in range(length):
            layers.append(get_conv(in_channels,
                                   in_channels,
                                   kernel_size))
            layers.append(get_activation())
        layers.append(FirstChannels(in_channels//2))
        layers.append(nn.PixelUnshuffle(2))
        super().__init__(*layers)


class SimplifiedConvNet(nn.Sequential):
    """
    Similar to LipConvnet, e.g. from SOC
    (https://arxiv.org/pdf/2105.11417.pdf)
    """

    def __init__(self, *args, seed: Union[int, None] = None, **kwargs):
        if seed is not None:
            torch.manual_seed(seed)
        self.hp = SimplifiedConvNetHyperparameters(*args, **kwargs)
        layers = get_layers(self.hp)
        super().__init__(OrderedDict(layers))


def get_layers(hp: SimplifiedConvNetHyperparameters):
    first_conv = hp.get_conv_first(
        in_channels=hp.base_width,
        out_channels=hp.base_width,
        kernel_size=1
    )

    kernel_sizes = [hp.kernel_size for _ in range(hp.nrof_blocks)]
    kernel_sizes[-1] = 1  # 2x2 blocks do not allow kernel size >= 3.
    downsize_blocks = [
        (f"Block{i + 1}", ConvBlock(
            hp.get_conv,
            hp.get_activation,
            hp.base_width * 2 ** i,
            hp.nrof_layers_per_block,
            # hp.kernel_size,
            kernel_sizes[i])
         ) for i in range(hp.nrof_blocks)
    ]
    if hp.get_linear is not None:
        classification_head = nn.Sequential(
            nn.Flatten(),
            hp.get_linear(hp.base_width * 2 ** hp.nrof_blocks,
                          hp.base_width * 2 ** hp.nrof_blocks)
        )
    else:
        classification_head = nn.Sequential(
            hp.get_conv_head(
                in_channels=hp.base_width * 2 ** hp.nrof_blocks,
                out_channels=hp.base_width * 2 ** hp.nrof_blocks,
                kernel_size=1),
            nn.Flatten()
        )

    layers = [
        ("ZeroConcatenation", ZeroChannelConcatenation(hp.base_width)),
        ("FirstConv", first_conv),
        ("FirstActivation", hp.get_activation()),
        *downsize_blocks,
        ('AdaptiveMaxPool', nn.AdaptiveMaxPool2d(1)),
        ('ClassificationHead', classification_head),
        ("FirstChannels", FirstChannels(hp.nrof_classes))
    ]
    return layers


DEFAULT_MODELS = OrderedDict([
    ('ConvNetXS', dict(nrof_layers_per_block=5, base_width=16, kernel_size=3)),
    ('ConvNetS', dict(nrof_layers_per_block=5, base_width=32, kernel_size=3)),
    ('ConvNetM', dict(nrof_layers_per_block=5, base_width=64, kernel_size=3)),
    ('ConvNetL', dict(nrof_layers_per_block=5, base_width=128, kernel_size=3)),

    # ('ConvNetXS_1x1',
    #  dict(nrof_layers_per_block=5, base_width=16, kernel_size=1)),
    # ('ConvNetS_1x1',
    #  dict(nrof_layers_per_block=5, base_width=32, kernel_size=1)),
    # ('ConvNetM_1x1',
    #  dict(nrof_layers_per_block=5, base_width=64, kernel_size=1)),
    # ('ConvNetL_1x1',
    #  dict(nrof_layers_per_block=5, base_width=128, kernel_size=1)),

    ('TConvNetXS', dict(nrof_layers_per_block=5,
     base_width=8, nrof_blocks=6, kernel_size=3)),
    ('TConvNetS', dict(nrof_layers_per_block=5,
     base_width=16, nrof_blocks=6, kernel_size=3)),
    ('TConvNetM', dict(nrof_layers_per_block=5,
     base_width=32, nrof_blocks=6, kernel_size=3)),
    ('TConvNetL', dict(nrof_layers_per_block=5,
     base_width=64, nrof_blocks=6, kernel_size=3)),
])


def conv_net(model_id: str, *args, **kwargs):
    """Returns a ConvNet model with the specified parameters."""
    kwargs = {**kwargs, **DEFAULT_MODELS[model_id]}
    model = SimplifiedConvNet(*args, **kwargs)
    model.__class__.__name__ = model_id
    return model


if __name__ == "__main__":
    from layers import SOC, MaxMin, CayleyLinear
    for model_name in DEFAULT_MODELS.keys():
        net = conv_net(model_name, get_conv=SOC,
                       get_activation=MaxMin).to('cuda:2')
        if model_name in ['ConvNetXS', 'ConvNetS', 'ConvNetM', 'ConvNetL']:
            size = 32
        elif model_name in ['TConvNetXS', 'TConvNetS', 'TConvNetM', 'TConvNetL']:
            size = 64
        else:
            continue
        x = torch.randn(1, 3, 64, 64).to('cuda:2')
        y = net(x)
        parameters = sum(p.numel()
                         for p in net.parameters() if p.requires_grad)
        print(f'{model_name}, Parameters {(parameters/1e6)} M')
        print()
    print()
