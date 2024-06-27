import torch
from torch import nn
from typing import Type, Optional, Callable
from collections import OrderedDict
import sys
from functools import partial
from layers import FirstChannels, ZeroChannelPad
import layers


class ConvBlock(nn.Sequential):
    def __init__(self,
                 conv: Type[nn.Module],
                 activation: Callable,
                 in_channels: int,
                 length: int,
                 kernel_size: int):
        layers = []
        for _ in range(length):
            layers.append(conv(in_channels,
                                   in_channels,
                                   kernel_size))
            layers.append(activation())
        layers.append(FirstChannels(in_channels//2))
        layers.append(nn.PixelUnshuffle(2))
        super().__init__(*layers)


class LipConvNet(nn.Sequential):
    r''''
    LipConvNet is adapted from `https://github.com/singlasahil14/improved_l2_robustness`
    '''
    def __init__(self,
                 conv: Type[nn.Module],
                 activation: Type[nn.Module],
                 linear: Optional[Type[nn.Linear]] = None,
                 base_width: int = 16,
                 nrof_blocks: int = 5,
                 nrof_layers_per_block: int = 3,
                 kernel_size: int = 3,
                 nrof_classes: Optional[int] = 10,
                 seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        # Not instantited classes
        self.activation = activation
        self.conv = conv
        self.linear = linear
        # Hyperparameters
        self.base_width = base_width
        self.nrof_blocks = nrof_blocks
        self.nrof_layers_per_block = nrof_layers_per_block
        self.kernel_size = kernel_size
        self.nrof_classes = nrof_classes
        layers = self.__get_layers__()
        super().__init__(OrderedDict(layers))

    def __get_layers__(self):
        first_conv = self.conv(self.base_width, self.base_width, 1)
        kernel_sizes = [self.kernel_size for _ in range(self.nrof_blocks)]
        kernel_sizes[-1] = 1  # 2x2 blocks do not allow kernel size >= 3.
        downsize_blocks = [
        (f"Block{i + 1}", ConvBlock(
            self.conv,
            self.activation,
            self.base_width * 2 ** i,
            self.nrof_layers_per_block,
            kernel_sizes[i])
            ) for i in range(self.nrof_blocks)
        ]
        in_features = self.base_width * 2 ** self.nrof_blocks
        if self.linear is not None:
            classification_head = nn.Sequential(nn.Flatten(), self.linear(in_features, in_features))
        else:
            classification_head = nn.Sequential(self.conv(in_features, in_features, 1), nn.Flatten())
        
        layers = [
            ("ZeroConcatenation", ZeroChannelPad(self.base_width)),
            ("FirstConv", first_conv),
            ("FirstActivation", self.activation()),
            *downsize_blocks,
            ('AdaptiveMaxPool', nn.AdaptiveMaxPool2d(1)),
            ('ClassificationHead', classification_head),
            ("FirstChannels", FirstChannels(self.nrof_classes))
            ]
        return layers


def _lip_conv_net(model_id: str, model_url: str, pretrained: bool=False, **kwargs) -> LipConvNet:
    model = LipConvNet(**kwargs)
    if pretrained:
        raise NotImplementedError
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_url))
    setattr(model, '__class__.__name__', model_id)
    return model

thismodule = sys.modules[__name__]
DATASET_NROF_CLASSES = dict(cifar10=10, cifar100=100, tiny_imagenet=200, imagenette=10)
DEFAULT_MODELS = OrderedDict([
    ('lipnetXS', dict(nrof_layers_per_block=5, base_width=16, kernel_size=3)),
    ('lipnetS', dict(nrof_layers_per_block=5, base_width=32, kernel_size=3)),
    ('lipnetM', dict(nrof_layers_per_block=5, base_width=64, kernel_size=3)),
    ('lipnetL', dict(nrof_layers_per_block=5, base_width=128, kernel_size=3)),
    ('t_lipnetXS', dict(nrof_layers_per_block=5, base_width=8, nrof_blocks=6, kernel_size=3)),
    ('t_lipnetS', dict(nrof_layers_per_block=5, base_width=16, nrof_blocks=6, kernel_size=3)),
    ('t_lipnetM', dict(nrof_layers_per_block=5, base_width=32, nrof_blocks=6, kernel_size=3)),
    ('t_lipnetL', dict(nrof_layers_per_block=5, base_width=64, nrof_blocks=6, kernel_size=3)),
])

LAYERS_DICT = dict(cpl='CPLConv2d',
                   aol='AOLConv2d', 
                   bcop='BCOP', 
                   cayley='CayleyConv', 
                   lot='LOT', 
                   sll='SLLConv2d', 
                   soc='SOC',
                   eco='ECO')

def get_url_models(model_id: str, dataset: str, layer_id: str) -> str:
    base_url = 'https://github.com/berndprach/1LipschitzLayersCompared/tree/main/data/runs'

    translate_model_dict = dict(lipnetXS='ConvNetXS',
                                lipnetS='ConvNetS',
                                lipnetM='ConvNetM',
                                lipnetL='ConvNetL',
                                t_lipnetXS='TConvNetXS',
                                t_lipnetS='TConvNetS',
                                t_lipnetM='TConvNetM',
                                t_lipnetL='TConvNetL')
    path = f'{base_url}/{dataset.upper()}/final_training_24h/'
    path += f'{translate_model_dict[model_id]}/{LAYERS_DICT[layer_id]}.pth'



for dataset in DATASET_NROF_CLASSES.keys():
    for model_id in ['lipnetXS', 'lipnetS', 'lipnetM', 'lipnetL']:
        model_url = f''
        for layer_id in LAYERS_DICT.keys():
            nrof_classes = DATASET_NROF_CLASSES[dataset]
            _model_id_aux = model_id if dataset != 'tiny_imagenet' else f't_{model_id}'
            default_kwargs = DEFAULT_MODELS[_model_id_aux]
            setattr(thismodule, 
                    f'{dataset}_{model_id}_{layer_id}', 
                    partial(_lip_conv_net, model_id = model_id, 
                                          model_url = get_url_models(model_id, dataset, layer_id),
                                          conv = getattr(layers, LAYERS_DICT[layer_id]),
                                          activation=layers.MaxMin,
                                          nrof_classes = DATASET_NROF_CLASSES[dataset],
                                          **default_kwargs
                    ))
            

