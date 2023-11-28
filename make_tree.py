import os

import yaml

from models.layers import available_conv2d_layers
from models.simplified_conv_net import DEFAULT_MODELS
from typing import Optional

# EPOCH_BUDGET_FILE = 'data/settings/epoch_budget.yml'
# conv_layers_names = available_conv2d_layers()
CIAFR10_models = ['ConvNetL', 'ConvNetS', 'ConvNetM', 'ConvNetXS']
TinyImageNet_models = ['TConvNetL', 'TConvNetS', 'TConvNetM', 'TConvNetXS']


def dump_yaml(kwargs, indent_level=0):
    template = ''
    for line in yaml.dump(kwargs).split('\n'):
        template += ' ' * indent_level + line + '\n'
    return template


def get_header(dataset, mode):
    template = f'''
## Training settings for {dataset['name']} and the {mode}
## These settings file are automatically generated you should not edit them
## If you want to change the settings, please edit the defaults.yml file
## or the epoch_budget.yml file'''
    return template + '\n'


def get_model(model_id, get_conv, get_linear, get_conv_first, get_conv_head, **kwargs):
    template = f'''
model:
  !model
  name: conv_net
  model_id: {model_id}
  get_conv: !layer {get_conv}
  get_activation: !layer MaxMin
  seed: !randint [0,999]\n'''
    if get_linear is not None:
        template += f'''  get_linear: !layer {get_linear}\n'''
    if get_conv_first is not None:
        template += f'''  get_conv_first: !layer {get_conv_first}\n'''
    if get_conv_head is not None:
        template += f'''  get_conv_head: !layer {get_conv_head}\n'''
    if len(kwargs) > 0:
        template += dump_yaml(kwargs, indent_level=2)
    return template + '\n'


def get_optimizer(name: str, lr: str, weight_dacay: str, **kwargs):
    template = f'''
optimizer:
  !optimizer
  name: {name}    
  lr: {lr}
  weight_decay: {weight_dacay}\n'''
    if len(kwargs) > 0:
        template += dump_yaml(kwargs, indent_level=2)
    return template


def get_scheduler(name: str, epochs: int, **kwargs):
    template = f'''
lr_scheduler:
  !scheduler
  name: {name}\n'''
    if name == 'OneCycleLR':
        template += f'''  epochs: {epochs}\n'''
    if len(kwargs) > 0:
        template += dump_yaml(kwargs, indent_level=2)
    return template


def get_loss(name: str, **kwargs):
    template = f'''
loss:
  !metric
  name: {name}\n'''
    if len(kwargs) > 0:
        template += dump_yaml(kwargs, indent_level=2)
    return template


def get_dataset(name: str, train: bool, **kwargs):
    template = f'''
{'trainset' if train else 'valset'}:
  !dataset
  name: {name}
  train: {train}\n'''
    if len(kwargs) > 0:
        template += dump_yaml(kwargs, indent_level=2)
    return template


def get_eval_metrics(**kwargs):
    template = f'''
eval_metrics:
  margin_at_50:
    !metric
    name: Margin
    aggregation: \'lambda x: -((-x).quantile(0.5))\'

  robstacc_36:
    !metric
    name: RobustAccuracy
    eps: 0.1411 

  robstacc_72:
    !metric
    name: RobustAccuracy
    eps: 0.2824

  robstacc_108:
    !metric
    name: RobustAccuracy
    eps: 0.4235

  robstacc_255:
    !metric
    name: RobustAccuracy
    eps: 1.0

  throughput:
    !metric
    name: Throughput\n\n'''
    return template


def make_setting(model_id, conv_layer: str,
                 epochs: int, lr: str,
                 weight_decay: str, mode: str,
                 defaults: dict):
    defaults = defaults.copy()
    get_linear = None
    get_conv_first = None
    get_conv_head = None
    get_conv = conv_layer
    if conv_layer == 'BCOP':
        get_linear = 'BjorckLinear'
    elif conv_layer == 'CayleyConv':
        get_linear = 'CayleyLinear'
    elif conv_layer == 'AOLConv2d':
        get_conv_first = 'AOLConv2dOrth'
        get_conv_head = 'AOLConv2dOrth'
        get_conv = 'AOLConv2dDirac'
    elif conv_layer == 'BnBConv2d':
        get_conv_first = 'BnBConv2dOrth'
        get_conv_head = 'BnBConv2dOrth'
        get_conv = 'BnBConv2dDirac'
    elif conv_layer == 'LOT' and 'ConvNetL' in model_id:
        get_conv = 'LOT2t'
    elif conv_layer == 'SandwichConv':
        get_linear = 'SandwichFc'

    # Creating the template for setting file
    template = get_header(defaults['trainset'], mode)
    defaults_model = defaults['model'] if 'model' in defaults.keys() else {}
    template += get_model(model_id, get_conv, get_linear,
                          get_conv_first, get_conv_head,
                          **defaults_model)
    defaults.pop('model') if 'model' in defaults.keys() else None
    template += f'''epochs: {epochs}\n'''
    template += get_optimizer(lr=lr, weight_dacay=weight_decay,
                              **defaults['optimizer'])
    defaults.pop('optimizer')
    template += get_scheduler(epochs=epochs, **defaults['lr_scheduler'])
    defaults.pop('lr_scheduler')
    template += get_loss(**defaults['loss'])
    defaults.pop('loss')
    template += get_dataset(train=True, **defaults['trainset'])
    if mode == 'final_training':
        template += get_dataset(train=False, **defaults['trainset'])
    defaults.pop('trainset')
    if len(defaults.keys()) > 0:
        template += f'''\n# Other settings\n'''
        template += yaml.dump(defaults)
    template += get_eval_metrics()
    return template


def get_lr_wd(model_id: str, conv_layer: str, mode: str, best_lr_wd_file: Optional[str] = None):
    import pandas as pd
    if mode == 'random_search':
        return '!randlog10 [-4, -1]', '!randlog10 [-5.5, -3.5]'
    elif mode == 'final_training':
        df = pd.read_csv(best_lr_wd_file)
        df.set_index(['model_id', 'get_conv'], inplace=True)
        lr = df.loc[(model_id, conv_layer), 'lr']
        wd = df.loc[(model_id, conv_layer), 'weight_decay']
        return lr, wd


def get_epochs(epoch_budget: dict, model_id: str, conv_layer: str, mode: str, training_time: int):
    if mode == 'random_search':
        # Initial budget was assumed for 2 hours training
        return epoch_budget[model_id][conv_layer]
    elif mode == 'final_training':
        return (epoch_budget[model_id][conv_layer]*training_time)//2


def create_directory_tree(root_dir: str, mode: str,
                          training_time: int, default: str,
                          layers: list[str], best_lr_wd_file: Optional[str] = None):
    # Create the directory tree and the settings files if they do not exist
    with open(default, 'r') as f:
        defaults = yaml.load(f, Loader=yaml.SafeLoader)
    default_dataset = defaults['trainset']['name']
    epochs_budget = f'data/settings/{default_dataset}/epoch_budget.yml'
    with open(epochs_budget, 'r') as f:
        epoch_budget = yaml.load(f, Loader=yaml.SafeLoader)
    if default_dataset == 'CIFAR10' or default_dataset == 'CIFAR100':
        default_models = CIAFR10_models
    elif default_dataset == 'TinyImageNet':
        default_models = TinyImageNet_models
    for model_id in default_models:
        if model_id not in epoch_budget.keys():
            # Skip if the model is not in the epoch_budget
            continue
        for conv_layer in layers:
            if conv_layer not in epoch_budget[model_id].keys():
                # Skip if the conv_layer is not in the epoch_budget
                continue
            epochs = get_epochs(epoch_budget, model_id,
                                conv_layer, mode, training_time)
            try:
                lr, wd = get_lr_wd(model_id, conv_layer, mode,
                                   best_lr_wd_file=best_lr_wd_file)
            except KeyError:
                print(f'No lr and wd found for {model_id} and {conv_layer}')
                continue
            model_dir = f'{root_dir}/{model_id}/{conv_layer}'
            settings = make_setting(model_id, conv_layer,
                                    epochs=epochs, lr=lr,
                                    weight_decay=wd, mode=mode,
                                    defaults=defaults)
            try:
                os.makedirs(model_dir)
            except OSError as error:
                print(error)
            print(
                f'Creating setting file for model:, {model_id}, {conv_layer}')
            try:
                with open(f'{model_dir}/settings.yml', 'x') as f:
                    f.write(settings)
            except FileExistsError as error:
                print(error)


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        help='root directory for storing all the runs.')
    parser.add_argument('--mode', type=str, default='random_search',
                        choices=['random_search', 'final_training'])
    parser.add_argument('--training-time', type=int,
                        default=10, help='training time in hours')
    parser.add_argument('--default', type=str, default='settings/defaults.yml',
                        help='''default settings file. Use this option if you want to change 
                        dataset, optimizer, scheduler, loss, etc.''')
    parser.add_argument('--best-lr-wd-file', type=str,
                        default=None,
                        help='best lr and wd csv file')
    parser.add_argument('--layers', type=str, nargs='+',
                        default=available_conv2d_layers(),
                        choices=available_conv2d_layers(),
                        help='List of layers to create the settings files for.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    # print(make_setting('ConvNetL', 'AOLConv2d'))
    create_directory_tree(**vars(args))
