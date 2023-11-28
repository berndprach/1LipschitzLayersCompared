import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
# from torchvision import transforms, datasets

# import datasets
from models.layers.utils import module_symmetric_power_iteration
# from datasets.CIFAR10 import get_data_loader
import datasets


def iterate_with_input_shape(modules: List, dummy_input):
    for module in modules:
        yield module, dummy_input.shape

        if not hasattr(module, "forward"):
            continue

        try:  # To skip modules such as parameterizations:
            dummy_input = module(dummy_input)

        except TypeError:
            pass
        except NotImplementedError:
            pass


def is_linear(module):
    if isinstance(module, torch.nn.Linear):
        return True
    if isinstance(module, torch.nn.Conv2d):
        return True

    for layer_abbreviation in ["SOC", "BCOP", "ECO"]:
        if layer_abbreviation in module.__class__.__name__:
            return True

    return False


def get_lipschitz_constant_per_layer(modules):
    lipschitz_constants = [1.]  # cumulative lipschitz constants

    model_input_shape = (1, 3, 32, 32)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_input = torch.ones(model_input_shape).to(device)

    for module, input_shape in iterate_with_input_shape(modules, model_input):
        if not is_linear(module):
            print(f"Skipping non-linear layer {module.__class__.__name__}.")
            lipschitz_constants.append(lipschitz_constants[-1])
            continue

        module_operator_norm = module_symmetric_power_iteration(
            module,
            input_size=input_shape[1:],
            nrof_iterations=10_000,  # (with early stopping)
        )
        lipschitz_constant = lipschitz_constants[-1] * module_operator_norm
        lipschitz_constants.append(lipschitz_constant)

    return lipschitz_constants


def plot_activation_variance_per_layer(model):
    activation_variances = get_activation_variance_for_model(model)
    draw_activation_variance_per_layer_plot(activation_variances)


def get_activation_variance_for_model(model):
    modules = [module for module in model.modules()
               if not isinstance(module, torch.nn.Sequential)]

    _, val_loader = datasets.get_train_val_loader(
        datasets.CIFAR10,
        batch_size=100,
    )

    data_x_batch = next(iter(val_loader))[0]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_x_batch = data_x_batch.to(device)

    activation_variances = get_activation_variance_per_layer(
        modules, data_x_batch
    )
    return activation_variances


def get_activation_variance_per_layer(modules, data_x_batch):
    activation_variances = []
    for module in modules:
        if not hasattr(module, "forward"):
            continue

        try:
            data_x_batch = module(data_x_batch)
        except TypeError:
            pass
        except NotImplementedError:
            pass

        activation_variances.append(data_x_batch.var(dim=0).sum().item())

    return activation_variances


def draw_lipschitz_constant_per_layer_plot(lipschitz_constants):

    plt.figure(figsize=(10, 5))
    # plt.title("Activation variance")
    plt.title("Cumulative Lipschitz Constant")
    draw_log2_plot(lipschitz_constants)

    plt.xlabel("Layer Index")
    plt.ylabel("Lipschitz Constant (Log2 scale)")

    plt.tight_layout()


def draw_activation_variance_per_layer_plot(activation_variances):

    plt.figure(figsize=(10, 5))
    plt.title("Activation Variances")
    draw_log2_plot(activation_variances)

    plt.xlabel("Layer Index")
    plt.ylabel("Activation Variance (Log2 scale)")

    plt.tight_layout()


def draw_log2_plot(values):
    log2_values = np.log2(values)
    plt.plot(log2_values)
    plt.scatter(torch.arange(len(log2_values)), log2_values)

    min_log2_variance = log2_values.min()
    max_log2_variance = log2_values.max()
    # y_ticks = torch.arange(
    # int(min_log2_variance), int(max_log2_variance) + 2)
    y_ticks = np.arange(int(min_log2_variance), int(max_log2_variance) + 2)
    plt.yticks(y_ticks, [f"$2^{{{y}}}$" for y in y_ticks])

    plt.grid(True, which="major", axis="both")


def plot_lipschitz_constant_per_layer(model):
    modules = [module for module in model.modules()
               if not isinstance(module, torch.nn.Sequential)]

    ls_constants = get_lipschitz_constant_per_layer(modules)
    draw_lipschitz_constant_per_layer_plot(ls_constants)




