
import unittest

import matplotlib.pyplot as plt
import torch

# from evaluator import lipschitz_constant
from evaluator import plot_lipschitz_constant
from models import layers


class TestEvaluateLipschitzConstant(unittest.TestCase):
    def test_iterate_with_input_shape(self):
        simple_model = get_simple_model()
        model_input_shape = (1, 3, 32, 32)
        model_input = torch.ones(model_input_shape)
        modules = [module for module in simple_model.modules()
                   if not isinstance(module, torch.nn.Sequential)]

        goal_input_shapes = [
            (1, 3, 32, 32),
            (1, 64, 30, 30),
            (1, 64, 30, 30),
            (1, 64, 28, 28),
            (1, 64, 28, 28),
            (1, 64*28*28),
        ]
        iterator = plot_lipschitz_constant.iterate_with_input_shape(
            modules, model_input
        )
        for i, (module, input_shape) in enumerate(iterator):
            print(module, input_shape)
            self.assertListEqual(list(input_shape),
                                 list(goal_input_shapes[i]))

    def test_plot_model(self):
        # self.plot_model_at_initialization(layers.AOLConv2d)
        # self.plot_model_at_initialization(layers.BCOP)
        self.plot_model_at_initialization(layers.SOC)
        # self.plot_model_at_initialization(StandardConv2d)
        plt.show()

        # self.plot_model_at_initialization(StandardConv2d)
        # plt.show()

    @staticmethod
    def plot_model_at_initialization(conv_cls):
        from models.layers import MaxMin
        from models import SimplifiedConvNet
        from models.simplified_conv_net import DEFAULT_MODELS

        model = SimplifiedConvNet(
            **DEFAULT_MODELS["ConvNetXS"],
            get_conv=conv_cls,
            get_activation=MaxMin,
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)

        model_input = torch.randn(1, 3, 32, 32).to(device)
        # model_output = eco_model(model_input)

        model_modules = [
            module for module in model.modules()
            if not isinstance(module, torch.nn.Sequential)
        ]

        for values in plot_lipschitz_constant.iterate_with_input_shape(
                model_modules, model_input):
            print(values)

        plot_lipschitz_constant.plot_lipschitz_constant_per_layer(model)


def get_simple_model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3),
        torch.nn.ReLU(),
        torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
        ),
        torch.nn.Flatten(),
    )


