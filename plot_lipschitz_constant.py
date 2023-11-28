
import os

import matplotlib.pyplot as plt
import torch

from models.layers import ECO, MaxMin, AOLConv2d
from models import SimplifiedConvNet
from models.simplified_conv_net import DEFAULT_MODELS
from evaluator.plot_lipschitz_constant import (
    plot_lipschitz_constant_per_layer,
    plot_activation_variance_per_layer,
)


SEARCH_PATH = os.path.join("data", "search")
ECO_PATH = os.path.join(SEARCH_PATH, "20230903_154144.1")
print(os.listdir(ECO_PATH))

OUTPUT_PATH = os.path.join("data", "evaluations", "lipschitz_constant")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


def load_eco_model():
    eco_s_model = SimplifiedConvNet(
        **DEFAULT_MODELS["ConvNetS"],
        get_conv=ECO,
        get_activation=MaxMin,
    )
    # eco_s_model.eval()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    eco_s_model.to(device)

    eco_input = torch.randn(1, 3, 32, 32).to(device)
    eco_output = eco_s_model(eco_input)
    # print(eco_output.shape)
    
    eco_s_model.eval()

    return eco_s_model


def load_eco_weights(eco_s_model):
    sd_path = os.path.join(ECO_PATH, "model_state_dict.pth")
    state_dict = torch.load(sd_path)

    eco_s_model.load_state_dict(state_dict)
    print("Loaded model weights!")
    return eco_s_model


def plot(model, postfix=""):

    modules = [module for module in model.modules()
               if not isinstance(module, torch.nn.Sequential)]
    for i, m in enumerate(modules):
        # print(i, m.name)
        print(i, m.__class__.__name__)
        

    plt.rcParams.update({'font.size': 22})
    
    plot_lipschitz_constant_per_layer(model)
    plt.ylabel("Upper bound")
    plt.savefig(
        os.path.join(OUTPUT_PATH, f"lipschitz_constant_eco{postfix}.pdf")
    )

    plot_activation_variance_per_layer(model)
    plt.ylabel("Activation Variance")
    plt.savefig(
        os.path.join(OUTPUT_PATH, f"activation_variance_eco{postfix}.pdf")
    )
    plt.close("all")


if __name__ == "__main__":
    eco_model = load_eco_model()
    plot(eco_model, postfix="_at_initialization")

    eco_model = load_eco_weights(eco_model)
    plot(eco_model, postfix="")



