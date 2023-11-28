
import itertools

from models.simplified_conv_net import SimplifiedConvNet, DEFAULT_MODELS
from models import layers


def get_all_model_layer_combinations(use_tiny_image_net: bool,
                                     remove_pointwise_kernel: bool = True,
                                     logging=None):

    if logging is None:
        def logging(*args, **kwargs):
            pass

    all_model_names = DEFAULT_MODELS.keys()

    if remove_pointwise_kernel:  # Remove 1x1 conv models:
        all_model_names = [n for n in all_model_names if "1x1" not in n]

    if use_tiny_image_net:
        all_model_names = [n for n in all_model_names if "TConv" in n]
    else:
        all_model_names = [n for n in all_model_names if "TConv" not in n]

    logging(f"Found {len(all_model_names)} models. "
            f"({', '.join(all_model_names)})")

    # all_layers = layers.available_conv2d_layers()
    all_layers = layers.ALL_COMPARED_LIPSCHITZ_LAYERS
    all_layers = ["StandardConv2d"] + all_layers
    logging(f"Found {len(all_layers)} layers. ({', '.join(all_layers)})")

    all_combinations = list(itertools.product(all_model_names, all_layers))
    logging(f"Found {len(all_combinations)} combinations: "
            f"({', '.join(str(c) for c in all_combinations[:4])} ...)")

    return all_combinations


def get_model(chosen_model_name, chosen_layer_name):
    linear_layers = {
        "CayleyConv": layers.CayleyLinear,
        "BCOP": layers.BnBLinearBCOP,  # <= This layer keeps memory somehow!
        "SandwichConv": layers.SandwichFc,
    }
    linear_cls = linear_layers.get(chosen_layer_name, None)

    conv_cls = getattr(layers, chosen_layer_name)

    return SimplifiedConvNet(get_conv=conv_cls,
                             get_activation=layers.MaxMin,
                             get_linear=linear_cls,
                             **DEFAULT_MODELS[chosen_model_name])
