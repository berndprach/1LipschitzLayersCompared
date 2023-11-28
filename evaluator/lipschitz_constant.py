
import torch

from models.layers.utils import module_symmetric_power_iteration_all_iterations


# @torch.nn.utils.parametrize.cached()
@torch.no_grad()
def bound_lipschitz_constant_all_iterations(model,
                                            model_input_size: int = 32,
                                            nrof_iterations=1000):
    """
    Number of iterations:
     - O(10^3) should give a good estimation for individual layers.
     - O(10^5) should give a good estimation for the whole model.
    """
    all_iterations = None

    modules = [module for module in model.modules()
               if not isinstance(module, torch.nn.Sequential)]

    # Hack: For our model, side_length * nrof_channels is constant.
    product = None  # side_length * nrof_channels

    exceptions = ['SOC', 'BCOP', 'ECO']
    for module in modules:
        model_is_linear = isinstance(module, torch.nn.Linear)
        model_is_conv = isinstance(module, torch.nn.Conv2d)
        model_is_exception = any(
            [exception in module.__class__.__name__ for exception in exceptions])

        if model_is_conv or model_is_linear or model_is_exception:
            nrof_channels = (
                module.in_channels
                if model_is_conv or model_is_exception
                else module.in_features
            )

            if product is None:
                product = nrof_channels * model_input_size

            side_length = product // nrof_channels  # 32 => 32, 64 => 16, ...

            if model_is_conv or model_is_exception:
                input_size = (nrof_channels, side_length, side_length)
            else:
                input_size = (nrof_channels,)

            power_iteration = module_symmetric_power_iteration_all_iterations
            layer_all_iterations = power_iteration(
                module,
                input_size,
                nrof_iterations=nrof_iterations,
                # dtype=torch.float64,
            )

            if all_iterations is None:
                all_iterations = layer_all_iterations
            else:
                all_iterations = all_iterations * layer_all_iterations

    return all_iterations


def bound_lipschitz_constant(*args, **kwargs):
    bound_all_iterations = bound_lipschitz_constant_all_iterations(
        *args, **kwargs)
    if bound_all_iterations is None:
        return torch.tensor(1.)
    else:
        return bound_all_iterations[-1]
