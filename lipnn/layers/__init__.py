
from .lipschitz.aol import AOLConv2d
from .lipschitz.bcop import BCOP
from .lipschitz.bjorck import BjorckLinear
from .lipschitz.cayley import CayleyConv, CayleyLinear
from .lipschitz.cpl import CPLConv2d
from .lipschitz.eco import ECO
from .lipschitz.lot import LOT
from .lipschitz.sandwich import SandwichConv, SandwichFc
from .lipschitz.sandwich_original import SandwichConv as SandwichConvOriginal
from .lipschitz.sandwich_original import SandwichFc as SandwichFcOriginal
from .lipschitz.sll import SLLConv2d
from .lipschitz.soc import SOC


from .activations import *
from .basic import *


def available_conv2d_layers() -> list[str]:
    r"""
    Returns a list of available convolutional layers in alphabetical order.
    """
    # return sorted(['AOLConv2d', 'BCOP', 'ECO', 'CayleyConv', 'LOT',
    #                'SOC', 'SLLConv2d', "CPLConv2d"])
    # return sorted(['AOLConv2d', 'BCOP', 'BnBConv2d', 'CayleyConv', 'LOT',
    #                'SOC', 'SLLConv2d', "CPLConv2d", "PowerMethodConv2d",
    #                "ECO"])
    return sorted(['AOLConv2d', 'BCOP', 'CayleyConv', 'LOT',
                   'SOC', 'SLLConv2d', "CPLConv2d",
                   "SandwichConv"])


ALL_COMPARED_LIPSCHITZ_LAYERS = sorted(
    ['AOLConv2d', 'BCOP', 'CayleyConv', 'LOT', 'SOC', 'SLLConv2d', "CPLConv2d"]
)
