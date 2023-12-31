
from .lipschitz.aol import AOLConv2d, AOLConv2dDirac, AOLConv2dOrth
from .lipschitz.bnb import BnBLinear, BnBLinearBCOP
from .lipschitz.bcop import BCOP
from .lipschitz.cayley import CayleyConv, CayleyLinear
from .lipschitz.cpl import CPLConv2d, CPLConv2d10k
from .lipschitz.eco import ECO
from .lipschitz.lot import LOT, LOT2t
from .lipschitz.sandwich import SandwichConv, SandwichFc
from .lipschitz.sandwich_original import SandwichConv as SandwichConvOriginal
from .lipschitz.sandwich_original import SandwichFc as SandwichFcOriginal
from .lipschitz.sll import SLLConv2d
from .lipschitz.soc import SOC

from .lipschitz.spectral_normal_control import *

from .activations import *
from .basic import *

from torch.nn import *


class StandardConv2d(Conv2d):
    def __init__(self,
                 *args,
                 # initializer: Optional[Callable] = None,
                 padding='same',
                 padding_mode='circular',
                 **kwargs) -> None:

        super().__init__(*args, padding=padding, padding_mode=padding_mode, **kwargs)


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
