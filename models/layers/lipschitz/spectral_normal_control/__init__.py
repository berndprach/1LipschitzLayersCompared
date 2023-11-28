r"""
Some functions taken from https://github.com/ColinQiyangLi/LConvNet.git
"""

# Need to import power_iteration before bnb:
from .power_method import power_iteration

from .fantastic4 import fantastic_four, l2_normalize
from .bnb import bjorck_orthonormalize
# from .power_method import power_iteration


