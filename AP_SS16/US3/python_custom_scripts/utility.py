import math
from decimal import Decimal
import scipy
from uncertainties import ufloat

def constant(name):
    """Returns ufloat containing value and its error of a physical constant
    Args:
            name    (str) The name of the constant which can be found at http://docs.scipy.org/doc/scipy/reference/constants.html
    """
    c = scipy.constants.physical_constants[name]
    return ufloat((c[0], c[2]))
