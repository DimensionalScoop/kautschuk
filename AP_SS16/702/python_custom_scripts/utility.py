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

    # default is round to even!!!
#   1.25  -> 1.2
#   1.251 -> 1.3
#   1.15  -> 1.2
#   1.151 -> 1.2
def round_figures(value, figures):
    """Returns rounded value
    Args:
            value   (float) The value which you want to be rounded
            figures (int)   The number of digits after the point
    """
    d = Decimal(value)
    if d == 0:
        return '0'
    d = round(d, figures - int(math.floor(math.log10(abs(d)))) - 1)
    return "{:f}".format(d)
