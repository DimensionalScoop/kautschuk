import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import uncertainties.unumpy as unp
from scipy.constants import K2C
from scipy.constants import C2K

from uncertainties import ufloat
import uncertainties
from uncertainties.unumpy import uarray
from scipy.optimize import curve_fit
import sys


def extract_error(data):
    if(isinstance(data[0], uncertainties.UFloat)):
        error = unp.std_devs(data)
        nominal = unp.nominal_values(data)
    else:
        nominal = data
        error = None
    return nominal, error


def autolimits(data, err=None):
    min_lim = min(data)
    max_lim = max(data)
    offset = (max(data) - min(data)) * 0.015
    if err is not None:
        offset += max(err)
    return [min_lim - offset, max_lim + offset]


def plot(x_messung, y_messung, theorie, xlabel, ylabel, filename):
    """Plottet diskrete Messwerte gegen eine kontinuierliche Messkurve

    Args:
        x_messung (uarray)
        y_messung (uarray)
        theorie (func(x)): Theoriefunktion, die x-Werte annimmt und y-Werte ausspuckt
        xlabel (string)
        ylabel (string)
        filename (string)

    Returns:
        TYPE: None
    """
    plt.clf()

    x_messung, x_error = extract_error(x_messung)
    y_messung, y_error = extract_error(y_messung)

    x_limit = autolimits(x_messung, err=x_error)
    x_flow = np.linspace(*x_limit, num=1000)
    y_messung = y_messung

    if theorie is not None:
        plt.plot(x_flow, theorie(x_flow), 'g-', label="Theoriekurve")
    plt.errorbar(x_messung, y_messung, xerr=x_error, yerr=y_error, fmt='r,',label="Fehler")
    plt.plot(x_messung, y_messung, 'r.', label="Messwerte")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(loc='best')

    plt.xlim(x_limit)
    plt.ylim(autolimits(y_messung, err=y_error))
    plt.grid()
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    # plt.show()
    plt.savefig('../plots/' + filename)
