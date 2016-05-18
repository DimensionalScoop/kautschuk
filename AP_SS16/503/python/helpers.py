import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import uncertainties
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import uarray

maxfev = 1000000


def autofit(x, y, fitFunction, p0_=None):
    """Returns params of the curvefit as ufloat."""
    if isinstance(y[0], uncertainties.UFloat):
        ny = [i.nominal_value for i in y]
        dy = [i.std_dev for i in y]
        params, covariance = curve_fit(fitFunction, x, ny, sigma=dy, absolute_sigma=True,
                                       p0=p0_, maxfev=maxfev)
    else:
        params, covariance = curve_fit(fitFunction, x, y, p0=p0_, maxfev=maxfev)
    errors = np.sqrt(np.diag(covariance))
    return uarray(params, errors)


def combine_measurements(values):
    """Combines a np.array of measurements into one ufloat"""
    return ufloat(mean(values), stdDevOfMean(values))


def mean(values):
    """Return the mean of values"""
    values = np.array(values)
    return sum(values) / len(values)


def stdDev(values):
    """Return estimated standard deviation"""
    values = np.array(values)
    b = 0
    m = mean(values)
    for x in values:
        b += (x - m) ** 2
    return np.sqrt(1 / (len(values) - 1) * b)


def stdDevOfMean(values):
    """Return estimated standard deviation of the mean (the important one!)"""
    return stdDev(values) / np.sqrt(len(values))


def cutErrors(values):
    """Converts an array of ufloat to an array of floats, discarding errors"""
    return np.array([v.nominal_value for v in values])


def estimate_sigmas(values, ableseunsicherheit):
    """Generates std deviations for analoge instruments. Returns a ufloatarray."""
    nominal = values
    magnitude = np.floor(np.log10(nominal))
    error = [ableseunsicherheit * 10**mag for mag in magnitude]

    return uarray(nominal, error)


def estimate_sigmas_only(values, ableseunsicherheit):
    """Generates std deviations for analoge instruments. Returns only an array with the errors."""
    nominal = values
    magnitude = np.floor(np.log10(nominal))
    error = [ableseunsicherheit * 10**mag for mag in magnitude]

    return error
