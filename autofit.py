import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import uncertainties
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import uarray

maxfev = 1000000

# by Max Pernklau


def autofit(x, y, fitFunction, p0_=None, weights=None, bounds=(-np.inf, np.inf), return_function=False):
    """Returns params of the curvefit as ufloat.
    Args:
        x (numpy.array of floats): The X axis of your data. Doesn't support errors for this axis, unfortunately.
        y (numpy.array of floats or ufloats): The Y axis of your data. Uses std_devs, if available, as weight for the fit. 
        fitFunction (function): The function you want to fit. First parameter of the function is always x, autofit will try to determine all other parameters as parameters of the fit
        p0_ (array of floats): Starting values for the parameter search. If you provide starting values, you must provide them for all the params you want to fit (so `len(p0_)` should be the number of parameters of `fitFunction`-1). Be sure to provide starting values as close to the real parameters the fit should return as possible. Otherwise the fit may 'converge' in a local minimum of the error function that is not the global minimum of the error function.
        weights (numpay.array of floats): Optional weight for the fit (will be ignored if you provide a ufloat array for y). Look for `absolute_sigma` in the curve_fit doc for an in-depth explanation.
        bounds (tuple of a float array of lower bounds and float array of upper bounds): Bounds for the parameter search. If provided curve_fit uses another algorithm. YMMV.
        return_function (bool): Returns the fitted function and the parameters. See Example.

    E.g.:
        Fit a sinus with an amplitude `A` and a angular frequency `w` and print the resulting fit evaluated at x=1 and then the fitted value for the amplitude: 
        `f = lambda x,A,w: A * np.sin(x * w)
        f_u = lambda x,A,w: A * unp.sin(x * w)
        params = autofit(data_x, data_y, f)
        print(f_u(1,*params))
        print(params[0])`
        Note that params is an array of ufloats, so you need uncertainties.unumpy.sin to evaluate the function instead of numpy.sin.

        Fit a simple polynomial and use return_function to get the fitted function directly: 
        `f = lambda x,a,b,c: a*x**2 + b*x + c
        function, params = autofit(data_x, data_y, f, return_function=True)
        print(function(1))`
    """
    if isinstance(y[0], uncertainties.UFloat):
        ny = [i.nominal_value for i in y]
        dy = [i.std_dev for i in y]
        params, covariance = curve_fit(fitFunction, x, ny, sigma=dy, absolute_sigma=True,
                                       p0=p0_, maxfev=maxfev,
                                       bounds=bounds)
    else:
        params, covariance = curve_fit(fitFunction, x, y, p0=p0_, maxfev=maxfev, sigma=weights, bounds=bounds)
    errors = np.sqrt(np.diag(covariance))

    params_with_error = uarray(params, errors)

    if return_function:
        return (lambda x: fitFunction(x, *uarray(params, errors)), params_with_error)
    return params_with_error
