import inspect

import numpy as np
import uncertainties
from scipy.optimize import fmin as simplex
from uncertainties import ufloat


# "fmin" is not a sensible name for an optimisation package.
# Rename fmin to "simplex"

# Define the objective function to be minimised by Simplex.
# params ... array holding the values of the fit parameters.
# X      ... array holding x-positions of observed data.
# Y      ... array holding y-values of observed data.
# Err    ... array holding errors of observed data.

# Max Pernklau: cobbled together from https://python4mpia.github.io/fitting_data/simplex-fitting.html


def squareError(fitFunction, fitParameterCount, params, X, Y, Err):
    """Computes chi-square (least-square problem) for an abitraty function fitFunction."""
    params = params[:fitParameterCount:]
    chi2 = 0.0
    for n in range(len(X)):
        x = X[n]
        y = fitFunction([x], *params)[0]  # mad conversions because input functions work on arrays instead of numbers

        chi2 = chi2 + (Y[n] - y) * (Y[n] - y) / (Err[n] * Err[n])
    return chi2


def optimize(x, y, fitFunction, p0=None, y_sigmas=None):
    """Optimizes fit parameters using assuming least-square using simplex algo.
    Returns array with fit parameters and the sum of the deviations squared (fit quality)"""

    yNominal = np.array(y)
    yError = [0.0001 for i in y]  # assuming 1 if no stdDev is given
    if isinstance(y[0], uncertainties.UFloat):
        yNominal = [i.nominal_value for i in y]
        yError = [i.std_dev for i in y]
    if y_sigmas is not None:
        yError = y_sigmas

    fitParameterCount = len(inspect.getargspec(fitFunction).args) - 1

    if p0 is None:
        p0 = [0 for i in range(fitParameterCount)]  # starting from 0 if no initial guess is given

    squareErrorForFitFunction = lambda params, xData, yData, error: squareError(fitFunction, fitParameterCount, params, xData, yData, error)
    fitParams = simplex(squareErrorForFitFunction, p0, args=(x, y, yError), disp=True, full_output=0, maxfun=1000000, maxiter=1000000)
    fitQual = fitQuality(x, y, lambda xIn: fitFunction(xIn, *fitParams))

    return [fitParams, fitQual]


def fitQuality(xData, yData, fit):
    assert(len(xData) > 2)

    n = len(xData)
    s2 = 0.0
    for i in range(len(xData)):
        x = xData[i]
        s2 += 1 / (n - 2) * (yData[i] - fit([x])[0])**2
    return s2


if __name__ == "__main__":  # unit testing
    polynom = lambda x, a, b, c: [a + r * b + r**2 * c for r in x]
    print(polynom([2], 1, 2, 3))
    X = [1, 2, 3, 4, 5]
    Y = [3, 4, 5, 6, 7]
    Err = [1, 1, 1, 1, 1]
    print(squareError(polynom, 3, [1, 2, 3, 4, 5, 6], X, Y, Err))

    fitFunction = lambda x, a, b: [r**2 * a - b / r for r in x]
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    Y = fitFunction(X, 1000, 0.00005)
    Err = np.array(X) * 1e-2
    #Y = [ufloat(nom, err) for nom, err in zip(Y_, Err)]
    print(Y)
    print("Simplex Test Fit:", optimize(X, Y, fitFunction))
