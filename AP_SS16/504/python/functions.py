import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties
from uncertainties.unumpy import uarray
from scipy.optimize import curve_fit
import os

# print("Cwd:", os.getcwd())
# print("Using matplotlibrc from ", mpl.matplotlib_fname())


fig = plt.figure()
clear = plt.close()
ax = fig.add_subplot(111)

def gimmeTHATcolumn(array,k):
    """Extracts the k-column of an 2D-array, returns list with those column-elements"""
    helparray = []
    for i in range(len(array)):
        helparray.append(array[i][k])
    return helparray

def meanDistance(x):
    x = np.array(x)
    sum = 0
    for a, b in zip(x, x[1:]):
        sum += (b - a) / len(x)
    return sum / len(x)


def autoplot(xValues, yValues, xLabel, yLabel, plotLabel="", errorbars=True, plotStyle='ro', errorStyle='g,', yScale='linear', **furtherPlotArgs):
    """Return a subplot object.

    :param errorbars=True: Plots error bars when true.
    :param yScale: e.g. 'log', 'dec'
    """
    xValues = np.array(xValues)
    yValues = np.array(yValues)
    errX = None
    errY = None
    if type(xValues[0]) == uncertainties.Variable or type(xValues[0]) == uncertainties.AffineScalarFunc:
        x = [item.nominal_value for item in xValues]
        errX = [item.std_dev for item in xValues]
    else:
        x = xValues
    if type(yValues[0]) == uncertainties.Variable or type(yValues[0]) == uncertainties.AffineScalarFunc:
        y = [item.nominal_value for item in yValues]
        errY = [item.std_dev for item in yValues]
    else:
        y = yValues

    ax.set_yscale(yScale)
    x_offset = (max(x) - min(x)) * 0.015
    ax.set_xlim(min(x) - x_offset, max(x) + x_offset)
    if yScale != 'log':
        y_offset = (max(y) - min(y)) * 0.015
        ax.set_ylim(min(y) - y_offset, max(y) + y_offset)

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    ax.legend(loc='best')

    if errorbars:
        if errX != None and errY != None:
            plt.errorbar(x, y, xerr=errX, yerr=errY, fmt=errorStyle)
        elif errY != None:
            plt.errorbar(x, y, yerr=errY, fmt=errorStyle)
            print(errY)
        elif errX != None:
            plt.errorbar(x, y, xerr=errX, fmt=errorStyle)
        else:
            raise "Should draw errorbars, but x, y are not ufloats!"
    ax.plot(x, y, plotStyle, label=plotLabel, **furtherPlotArgs)

    fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    return fig


def linearFit(x, a, b):
    return a * x + b


def isUfloat(var):
    return type(var) == uncertainties.core.Variable or type(var) == uncertainties.core.AffineScalarFunc

maxfev = 1000000


def autofit(x, y, fitFunction, p0=None):
    """Returns params of the curvefit as ufloat."""
    if isUfloat(y[0]):
        ny = [i.nominal_value for i in y]
        dy = [i.std_dev for i in y]
        params, covariance = curve_fit(fitFunction, x, ny, sigma=dy, absolute_sigma=True,
                                       p0=p0, maxfev=maxfev)
    else:
        params, covariance = curve_fit(fitFunction, x, y, p0=p0, maxfev=maxfev)
    errors = np.sqrt(np.diag(covariance))
    return uarray(params, errors)


def array(values, offset, magnitude):
    """Return numpy array

    offset: is added to all items
    magnitude: all items are multiplied by 10^magnitude"""
    res = np.array(values).astype(float) + offset
    res *= 10 ** magnitude
    return res


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


def errorString(value):
    """Return a more beautiful number with error"""
    return str(value.nominal_value) + "±" + str(value.std_dev)


def abweichung(value, lit):
    """Returns diveation of an experimental value from a literature value."""
    return '{:.3f}'.format((lit - value.nominal_value) / lit * 100) + "%"


def modifiyItems(dic, keyFunction, valueFunction):
    """Applies *funkction(key,value) to each key or value in dic"""
    return {keyFunction(key, value): valueFunction(key, value) for key, value in dic.items()}


# find peaks
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

# if __name__=="__main__":
#    from matplotlib.pyplot import plot, scatter, show
#    series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
#    maxtab, mintab = peakdet(series,.3)
#    plot(series)
#    scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
#    scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
#    show()


def getPeakVal(peaksmax):
    """gets the values of the peaks for the x and y axes"""
    peakst = []
    for i in range(len(peaksmax)):
        peakst.append(peaksmax[i][0])
    peaksT = []
    for i in range(len(peaksmax)):
        peaksT.append(peaksmax[i][1])
    return peakst, peaksT


def get_noms(values):
    return array([i.nominal_value for i in values])


def get_std_dev(values):
    return array([i.std_dev for i in values])
