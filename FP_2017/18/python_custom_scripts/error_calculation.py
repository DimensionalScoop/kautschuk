import numpy as np
from uncertainties import unumpy as unp
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)
import scipy.stats

#Fehler des Mittelwertes
def MeanError (Data):
    assert isinstance(Data, (np.ndarray, np.generic) ) , "You need to give a numpy array for calculation of mean error."
    assert Data.size , "Empty arrays are not allowed for calculation of mean error."
    Mean = np.mean(Data)
    N = np.size(Data)
    qSum = np.sum( (Data - Mean)**2 )
    Error = 1 / np.sqrt(N) * np.sqrt(1 / (N-1) * qSum  )
    return Error

def mean(values, axis=0):
    """Returns mean values and their mean errors of a given array. Return value will be a unp.uarray
    Args:
            values:     (list)  Array containing numbers whose mean is desired.
            axis:       (int)   Axis along which the means are computed. The default is to compute the mean of the flattened array.
    """
    return unp.uarray(np.mean(noms(values), axis=axis), scipy.stats.sem(noms(values), axis=axis))

# RMSE berechnen (root mean square error) berechnen, Abweichung von Vorhersage
def rmse(yTrue, yPred):
    return np.sqrt(np.mean((yTrue-yPred)**2))
