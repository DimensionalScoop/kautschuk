import numpy as np

#Fehler des Mittelwertes
def MeanError (Data):
    assert isinstance(Data, (np.ndarray, np.generic) ) , "You need to give a numpy array for calculation of mean error."
    assert Data.size , "Empty arrays are not allowed for calculation of mean error."
    Mean = np.mean(Data)
    N = np.size(Data)
    qSum = np.sum( (Data - Mean)**2 )
    Error = 1 / np.sqrt(N) * np.sqrt(1 / (N-1) * qSum  )
    return Error

# RMSE (root mean square error) berechnen, Abweichung von Vorhersage
def rmse(yTrue, yPred):
    return np.sqrt(np.mean((yTrue-yPred)**2))
