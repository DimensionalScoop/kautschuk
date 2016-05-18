from numpy import array, pi
from uncertainties import ufloat
from uncertainties.unumpy import uarray
import numpy as np
import uncertainties.unumpy as unp


L = 1.75e-3
C1 = 22.0e-9
C2 = 9.39e-9
std_dev_frequenzeinstellung = 60
std_dev_wellenwiderstand = 5


def calc_std_dev_oszilloskop(werte):
    """Generiert Standardabweichungen für mit dem Oszilloskop gemessene Werte.

    Args:
        werte (array): Messwerte.

    Returns:
        uarray: Messwerte mit Fehler
    """
    nominal = werte
    magnitude = np.floor(np.log10(nominal))
    error = [0.05 * 10**mag for mag in magnitude]

    return uarray(nominal, error)


class dummy():
    pass

# Aufgabe a)
grenzfrequenzmessung = dummy()
grenzfrequenzmessung.Z_End = ufloat(289, std_dev_wellenwiderstand)
grenzfrequenzmessung.Z_Anf = ufloat(282, std_dev_wellenwiderstand)
grenzfrequenzmessung.gemessene_f_homogene_Kette = ufloat(52165, 50)
grenzfrequenzmessung.gemessene_f_inomogene_Kette = ufloat(39949, 50)


# Aufgabe b)
dispersion = dummy()
dispersion.phase    = array([pi, 0, pi, 0, pi, 0, pi, 0])
dispersion.frequenz = uarray([5066, 10031, 14916, 19610, 24174, 28438, 32533, 36150], [std_dev_frequenzeinstellung] * 8)
dispersion.Z_End = ufloat(277, 5)
dispersion.glieder_anzahl = 16

# Aufgabe c)
eigenfrequenzen_homogen = uarray([5058, 10090, 14905, 19566, 24186, 28469], [std_dev_frequenzeinstellung] * 6)

# Aufgabe d)
gliedamplitude_homogen = dummy()
gliedamplitude_homogen.amplituden_erste_eigenfrequenz = 1e-3 * array([5600, 5480, 5160, 4650, 3950, 3100, 2160, 1100, 30, 1050, 2130, 3080, 3960, 4680, 5180, 5500, 5560])
gliedamplitude_homogen.amplituden_zweite_eigenfrequenz = 0.5e-3 * array([2900, 2740, 2220, 1250, 111, 1060, 2120, 2700, 2900, 2850, 2200, 1220, 51, 1060, 2020, 2600, 2850])
gliedamplitude_homogen.amplituden_erste_eigenfrequenz = calc_std_dev_oszilloskop(gliedamplitude_homogen.amplituden_erste_eigenfrequenz) * 0.5# Es wurden peak-to-peak-Werte gemessen
gliedamplitude_homogen.amplituden_zweite_eigenfrequenz = calc_std_dev_oszilloskop(gliedamplitude_homogen.amplituden_zweite_eigenfrequenz) * 0.5

# Aufgabe e)
gliedamplitude_inhomogen = dummy()
gliedamplitude_inhomogen.Z_End = ufloat(285, std_dev_wellenwiderstand)
gliedamplitude_inhomogen.amplituden_erste_eigenfrequenz = 1e-3 * array([155, 155, 158, 162, 168, 172, 178, 182, 180, 184, 180, 176, 170, 166, 160, 158, 158])
gliedamplitude_inhomogen.amplituden_erste_eigenfrequenz = calc_std_dev_oszilloskop(gliedamplitude_inhomogen.amplituden_erste_eigenfrequenz) * 0.5
