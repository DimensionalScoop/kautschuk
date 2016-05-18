import numpy as np


def Z(omega, L, C):
    """Berechnet Wellenwiderstand für homogene Kette"""
    return np.sqrt(L / C) * 1 / np.sqrt(1 - 1 / 4 * omega**2 * L * C)


def Z_C1C2(L, C1, C2):
    """Berechnet Wellenwiderstand für Kette mit wechselnden C1 - C2 - Gliedern für Omega=0"""
    return np.sqrt(2 * L / (C1 + C2))


def omega(L, C, theta):
    """Dispersionsrelation

    Args:
        L (TYPE): Description
        C (TYPE): Description
        theta (TYPE): Phasenverschiebung

    Returns:
        ufloat: omega Kreisfrequenz
    """
    return np.sqrt(2 / (L * C) * (1 - np.cos(theta)))


def theta(L, C, f):
    """Dispersionsrelation

    Args:
        L (TYPE): Description
        C (TYPE): Description
        f (TYPE): Frequenz

    Returns:
        ufloat: theta Phasenverschiebung
    """
    return np.arccos(-(f * 2 * np.pi)**2 * L * C / 2 + 1)


def f_dispersion(L, C, theta):
    """Dispersionsrelation"""
    return np.sqrt(2 / L / C * (1 - np.cos(theta))) / 2 / np.pi


def grenzfrequenz(L, C):
    return 2 * np.sqrt(1 / (L * C)) / (2 * np.pi)


def grenzfrequenz_inhomogen(L, C):
    return np.sqrt(2 / (L * C)) / (2 * np.pi)


def phasengeschwindigkeit_theorie(L, C, f):
    return 2 * np.pi * f / theta(L, C, f)


def phasengeschwindigkeit(f, theta):
    return 2 * np.pi * f / theta
