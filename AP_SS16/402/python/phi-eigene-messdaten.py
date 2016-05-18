import numpy as np
import uncertainties.unumpy as unp
from numpy import array
import helpers as hel

# Hauptprogramm benutzt Messdaten von Saskia, Sonja

_phi_r = np.deg2rad(array([262.3, 266.5, 259]))
_phi_l = np.deg2rad(array([131, 128, 133.3]))
phi = 0.5 * hel.combine_measurements(_phi_r - _phi_l)

print(phi / np.pi * 180)
