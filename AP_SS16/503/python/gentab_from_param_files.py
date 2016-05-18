import logging
import os
from logging import info

import numpy as np
import uncertainties as ucs
from scipy.optimize import curve_fit
from uncertainties.unumpy import uarray

import data
import fit
import helpers as hel
import simplex
from table import make_table

# remove this line if you aren't Max
os.chdir("/home/elayn/Projects/Uni/fluffy-giggle/Bearbeitung/V406/python/")

fit_array = []

slit_count = 0
for slit in data.single_slits:
    slit_count += 1
    fit_params = np.genfromtxt("data/single_slit" + str(slit_count) + ".fit_params")
    qual = simplex.fitQuality(slit.angles, slit.currents, lambda xIn: fit.single_slit_fit(xIn, *fit_params))
    fit_array.append(np.append(np.append([slit_count], fit_params), [qual]))

fit_array = np.array(fit_array)
print(fit_array)

make_table(fit_array, "../table/fit_data.tex", figures=[0, 3, 3, 3, 3])
