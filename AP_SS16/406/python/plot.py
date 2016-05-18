import os

import numpy as np
from uncertainties.unumpy import uarray

import data
import helpers as hel
import plot_helpers as plot
import simplex
from fit import double_slit_fit, single_slit_fit

# remove this line if you aren't Max
os.chdir("/home/elayn/Projects/Uni/fluffy-giggle/Bearbeitung/V406/python/")

slit_count = 0
for slit in data.single_slits:
    slit_count += 1
    fit_params = np.genfromtxt("data/single_slit" + str(slit_count) + ".fit_params")
    plot.plot(slit.angles, uarray(slit.currents, slit.currents_sigma), lambda phi: single_slit_fit(phi, *fit_params), "Phi in Rad", "Zur Helligkeit proportionaler Diodenstrom in A", "../plots/single_slit" + str(slit_count) + ".pdf")

    # plot minimizer function
    # b = np.linspace(1e-3, 1e-6, 1e4)
    # qual = simplex.fitQuality(slit.angles, slit.currents, lambda xIn: single_slit_fit(xIn, b, *fit_params[:-1:]))
    # plot.plot(b, qual, None, "Phi in Rad", "Zur Helligkeit proportionaler Diodenstrom in A", "../plots/single_slit-error" + str(slit_count) + ".pdf")

    qual = simplex.fitQuality(slit.angles, slit.currents, lambda xIn: single_slit_fit(xIn, *fit_params))
    print(qual)

slit = data.double_slit
fit_params = np.genfromtxt("data/double_slit.fit_params")
plot.plot(slit.angles, uarray(slit.currents, slit.currents_sigma), lambda phi: double_slit_fit(phi, *fit_params), "Phi in Rad", "Zur Helligkeit proportionaler Diodenstrom in A", "../plots/double_slit.pdf")
qual = simplex.fitQuality(slit.angles, slit.currents, lambda xIn: double_slit_fit(xIn, *fit_params))
print(qual)
