import logging
import os
from logging import info

import numpy as np

import data
import helpers as hel
import simplex

# remove this line if you aren't Max
os.chdir("/home/elayn/Projects/Uni/fluffy-giggle/Bearbeitung/V406/python/")
logging.basicConfig(filename='test.log', filemode='w', format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)


def phi(x_, x0):
    return [(x - x0) / data.screen_distance for x in x_]


def single_slit_fit(phi_, b, amplitude, offset_phi):
    return [amplitude**2 * np.sinc(b * phi / data.lambda_)**2 for phi in phi_]
    #offset_current = 0
    # return [offset_current + amplitude**2 * b**2
    #        * data.lambda_ / (np.pi * b * np.sin(phi + offset_phi))**2
    #        * np.sin(np.pi * b * np.sin(phi + offset_phi) / data.lambda_)**2 for phi in phi_]


def double_slit_fit(phi_, b, slit_dist, amplitude, offset_phi):
    slit_dist += b
    offset_current = 0
    return [offset_current + amplitude *
            np.cos(np.pi * slit_dist * np.sin(offset_phi + phi) / data.lambda_)**2
            * (data.lambda_ / (np.pi * b * np.sin(phi + offset_phi)))**2
            * np.sin(np.pi * b * np.sin(offset_phi + phi) / data.lambda_)**2 for phi in phi_]


def mulitopt(fitinput, min_, max_, steps, other_guesses, y_sigmas):
    better_fits_found = -1
    best_fit_quality = 1e10
    best_fit_params = []
    inital_guesses = np.exp(np.linspace(np.log(min_), np.log(max_), steps))

    for guess in inital_guesses:
        params, quality = simplex.optimize(*fitinput, y_sigmas=y_sigmas, p0=np.append([guess], other_guesses))
        if quality < best_fit_quality:
            best_fit_quality = quality
            best_fit_params = params
            better_fits_found += 1
    info("Better Fits found: %s", better_fits_found)
    return [best_fit_params, best_fit_quality]



for slit in data.slits:
    count = len(slit.currents)
    midpoint = (slit.midpoint_index + 1)
    spacing = slit.spacing

    angles_left = np.linspace(-(midpoint) * spacing, -spacing, midpoint)
    angles_right = np.linspace(0, (count - midpoint) * spacing, count - midpoint)
    slit.angles = np.arctan(np.append(angles_left, angles_right) / data.screen_distance)

    slit.currents_sigma = hel.estimate_sigmas_only(slit.currents, data.analoge_abberation)

if __name__ == "__main__":
    slit_count = 0

    for slit in data.single_slits:
        slit_count += 1
        fit_params, quality = mulitopt([slit.angles, slit.currents, single_slit_fit], 1e-6, 1e-3, 100, [10, 1e-10], slit.currents_sigma)
        info("Fit Quality %s", quality)
        info("Slit Size %s", fit_params[0])
        fit_params[0] = 4e-4
        phi_offset = fit_params[2]

        np.savetxt("data/single_slit" + str(slit_count) + ".fit_params", fit_params, header='Slit Width, Amplitude, Offset Phi, Offset Current')

    slit = data.double_slit
    fit_params, quality = mulitopt([slit.angles, slit.currents, double_slit_fit], 1e-5, 1e-3, 100, [0.00048, 10, 1e-10], slit.currents_sigma)
    info("Fit Quality %s", quality)
    info("Slit Size %s", fit_params[0])
    info("Slit Distance %s", fit_params[1])
    np.savetxt("data/double_slit.fit_params", fit_params, header='Slit Width, Slit Distance, Amplitude, Offset Phi')
