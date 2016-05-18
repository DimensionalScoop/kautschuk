import uncertainties as u
import numpy as np
import uncertainties.unumpy as unp
import uncertainties.umath as umath

import data as d
import calculations as c
import plot as plot
import helpers as h
from table import (
    make_table,
    make_SI,
    write,
)


print("Maximale Messdifferenz:", max(d.eigenfrequenzen_homogen - d.dispersion.frequenz[:-2:]))


print("--- Grenzfrequenz mit", d.L, "H,", d.C1, "F", "---")
print("Gerechnet:", c.grenzfrequenz(d.L, d.C1))
print("Oszilloskop:", d.grenzfrequenzmessung.gemessene_f_homogene_Kette)
print("Abgelesen:", 51.52e3)
print("Abweichung zu C1:", d.grenzfrequenzmessung.gemessene_f_homogene_Kette - c.grenzfrequenz(d.L, d.C1))

print("--- Grenzfrequenz mit", d.L, "H,", d.C1, "F", d.C2, "F", "---")
print("Gerechnet:", c.grenzfrequenz_inhomogen(d.L, d.C1))
print("Oszilloskop:", d.grenzfrequenzmessung.gemessene_f_inomogene_Kette)
print("Abweichung zu C1:")

print("--- Plot Frequenz gegen Phase ---")


def f_dispersion_kHz(t):
    return c.f_dispersion(d.L, d.C1, t) * 1e-3

phase = []
for i in range(len(d.dispersion.phase)):
    phase.append((i + 1) * np.pi)  # Jede Lissajou-Figur liegt um pi von der nächsten entfernt
    phase[-1] /= d.dispersion.glieder_anzahl

plot.plot(phase, d.dispersion.frequenz * 1e-3, f_dispersion_kHz,
          r"Phasenverschiebung in Radian", r"Anregungsfrequenz in kHz", "dispersion.pdf")


make_table((
    range(1, len(d.dispersion.phase) + 1),
    d.dispersion.frequenz * 1e-3,
    phase,
), "../table/dispersion.tex",
    figures=[1, 1, 2])


print("--- Plot Frequenz gegen Phasengeschwindigkeit ---")


def phase_theorie(f):
    return c.phasengeschwindigkeit_theorie(d.C1, d.L, f * 1e3)

x = d.dispersion.frequenz  # d.eigenfrequenzen_homogen werden nicht benötigt! Sie sind identisch mit ~
y = c.phasengeschwindigkeit(x, phase)
plot.plot(x * 1e-3, y, phase_theorie,
          r"Anregungsfrequenz in kHz", "Phasengeschwindigkeit in Gliedern/s", "phasengesch.pdf")


print("--- Plot Amplitude an den Kettengliedern ---")


def sin_fit(x, amplitude, offset):
    omega = 2 * np.pi / 17.
    return offset + amplitude * np.cos(omega * np.array(x))


def plotAmplitudes(amplitude, name):
    glied = range(1, len(amplitude) + 1)
    plot.plot(glied, amplitude, None,
              r"# des Glieds", r"Schwingungsamplitude in V", "amplitude " + name + ".pdf")

plotAmplitudes(d.gliedamplitude_homogen.amplituden_erste_eigenfrequenz, "homogen, erste eigen f")
plotAmplitudes(d.gliedamplitude_homogen.amplituden_zweite_eigenfrequenz, "homogen, zweite eigen f")
plotAmplitudes(d.gliedamplitude_inhomogen.amplituden_erste_eigenfrequenz, "inhomogen, erste eigen f")


print("--- Eigenfrequenzen ---")
delta = [u.ufloat(0, 0)]
for i in range(1, len(d.eigenfrequenzen_homogen)):
    delta.append(d.eigenfrequenzen_homogen[i] - d.eigenfrequenzen_homogen[i - 1])


print(delta)

make_table([d.eigenfrequenzen_homogen,
            delta], '../data/eigenf.tex', [1, 1])
