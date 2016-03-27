import matplotlib as mpl
mpl.rcdefaults()
mpl.rcParams.update(mpl.rc_params_from_file('meine-matplotlibrc'))
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)
import Regression as reg

from curve_fit import ucurve_fit
from table import (
    make_table,
    make_full_table,
    make_composed_table,
    make_SI,
    write,
)

## FREQUENZLY USED CODE
#
## IMPORT
# t, U, U_err = np.genfromtxt('data.txt', unpack=True)
# t *= 1e-3
#
#
## ERRORS
# R_unc = ufloat(R[0],R[2])
# U = 1e3 * unp.uarray(U, U_err)
# Rx_mean = np.mean(Rx)       # Mittelwert und syst. Fehler
# Rx_mean_err = MeanError(noms(Rx))    # Fehler des Mittelwertes
#
#
## CURVE FIT
# def f(t, a, b, c, d):
#     return a * np.sin(b * t + c) + d
#
# params = ucurve_fit(f, t, U, p0=[1, 1e3, 0, 0])
# params = ucurve_fit(reg.reg_linear, x, y)         # linearer Fit
# params = ucurve_fit(reg.reg_quadratic, x, y)      # quadratischer Fit
# params = ucurve_fit(reg.reg_cubic, x, y)          # kubischer Fit
# a, b, c, d = params
# write('build/loesung-a.tex', make_SI(a * 1e-3, r'\kilo\volt', figures=1))     # type in Anz. signifikanter Stellen
# write('build/loesung-b.tex', make_SI(b * 1e-3, r'\kilo\hertz', figures=1))
# write('build/loesung-c.tex', make_SI(c,        r'', figures=1))
# write('build/loesung-d.tex', make_SI(d * 1e-3, r'\kilo\volt', figures=2))
#
#
## PLOTTING
# t_plot = np.linspace(-0.5, 2 * np.pi + 0.5, 1000) * 1e-3
# plt.clf
### AUTOMATICLY CHOSING LIMITS WITH EXISTING ARRAY T1
## t_plot = np.linspace(np.amin(T1), np.amax(T1), 100)
## plt.xlim(t_plot[0]-1/np.size(T1)*(t_plot[-1]-t_plot[0]), t_plot[-1]+1/np.size(T1)*(t_plot[-1]-t_plot[0]))
#
# plt.plot(t_plot * 1e3, f(t_plot, *noms(params)) * 1e-3, 'b-', label='Fit')
# plt.plot(t * 1e3, U * 1e3, 'rx', label='Messdaten')
## plt.errorbar(B * 1e3, noms(y) * 1e5, fmt='rx', yerr=stds(y) * 1e5, label='Messdaten')
## plt.xscale('log')    # logarithmische x-Achse
# plt.xlim(t_plot[0] * 1e3, t_plot[-1] * 1e3)
# plt.xlabel(r'$t \:/\: \si{\milli\second}$')
# plt.ylabel(r'$U \:/\: \si{\kilo\volt}$')
# plt.legend(loc='best')
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/loesung-plot.pdf')
#
#
## WRITING TABLES
# t1, t2 = np.array_split(t * 1e3, 2)
# U1, U2 = np.array_split(U * 1e-3, 2)
# write('build/loesung-table.tex', make_table([t1, U1, t2, U2], [3, None, 3, None]))  # type in Nachkommastellen
### ONLY ONE COLUMN IN A TABLE:
## a=np.array([Wert_d[0]])
## b=np.array([Rx_mean])
## c=np.array([Rx_mean_err])
## d=np.array([Lx_mean*1e3])
## e=np.array([Lx_mean_err*1e3])
## write('build/Tabelle_err_d.tex', make_table([a,b,c,d,e],[0, 1, 0, 1, 1]))
## FULLTABLE
# write('build/Tabelle_b_texformat.tex', make_full_table(
#     'Messdaten Kapazitätsmessbrücke.',
#     'table:A2',
#     'build/Tabelle_b.tex',
#     [1,2,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
##                              # die Multicolumns sein sollen
#     ['Wert',
#     r'$C_2 \:/\: \si{\nano\farad}$',
#     r'$R_2 \:/\: \si{\ohm}$',
#     r'$R_3 / R_4$', '$R_x \:/\: \si{\ohm}$',
#     r'$C_x \:/\: \si{\nano\farad}$']))

# # Relative Fehler
# RelFehler_G = (G_mess - G_lit) / G_lit
# RelFehler_B = (B_mess - B_lit) / B_lit
# write('build/RelFehler_G.tex', make_SI(RelFehler_G*100, r'\percent', figures=1))
# write('build/RelFehler_B.tex', make_SI(RelFehler_B*100, r'\percent', figures=1))
#
#
## ARRAY FUNCTIONS
# np.arange(2,10)                   # Erzeugt aufwärts zählendes Array von 2 bis 10
# np.zeros(15)                      # Erzeugt Array mit 15 Nullen
# np.ones(15)                       # Erzeugt Array mit 15 Einsen
#
# np.amin(array)                    # Liefert den kleinsten Wert innerhalb eines Arrays
# np.argmin(array)                  # Gibt mir den Index des Minimums eines Arrays zurück
# np.amax(array)                    # Liefert den größten Wert innerhalb eines Arrays
# np.argmax(array)                  # Gibt mir den Index des Maximums eines Arrays zurück
#
# a1,a2 = np.array_split(array, 2)  # Array in zwei Hälften teilen
# np.size(array)                    # Anzahl der Elemente eines Arrays ermitteln
#
## ARRAY INDEXING
# y[n - 1::n]                       # liefert aus einem Array jeden n-ten Wert als Array
#
#
## DIFFERENT STUFF
# R = const.physical_constants["molar gas constant"]      # Array of value, unit, error
