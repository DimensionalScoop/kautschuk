##################################################### Import system libraries ######################################################
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
################################################ Finish importing system libraries #################################################

################################################ Adding subfolder to system's path #################################################
import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

 # use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"python_custom_scripts")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
############################################# Finish adding subfolder to system's path #############################################

##################################################### Import custom libraries ######################################################
from curve_fit import ucurve_fit
from table import (
    make_table,
    make_full_table,
    make_composed_table,
    make_SI,
    write,
    search_replace_within_file,
)
from regression import (
    reg_linear,
    reg_quadratic,
    reg_cubic
)
from error_calculation import(
    mean,
    MeanError
)
from utility import(
    constant
)
################################################ Finish importing custom libraries #################################################

###########################   a)   ############################
Z, I = np.genfromtxt('messdaten/a.txt', unpack=True)        # I in µA
I = I*1e-6                      # in A
U = np.array(range(320, 710, 10))
delta_t = 10                    # fix for a)
Z_err = np.sqrt(Z)              # poisson verteilt
Z_unc = unp.uarray(Z, Z_err)    # mit Fehlern versehen
N = Z_unc / delta_t             # Zählrate
delta_Q = I/N/constant('elementary charge')*1e-10                   # in 10^10 e

write('build/Tabelle_a.tex', make_table([U, Z_unc, N, I*1e6, delta_Q],[0, 1, 1, 1, 1]))
write('build/Tabelle_a_texformat.tex', make_full_table(
    caption = 'Messdaten für die Charakteristik des Zählrohrs.',
    label = 'table:a',
    source_table = 'build/Tabelle_a.tex',
    stacking = [1,2,4],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
    units = [r'$U \:/\: \si{\volt}$',
    r'$Z$',
    r'$N \:/\: \si{\per\second}$',
    r'$I \:/\: 10^{10}\si{\micro\ampere}$',
    r'$\Delta Q \:/\: \si{\elementarycharge}$']))


##### Fit ####
no_of_first_ignored_values = 3
no_of_last_ignored_values = 5
params = ucurve_fit(reg_linear, U[no_of_first_ignored_values:-no_of_last_ignored_values],
N[no_of_first_ignored_values:-no_of_last_ignored_values])   # skip first 3 and last 5 values as they dont belong to the plateau
write('build/plateaulaenge.tex', make_SI(U[-no_of_last_ignored_values] - U[no_of_first_ignored_values], r'\volt', figures = 0))
a, b = params
write('build/parameter_a.tex', make_SI(a, r'\per\second\per\volt', figures=1))
write('build/plateaumitte.tex', make_SI(reg_linear(500, *noms(params)), r'\per\second', figures=1) )   # Der Wert in der Hälfte des Plateaus als Referenz für die plateausteigung
write('build/plateausteigung.tex', make_SI(a*100*100/reg_linear(500, *noms(params)), r'\percent', figures=1))  #   %/100V lässt sich nicht mit siunitx machen -> wird in latex hart reingeschrieben
write('build/parameter_b.tex', make_SI(b, r'\per\second', figures=1))

#### Plot ####
t_plot = np.linspace(350, 650, 2)
plt.xlim(290,710)
plt.plot(t_plot, reg_linear(t_plot, *noms(params)), 'b-', label='Fit')
U[no_of_first_ignored_values:-no_of_last_ignored_values]
plt.errorbar(U[no_of_first_ignored_values:-no_of_last_ignored_values],
    noms(N[no_of_first_ignored_values:-no_of_last_ignored_values]),
    fmt='bx', yerr=stds(N[no_of_first_ignored_values:-no_of_last_ignored_values]),
    label='Messdaten auf dem Plateau')
plt.errorbar(U[0:no_of_first_ignored_values],
    noms(N[0:no_of_first_ignored_values]),
    fmt='rx', yerr=stds(N[0:no_of_first_ignored_values]),
    label='Messdaten außerhalb des Plateaus')
plt.errorbar(U[-no_of_last_ignored_values:],
    noms(N[-no_of_last_ignored_values:]),
    fmt='rx', yerr=stds(N[-no_of_last_ignored_values:]))
plt.ylabel(r'$N \:/\: \si{\per\second}$')
plt.xlabel(r'$U \:/\: \si{\volt}$')
plt.grid()
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/aufgabenteil_a_plot.pdf')

plt.clf()
plt.errorbar(U, noms(delta_Q), fmt='rx', yerr=stds(delta_Q), label='Messdaten')
plt.ylabel(r'$\Delta Q \:/\: 10^{10}\si{\elementarycharge}$')
plt.xlabel(r'$U \:/\: \si{\volt}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/aufgabenteil_e_plot.pdf')


###########################   c)   ############################
# measurement data from the oscilloscope
T_tot_osci = unp.uarray([2.6], [0.5])   # in cm
T_tot_osci_inseconds = T_tot_osci*50    # in µs
T_erh_osci = unp.uarray([4.4], [1.5])   # großzügiger Fehler ftw (willkürlich)
T_erh_osci_inseconds = T_erh_osci*50    # in µs
scale = np.array([50])                  # µs/cm

write('build/Erholungszeit.tex', make_SI(T_erh_osci_inseconds[0], r'\micro\second', figures=1))
write('build/Totzeit_c.tex', make_SI(T_tot_osci_inseconds[0], r'\micro\second', figures=1))

write('build/Tabelle_c.tex', make_table([scale, T_tot_osci, T_tot_osci_inseconds, T_erh_osci, T_erh_osci_inseconds],[0, 1, 1, 1, 1]))
write('build/Tabelle_c_texformat.tex', make_full_table(
    caption = 'Messdaten aus der qualitativen Analyse der Oszilloskop Bilder. Totzeit und Erholungszeit.',
    label = 'table:c',
    source_table = 'build/Tabelle_c.tex',
    stacking = [1,2,3,4],
    units = [r'$\text{scale} \:/\: \si{\micro\second\per\centi\meter}$',
    r'$T_\text{scaled} \:/\: \si{\centi\meter}$',
    r'$T \:/\: \si{\micro\second}$',
    r'$T_{E,\text{scaled}} \:/\: \si{\centi\meter}$',
    r'$T_E \:/\: \si{\micro\second}$']))


###########################   d)   ############################
# The following values were measured
N1 = unp.uarray([8929], [np.sqrt(8929)])
N1 = N1/delta_t
N2 = unp.uarray([11358], [np.sqrt(11358)])
N2 = N2/delta_t
N12 = unp.uarray([18271], [np.sqrt(18271)])
N12 = N12/delta_t
Totzeit = (N1+N2-N12) / (2*N1*N2)           # in seconds
Totzeit *= 1e6                              # in µs
write('build/Totzeit_d.tex', make_SI(Totzeit[0], r'\micro\second', figures = 1))

write('build/Tabelle_d.tex', make_table([N1, N2, N12, Totzeit],[1, 1, 1, 1]))
write('build/Tabelle_d_texformat.tex', make_full_table(
    caption = 'Messdaten der 2-Quell-Methode und resultierende Totzeit.',
    label = 'table:d',
    source_table = 'build/Tabelle_d.tex',
    stacking = [0,1,2,3],
    units = [r'$N_1 \:/\: \si{\per\second}$',
    r'$N_2 \:/\: \si{\per\second}$',
    r'$N_{1+2} \:/\: \si{\per\second}$',
    r'$T \:/\: \si{\micro\second}$']))

# Relativer Fehler
RelFehler = (Totzeit[0] - T_tot_osci_inseconds[0]) / (1/2 * noms(Totzeit[0] + T_tot_osci_inseconds[0]))
write('build/RelFehler.tex', make_SI(abs(RelFehler)*100, r'\percent', figures=1))



################################ FREQUENTLY USED CODE ################################
#
########## IMPORT ##########
# t, U, U_err = np.genfromtxt('data.txt', unpack=True)
# t *= 1e-3


########## ERRORS ##########
# R_unc = ufloat(R[0],R[2])
# U = 1e3 * unp.uarray(U, U_err)
# Rx_mean = np.mean(Rx)                 # Mittelwert und syst. Fehler
# Rx_mean_with_error = mean(Rx, 0)      # unp.uarray mit Fehler und Fehler des Mittelwertes, die 0 gibt an, dass in einem R^2 array jeweils die Zeilen gemittelt werden sollen
# Rx_mean_err = MeanError(noms(Rx))     # nur der Fehler des Mittelwertes
#
## Relative Fehler zum späteren Vergleich in der Diskussion
# RelFehler_G = (G_mess - G_lit) / G_lit
# RelFehler_B = (B_mess - B_lit) / B_lit
# write('build/RelFehler_G.tex', make_SI(RelFehler_G*100, r'\percent', figures=1))
# write('build/RelFehler_B.tex', make_SI(RelFehler_B*100, r'\percent', figures=1))


########## CURVE FIT ##########
# def f(t, a, b, c, d):
#     return a * np.sin(b * t + c) + d
#
# params = ucurve_fit(f, t, U, p0=[1, 1e3, 0, 0])   # p0 bezeichnet die Startwerte der zu fittenden Parameter
# params = ucurve_fit(reg_linear, x, y)             # linearer Fit
# params = ucurve_fit(reg_quadratic, x, y)          # quadratischer Fit
# params = ucurve_fit(reg_cubic, x, y)              # kubischer Fit
# a, b = params
# write('build/parameter_a.tex', make_SI(a * 1e-3, r'\kilo\volt', figures=1))       # type in Anz. signifikanter Stellen
# write('build/parameter_b.tex', make_SI(b * 1e-3, r'\kilo\hertz', figures=2))      # type in Anz. signifikanter Stellen


########## PLOTTING ##########
# plt.clf                   # clear actual plot before generating a new one
#
## automatically choosing limits with existing array T1
# t_plot = np.linspace(np.amin(T1), np.amax(T1), 100)
# plt.xlim(t_plot[0]-1/np.size(T1)*(t_plot[-1]-t_plot[0]), t_plot[-1]+1/np.size(T1)*(t_plot[-1]-t_plot[0]))
#
## hard coded limits
# t_plot = np.linspace(-0.5, 2 * np.pi + 0.5, 1000) * 1e-3
#
## standard plotting
# plt.plot(t_plot * 1e3, f(t_plot, *noms(params)) * 1e-3, 'b-', label='Fit')
# plt.plot(t * 1e3, U * 1e3, 'rx', label='Messdaten')
## plt.errorbar(B * 1e3, noms(y) * 1e5, fmt='rx', yerr=stds(y) * 1e5, label='Messdaten')        # mit Fehlerbalken
## plt.xscale('log')                                                                            # logarithmische x-Achse
# plt.xlim(t_plot[0] * 1e3, t_plot[-1] * 1e3)
# plt.xlabel(r'$t \:/\: \si{\milli\second}$')
# plt.ylabel(r'$U \:/\: \si{\kilo\volt}$')
# plt.legend(loc='best')
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/aufgabenteil_a_plot.pdf')


########## WRITING TABLES ##########
### IF THERE IS ONLY ONE COLUMN IN A TABLE (workaround):
## a=np.array([Wert_d[0]])
## b=np.array([Rx_mean])
## c=np.array([Rx_mean_err])
## d=np.array([Lx_mean*1e3])
## e=np.array([Lx_mean_err*1e3])
#
# write('build/Tabelle_b.tex', make_table([a,b,c,d,e],[0, 1, 0, 1, 1]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
# write('build/Tabelle_b_texformat.tex', make_full_table(
#     caption = 'Messdaten Kapazitätsmessbrücke.',
#     label = 'table:A2',
#     source_table = 'build/Tabelle_b.tex',
#     stacking = [1,2,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
#     units = ['Wert',
#     r'$C_2 \:/\: \si{\nano\farad}$',
#     r'$R_2 \:/\: \si{\ohm}$',
#     r'$R_3 / R_4$', '$R_x \:/\: \si{\ohm}$',
#     r'$C_x \:/\: \si{\nano\farad}$'],
#     replaceNaN = True,                      # default = false
#     replaceNaNby = 'not a number'))         # default = '-'
#
## Aufsplitten von Tabellen, falls sie zu lang sind
# t1, t2 = np.array_split(t * 1e3, 2)
# U1, U2 = np.array_split(U * 1e-3, 2)
# write('build/loesung-table.tex', make_table([t1, U1, t2, U2], [3, None, 3, None]))  # type in Nachkommastellen
#
## Verschmelzen von Tabellen (nur Rohdaten, Anzahl der Zeilen muss gleich sein)
# write('build/Tabelle_b_composed.tex', make_composed_table(['build/Tabelle_b_teil1.tex','build/Tabelle_b_teil2.tex']))


########## ARRAY FUNCTIONS ##########
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


########## ARRAY INDEXING ##########
# y[n - 1::n]                       # liefert aus einem Array jeden n-ten Wert als Array


########## DIFFERENT STUFF ##########
# R = const.physical_constants["molar gas constant"]      # Array of value, unit, error
# search_replace_within_file('build/Tabelle_test.tex','find me','found you')    # Selbsterklärend
