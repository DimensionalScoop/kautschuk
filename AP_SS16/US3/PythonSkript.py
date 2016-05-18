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
)
from regression import (
    reg_linear,
    reg_quadratic,
    reg_cubic
)
from error_calculation import(
    MeanError
)
################################################ Finish importing custom libraries #################################################

### Gegebene Daten
rho_L = 1.15            # in g/cm^3
c_L = 1800              # m/s
eta = 12                # mPa s
c_P = 2700              # m/2
l = 30.7                # mm
di_k = 7e-3             # m
di_m = 10e-3            # m
di_g = 16e-3            # m
da_k = 10e-3            # m
da_m = 15e-3            # m
da_g = 20e-3            # m

nu_0 = 2e6              # Hz

import scipy.stats
# def mean(values):
#     return ufloat(np.mean(noms(values)), scipy.stats.sem(noms(values)))
def mean(values, axis=0):
    return unp.uarray((np.mean(noms(values), axis=axis), scipy.stats.sem(noms(values), axis=axis)))

Theta = np.array([30, 15, 60])*np.pi/180                # rad
alpha = np.pi/2 - np.arcsin(np.sin(Theta) * c_L/c_P)    # rad

# wiederverwendbare Funktion zum Erledigen der Aufgaben für verschiedene Innenwinkel
def do_job_a(filename, error, j, filename_out = None):
    # Einlesen der Messdaten
    P, Delta_f_30, Delta_f_15, Delta_f_60 = np.genfromtxt(filename, unpack=True)

    #
    di = [7,10,16]
    colors = ['rx', 'bx', 'gx']

    Delta_f_30_error = Delta_f_30*error
    Delta_f_30 = unp.uarray(Delta_f_30, Delta_f_30_error)
    Delta_f_15_error = Delta_f_15*error
    Delta_f_15 = unp.uarray(Delta_f_15, Delta_f_15_error)
    Delta_f_60_error = Delta_f_60*error
    Delta_f_60 = unp.uarray(Delta_f_60, Delta_f_60_error)

    v= unp.uarray(np.zeros(3), np.zeros(3))
    v[0] = c_L / 2 / nu_0 * Delta_f_30 / np.cos(alpha[0])
    v[1] = c_L / 2 / nu_0 * Delta_f_15 / np.cos(alpha[1])
    v[2] = c_L / 2 / nu_0 * Delta_f_60 / np.cos(alpha[2])

    v_mean = mean([v[0], v[1], v[2]], 0)

    # TABLES
    write('build/Tabelle_a_'+str(di[j])+'.tex',
        make_table(
        [P,Delta_f_30,Delta_f_15,Delta_f_60,v[0],v[1],v[2], v_mean],
        [0, 1, 1, 1, 1, 1, 1, 1]))
    write('build/Tabelle_a_'+str(di[j])+'_texformat.tex', make_full_table(
        r'Messdaten und daraus errechnete Geschwindikgiet für $\d_i = $'+str(di[j])+r'$\si{\milli\meter}$.',
        'table:A'+str(j),
        'build/Tabelle_a_'+str(di[j])+'.tex',
        [1,2,3,4,5,6,7],
        [r'$\frac{P}{P_\text{max}} \:/\: \si{\percent}$',
        r'$\Delta f_{30°} \:/\: \si{\hertz}$',
        r'$\Delta f_{15°} \:/\: \si{\hertz}$',
        r'$\Delta f_{60°} \:/\: \si{\hertz}$',
        r'$v_{30°} \:/\: \si{\meter\per\second}$',
        r'$v_{15°} \:/\: \si{\meter\per\second}$',
        r'$v_{60°} \:/\: \si{\meter\per\second}$',
        r'$\overline{v} \:/\: \si{\meter\per\second}$']))

    # Plotting
    plt.figure(1)
    y = Delta_f_30 / np.cos(alpha[0])
    plt.errorbar(noms(v[0]), noms(y), fmt=colors[j], xerr = stds(v[0]), yerr=stds(y), label=r'$d_i = ' + str(di[j]) + r'\si{\milli\meter}$')

    plt.figure(2)
    y = Delta_f_15 / np.cos(alpha[1])
    plt.errorbar(noms(v[1]), noms(y), fmt=colors[j], xerr = stds(v[1]), yerr=stds(y), label=r'$d_i = ' + str(di[j]) + r'\si{\milli\meter}$')

    plt.figure(3)
    y = Delta_f_60 / np.cos(alpha[2])
    plt.errorbar(noms(v[2]), noms(y), fmt=colors[j], xerr = stds(v[2]), yerr=stds(y), label=r'$d_i = ' + str(di[j]) + r'\si{\milli\meter}$')

    i = 1
    if (filename_out):
        for name in filename_out:
            plt.figure(i)
            plt.xlabel(r'$v \:/\: \si{\meter\per\second}$')
            plt.ylabel(r'$\Delta\nu / \cos{\alpha} \:/\: \si{\kilo\volt}$')
            plt.legend(loc='best')
            plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
            plt.savefig(name)
            i += 1


def do_job_b(filename, error, P, limits):
    # Einlesen der Messdaten
    Tiefe, Delta_f, Intensity = np.genfromtxt(filename, unpack=True)

    colors = ['rx', 'bx', 'gx']

    Delta_f_error = Delta_f*error
    Delta_f = unp.uarray(Delta_f, Delta_f_error)

    v = c_L / 2 / nu_0 * Delta_f / np.cos(alpha[1])      # 15 ° Winkel

    ###### Fit im Intervall limits mit quadratischer Funktion gemäß dem Gesetz von Hagen-Poiseuille
    i = 0
    start = 0
    end = 0
    for x in Tiefe:
        if (x == limits[0]):
            start = i
        if (x == limits[1]):
            end = i
        i += 1
    params = ucurve_fit(reg_quadratic, Tiefe[start:(end+1)], v[start:(end+1)])          # quadratischer Fit
    a, x0, c = params
    write('build/parameter_a.tex', make_SI(a * 1e-3, r'\kilo\volt', figures=1))
    ##### Ende Fit ########

    # Plotting
    plt.clf
    fig, ax1 = plt.subplots()
    t_plot = np.linspace(limits[0]-0.5, limits[1]+0.5, 50)

    # Momentangeschwindigkeiten
    Ins1 = ax1.plot(Tiefe, noms(v), 'rx', label='Momentangeschwindigkeit')
    Ins2 = ax1.plot(t_plot, reg_quadratic(t_plot, *noms(params)), 'r--', label='Fit')
    ax1.set_xlabel(r'$\text{Laufzeit} \:/\: \si{\micro\second}$')
    ax1.set_ylabel(r'$v \:/\: \si{\meter\per\second}$')
    if ( P==45 ):
        ax1.set_ylim(0.45, 0.9)       # hard coded stuff ftl !

    # Streuintensitäten
    ax2 = ax1.twinx()
    Ins3 = ax2.plot(Tiefe, Intensity, 'b+', label='Intensität')
    ax2.set_ylabel(r'$I \:/\: \si{\kilo\volt\squared\per\second}$')

    # Theoretische Grenzen des Rohres einzeichnen
    ax1.plot((noms(x0)-5/1.5, noms(x0)-5/1.5), (ax1.get_ylim()[0], ax1.get_ylim()[1]), 'k:', linewidth=1)
    ax1.plot((noms(x0)+5/1.5, noms(x0)+5/1.5), (ax1.get_ylim()[0], ax1.get_ylim()[1]), 'k:', linewidth=1)
    ax1.plot((noms(x0)-5/1.5-2.5/2.5, noms(x0)-5/1.5-2.5/2.5), (ax1.get_ylim()[0], ax1.get_ylim()[1]), 'k:', linewidth=1)
    ax1.plot((noms(x0)+5/1.5+2.5/2.5, noms(x0)+5/1.5+2.5/2.5), (ax1.get_ylim()[0], ax1.get_ylim()[1]), 'k:', linewidth=1)

    Ins = Ins1 + Ins2 + Ins3
    labs = [l.get_label() for l in Ins]
    ax1.legend(Ins, labs, loc='upper left')
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    plt.savefig('build/Plot_b_P'+str(P)+'.pdf')

do_job_a('messdaten/Delta_f_7mm.txt', 0.07, 0)
do_job_a('messdaten/Delta_f_10mm.txt', 0.07, 1)
do_job_a('messdaten/Delta_f_16mm.txt', 0.07, 2, ['build/Plot_a_30deg.pdf', 'build/Plot_a_15deg.pdf', 'build/Plot_a_60deg.pdf'])
do_job_b('messdaten/stroemungsprofil45.txt', 0.07, 45, [14, 17])
do_job_b('messdaten/stroemungsprofil70.txt', 0.07, 70, [14, 17])

# Tabelle für Messdaten
Tiefe, Delta_f_45, Intensity_45 = np.genfromtxt('messdaten/stroemungsprofil45.txt', unpack=True)
Tiefe, Delta_f_70, Intensity_70 = np.genfromtxt('messdaten/stroemungsprofil70.txt', unpack=True)
error = 0.07
Delta_f_45_error = Delta_f_45*error
Delta_f_45 = unp.uarray(Delta_f_45, Delta_f_45_error)
Delta_f_70_error = Delta_f_70*error
Delta_f_70 = unp.uarray(Delta_f_70, Delta_f_70_error)
Intensity_45_error = Intensity_45*error
Intensity_45 = unp.uarray(Intensity_45, Intensity_45_error)
Intensity_70_error = Intensity_70*error
Intensity_70 = unp.uarray(Intensity_70, Intensity_70_error)

write('build/Tabelle_messdaten.tex', make_table([Tiefe, Delta_f_45, Intensity_45, Delta_f_70, Intensity_70],[0, 1, 1, 1, 1]))
write('build/Tabelle_messdaten_texformat.tex', make_full_table(
    'Messdaten zum Strömungsprofil.',
    'table:messdaten_b',
    'build/Tabelle_messdaten.tex',
    [1,2,3,4],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [r'$\text{Laufzeit} \:/\: \si{\micro\second}$',
    r'$\Delta f_{45\si{\percent}} \:/\: \si{\hertz}$',
    r'$I_{45\si{\percent}} \:/\: \si{\kilo\square\volt\per\second}$',
    r'$\Delta f_{70\si{\percent}} \:/\: \si{\hertz}$',
    r'$I_{70\si{\percent}} \:/\: \si{\kilo\square\volt\per\second}$']))



################################ FREQUENTLY USED CODE ################################
#
########## IMPORT ##########
# t, U, U_err = np.genfromtxt('data.txt', unpack=True)
# t *= 1e-3


########## ERRORS ##########
# R_unc = ufloat(R[0],R[2])
# U = 1e3 * unp.uarray(U, U_err)
# Rx_mean = np.mean(Rx)                 # Mittelwert und syst. Fehler
# Rx_mean_err = MeanError(noms(Rx))     # Fehler des Mittelwertes
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
#     'Messdaten Kapazitätsmessbrücke.',
#     'table:A2',
#     'build/Tabelle_b.tex',
#     [1,2,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
#                               # die Multicolumns sein sollen
#     ['Wert',
#     r'$C_2 \:/\: \si{\nano\farad}$',
#     r'$R_2 \:/\: \si{\ohm}$',
#     r'$R_3 / R_4$', '$R_x \:/\: \si{\ohm}$',
#     r'$C_x \:/\: \si{\nano\farad}$']))
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
