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

#### DATENEINGABE UND AUSLESE ####

lamb = 633*1e-9         # angegebene Wellenlänge in m
L = 0.93                # gemessener Abstand in m
Dunkelstrom = 7.1e-9    # Dunkelstrom in A (gemessen)
pi = np.pi

x1 = np.linspace(-12.5, 12.5, num=51)*1e-3  # in m
x2 = np.linspace(-10, 10, num=81)*1e-3      # in m
phi1 = x1/L                                 # in rad
phi2 = x2/L                                 # in rad
# For plotting smoother fits:
phi1_plot = np.linspace(phi1[0], phi1[-1], 200)     # in rad
phi2_plot = np.linspace(phi2[0], phi2[-1], 300)     # in rad

b_k_hst, b_m_hst, b_g_hst, b_ds_hst, s_ds_hst = (np.genfromtxt('messdaten/hersteller.txt', unpack=True))*1e-3   # in m
b_k_mic, b_m_mic, b_g_mic, b_ds_mic, s_ds_mic = (np.genfromtxt('messdaten/mikroskop.txt', unpack=True))   # in cm
I_k, Scale_k = np.genfromtxt('messdaten/EP_klein.txt', unpack=True)
I_m, Scale_m = np.genfromtxt('messdaten/EP_mittel.txt', unpack=True)
I_g, Scale_g = np.genfromtxt('messdaten/EP_gross.txt', unpack=True)
I_ds, Scale_ds = np.genfromtxt('messdaten/DS.txt', unpack=True)


# Offset zeta_0 für die verschiedenen Messungen
# zeta_0 durch Hingucken abschätzen (sorry - hier manuell nachdenken und in die Messdaten/Plots gucken!)
zeta_0 = np.array([-0.3e-3, 0.25*1e-3, 0, (x2[35]+x2[36])/2])

I_k *= Scale_k      # in Ampere
I_m *= Scale_m      # in Ampere
I_g *= Scale_g      # in Ampere
I_ds *= Scale_ds    # in Ampere

# Korrektur wegen des Dunkelstroms
I_g = I_g - Dunkelstrom
I_m = I_m - Dunkelstrom
I_k = I_k - Dunkelstrom
I_ds = I_ds - Dunkelstrom

#### FUNCTIONS ####

def theory_einfach(phi, A_0, b):
  I = np.array([])
  for i in range(len(phi)):
    if phi[i] == 0:
      I = np.append(I, A_0**2 * b**2)
    else:
      I = np.append(I, (A_0 * lamb / pi / np.sin(phi[i]) * np.sin(pi * b * np.sin(phi[i]) / lamb))**2 )
  return I


def theory_doppel(phi, A_0, b, s):
  I = np.array([])
  for i in range(len(phi)):
    if phi[i] == 0:
      I = np.append(I, A_0**2 * b**2)
    else:
      I = np.append(I, (A_0 * np.cos(pi * s * np.sin(phi[i]) / lamb) * lamb / pi / np.sin(phi[i]) * np.sin(pi * b * np.sin(phi[i]) / lamb))**2)
  return I

############# MICROSCOPE MEASUREMENTS ##################
scale_mic = 0.5/1.3                     # mm pro cm bei Vergrößerung 3,2
scale_mic *= 3.2/4                      # jetzt für Vergrößerung 4
b_microscope = np.array([b_k_mic, b_m_mic, b_g_mic, b_ds_mic])
b_microscope *= scale_mic * 1e-3        # in m
s_ds_mic *= scale_mic * 1e-3            # in m


############# STARTING FITS ##################
plot_it = True
#### KLEINER EINZELSPALT ####
zeta_0_k = zeta_0[0]                        # in m
phi_k = phi1 - zeta_0_k/L                   # in rad
phi_k_plot = phi1_plot - zeta_0_k/L         # in rad

params_k = ucurve_fit(theory_einfach, phi_k, I_k, p0=[np.sqrt(max(I_k)) / b_k_hst, b_k_hst])
A0_k, b_k = params_k

if (plot_it):
    plt.clf()
    plt.plot(phi1_plot*1e3, theory_einfach(phi_k_plot, *noms(params_k))*1e6, 'b-', label='Fit')
    plt.plot(phi1*1e3, I_k*1e6, 'rx', label='Messdaten')
    plt.xlabel(r'$\varphi \:/\: \si{\milli\radian}$')
    plt.ylabel(r'$I \:/\: \si{\micro\ampere}$')
    plt.legend(loc='best')
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    plt.savefig('build/plot_klein.pdf')


#### MITTLERER EINZELSPALT ####
zeta_0_m = zeta_0[1]            # in m
phi_m = phi1 - zeta_0_m/L
phi_m_plot = phi1_plot - zeta_0_m/L

params_m = ucurve_fit(theory_einfach, phi_m, I_m, p0=[np.sqrt(max(I_m)) / b_m_hst, b_m_hst])
A0_m, b_m = params_m

if (plot_it):
    plt.clf()
    plt.plot(phi1_plot*1e3, theory_einfach(phi_m_plot, *noms(params_m))*1e6, 'b-', label='Fit')
    plt.plot(phi1*1e3, I_m*1e6, 'rx', label='Messdaten')
    plt.xlabel(r'$\varphi \:/\: \si{\milli\radian}$')
    plt.ylabel(r'$I \:/\: \si{\micro\ampere}$')
    plt.legend(loc='best')
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    plt.savefig('build/plot_mittel.pdf')


#### GROßER EINZELSPALT ####
zeta_0_g = zeta_0[2]                            # in m
phi_g = phi1 - zeta_0_g / L        # in rad
phi_g_plot = phi1_plot - zeta_0_g/L     # in rad

params_g = ucurve_fit(theory_einfach, phi_g, I_g, p0=[np.sqrt(max(I_g)) / b_g_hst, b_g_hst])
A0_g, b_g = params_g

if (plot_it):
    plt.clf()
    # Vergleichsplot mit den angegebenen Daten :
    # plt.plot(phi1_plot*1e3, theory_einfach(phi_g_plot, np.sqrt(max(I_g)) / b_g_hst , b_g_hst) * 1e6, 'b-', label='Fit')
    plt.plot(phi1_plot*1e3, theory_einfach(phi_g_plot, *noms(params_g))*1e6, 'b-', label='Fit')
    plt.plot(phi1*1e3, I_g*1e6, 'rx', label='Messdaten')
    plt.xlabel(r'$\varphi \:/\: \si{\milli\radian}$')
    plt.ylabel(r'$I \:/\: \si{\micro\ampere}$')
    plt.ylim(-2,16)
    plt.legend(loc='best')
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    plt.savefig('build/plot_gross.pdf')


#### DOPPELSPALT ####
zeta_0_ds = zeta_0[3]                            # in m
phi_ds = phi2 - zeta_0_ds / L        # in rad
phi_ds_plot = phi2_plot - zeta_0_ds/L     # in rad

params_ds = ucurve_fit(theory_doppel, phi_ds, I_ds, p0=[np.sqrt(max(I_ds)) / b_ds_hst, b_ds_hst, s_ds_hst], method='dogbox')
A0_ds, b_ds, s_ds = params_ds


if (plot_it):
    plt.clf()
    # Vergleichsplot mit den angegebenen Daten :
    # plt.plot(phi2_plot*1e3, theory_doppel(phi_ds_plot, A0_ds_fit , b_ds_fit, s_ds_fit) * 1e6, 'b-', label='Fit Doppelspalt')
    plt.plot(phi_ds_plot*1e3, theory_doppel(phi_ds_plot, *noms(params_ds))*1e6, 'b-', label='Fit Doppelspalt')
    plt.plot(phi_ds_plot*1e3, theory_einfach(phi_ds_plot, noms(A0_ds), noms(b_ds))*1e6, 'g-', label='Fit Einzelspalt')
    plt.plot((phi2 - zeta_0_ds/L)*1e3, I_ds*1e6, 'rx', label='Messdaten')
    plt.xlabel(r'$\varphi \:/\: \si{\milli\radian}$')
    plt.ylabel(r'$I \:/\: \si{\micro\ampere}$')
    plt.legend(loc='best')
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    plt.savefig('build/plot_ds.pdf')


####################### WRITING RESULT TABLES #######################
slits = ['klein', 'mittel', 'groß', 'doppel']
b = np.array([b_k, b_m, b_g, b_ds])                             # in m
b_hst = np.array([b_k_hst, b_m_hst, b_g_hst, b_ds_hst])         # in m
b_err = np.abs(((b-b_hst) / b_hst))                             # relative error
A0 = np.array([A0_k, A0_m, A0_g, A0_ds])                        # in A/m
s_err = np.abs(np.array([(s_ds - s_ds_hst) / s_ds_hst]))        # relative error
s_mic = np.array([s_ds_mic])
s = np.array([s_ds])
s_hst = np.array([s_ds_hst])


write('build/Tabelle_results.tex', make_table([slits, zeta_0*1e3, A0, b_microscope*1e3, b*1e3, b_hst*1e3, b_err*1e2], [0, 2, 1, 2, 1, 2, 1]))
write('build/Tabelle_results_texformat.tex', make_full_table(
    caption = r'Herstellerangaben, Mikroskopmessungen, Firparameter und der Fehler zwischen Fit und Herstellerangabe für die Spaltbreite $b$.',
    label = 'table:A1',
    source_table = 'build/Tabelle_results.tex',
    stacking = [2,4,6],
    units = [
    'Spalt',
    r'$\zeta_0 \:/\: \si{\milli\meter}$',
    r'$A_0 \:/\: \si{\ampere\per\meter}$',
    r'$b_\text{mic} \:/\: \si{\milli\meter}$',
    r'$b_\text{mess} \:/\: \si{\milli\meter}$',
    r'$b_\text{hst} \:/\: \si{\milli\meter}$',
    r'$|\varepsilon_b| \:/\: \si{\percent}$']))

write('build/Tabelle_results_s.tex', make_table([s_mic*1e3, s*1e3, s_hst*1e3, s_err*1e2], [2, 1, 2, 1]))
write('build/Tabelle_results_s_texformat.tex', make_full_table(
    caption = r'Herstellerangabe, Mikroskopmessung, Firparameter und der Fehler zwischen Fit und Herstellerangabe für den Abstand $s$ des Doppelspalts.',
    label = 'table:A2',
    source_table = 'build/Tabelle_results_s.tex',
    stacking = [1,3],
    units = [
    r'$s_\text{mic} \:/\: \si{\milli\meter}$',
    r'$s_\text{mess} \:/\: \si{\milli\meter}$',
    r'$s_\text{hst} \:/\: \si{\milli\meter}$',
    r'$|\varepsilon_s| \:/\: \si{\percent}$']))

all_errors = np.append(b_err, s_err)
Beschriftung = [
'Spaltbreite des kleinen Spalts',
'Spaltbreite des mittelgroßen Spalts',
'Spaltbreite des großen Spalts',
'Spaltbreite des Doppelspalts',
'Spaltabstand des Doppelspalts']

write('build/Tabelle_errors.tex', make_table([Beschriftung, all_errors*1e2], [0, 1]))
write('build/Tabelle_errors_texformat.tex', make_full_table(
    caption = 'Relative Fehler des Rückschlusses auf die Spaltabmessungen.',
    label = 'table:A3',
    source_table = 'build/Tabelle_errors.tex',
    stacking = [1],
    units = [
    'Messgröße',
    r'$|\varepsilon| \:/\: \si{\percent}$']))

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


## convenience file writing for standard make Files
f = open('build/.pysuccess', 'w')
f.write('MarktkraM')
f.close()
