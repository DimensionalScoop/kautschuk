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

def p_saet(Temperatur):
    """
        Args:
            Temperatur: Temperatur [K]
        Returns:
            Sättigungsdampfdruck für Quecksilber [bar]
    """
    p = 5.5 * 10**(7) * np.exp(-6876/T) / 1000
    return p

def w_quer(p_saet):
    """
        Args:
            P_Saet: Sättigungsdampfdruck [bar]
        Returns:
            Mittlere freie Weglänge [m]
    """
    w_quer = 0.0029/(p_saet*1000)
    return w_quer/100

########## Aufgabenteil 0) ##########

T = np.genfromtxt('messdaten/0.txt', unpack=True)
T += 273.15
p_saet = p_saet(T)
w_quer = w_quer(p_saet)




write('build/Tabelle_0.tex', make_table([T,p_saet*1000,w_quer*1000],[2, 3, 3]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_0_texformat.tex', make_full_table(
     'Bestimmung der Sättigungsdampfdrücke sowie der mittleren Weglängen.',
     'tab:0',
     'build/Tabelle_0.tex',
     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                               # die Multicolumns sein sollen
     ['T  /  \si{\kelvin}',
     r'$p_{\text{sätt}} \:/\: 10^{-3} \si{\bar}$',
     r'$\bar{w} \:/\: 10^{-3} \si{\metre} $']))

########## Aufgabenteil a) ##########

U_a, I_a, I_a_plus_delta = np.genfromtxt('messdaten/a_1.txt', unpack=True) # Ströme in Nanoampere

plt.clf                   # clear actual plot before generating a new one
t_plot = np.linspace(np.amin(U_a), np.amax(U_a), 100)
plt.xlim(t_plot[0]-1/np.size(U_a)*(t_plot[-1]-t_plot[0]), t_plot[-1]+1/np.size(U_a)*(t_plot[-1]-t_plot[0]))

plt.plot(U_a, I_a_plus_delta, 'rx', label=r'Messwerte für $T = \SI{26.1}{\celsius}$')
plt.xlabel(r'$U_a \:/\: \si{\volt}$')
plt.ylabel(r'$\increment I_a \:/\: \si{\nano\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/aufgabenteil_a_plot.pdf')

write('build/Tabelle_a1.tex', make_table([U_a, I_a_plus_delta],[2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_a1_texformat.tex', make_full_table(
     r'Messwerte für die Integrale Energieverteilung bei $T = \SI{26.1}{\celsius}$.',
     'tab:1',
     'build/Tabelle_a1.tex',
     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                               # die Multicolumns sein sollen
     [r'$U_a  /  \si{\volt}$',
     r'$\increment I_a \:/\: 10^{-9} \si{\ampere}$']))




U_a_2, I_a_2 = np.genfromtxt('messdaten/a_2.txt', unpack=True) # Ströme in Nanoampere

plt.clf()                   # clear actual plot before generating a new one
t_plot = np.linspace(np.amin(U_a_2), np.amax(U_a_2), 100)
plt.xlim(t_plot[0]-1/np.size(U_a_2)*(t_plot[-1]-t_plot[0]), t_plot[-1]+1/np.size(U_a_2)*(t_plot[-1]-t_plot[0]))

plt.plot(U_a_2, I_a_2, 'rx', label=r'Messwerte für $T = \SI{145.5}{\celsius}$')
plt.xlabel(r'$U_a \:/\: \si{\volt}$')
plt.ylabel(r'$\increment I_a \:/\: \si{\nano\ampere}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/aufgabenteil_a_plot_2.pdf')

write('build/Tabelle_a2.tex', make_table([U_a_2, I_a_2],[2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_a2_texformat.tex', make_full_table(
     r'Messwerte für die Integrale Energieverteilung bei $T = \SI{145.5}{\celsius}$.',
     'tab:2',
     'build/Tabelle_a2.tex',
     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                               # die Multicolumns sein sollen
     [r'$U_a  /  \si{\volt}$',
     r'$\increment I_a \:/\: 10^{-9} \si{\ampere}$']))


########## Aufgabenteil b) ##########

U_max_1, U_max_2 = np.genfromtxt('messdaten/b.txt', unpack=True) # Lage der Maxima
T_1 = 161+273.15
T_2 = 178+273.15

write('build/Tabelle_b1.tex', make_table([U_max_2],[2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
#write('build/Tabelle_b1_texformat.tex', make_full_table(
#     r'Maxima der Franck-Hertz-Kurve bei $T = \SI{178}{\celsius}$.',
#     'tab:3',
#     'build/Tabelle_b1.tex',
#     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
#                               # die Multicolumns sein sollen
#     [r'$U_max  /  \si{\volt}$']))

U_max_2_deltas = ([U_max_2[1]-U_max_2[0], U_max_2[2]-U_max_2[1], U_max_2[3]-U_max_2[2]])
max1 = U_max_2[0]
max2 = U_max_2[1]
max3 = U_max_2[2]
max4 = U_max_2[3]

write('build/b_max1.tex', make_SI(max1, r'\volt', figures=1))
write('build/b_max2.tex', make_SI(max2, r'\volt', figures=1))
write('build/b_max3.tex', make_SI(max3, r'\volt', figures=1))
write('build/b_max4.tex', make_SI(max4, r'\volt', figures=1))


write('build/Tabelle_b2.tex', make_table([U_max_2_deltas],[2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
#write('build/Tabelle_b2_texformat.tex', make_full_table(
#     r'Abstände der Maxima der Franck-Hertz-Kurve bei $T = \SI{178}{\celsius}$.',
#     'tab:4',
#     'build/Tabelle_b2.tex',
#     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
#                               # die Multicolumns sein sollen
#     [r'$\Delta U_max  /  \si{\volt}$']))

U_max_2_delta_mean         = np.mean(U_max_2_deltas)
U_max_2_delta_err          = MeanError(noms(U_max_2_deltas))     #Fehler des Mittelwertes  KOMISCHERWEISE 0?
U_max_2_delta_err          = np.std(U_max_2_deltas)
U_max_2_delta_delta_ablese = 0.2332                    #Ablesefehler

#write('build/b_U_max_2_delta_mean.tex', make_SI(U_max_2_delta_mean, r'\volt', figures=5 ))
#write('build/b_U_max_2_delta_err.tex', make_SI(U_max_2_delta_err, r'\volt', figures=5 ))

U_max_delta = ufloat(U_max_2_delta_mean, U_max_2_delta_err)
write('build/b_U_max_delta.tex', make_SI(U_max_delta, r'\volt', figures=1 ))

h      = 6.62607004*10**(-34)
e      = 1.6021766208*10**(-19)
nu     = e*U_max_delta/h
#nu     = e*U_max_2_delta_mean/h # hab ich nur zum Testen gemacht
#nu     = e*5.25/h
c      = 299792458
laenge = c/nu
laenge_lit = 435.835 * 10**(-9)
rel_err_laenge = abs(noms(U_max_delta) - 4.9)/4.9 *100

write('build/b_wellenlaenge.tex', make_SI(laenge*10**(9), r'\nano\metre', figures=1 ))  # Wert passt ungefähr (bei uns 241nm, eigentlich so 435nm), mit der Abweichung sollte das drin liegen, ohne den ersten Wert treffen wir den fast perfekt 436nm
write('build/b_anregung_rel.tex', make_SI(rel_err_laenge, r'\percent', figures=1 ))

########## Aufgabenteil c) ##########

U_peak = 14.1
K      = 3.1
E_ion  = (U_peak - K)
E_ion_lit = 10.438
rel_E_ion_err = abs(E_ion - E_ion_lit)/E_ion_lit * 100
write('build/c_ion_rel_err.tex', make_SI(rel_E_ion_err, r'\percent', figures=1 ))
write('build/c_ion.tex', make_SI(E_ion, r'\electronvolt', figures=0 ))
