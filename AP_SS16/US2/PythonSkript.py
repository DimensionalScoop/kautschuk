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


###########Daten des Blocks################

tiefe  = 4.03*10**(-2)
breite = 15.02*10**(-2)
hoehe  = 8.035*10**(-2)

##########A-Scan##################

c = 2730
t_start = 1.6*10**(-6)
t_end = 59.4*10**(-6)

write('build/t_start.tex', make_SI(t_start*10**6, r'\micro\second', figures=2))
write('build/t_end.tex', make_SI(t_end*10**6, r'\micro\second', figures=2))

hoehe_mess = c*(t_end-t_start)/2

write('build/hoehe_mess.tex', make_SI(hoehe_mess*10**2, r'\centi\metre', figures=2))

hoehe_mess_rel = abs(hoehe_mess-hoehe)/hoehe * 100
write('build/hoehe_mess_rel.tex', make_SI(hoehe_mess_rel, r'\percent', figures=1))

D_o , t_o, t_u = np.genfromtxt('messdaten/a.txt', unpack=True)
D_o = D_o*10**(-2)
t_o = t_o*10**(-6)
t_u = t_u*10**(-6)

D_mess_o = D_o
D_mess_u = D_o


D_mess_o = c*t_o/2
D_mess_u = c*t_u/2
D_loch = hoehe_mess - (D_mess_o + D_mess_u)


D_mess_o = D_mess_o*10**(2)
D_mess_u = D_mess_u*10**(2)
D_o = D_o*10**2
D_loch = D_loch*10**3
List = [1,2,3,4,5,6,7,8,9,10,11]
write('build/Tabelle_a.tex', make_table([List, D_o, D_mess_o, D_mess_u, D_loch],[0,2,2,2,2])) #cm, cm ,cm, mm
write('build/Tabelle_a_texformat.tex', make_full_table(
    'Messdaten Tiefenmessungen.',
    'table:1',
    'build/Tabelle_a.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$\text{Stelle}$',
    r'$D_{\text{oben}} \:/\: \si{\centi\metre}$',
    r'$D_{\text{oben,gem}} \:/\: \si{\centi\metre}$',
    r'$D_{\text{unten,gem}} \:/\: \si{\centi\metre}$',
    r'$D_{\text{loch,gem}} \:/\: \si{\milli\metre}$']))

D_o_rel_a = abs(D_mess_o-D_o)/D_o * 100
D_o_rel_a = np.mean(D_o_rel_a)
write('build/D_o_rel_a.tex', make_SI(D_o_rel_a, r'\percent', figures=2)) #um herauszufinden, welche Methode besser ist (Spoiler: Durchschnittlich 4% Abweichung)

#####################B-Scan##################

t_start2 = 5.5*0.5*10**(-6)
t_end2 = (55+0.5*7)*10**(-6)
write('build/t_start2.tex', make_SI(t_start2*10**6, r'\micro\second', figures=2))
write('build/t_end2.tex', make_SI(t_end2*10**6, r'\micro\second', figures=2))

hoehe_mess2 = c*(t_end2-t_start2)/2

write('build/hoehe_mess2.tex', make_SI(hoehe_mess2*10**2, r'\centi\metre', figures=2))

hoehe_mess_rel2 = abs(hoehe_mess2-hoehe)/hoehe * 100
write('build/hoehe_mess_rel2.tex', make_SI(hoehe_mess_rel2, r'\percent', figures=1))

t_o, t_u = np.genfromtxt('messdaten/b.txt', unpack=True)

D_o = D_o*10**(-2)
t_o = t_o*10**(-6)
t_u = t_u*10**(-6)

D_mess_o = D_o
D_mess_u = D_o


D_mess_o = c*t_o/2
D_mess_u = c*t_u/2
D_loch = hoehe_mess2 - (D_mess_o + D_mess_u)



D_mess_o = D_mess_o*10**(2)
D_mess_u = D_mess_u*10**(2)
D_o = D_o*10**2
D_loch = D_loch*10**3
List = [1,2,3,4,5,6,7,8,9,10,11]
write('build/Tabelle_b.tex', make_table([List, D_o, D_mess_o, D_mess_u, D_loch],[0,2,2,2,2])) #cm, cm ,cm, mm
write('build/Tabelle_b_texformat.tex', make_full_table(
    'Messdaten Tiefenmessungen.',
    'table:2',
    'build/Tabelle_b.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$\text{Stelle}$',
    r'$D_{\text{oben}} \:/\: \si{\centi\metre}$',
    r'$D_{\text{oben,gem}} \:/\: \si{\centi\metre}$',
    r'$D_{\text{unten,gem}} \:/\: \si{\centi\metre}$',
    r'$D_{\text{loch,gem}} \:/\: \si{\milli\metre}$']))

D_o_rel_b = abs(D_mess_o-D_o)/D_o * 100
D_o_rel_b = np.mean(D_o_rel_b)
write('build/D_o_rel_b.tex', make_SI(D_o_rel_b, r'\percent', figures=2))  #um herauszufinden, welche Methode besser ist (Spoiler: Durchschnittlich 15% Abweichung)


###########Kack-Herz###############

ESD = np.genfromtxt('messdaten/ESD.txt', unpack=True)
th  = np.genfromtxt('messdaten/th.txt', unpack=True)

write('build/Tabelle_th.tex', make_table([th],[0]))
#write('build/Tabelle_th_texformat.tex', make_full_table(
#    'Zeitliche Abstände der Schläge.',
#    'table:4',
#    'build/Tabelle_th.tex',
#    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
#                              # die Multicolumns sein sollen
#    [
#    r'$t_{\text{Herz}} \:/\: \si{\second}$']))
#
write('build/Tabelle_ESD.tex', make_table([ESD],[0]))
#write('build/Tabelle_ESD_texformat.tex', make_full_table(
#    'Größe der Amplituden.',
#    'table:3',
#    'build/Tabelle_ESD.tex',
#    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
#                              # die Multicolumns sein sollen
#    [
#    r'$\text{Amplituden} \:/\: \si{\micro\second}$']))

c_wasser = 1484
write('build/c_wasser.tex', make_SI(c_wasser, r'\metre\per\second', figures=0))

ESD = ESD*10**(-6)
ESD = ESD*c_wasser/2
ESD = ufloat(np.mean(ESD), np.std(ESD))
hf  = 1/th
hf  = ufloat(np.mean(hf), np.std(hf))

write('build/ESD.tex', make_SI(ESD*10**2, r'\centi\metre', figures=1))
write('build/hf.tex', make_SI(hf*100, r'\second\tothe{-1}', figures=2))

#durchmesser = 4.94*10**(-2)
#write('build/h_durchmesser.tex', make_SI(durchmesser*100, r'\centi\metre', figures=2))

ESV = 4/3*np.pi*(ESD/2)**3
write('build/ESV.tex', make_SI(ESV*10**6, r'\milli\litre', figures=2))

HZV = ESV*hf
write('build/HZV.tex', make_SI(HZV*10**6, r'\milli\litre\per\second', figures=2))

###############HerzAnHerz A Scan ####
s_1 = 1.6*10**(-6)
s_2 = 34.8*10**(-6)
h_bla = (s_2-s_1)*c_wasser*0.5
write('build/h_bla.tex', make_SI(h_bla*10**2, r'\centi\metre', figures=2))
write('build/s_1.tex', make_SI(s_1*10**6, r'\micro\second', figures=2))
write('build/s_2.tex', make_SI(s_2*10**6, r'\micro\second', figures=2))
