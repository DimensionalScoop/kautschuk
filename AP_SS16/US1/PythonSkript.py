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

### VORARBEITEN ####

h_zylinder, t_zylinder = np.genfromtxt('messdaten/a.txt', unpack=True)

h_zylinder = h_zylinder*10**(-3)
t_zylinder = t_zylinder*10**(-6)


##### a #####

v_zylinder = 2*h_zylinder/t_zylinder

write('build/Tabelle_0.tex', make_table([h_zylinder*10**3, t_zylinder*10**6, v_zylinder],[2, 1, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_0_texformat.tex', make_full_table(
     'Bestimmung der Schallgeschwindigkeit mittels Impuls-Echo-Verfahren.',
     'tab:0',
     'build/Tabelle_0.tex',
     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                               # die Multicolumns sein sollen
     [r'$h_{\text{zylinder}} \:/\: 10^{-3} \si{\metre}$',
     r'$\increment t \:/\: 10^{-6} \si{\second} $',
     r'$c_\text{Acryl} \:/\:\si{\metre\per\second} $']))

c_arcyl_1 = ufloat(np.mean(v_zylinder), np.std(v_zylinder))
write('build/c_acryl_1.tex', make_SI(c_arcyl_1, r'\metre\per\second', figures=2))      # type in Anz. signifikanter Stellen

params = ucurve_fit(reg_linear, 0.5*t_zylinder, h_zylinder)             # linearer Fit
a, b = params
write('build/parameter_a.tex', make_SI(a, r'\metre\per\second', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b.tex', make_SI(b, r'\metre', figures=2))      # type in Anz. signifikanter Stellen

v_lit   = 2730
v_rel_3 = abs(np.mean(a)-v_lit)/v_lit *100
write('build/v_rel_3.tex', make_SI(v_rel_3, r'\percent', figures=2))

t_plot = np.linspace(0.9*np.amin(0.5*t_zylinder), np.amax(0.5*t_zylinder)*1.1, 100)
plt.plot(t_plot, t_plot*a.n+b.n, 'b-', label='Linearer Fit')
plt.plot(0.5*t_zylinder, h_zylinder, 'rx', label='Messdaten')
# t_plot = np.linspace(-0.5, 2 * np.pi + 0.5, 1000) * 1e-3
#
## standard plotting
# plt.plot(t_plot * 1e3, f(t_plot, *noms(params)) * 1e-3, 'b-', label='Fit')
# plt.plot(t * 1e3, U * 1e3, 'rx', label='Messdaten')
## plt.errorbar(B * 1e3, noms(y) * 1e5, fmt='rx', yerr=stds(y) * 1e5, label='Messdaten')        # mit Fehlerbalken
## plt.xscale('log')                                                                            # logarithmische x-Achse
# plt.xlim(t_plot[0] * 1e3, t_plot[-1] * 1e3)
# plt.xlabel(r'$t \:/\: \si{\milli\selinder, 'rx', label='Messdaten')
plt.xlim(t_plot[0], t_plot[-1])
plt.xlabel(r'$\frac{1}{2} t \:/\: \si{\second}$')
plt.ylabel(r'$h \:/\: \si{\metre}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/ausgleich.pdf')



v_rel_1 = abs(np.mean(v_zylinder)-v_lit)/v_lit *100
write('build/v_rel_1.tex', make_SI(v_rel_1, r'\percent', figures=2))
write('build/v_lit.tex', make_SI(v_lit, r'\metre\per\second', figures=0))

##############Durchschallungs-Methode####################

h_zylinder, t_zylinder = np.genfromtxt('messdaten/b.txt', unpack=True)

h_zylinder = h_zylinder*10**(-3)
t_zylinder = t_zylinder*10**(-6)/2

v_zylinder = h_zylinder/t_zylinder


write('build/Tabelle_1.tex', make_table([h_zylinder*10**3, t_zylinder*10**6, v_zylinder],[2, 1, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_1_texformat.tex', make_full_table(
     'Bestimmung der Schallgeschwindigkeit mittels Durchschallungs-Methode.',
     'tab:1',
     'build/Tabelle_1.tex',
     [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                               # die Multicolumns sein sollen
     [r'$h_{\text{zylinder}} \:/\: 10^{-3} \si{\metre}$',
     r'$\increment t \:/\: 10^{-6} \si{\second} $',
     r'$c_\text{Acryl} \:/\: \si{\metre\per\second} $']))

c_arcyl_2 = ufloat(np.mean(v_zylinder), np.std(v_zylinder))
write('build/c_acryl_2.tex', make_SI(c_arcyl_2, r'\metre\per\second', figures=2))      # type in Anz. signifikanter Stellen

v_rel_2 = abs(np.mean(v_zylinder)-v_lit)/v_lit *100
write('build/v_rel_2.tex', make_SI(v_rel_2, r'\percent', figures=2))


################Abschwächungskoeffizient################
U_1 = 1.214
U_2 = 1.105
t_1 = 1.3 * 10**(-6)
t_2 = 46.2 * 10**(-6)

alpha = np.log(U_1/U_2)/(t_1-t_2)
write('build/alpha.tex', make_SI(alpha, r'\second\tothe{-1}', figures=1))

################Auge##################
t_auge = np.genfromtxt('messdaten/auge.txt', unpack=True)
t_auge = t_auge*10**(-6)
c_linse = 2500
c_gk = 1410

s_12 = (t_auge[1]-t_auge[0])*c_gk
s_23 = (t_auge[2]-t_auge[1])*c_linse
s_34 = (t_auge[3]-t_auge[2])*c_linse
s_45 = (t_auge[4]-t_auge[3])*c_linse
s_36 = (t_auge[5]-t_auge[2])*c_gk

write('build/c_linse.tex', make_SI(c_linse, r'\metre\per\second', figures=0))
write('build/c_gk.tex', make_SI(c_gk, r'\metre\per\second', figures=0))

write('build/s_12.tex', make_SI(s_12, r'\metre', figures=3))
write('build/s_23.tex', make_SI(s_23, r'\metre', figures=3))
write('build/s_34.tex', make_SI(s_34, r'\metre', figures=3))
write('build/s_45.tex', make_SI(s_45, r'\metre', figures=3))
write('build/s_36.tex', make_SI(s_36, r'\metre', figures=3))

### FFT - For Fucks... Time?####
fft = np.genfromtxt('messdaten/fft.txt', unpack=True)

write('build/fft_1.tex', make_SI(fft[0], r'\mega\hertz', figures=2))
write('build/fft_2.tex', make_SI(fft[1], r'\mega\hertz', figures=2))
write('build/fft_3.tex', make_SI(fft[2], r'\mega\hertz', figures=2))
write('build/fft_4.tex', make_SI(fft[3], r'\mega\hertz', figures=2))
write('build/fft_5.tex', make_SI(fft[4], r'\mega\hertz', figures=2))
write('build/fft_6.tex', make_SI(fft[5], r'\mega\hertz', figures=2))


fft = fft * 10**6
delta_f = np.array([ fft[1]-fft[0], fft[2]-fft[1], fft[3]-fft[2], fft[4]-fft[3], fft[5]-fft[4] ])
mean_delta_f = np.mean(delta_f)
std_delta_f = np.std(delta_f)
delta_f = ufloat(mean_delta_f, std_delta_f)
s_probe = 2730/delta_f
write('build/s_probe.tex', make_SI(s_probe, r'\metre', figures=2))



### Cepstrum ###
f_cep = 10/4.9 * 0.8 + 10 #Peak in µs
s_cep = f_cep * 2730 * 10**(-6)
write('build/s_cep.tex', make_SI(s_cep, r'\metre', figures=3))
