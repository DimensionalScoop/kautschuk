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

#Nulleffekt nach einer Wartezeit von 900s
Nu=460
t_0=900 #in s
N_Offset_Indium = (Nu/t_0)*220
write('build/Fehler_Indium.tex', make_SI(N_Offset_Indium, r'', figures=1))
N_Offset_Silber = (Nu/t_0)*9
#Import Data
#Indium = Ind
#Silber = Si

Ind_nom, t = np.genfromtxt('messdaten/Indium.txt', unpack=True)
Ind_nom = Ind_nom -  N_Offset_Indium
Ind = unp.uarray(Ind_nom, np.sqrt(Ind_nom))
# Ind = unp.uarray(Ind_nom, N_Offset_Indium)


write('build/Tabelle_Indium.tex', make_table([Ind,t],[1, 0]))
write('build/Tabelle_Indium_texformat.tex', make_full_table(
    caption = 'Messdaten von Indium unter Berücksichtigung des Nulleffekts.',
    label = 'table:Indium',
    source_table = 'build/Tabelle_Indium.tex',
    stacking = [0],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
    units = [
    r'$N_\textrm{\Delta t}$',
    r'$t \:/\: \si{\second}$']))         # default = '-'





Si_nom, t_Si = np.genfromtxt('messdaten/Silber.txt', unpack=True)
Si_nom = Si_nom - N_Offset_Silber
Si = unp.uarray(Si_nom, np.sqrt(Si_nom))
# Si = unp.uarray(Si_nom, N_Offset_Silber)

write('build/Tabelle_Silber.tex', make_table([Si[:23],t_Si[:23],Si[23:],t_Si[23:]],[1,0,1,0]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_Silber_texformat.tex', make_full_table(
    caption = 'Messdaten von Silber unter Berücksichtigung des Nulleffekts.',
    label = 'table:Silber',
    source_table = 'build/Tabelle_Silber.tex',
    stacking = [0,2],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
    units = [
    r'$N_\textrm{\Delta t}$',
    r'$t \:/\: \si{\second}$',
    r'$N_\textrm{\Delta t}$',
    r'$t \:/\: \si{\second}$']))         # default = '-'



params = ucurve_fit(reg_linear, t, np.log(noms(Ind)))                                       # linearer Fit
m,b = params
write('build/Indium_m.tex', make_SI(m*1e4, r'\per\second', '1e-4' ,figures=1))
write('build/Indium_b.tex', make_SI(b, r'', figures=1))
t_plot = np.linspace(0, 4000, 2)
plt.plot(t_plot, np.exp(reg_linear(t_plot, *noms(params))), 'b-', label='Fit')
plt.errorbar(t, noms(Ind), fmt='rx', yerr=stds(Ind), label='Messdaten')                      # mit Fehlerbalken
plt.yscale('log')                                                                            # logarithmische x-Achse
plt.xlabel(r'$t \:/\: \si{\second}$')
plt.ylabel(r'$ \text{ln}(N_\textrm{\Delta t}) $')
plt.legend(loc='best')
plt.savefig('build/Indium_plot.pdf')
plt.clf()

lambda_Ind = -m

T = np.log(2)/lambda_Ind # in s
write('build/Halbwertszeit_Indium_s.tex', make_SI(T, r'\second', figures=1))
T = T/60 # in min
write('build/Halbwertszeit_Indium_h.tex', make_SI(T, r'\minute', figures=1))

write('build/Startwert_Indium.tex', make_SI(np.e**b, r'', figures=1))

##############Silber#############################

plt.errorbar(t_Si, noms(Si), fmt='rx', yerr=stds(Si), label='Messdaten')        # mit Fehlerbalken
plt.yscale('log')                                                                            # logarithmische x-Achse
plt.xlabel(r'$t \:/\: \si{\second}$')
plt.ylabel(r'$ \text{ln}(N_\textrm{\Delta t}) $')
plt.legend(loc='best')
plt.savefig('build/Silber_plot.pdf')
plt.clf()



#### langsamer Zerfall####

params = ucurve_fit(reg_linear, t_Si[19:], np.log(noms(Si[19:])))                                       # linearer Fit
m,b = params

print('Überprüfe m')
print(m)
write('build/Silber_108_m.tex', make_SI(m*1e4, r'\per\second', 'e-4' ,figures=1))
write('build/Silber_108_b.tex', make_SI(b, r'', figures=1))

t_plot = np.linspace(160, 450, 2)
plt.plot(t_plot, np.exp(reg_linear(t_plot, *noms(params))), 'b-', label='Fit')

plt.errorbar(t_Si[19:], noms(Si[19:]), fmt='rx', yerr=stds(Si[19:]), label='Messdaten')        # mit Fehlerbalken
plt.yscale('log')                                                                            # logarithmische x-Achse
plt.xlabel(r'$t \:/\: \si{\second}$')
plt.ylabel(r'$ \text{ln}(N_\textrm{\Delta t}) $')
plt.legend(loc='best')
plt.savefig('build/Silber_plot_108.pdf')
plt.clf()



lambda_Si_108 = -m
print('Halloooo')
print(lambda_Si_108)
print(noms(lambda_Si_108))
T = np.log(2)/lambda_Si_108 # in s
T_108 = T


write('build/lambda_Si_108.tex', make_SI(lambda_Si_108.nominal_value, r'\per\second', figures=1))
write('build/Halbwertszeit_Silber_108.tex', make_SI(T, r'\second', figures=1))
write('build/Halbwertszeit_Silber_108_ohne.tex', make_SI(T, r'', figures=1))
write('build/Startwert_Silber_108.tex', make_SI(np.e**b, r'', figures=1))
write('build/Startwert_Silber_108_nom.tex', make_SI((np.e**b).nominal_value, r'', figures=1))
a_0 = np.e**b


#### Schneller Zerfall####

def Rechnung(t,A,M):
    return A * np.e**(-M * t)

Si_fast = unp.uarray(np.zeros(16),np.zeros(16))

for x in range(0,16):
    Si_fast[x] = noms(Si[x]) - Rechnung(t_Si[x], a_0, lambda_Si_108)
    print(noms(Si[x]) - Rechnung(t_Si[x], a_0, lambda_Si_108))

# print('Fast')
# print(Si_fast)
params = ucurve_fit(reg_linear, t_Si[:16], np.log(noms(Si_fast[:16])))                                       # linearer Fit
m,b = params
write('build/Silber_110_m.tex', make_SI(m*1e4, r'\per\second', 'e-4' ,figures=1))
write('build/Silber_110_b.tex', make_SI(b, r'', figures=1))

t_plot = np.linspace(0, 160, 2)
plt.plot(t_plot, np.exp(reg_linear(t_plot, *noms(params))), 'b-', label='Fit')
plt.plot(t_Si[0:16:2], noms(Si_fast[0:16:2]), 'rx')

# Si_fast[8]= (noms(Si_fast[8]),5)
# Si_fast[10]= (noms(Si_fast[10]),2)
# Si_fast[14]= (noms(Si_fast[14]),1)
print(Si_fast)
plt.errorbar(t_Si[1:16:2], noms(Si_fast[1:16:2]), fmt='rx', yerr=stds(Si_fast[1:16:2]), label='Messdaten')        # mit Fehlerbalken
plt.yscale('log')                                                                            # logarithmische x-Achse
plt.xlabel(r'$ t \:/\: \si{\second}$')
plt.ylabel(r'$ \text{ln}(N_\textrm{\Delta t}) $')
plt.legend(loc='best')
plt.savefig('build/Silber_plot_110.pdf')
plt.clf()

lambda_Si_110 = -m
T = np.log(2)/lambda_Si_110 # in s
# lambda_Si_110_nom = noms(lambda_Si_110)
# st_nom = noms(np.e**b)
write('build/lambda_Si_110.tex', make_SI(lambda_Si_110.nominal_value, r'\per\second', figures=1))
write('build/Halbwertszeit_Silber_110.tex', make_SI(T, r'\second', figures=1))
write('build/Halbwertszeit_Silber_110_ohne.tex', make_SI(T, r'', figures=1))
write('build/Startwert_Silber_110.tex', make_SI(np.e**b, r'', figures=1))
write('build/Startwert_Silber_110_nom.tex', make_SI((np.e**b).nominal_value, r'', figures=1))
A_110 = np.e**b

#####Summenkurve####
params = ucurve_fit(reg_linear, t_Si, np.log(noms(Si)))                                       # linearer Fit
m,b = params

def Summe(t):
    return noms(A_110 * np.e**(-lambda_Si_110 * t) + a_0 * np.e**(-lambda_Si_108 * t))

print('UP')
# t_plot = np.linspace(0, 450, 1000)
# plt.plot(t_plot, np.exp(reg_linear(t_plot, noms(m), noms(b))), 'b-', label='Fit')

t_plot = np.linspace(0, 450, 1000)
plt.plot(t_plot, Summe(t_plot), 'b-', label='Fit')

plt.errorbar(t_Si, noms(Si), fmt='rx', yerr=stds(Si), label='Messdaten')        # mit Fehlerbalken
# plt.yscale('log')                                                                            # logarithmische x-Achse
plt.xlabel(r'$t \:/\: \si{\second}$')
plt.ylabel(r'$ N_\textrm{\Delta t} $')
plt.legend(loc='best')
plt.savefig('build/Silber_mit_Ausgleichsgrade.pdf')
plt.clf()


################################ FREQUENTLY USED CODE ################################
#
########## IMPORT ##########
# t, U, U_err = np.genfromtxt('data.txt', unpack=True)
# t *= 1e-3


########## ERRORS ##########
# R_unc = ufloat(R[0],R[2])
# U = 1e3 * unp.uarray(U, U_err)//
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
