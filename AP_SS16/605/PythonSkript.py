# Import system librar
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
import math
# Finish importing system l

# Adding subfolder to syste
import os
import sys
import inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

 # use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(
    inspect.getfile(inspect.currentframe()))[0], "python_custom_scripts")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
# Finish adding subfolder to s

# Import custom librar
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
# Finish importing custom l

# physikalische Konstanten
h = const.physical_constants["Planck constant in eV s"][0]
c = const.physical_constants["speed of light in vacuum"][0]
R = const.physical_constants["Rydberg constant times hc in eV"][0]
a = const.physical_constants["fine-structure constant"][0]          # w/o units

# Alle Daten einlesen
roh_winkel_helium = np.genfromtxt(
    'messdaten/WinkelHelium.txt', unpack=True)
roh_winkel_kalium, s_kalium = np.genfromtxt(
    'messdaten/WinkelKalium.txt', unpack=True)
roh_winkel_natrium, s_natrium = np.genfromtxt(
    'messdaten/WinkelNatrium.txt', unpack=True)
roh_winkel_rubidium, s_rubidium = np.genfromtxt(
    'messdaten/WinkelRubidium.txt', unpack=True)
roh_winkel_kalib, t_kalib = np.genfromtxt(
    'messdaten/Kalibrierung.txt', unpack=True)
# Wir haben das Reflexionsgitter verwendet und müssen daher ein paar
# Winkel umrechnen...
# aus Rohdaten die gesuchten phi Winkel machen:
delta = 338.2               # Winkel bei Totalreflexion
alpha = 400 - delta         # Winkel zw. einfallendem und reflektiertem Strahl
beta = 90 - alpha / 2       # Reflexionswinkel

def adapt_angle(delta_strich):
    return np.abs(400 - delta_strich - 90 - alpha / 2)

phi_helium = -np.array(adapt_angle(roh_winkel_helium))* np.pi / 180.        # das - kommt von k = -1
phi_kalium = -np.array(adapt_angle(roh_winkel_kalium))* np.pi / 180.        # das - kommt von k = -1
phi_natrium = -np.array(adapt_angle(roh_winkel_natrium))* np.pi / 180.      # das - kommt von k = -1
phi_rubidium = -np.array(adapt_angle(roh_winkel_rubidium))* np.pi / 180.    # das - kommt von k = -1
phi_kalib = -np.array(adapt_angle(roh_winkel_kalib))* np.pi / 180.          # das - kommt von k = -1

########## Aufgabenteil a) ##########
# Bestimmung der Gitterkonstante

# bekannte Wellenlängen der Helium Spektrallinien (hier muss die
# Reihenfolge natürlich übereinstimmen mit derjenigen der Datei
# WinkelHelium.txt):
lambda_helium = np.array([438.8, 447.1, 471.3, 492.2,
                          501.6, 504.8, 587.6, 667.8, 706.5]) * 1e-9    # in m

# sinus für den plot und den fit
sin_phi_helium = np.array(np.sin(phi_helium))
# fit sin(phi) gegenüber lambda zur Bestimmung von g
params_gitterkonstante = ucurve_fit(
    reg_linear, sin_phi_helium, lambda_helium)

g, offset = params_gitterkonstante                  # g in m, offset Einheitenfrei
write('build/gitterkonstante.tex', make_SI(g * 1e9, r'\nano\meter', figures=1))
write('build/offset.tex', make_SI(offset * 1e9, r'\nano\meter', figures=1))
write('build/Tabelle_messdaten_kalium.tex', make_table([phi_kalium*180/np.pi],[1]))
write('build/Tabelle_messdaten_natrium.tex', make_table([phi_natrium*180/np.pi],[1]))
write('build/Tabelle_messdaten_rubidium.tex', make_table([phi_rubidium*180/np.pi],[1]))

##### PLOT lineare Regression #####
t_plot = np.linspace(np.amin(sin_phi_helium), np.amax(sin_phi_helium), 2)
plt.xlim(t_plot[0] - 1 / np.size(sin_phi_helium) * (t_plot[-1] - t_plot[0]),
         t_plot[-1] + 1 / np.size(sin_phi_helium) * (t_plot[-1] - t_plot[0]))
plt.plot(t_plot,
         reg_linear(t_plot, *noms(params_gitterkonstante))* 1e9,
         'b-', label='Fit')
plt.plot(sin_phi_helium,
         lambda_helium * 1e9,
         'rx', label='Messdaten')
plt.ylabel(r'$\lambda \:/\: \si{\nano\meter}$')
plt.xlabel(r'$\sin(\varphi)$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/aufgabenteil_a_plot.pdf')
plt.clf()
#### Ende Plot ####

#### TABELLE ####
write('build/Tabelle_a.tex', make_table([lambda_helium*1e9, -phi_helium, -sin_phi_helium],[1, 3, 3]))
write('build/Tabelle_a_texformat.tex', make_full_table(
    'Messdaten zur Bestimmung der Gitterkonstante.',
    'table:gitterkonstante',
    'build/Tabelle_a.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [r'$\lambda \:/\: \si{\nano\meter}$',
    r'$|\varphi| \:/\: \si{\radian}$',
    r'$|\sin(\varphi)|$']))
#### Ende Tabelle ####





########## Aufgabenteil b) ##########
# Kalibrierung des Okularmikrometers
lambda_kalibrierung = np.array(
    [438.8, 447.1, 501.6, 504.8]) * 1e-9       # in m
Eichgroesse = unp.uarray(np.zeros(np.size(lambda_kalibrierung)/2), np.zeros(np.size(lambda_kalibrierung)/2))    # in m Skt^(-1)     Skt = Sektor
phi_mean = unp.uarray(np.zeros(np.size(phi_kalib)/2), np.zeros(np.size(phi_kalib)/2))                           # in rad
eichgroesse_mit_fehler = ufloat(0,0)                                                                            # in m Skt^(-1)     Skt = Sektor
i = 0
for eichgroesse in Eichgroesse:
    j = 2*i         # index für die 2*size(Eichgroesse) arrays, in unserem Fall arrays mit 4 Einträgen
    phi_mean[i] = ufloat(np.mean(phi_kalib[j:j+2]), np.std(phi_kalib[j:j+2]))    # Indizierung für i= 0: von 0 bis < 2  also 0 bis 1, dh. wir bilden Mittelwert aus zwei Werten
    Eichgroesse[i] = np.abs(lambda_kalibrierung[j] - lambda_kalibrierung[j+1]) / ( np.abs(t_kalib[j] - t_kalib[j+1]) * unp.cos(phi_mean[i]) )   # Formel aus "zu b:"
    eichgroesse_mit_fehler += Eichgroesse[i]
    i += 1
eichgroesse_mit_fehler *= 1/(np.size(lambda_kalibrierung)/2)      # in m
write('build/Eichgroesse.tex', make_SI(eichgroesse_mit_fehler*1e9, r'\nano\meter', figures=1))

# Tabelle für LaTeX:
phi1 = np.array([phi_kalib[0], phi_kalib[2]])       # crappy weil hart codiert aber was will man machen
phi2 = np.array([phi_kalib[1], phi_kalib[3]])       # in rad
lambda1 = [lambda_kalibrierung[0]*1e9, lambda_kalibrierung[2]*1e9]     # in nm
lambda2 = [lambda_kalibrierung[1]*1e9, lambda_kalibrierung[3]*1e9]     # in nm
t1 = [t_kalib[0], t_kalib[2]]
t2 = [t_kalib[1], t_kalib[3]]
write('build/Tabelle_b.tex', make_table([-phi1,lambda1,t1,-phi2,lambda2,t2,-phi_mean],[3, 1, 1, 3, 1, 1, 1]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_b_texformat.tex', make_full_table(
    'Messdaten zur Bestimmung der Eichgröße.',
    'table:eichgroesse',
    'build/Tabelle_b.tex',
    [6],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [r'$\lambda_1 \:/\: \si{\nano\meter}$',
    r'$|\varphi_1| \:/\: \si{\radian}$',
    r'$t_1 \:/\: \text{Skt}^{-1}$',
    r'$\lambda_2 \:/\: \si{\nano\meter}$',
    r'$|\varphi_2| \:/\: \si{\radian}$',
    r'$t_2 \:/\: \text{Skt}^{-1}$',
    r'$|\overline{\varphi_{1,2}}| \:/\: \si{\radian}$']))




########## Aufgabenteil c) ##########
# Beugungswinkel von Natrium, Kalium, Rubidium
def make_mean_from_two_values_in_array(array_in, array_out):
    i=0
    for bla in array_out:
        j = 2*i
        array_out[i] = ufloat(np.mean(array_in[j:j+2]), np.std(array_in[j:j+2]))
        i += 1
    return array_out

# arrays mit der richtigen Größe initialisieren
phi_kalium_mean = unp.uarray(np.zeros(np.size(phi_kalium)/2), np.zeros(np.size(phi_kalium)/2))
phi_natrium_mean = unp.uarray(np.zeros(np.size(phi_natrium)/2), np.zeros(np.size(phi_natrium)/2))
phi_rubidium_mean = unp.uarray(np.zeros(np.size(phi_rubidium)/2), np.zeros(np.size(phi_rubidium)/2))
lambda_kalium_mean = unp.uarray(np.zeros(np.size(phi_kalium)/2), np.zeros(np.size(phi_kalium)/2))
lambda_natrium_mean = unp.uarray(np.zeros(np.size(phi_natrium)/2), np.zeros(np.size(phi_natrium)/2))
lambda_rubidium_mean = unp.uarray(np.zeros(np.size(phi_rubidium)/2), np.zeros(np.size(phi_rubidium)/2))

# Mittelwerte aus den 2 Dublettwinkeln bilden
phi_kalium_mean = make_mean_from_two_values_in_array(phi_kalium, phi_kalium_mean)         # in rad
phi_natrium_mean = make_mean_from_two_values_in_array(phi_natrium, phi_natrium_mean)      # in rad
phi_rubidium_mean = make_mean_from_two_values_in_array(phi_rubidium, phi_rubidium_mean)   # in rad

# Daraus die zugehörigen Wellenlängen errechnen
lambda_kalium_mean = g*unp.sin(phi_kalium_mean) + offset        # in m
lambda_natrium_mean = g*unp.sin(phi_natrium_mean) + offset      # in m
lambda_rubidium_mean = g*unp.sin(phi_rubidium_mean) + offset    # in m

# zur Berechnung von Delta_s
def get_difference_between_two_values_in_array(array_in, array_out):
    i=0
    for bla in array_out:
        j = 2*i
        array_out[i] = array_in[j] - array_in[j+1]
        i += 1
    return array_out

# mehr arrays initialisieren!
delta_s_kalium = np.array(np.zeros(np.size(s_kalium)/2))
delta_s_natrium = np.array(np.zeros(np.size(s_natrium)/2))
delta_s_rubidium = np.array(np.zeros(np.size(s_rubidium)/2))
# delta s berechnen
delta_s_kalium = np.abs(get_difference_between_two_values_in_array(s_kalium, delta_s_kalium))
delta_s_natrium = np.abs(get_difference_between_two_values_in_array(s_natrium, delta_s_natrium))
delta_s_rubidium = np.abs(get_difference_between_two_values_in_array(s_rubidium, delta_s_rubidium))
# aus der Gleichung direkt überhalb Kapitel 12 delta lambda berechnen
delta_lambda_kalium = eichgroesse_mit_fehler * delta_s_kalium * unp.cos(phi_kalium_mean)        # in m
delta_lambda_natrium = eichgroesse_mit_fehler * delta_s_natrium * unp.cos(phi_natrium_mean)     # in m
delta_lambda_rubidium = eichgroesse_mit_fehler * delta_s_rubidium * unp.cos(phi_rubidium_mean)  # in m
# Delta E berechnen aus der Gleichung direkt überhalb von Kap. 7
delta_E_kalium = h*c*delta_lambda_kalium/(lambda_kalium_mean**2)            # in eV (wegen h in eVs)
delta_E_natrium = h*c*delta_lambda_natrium/(lambda_natrium_mean**2)         # # in eV (wegen h in eVs)
delta_E_rubidium = h*c*delta_lambda_rubidium/(lambda_rubidium_mean**2)      # # in eV (wegen h in eVs)

# final step : calculate sigma_2
def sigma_2 (n, l, Delta_E, z):
    return z-unp.sqrt(unp.sqrt(Delta_E * l * (l+1) * n**3 / (R*a**2)))

sigma_2_kalium = sigma_2(4, 1, delta_E_kalium, 19)  # Kalium : z=19, 4P1/2 Übergang (P bedeutet l=1)
sigma_2_natrium = sigma_2(3, 1, delta_E_natrium, 11)  # Kalium : z=11, 3P1/2 Übergang (P bedeutet l=1)
sigma_2_rubidium = sigma_2(5, 1, delta_E_rubidium, 37)  # Kalium : z=37, 5P1/2 Übergang (P bedeutet l=1)

# da der Fehler systematische Fehler der Eingangsgrößen klein ist, geben wir nur den Fehler des Mittelwerts an.
sigma_2_kalium_mean = ufloat(np.mean(noms(sigma_2_kalium)), np.std(noms(sigma_2_kalium)))
sigma_2_natrium_mean = ufloat(np.mean(noms(sigma_2_natrium)), np.std(noms(sigma_2_natrium)))
# Für Rubidium haben wir nur einen Messwert, daher gibts auch nichts zu mitteln
sigma_2_rubidium_unc = ufloat(noms(sigma_2_rubidium), stds(sigma_2_rubidium))

write('build/sigma_kalium.tex', make_SI(sigma_2_kalium_mean, r'', figures=1))
write('build/sigma_natrium.tex', make_SI(sigma_2_natrium_mean, r'', figures=1))
write('build/sigma_rubidium.tex', make_SI(sigma_2_rubidium_unc, r'', figures=1))

#### TABELLEn ####
write('build/Tabelle_c_kalium.tex', make_table([
                                                -phi_kalium_mean,
                                                lambda_kalium_mean*1e9,
                                                delta_s_kalium,
                                                delta_lambda_kalium*1e9,
                                                delta_E_kalium*1e3,
                                                sigma_2_kalium
                                                ],
                                                [1, 1, 2, 1, 1, 1]))
write('build/Tabelle_c_kalium_texformat.tex', make_full_table(
    r'Messdaten und abgeleitete Größen -- Kalium ($z=19\;n=4\;l=1$).',
    'table:kalium',
    'build/Tabelle_c_kalium.tex',
    [0,1,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [r'$|\overline{\varphi}| \:/\: \si{\radian}$',
    r'$\overline{\lambda} \:/\: \si{\nano\meter}$',
    r'$\Delta s \:/\: \text{Skt}$',
    r'$\Delta\lambda \:/\: \si{\nano\meter}$',
    r'$\Delta E_\text{D} \:/\: \si{\milli\electronvolt}$',
    r'$\sigma_2$']))

write('build/Tabelle_c_natrium.tex', make_table([
                                                -phi_natrium_mean,
                                                lambda_natrium_mean*1e9,
                                                delta_s_natrium,
                                                delta_lambda_natrium*1e9,
                                                delta_E_natrium*1e3,
                                                sigma_2_natrium
                                                ],
                                                [1, 1, 2, 1, 1, 1]))
write('build/Tabelle_c_natrium_texformat.tex', make_full_table(
    r'Messdaten und abgeleitete Größen -- Natrium ($z=11\;n=3\;l=1$).',
    'table:natrium',
    'build/Tabelle_c_natrium.tex',
    [0,1,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [r'$|\overline{\varphi}| \:/\: \si{\radian}$',
    r'$\overline{\lambda} \:/\: \si{\nano\meter}$',
    r'$\Delta s \:/\: \text{Skt}$',
    r'$\Delta\lambda \:/\: \si{\nano\meter}$',
    r'$\Delta E_\text{D} \:/\: \si{\milli\electronvolt}$',
    r'$\sigma_2$']))

write('build/Tabelle_c_rubidium.tex', make_table([
                                                -phi_rubidium_mean,
                                                lambda_rubidium_mean*1e9,
                                                delta_s_rubidium,
                                                delta_lambda_rubidium*1e9,
                                                delta_E_rubidium*1e3,
                                                sigma_2_rubidium
                                                ],
                                                [1, 1, 2, 1, 1, 1]))
write('build/Tabelle_c_rubidium_texformat.tex', make_full_table(
    r'Messdaten und abgeleitete Größen -- Rubidium ($z=37\;n=5\;l=1$).',
    'table:rubidium',
    'build/Tabelle_c_rubidium.tex',
    [0,1,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [r'$|\overline{\varphi}| \:/\: \si{\radian}$',
    r'$\overline{\lambda} \:/\: \si{\nano\meter}$',
    r'$\Delta s \:/\: \text{Skt}$',
    r'$\Delta\lambda \:/\: \si{\nano\meter}$',
    r'$\Delta E_\text{D} \:/\: \si{\milli\electronvolt}$',
    r'$\sigma_2$']))
#### Ende Tabellen ####

#### Relative Fehler ####
relFehlerEdKalium = (np.amax(delta_E_kalium) - np.amin(delta_E_kalium))/np.mean(delta_E_kalium)
relFehlerEdNatrium = (np.amax(delta_E_natrium) - np.amin(delta_E_natrium))/np.mean(delta_E_natrium)
write('build/RelFehlerEdKalium.tex', make_SI(relFehlerEdKalium*1e2, r'\percent', figures=1))
write('build/RelFehlerEdNatrium.tex', make_SI(relFehlerEdNatrium*1e2, r'\percent', figures=1))

################################ FREQUENTLY USED CODE ####################
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
# Relative Fehler zum späteren Vergleich in der Diskussion
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
# write('build/parameter_b.tex', make_SI(b * 1e-3, r'\kilo\hertz',
# figures=2))      # type in Anz. signifikanter Stellen


########## PLOTTING ##########
# plt.clf                   # clear actual plot before generating a new one
#
# automatically choosing limits with existing array T1
# t_plot = np.linspace(np.amin(T1), np.amax(T1), 100)
# plt.xlim(t_plot[0]-1/np.size(T1)*(t_plot[-1]-t_plot[0]), t_plot[-1]+1/np.size(T1)*(t_plot[-1]-t_plot[0]))
#
# hard coded limits
# t_plot = np.linspace(-0.5, 2 * np.pi + 0.5, 1000) * 1e-3
#
# standard plotting
# plt.plot(t_plot * 1e3, f(t_plot, *noms(params)) * 1e-3, 'b-', label='Fit')
# plt.plot(t * 1e3, U * 1e3, 'rx', label='Messdaten')
# plt.errorbar(B * 1e3, noms(y) * 1e5, fmt='rx', yerr=stds(y) * 1e5, label='Messdaten')        # mit Fehlerbalken
# plt.xscale('log')                                                                            # logarithmische x-Achse
# plt.xlim(t_plot[0] * 1e3, t_plot[-1] * 1e3)
# plt.xlabel(r'$t \:/\: \si{\milli\second}$')
# plt.ylabel(r'$U \:/\: \si{\kilo\volt}$')
# plt.legend(loc='best')
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/aufgabenteil_a_plot.pdf')


########## WRITING TABLES ##########
# IF THERE IS ONLY ONE COLUMN IN A TABLE (workaround):
# a=np.array([Wert_d[0]])
# b=np.array([Rx_mean])
# c=np.array([Rx_mean_err])
# d=np.array([Lx_mean*1e3])
# e=np.array([Lx_mean_err*1e3])
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
# Aufsplitten von Tabellen, falls sie zu lang sind
# t1, t2 = np.array_split(t * 1e3, 2)
# U1, U2 = np.array_split(U * 1e-3, 2)
# write('build/loesung-table.tex', make_table([t1, U1, t2, U2], [3, None, 3, None]))  # type in Nachkommastellen
#
# Verschmelzen von Tabellen (nur Rohdaten, Anzahl der Zeilen muss gleich sein)
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
# np.size(array)                    # Anzahl der Elemente eines Arrays
# ermitteln


########## ARRAY INDEXING ##########
# y[n - 1::n]                       # liefert aus einem Array jeden n-ten
# Wert als Array


########## DIFFERENT STUFF ##########
# R = const.physical_constants["molar gas constant"]      # Array of
# value, unit, error
