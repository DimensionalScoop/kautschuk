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
import math
from scipy.interpolate import UnivariateSpline
# Planck
h = 4.135667516e-15 # eV second
# vacuum velo of light
c = 299792458 # metre per second
# diffraction distance
d = 201.4e-12 # metre
#elementary charge
e = 1.6e-19#coulomb
#Rydbergonstante
r = 13.6 #eV
#sommerfeldsche Feinstrukturkonstante
s_k = 7.29e-3



zwei_theta, impulsrate = np.genfromtxt('messdaten/1_Messung_werte.txt', unpack=True)

write('build/Tabelle_messung_1.tex', make_table([zwei_theta,impulsrate],[1, 0]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_messung_1_texformat.tex', make_full_table(
    'Messdaten Bragg Bedingung.',
    'table:A2',
    'build/Tabelle_messung_1.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$\theta \:/\: \si{\degree}$',
    r'$Zaehlrate$']))

theta, Z = np.loadtxt("messdaten/Bremsberg_werte.txt", unpack=True)

theta = theta/2


plt.xlabel(r'$\theta \:/\: \si{\degree}$')
plt.ylabel(r'$Impulsrate \:/\: \si{\kilo\gram\meter\per\second\tothe{2}}$')
# plt.xlabel(r'$t \:/\: \si{\milli\second}$')
#plt.title("Emissionsspektrum einer Cu-Anode bei 35 kV")
plt.grid()
plt.xticks()
plt.yticks()
plt.annotate(r'$K_\alpha$', xy=(46.5/2, 6499))
plt.annotate(r'$K_\beta$', xy=(41.5/2, 2000))
plt.annotate(r'Bremsberg', xy=(20/2, 750))
plt.plot(theta, Z,'b-', label='Interpolation')
plt.legend(loc='best')


plt.savefig("build/cu-emission.pdf")
plt.close()

# print("hallo")
# #np.arcsin(0.5)
# print(np.arcsin(1))
# print(np.sin(90))
# import math
# print("try")
# print(math.sin(90))
# print(math.cos(math.radians(1)))
####Grenzwinkel-Bestimmung####

print("Ergebniss")
#lamb1 = math.sin(math.radians(5.4)) * 2* 201.4*10**(-12)
lamb_min = 2*d*np.sin(np.deg2rad(5.4))
E_max = h*c/lamb_min
write('build/lambda_min.tex', make_SI(lamb_min*1e12, r'\pico\meter', figures=2))       # type in Anz. signifikanter Stellen
write('build/E_max.tex', make_SI(E_max*1e-3, r'\kilo\electronvolt', figures=2))       # type in Anz. signifikanter Stellen



####Halbwertsbreite####

def halbwertsbreite(x, y):
    spline = UnivariateSpline(x, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots

    lambda1 = 2*d*np.sin(np.deg2rad(r1))
    lambda2 = 2*d*np.sin(np.deg2rad(r2))
    E1 = h*c/lambda1
    E2 = h*c/lambda2
    DE = E1 - E2
    print ('Halbwertswinkel: {0:.5e} deg, {1:.5e} deg'.format(r1, r2))
    print ('Halbwertsbreite: {0:.5e}'.format(np.abs(r1-r2)))
    print (u'Energieaufloesung: {0:.5e} eV'.format(DE))

    xnew = np.linspace(min(x), max(x))
    ynew = spline(xnew)

    plt.plot(x, y, 'rx', label='Messdaten')
    plt.plot(xnew, ynew+np.max(y)/2,'b-', label='Interpolation')
    plt.axvline(r1)
    plt.axvline(r2)

    plt.grid()
    plt.legend()
    plt.xlabel("doppelter Kristallwinkel in Grad")
    plt.ylabel(u"Zählrate")
###############################################################



spline = UnivariateSpline(theta[84:90], Z[84:90]-np.max(Z[84:90])/2, s=0)
r1, r2 = spline.roots() # find the roots

lambda1 = 2*d*np.sin(np.deg2rad(r1))
lambda2 = 2*d*np.sin(np.deg2rad(r2))
E1 = h*c/lambda1
E2 = h*c/lambda2
DE = E1 - E2
print ('Halbwertswinkel: {0:.5e} deg, {1:.5e} deg'.format(r1, r2))
print ('Halbwertsbreite: {0:.5e}'.format(np.abs(r1-r2)))
print (u'Energieaufloesung: {0:.5e} eV'.format(DE))

xnew = np.linspace(min(theta[84:90]), max(theta[84:90]))
ynew = spline(xnew)

plt.plot(theta[84:90], Z[84:90], 'rx', label='Messdaten')
plt.plot(xnew, ynew+np.max(Z[84:90])/2,'b-', label='Interpolation')
plt.axvline(r1)
plt.axvline(r2)

plt.grid()
plt.legend(loc='best')
# plt.xlabel("doppelter Kristallwinkel in Grad")
# plt.ylabel(u"Zählrate")
plt.xlabel(r'$\theta \:/\: \si{\degree}$')
plt.ylabel(r'$Impulsrate \:/\: \si{\kilo\gram\meter\per\second\tothe{2}}$')


write('build/Halbwertswinkel_beta_1.tex', make_SI(r1, r'\degree', figures=2))
write('build/Halbwertswinkel_beta_2.tex', make_SI(r2, r'\degree', figures=2))
write('build/Halbwertsbreite_beta.tex', make_SI(np.abs(r1-r2), r'\degree', figures=2))
write('build/Energieaufloesung_beta.tex', make_SI(DE*1e-3, r'\kilo\electronvolt', figures=2))
plt.savefig("build/halbwertsbreiten_beta.pdf")
plt.close()

#halbwertsbreite(theta[96:101], Z[96:101])
spline = UnivariateSpline(theta[96:101], Z[96:101]-np.max(Z[96:101])/2, s=0)
r1, r2 = spline.roots() # find the roots

lambda1 = 2*d*np.sin(np.deg2rad(r1))
lambda2 = 2*d*np.sin(np.deg2rad(r2))
E1 = h*c/lambda1
E2 = h*c/lambda2
DE = E1 - E2
print ('Halbwertswinkel: {0:.5e} deg, {1:.5e} deg'.format(r1, r2))
print ('Halbwertsbreite: {0:.5e}'.format(np.abs(r1-r2)))
print (u'Energieaufloesung: {0:.5e} eV'.format(DE))

xnew = np.linspace(min(theta[96:101]), max(theta[96:101]))
ynew = spline(xnew)

plt.plot(theta[96:101], Z[96:101], 'rx', label='Messdaten')
plt.plot(xnew, ynew+np.max(Z[96:101])/2,'b-', label='Interpolation')
plt.axvline(r1)
plt.axvline(r2)

plt.grid()
plt.legend(loc='best')
plt.xlabel(r'$\theta \:/\: \si{\degree}$')
plt.ylabel(r'$Impulsrate \:/\: \si{\kilo\gram\meter\per\second\tothe{2}}$')

write('build/Halbwertswinkel_alpha_1.tex', make_SI(r1, r'\degree', figures=2))
write('build/Halbwertswinkel_alpha_2.tex', make_SI(r2, r'\degree', figures=2))
write('build/Halbwertsbreite_alpha.tex', make_SI(np.abs(r1-r2), r'\degree', figures=2))
write('build/Energieaufloesung_alpha.tex', make_SI(DE*1e-3, r'\kilo\electronvolt', figures=2))
write('build/Absorptionsenergie_Kupfer.tex', make_SI(E1*1e-3, r' ', figures=2))
plt.savefig("build/halbwertsbreiten_alpha.pdf")
plt.close()

##################### Abschirmkonstante
theta_alpha = 47.2/2
theta_beta = 42.8/2
write('build/theta_alpha.tex', make_SI(theta_alpha, r'\degree', figures=2))
write('build/theta_beta.tex', make_SI(theta_beta, r'\degree', figures=2))

lambda_alpha = 2*d*np.sin(np.deg2rad(theta_alpha))
lambda_beta = 2*d*np.sin(np.deg2rad(theta_beta))
E_alpha = h*c/lambda_alpha
E_beta = h*c/lambda_beta

sigma_1 = 29 - np.sqrt(E_beta/r)
sigma_2 = 29 -2* np.sqrt((r*((29-sigma_1)**2) - E_alpha)/r)

write('build/sigma_1.tex', make_SI(sigma_1, r' ', figures=2))
write('build/sigma_2.tex', make_SI(sigma_2, r' ', figures=2))

##Literaturwerte
sigma_1_lit = 29 - np.sqrt(8903/r)
sigma_2_lit = 29 -2* np.sqrt((r*((29-sigma_1)**2) - 8046)/r)

write('build/sigma_1_lit.tex', make_SI(sigma_1_lit, r' ', figures=2))
write('build/sigma_2_lit.tex', make_SI(sigma_2_lit, r' ', figures=2))

#write('build/Energiedifferenz.tex', make_SI(6268-1919, r'\electronvolt', figures=2)) # abgelesen
#######################

# Das Absorptionsspektrum, Graphiken
## Germanium
plt.clf
theta_ger, Z_ger = np.genfromtxt('messdaten/Germanium.txt', unpack=True)
theta_ger = theta_ger/2

# plt.plot(theta_ger, Z_ger)
plt.xlabel(r'$\theta \:/\: \si{\degree}$')
plt.ylabel(r'$Impulsrate \:/\: \si{\kilo\gram\meter\per\second\tothe{2}}$')

plt.grid()
# plt.xticks()
# plt.yticks()

plt.plot(theta_ger, Z_ger,'b-', label='Messdaten')
plt.legend(loc='best')
plt.savefig("build/Germanium.pdf")
plt.close()


## Zink
theta_zink, Z_zink = np.genfromtxt('messdaten/Zink.txt', unpack=True)
theta_zink = theta_zink/2

plt.xlabel(r'$\theta \:/\: \si{\degree}$')
plt.ylabel(r'$Impulsrate \:/\: \si{\kilo\gram\meter\per\second\tothe{2}}$')

plt.grid()
plt.xticks()
plt.yticks()

plt.plot(theta_zink, Z_zink,'b-', label='Messdaten')
plt.legend(loc='best')
plt.savefig("build/Zink.pdf")
plt.close()


##Zirkonium
theta_zir, Z_zir = np.genfromtxt('messdaten/Zirkonium.txt', unpack=True)
theta_zir = theta_zir/2

plt.xlabel(r'$\theta \:/\: \si{\degree}$')
plt.ylabel(r'$Impulsrate \:/\: \si{\kilo\gram\meter\per\second\tothe{2}}$')

plt.grid()
plt.xticks()
plt.yticks()

plt.plot(theta_zir, Z_zir,'b-', label='Messdaten')
plt.legend(loc='best')
plt.savefig("build/Zirkonium.pdf")
plt.close()


##Gold
theta_gold, Z_gold = np.genfromtxt('messdaten/Gold.txt', unpack=True)
theta_gold = theta_gold/2

plt.xlabel(r'$\theta \:/\: \si{\degree}$')
plt.ylabel(r'$Impulsrate \:/\: \si{\kilo\gram\meter\per\second\tothe{2}}$')

plt.grid()
plt.xticks()
plt.yticks()

plt.plot(theta_gold, Z_gold,'b-', label='Messdaten')
plt.legend(loc='best')
plt.savefig("build/Gold.pdf")
plt.close()

#### Energiebestimmung

def Grade(x_1, y_1, x_2, y_2):
    m = (y_2-y_1)/(x_2-x_1)
    b = y_1 - m*x_1

    y = (y_2 + y_1)/2
    x = (y-b)/m
    return x

##Germanium
theta_ger_x = Grade(theta_ger[32], Z_ger[32], theta_ger[35], Z_ger[35])
lambda_ger = 2*d*np.sin(np.deg2rad(theta_ger_x))
E_ger = h*c/lambda_ger
write('build/Absorptionsenergie_Germanium.tex', make_SI(E_ger*1e-3, r'\kilo\electronvolt', figures=2))
write('build/Absorptionsenergie_Germanium_ohne.tex', make_SI(E_ger*1e-3, r' ', figures=2))

##Zink
theta_zink_x = Grade(theta_zink[30], Z_zink[30], theta_zink[35], Z_zink[35])
lambda_zink = 2*d*np.sin(np.deg2rad(theta_zink_x))
E_zink = h*c/lambda_zink
write('build/Absorptionsenergie_Zink.tex', make_SI(E_zink*1e-3, r'\kilo\electronvolt', figures=2))
write('build/Absorptionsenergie_Zink_ohne.tex', make_SI(E_zink*1e-3, r' ', figures=2))

##Zirkonium
theta_zir_x = Grade(theta_zir[23], Z_zir[23], theta_zir[27], Z_zir[27])
lambda_zir = 2*d*np.sin(np.deg2rad(theta_zir_x))
E_zir = h*c/lambda_zir
write('build/Absorptionsenergie_Zirkonium.tex', make_SI(E_zir*1e-3, r'\kilo\electronvolt', figures=2))
write('build/Absorptionsenergie_Zirkonium_ohne.tex', make_SI(E_zir*1e-3, r' ', figures=2))

#### Bestimmung der Abschirmkonstante

sigma_ger = 32 - np.sqrt((E_ger/r) -((s_k**2)/4)*32**4)
sigma_zink = 30 - np.sqrt((E_zink/r) -((s_k**2)/4)*30**4)
sigma_zir = 40 - np.sqrt((E_zir/r) -((s_k**2)/4)*40**4)

write('build/Abschirmkonstante_Germanium.tex', make_SI(sigma_ger, r' ', figures=2))
write('build/Abschirmkonstante_Zink.tex', make_SI(sigma_zink, r' ', figures=2))
write('build/Abschirmkonstante_Zirkonium.tex', make_SI(sigma_zir, r' ', figures=2))

#Moseley-Diagramm

E_k = (E_zink, E_ger, E_zir)
Z   = (30,32,40) # Zn, Ge, Zr
E_k_wurzel = np.sqrt(E_k)
params = ucurve_fit(reg_linear, Z, E_k_wurzel)
m,b = params
write('build/hcRydbergonstante.tex', make_SI(4/3*m**2, r'\electronvolt', figures=1))
write('build/Rydbergonstante.tex', make_SI(4/3*m**2/(h*c), r'\per\meter', figures=1))

plt.clf
t_plot = np.linspace(25,45, 100)
plt.plot(t_plot , reg_linear(t_plot, *noms(params)), 'b-', label='Fit')
plt.plot(Z, E_k_wurzel, 'rx', label='Messdaten')
plt.xlabel(r'Kernladungszahl  $Z$')
plt.ylabel(r'$\sqrt{E_\textrm{k} \:/\: \si{\kilo\electronvolt}}$')
plt.legend(loc='best')
plt.savefig("build/Moseley_Diagramm.pdf")
plt.close

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
