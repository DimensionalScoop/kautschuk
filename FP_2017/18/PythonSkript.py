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
    reg_cubic,
    reg_gauss
)
from error_calculation import(
    mean,
    MeanError
)
from utility import(
    constant
)
from PeakDetect import(
    peakdetect
)
################################################ Finish importing custom libraries #################################################

Teilchen_pro_Kanal = np.genfromtxt('messdaten/1.txt', unpack=True)
Energie,EmW = np.genfromtxt('messdaten/Tabelle1_kontrolldaten.txt', unpack=True)
Kanal=np.arange(0,8192)

Messzeit_1 = 3637 #sekunden
# print(np.size(Kanal))
#
# a,b=peakdetect(Teilchen_pro_Kanal, Kanal, lookahead=36, delta=20)
#
# print(a)

# Peaksx=[60,290,414,502,522,637,655,825,995,1159,1382,1968,2308,2611,3232]
# Peaksy=[112,118,3943,89,65,66,60,541,37,1091,81,29,25,139,113]

# Peaksx=[140,290,414,825,1159,1236,1382,1492,2611,2909,3232,3639,3726,4718]
# Peaksy=[233,118,3948,541,1091,49,81,104,139,50,113,70,78,84]

Peaksx=np.array([414,825,1159,1492,2611,2909,3232,3639,3726,4718])
Peaksy=np.array([3948,541,1091,104,139,50,113,70,78,84])
Peaksy=Peaksy/Messzeit_1

plt.plot(Kanal, Teilchen_pro_Kanal[Kanal]/Messzeit_1,'r', linewidth=0.5, label='Messdaten')
plt.plot(Peaksx, Peaksy,'bx', linewidth=0.5, label='verwendete Maxima')
plt.xlabel(r'$\text{Kanalnummer}$')
plt.ylabel(r'$\text{Zählrate} \:/\: \si{\per\second}$')
plt.legend(loc='best')
plt.savefig('build/Messdaten_Teil_1_Rohdaten.pdf')
plt.clf()

write('build/Tabelle_Energie_EmW.tex', make_table([Peaksx,Energie,EmW],[0,2,1]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_Energie_EmW_texformat.tex', make_full_table(
    caption = 'Verwendeten Werte für die lineare Ausgleichsrechnung.',
    label = 'table:A1',
    source_table = 'build/Tabelle_Energie_EmW.tex',
    stacking = [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
    units = [
    r'$\text{Kanalnummer}$',
    r'$\text{Energie} \:/\: \si{\kilo\electronvolt}$',
    r'$ \text{W} \:/\: \si{\percent}$'],
    replaceNaN = True,                      # default = false
    replaceNaNby = 'not a number'))         # default = '-'






t_plot = np.linspace(0, 5000, 2)
params1 = ucurve_fit(reg_linear, Peaksx, Energie)
plt.plot(t_plot, reg_linear(t_plot, *noms(params1)), 'b-', label='Fit')
plt.plot(Peaksx, Energie, 'rx', label='Messdaten')
plt.xlabel(r'$\text{Kanalnummer}$')
plt.ylabel(r'$\text{Energie} \:/\: \si{\kilo\electronvolt}$')
plt.legend(loc='best')
plt.savefig('build/fit.pdf')
plt.clf()
write('build/fitparameter_m.tex', make_SI(params1[0], r'', figures=1))
write('build/fitparameter_b.tex', make_SI(params1[1]*(-1), r'\kilo\electronvolt', figures=1))
#--------------------------------------------------------------------------------------------------

Akt_Bq_2000 = ufloat(4130,60)
write('build/Akt_Bq_2000.tex', make_SI(Akt_Bq_2000, r'\becquerel', figures=1))
Halbwertszeit_Bq= ufloat(4943,5)
write('build/Halbwertszeit_Bq.tex', make_SI(Halbwertszeit_Bq, r'\day', figures=1))
Akt_Bq = Akt_Bq_2000*unp.exp(-(np.log(2)/Halbwertszeit_Bq)*6077)
write('build/Akt_Bq.tex', make_SI(Akt_Bq, r'\becquerel', figures=1))

a=11.5 #cm
r=2.25 #cm
Raumw = 0.5*(1-(a/np.sqrt(a**2+r**2)))
write('build/Raumwinkel.tex', make_SI(Raumw, r'', figures=2))
# plt.plot(Kanal[400:430], Teilchen_pro_Kanal[Kanal[400:430]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_1_peak1.pdf')
# plt.clf()
Inhalt_peak1 = sum(Teilchen_pro_Kanal[Kanal[409:419]])
# print(Inhalt_peak1)

# plt.plot(Kanal[810:840], Teilchen_pro_Kanal[Kanal[810:840]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_1_peak2.pdf')
# plt.clf()
Inhalt_peak2 = sum(Teilchen_pro_Kanal[Kanal[820:830]])


# plt.plot(Kanal[1145:1175], Teilchen_pro_Kanal[Kanal[1145:1175]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_1_peak3.pdf')
# plt.clf()
Inhalt_peak3 = sum(Teilchen_pro_Kanal[Kanal[1152:1165]])

# plt.plot(Kanal[1477:1507], Teilchen_pro_Kanal[Kanal[1477:1507]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_1_peak4.pdf')
# plt.clf()
Inhalt_peak4 = sum(Teilchen_pro_Kanal[Kanal[1485:1498]])

# plt.plot(Kanal[2596:2626], Teilchen_pro_Kanal[Kanal[2596:2626]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_1_peak5.pdf')
# plt.clf()
Inhalt_peak5 = sum(Teilchen_pro_Kanal[Kanal[2601:2623]])

# plt.plot(Kanal[2894:2924], Teilchen_pro_Kanal[Kanal[2894:2924]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_1_peak6.pdf')
# plt.clf()
Inhalt_peak6 = sum(Teilchen_pro_Kanal[Kanal[2902:2915]])

# plt.plot(Kanal[3217:3247], Teilchen_pro_Kanal[Kanal[3217:3247]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_1_peak7.pdf')
# plt.clf()
Inhalt_peak7 = sum(Teilchen_pro_Kanal[Kanal[3220:3240]])

# plt.plot(Kanal[3624:3654], Teilchen_pro_Kanal[Kanal[3624:3654]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_1_peak8.pdf')
# plt.clf()
Inhalt_peak8 = sum(Teilchen_pro_Kanal[Kanal[3627:3648]])

# plt.plot(Kanal[3711:3741], Teilchen_pro_Kanal[Kanal[3711:3741]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_1_peak9.pdf')
# plt.clf()
Inhalt_peak9 = sum(Teilchen_pro_Kanal[Kanal[3712:3736]])

# plt.plot(Kanal[4703:4733], Teilchen_pro_Kanal[Kanal[4703:4733]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_1_peak10.pdf')
# plt.clf()
Inhalt_peak10 = sum(Teilchen_pro_Kanal[Kanal[4706:4731]])
Z=np.array([Inhalt_peak1,Inhalt_peak2,Inhalt_peak3,Inhalt_peak4,Inhalt_peak5,Inhalt_peak6,Inhalt_peak7,Inhalt_peak8,Inhalt_peak9,Inhalt_peak10])

Z=Z/Messzeit_1

#Emission
Q=Z/(Raumw*Akt_Bq*EmW/100)
print(Z)
print(Raumw)
print(Akt_Bq)
print(EmW)
print(Q)

def f(E, a, b, c):
    return a*E**b +c

t_plot = np.linspace(200, 1500, 1500)
params2 = ucurve_fit(f, Energie[1:10], noms(Q[1:10]),  p0=[200, -1, 0.55])
plt.plot(t_plot, f(t_plot, *noms(params2)), 'b-', label='Fit')

# plt.plot(Energie, Q,'r', linewidth=0.5, label='Messdaten')

plt.errorbar(Energie[1:10], noms(Q[1:10]), fmt='rx', yerr=stds(Q[1:10]), label='Messdaten')
plt.xlabel(r'$\text{Energie} \:/\: \si{\kilo\electronvolt}$')
plt.ylabel(r'$\text{Q(E)}$')
plt.legend(loc='best')
plt.savefig('build/Energie_Effizenz.pdf')
plt.clf()


write('build/Fitparamter_Effizienz_a.tex', make_SI(params2[0], r'\per\kilo\electronvolt', figures=1))
write('build/Fitparamter_Effizienz_b.tex', make_SI(params2[1], r'', figures=1))
write('build/Fitparamter_Effizienz_c.tex', make_SI(params2[2], r'', figures=1))

write('build/Tabelle_Effizienz_a.tex', make_table([Peaksx,Energie,Z,EmW,Q],[0, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_Effizienz_a_texformat.tex', make_full_table(
    caption = 'Messdaten zur Bestimmung der Effizienz.',
    label = 'table:Effizienz_a',
    source_table = 'build/Tabelle_Effizienz_a.tex',
    stacking = [4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
    units = [
    r'$Kanalnummer$',
    r'$Energie \:/\: \si{\kilo\electronvolt}$',
    r'$Z  \:/\: \si{\per\second}$',
    r'$W \:/\: \si{\percent}$',
    r'$Q$'],
    replaceNaN = True,                      # default = false
    replaceNaNby = 'not a number'))         # default = '-'



# #-----------------------------------------------------------------------------------------------
# #Aufgabenteil b
def Eff(E):
    return params2[0]*E**params2[1] +params2[2]
def Energiefkt(k):
    return params1[0]*k + params1[0]
#
def Umkehrgauss(Wert,a,b,c):
    return unp.sqrt(np.abs((unp.log(Wert/a))/(-b))) +c

Messzeit_2 =6707 #sekunden

Teilchen_pro_Kanal = np.genfromtxt('messdaten/2.txt', unpack=True)
plt.plot(Kanal, Teilchen_pro_Kanal[Kanal]/Messzeit_2,'r', linewidth=0.5)
plt.xlabel(r'$\text{Kanalnummer}$')
plt.ylabel(r'$\text{Zählrate} \:/\: \si{\per\second}$')
plt.savefig('build/Messdaten_Teil_2_Rohdaten.pdf')
plt.clf()

plt.plot(Kanal[2203:2263], Teilchen_pro_Kanal[Kanal[2203:2263]]/Messzeit_2,'r', linewidth=0.5)
plt.xlabel(r'$\text{Kanalnummer}$')
plt.ylabel(r'$\text{Zählrate} \:/\: \si{\per\second}$')
plt.savefig('build/Messdaten_Teil_2_Photopeak.pdf')
plt.clf()


plt.plot(Kanal[575:815], Teilchen_pro_Kanal[Kanal[575:815]]/Messzeit_2,'r', linewidth=0.5)
plt.xlabel(r'$\text{Kanalnummer}$')
plt.ylabel(r'$\text{Zählrate} \:/\: \si{\per\second}$')
plt.savefig('build/Messdaten_Teil_2_Ruckstreupeak.pdf')
plt.clf()

E_gamma = Energiefkt(2220)
write('build/E_gamma.tex', make_SI(E_gamma, r'\kilo\electronvolt', figures=1))


t_plot = np.linspace(2200, 2235, 500)
params = ucurve_fit(reg_gauss, Kanal[2200:2235], Teilchen_pro_Kanal[Kanal[2200:2235]]/Messzeit_2, p0=[0.5, 0.2, 2220])
plt.plot(t_plot, reg_gauss(t_plot, *noms(params)), 'b-', label='Fit')
plt.plot(Kanal[2203:2263], Teilchen_pro_Kanal[Kanal[2203:2263]]/Messzeit_2,'r', linewidth=0.5, label='Messdaten')
plt.xlabel(r'$\text{Kanalnummer}$')
plt.ylabel(r'$\text{Zählrate} \:/\: \si{\per\second}$')
plt.legend(loc='best')
plt.savefig('build/fit2.pdf')
plt.clf()

write('build/Fitparamter_Photo_a.tex', make_SI(params[0], r'\per\second', figures=1))
write('build/Fitparamter_Photo_b.tex', make_SI(params[1], r'', figures=1))
write('build/Fitparamter_Photo_c.tex', make_SI(params[2], r'', figures=1))


#E_1_2 = 1/unp.sqrt(2*params[2])


E_1_2_theo = 2.35*np.sqrt(0.1*660000*2.9)
E_1_10_theo = E_1_2_theo * 1.823
write('build/Halbwertsbreite_theo.tex', make_SI(E_1_2_theo*1e-3, r'\kilo\electronvolt', figures=1))
write('build/Zehntelwertsbreite_theo.tex', make_SI(E_1_10_theo*1e-3, r'\kilo\electronvolt', figures=1))
# print(Umkehrgauss(params[2]/10,*params))
#
# a=params[2]+params[2]-2214
# print(reg_gauss(2214,*noms(params)))
# print(reg_gauss(noms(a),*noms(params)))
# print(reg_gauss(noms(params[2]),*noms(params)))




E_1_2_a= (Umkehrgauss(params[0]/2,*params))
E_1_2_b= params[2]-(E_1_2_a-params[2])
E_1_2 = Energiefkt(E_1_2_a) - Energiefkt(E_1_2_b)
write('build/Halbwertsbreite.tex', make_SI(E_1_2, r'\kilo\electronvolt', figures=1))
E_1_10 = E_1_2*1.823

# E_1_10_a= (Umkehrgauss(params[0]/10,*params))
# E_1_10_b= params[2]-(E_1_10_a-params[2])
# E_1_10 = E_1_10_a - E_1_10_b
write('build/Zehntelwertsbreite.tex', make_SI(E_1_2*1.823, r'\kilo\electronvolt', figures=1))
# print(E_1_10/E_1_2)
write('build/Quotient.tex', make_SI(E_1_10/E_1_2, r'', figures=1))
write('build/Abweichung_Quotient.tex', make_SI(((E_1_10/E_1_2)/1.823)*1e2, r'\percent', figures=1))
write('build/Inhalt_photopeak.tex', make_SI(unp.sqrt(np.pi/params[1])*params[0], r'\per\second', figures=1))
Z_Photopeak = unp.sqrt(np.pi/params[1])*params[0]
write('build/Vergleich_halbwertsbreiten_photo.tex', make_SI(100-(E_1_2/E_1_2_theo)*100, r'\kilo\electronvolt', figures=1))
# #------Compton---------------
plt.plot(Kanal[1400:1700], Teilchen_pro_Kanal[Kanal[1400:1700]]/Messzeit_2,'r', linewidth=0.5)

#plt.axvline(x=1558, ymin=0, ymax = Teilchen_pro_Kanal[Kanal[1558]]/Messzeit_2, linewidth=1, color='b', label='Comptonkante')
plt.xlabel(r'$\text{Kanalnummer}$')
plt.ylabel(r'$\text{Zählrate} \:/\: \si{\per\second}$')

plt.savefig('build/Messdaten_Teil_2_Comptonkante.pdf')
plt.clf()
Kanal_Comptonkante = ufloat(1558,25)
write('build/Kanal_Comptonkante.tex', make_SI(Kanal_Comptonkante, r'', figures=1))
E_Compton = Energiefkt(Kanal_Comptonkante)
write('build/E_Comptonkante.tex', make_SI(E_Compton, r'\kilo\electronvolt', figures=1))



c=299792458 #m/s
m_e = 511 #keV
E_gamma = 660 #keV
eps=E_gamma/(m_e)
E_Compton_theo = (2*E_gamma*eps)/(1+2*eps)
write('build/E_Comptonkante_theo.tex', make_SI(E_Compton_theo, r'\kilo\electronvolt', figures=1))
write('build/E_Comptonkante_prozent.tex', make_SI(100-E_Compton/E_Compton_theo*100, r'\percent', figures=1))
# rückstreupeak
Kanal_Compton_ruck = ufloat(637,15)
write('build/Kanal_Compton_ruck.tex', make_SI(Kanal_Compton_ruck, r'', figures=1))

E_Compton_rück_mess = Energiefkt(Kanal_Compton_ruck)
write('build/Compton_ruck_direkt.tex', make_SI(E_Compton_rück_mess, r'\kilo\electronvolt', figures=1))
E_gamma_strich = E_gamma - E_Compton
E_gamma_strich_theo = E_gamma - E_Compton_theo
eps_strich=E_gamma_strich/(m_e)
eps_strich_theo=E_gamma_strich_theo/(m_e)
E_Compton_rück = (2*E_gamma_strich)/(1+2*eps_strich)
write('build/Compton_ruck_indirekt.tex', make_SI(E_Compton_rück, r'\kilo\electronvolt', figures=1))
E_Compton_theo_rück = (2*E_gamma_strich_theo)/(1+2*eps_strich_theo)
write('build/Compton_ruck_theo.tex', make_SI(E_Compton_theo_rück, r'\kilo\electronvolt', figures=1))
write('build/Compton_ruck_abw.tex', make_SI(100-(E_Compton_rück_mess/E_Compton_theo_rück)*100, r'\percent', figures=1))
print('moin2')
print(E_Compton)
print(E_Compton_theo)
print('Rückstreupeak exp (abgeleitet aus experimenteller compton kante): ' +str(E_Compton_rück))
print('Rückstreupeak theo: ' + str(E_Compton_theo_rück))
print('Rückstreupeak mess (abgelesen): ' + str(E_Compton_rück_mess))
write('build/Compton_ruck_abw_2.tex', make_SI(100-(E_Compton_rück/E_Compton_theo_rück)*100, r'\percent', figures=1))
Mittelwert_Compton_ruck = (E_Compton_rück + E_Compton_rück_mess)/2
write('build/Mittelwert_Compton_ruck.tex', make_SI(Mittelwert_Compton_ruck, r'\kilo\electronvolt', figures=1))
write('build/Compton_ruck_abw_mittel.tex', make_SI(100-(Mittelwert_Compton_ruck/E_Compton_theo_rück)*100, r'\percent', figures=1))
Z_Comptonkontinuum= sum(Teilchen_pro_Kanal[Kanal[51:1558]]/Messzeit_2)
write('build/Z_Comptonkontinuum.tex', make_SI(Z_Comptonkontinuum, r'\per\second', figures=1))


# #-----------------Absorptionswahrscheinlichkeit
Q_Photo = (1-np.exp(-0.008*3.9))*100
write('build/Q_Photo.tex', make_SI((1-np.exp(-0.008*3.9))*100, r'\percent', figures=1))
Q_Compton = (1-np.exp(-0.38*3.9))*100
write('build/Q_Compton.tex', make_SI((1-np.exp(-0.38*3.9))*100, r'\percent', figures=1))

write('build/Q_ver.tex', make_SI(Q_Compton/Q_Photo, r'', figures=1))
b=Q_Compton/Q_Photo
write('build/Z_ver.tex', make_SI(Z_Comptonkontinuum/Z_Photopeak, r'', figures=1))
a=Z_Comptonkontinuum/Z_Photopeak
write('build/Z_Q_wahr.tex', make_SI(100-(a/b*100), r'\percent', figures=1))

# #-------------- d) Barium-Messung
Messzeit_3 = 2203 #sekunden
Energie,EmW = np.genfromtxt('messdaten/Teil_d_Energien_Wahr.txt', unpack=True)
Teilchen_pro_Kanal = np.genfromtxt('messdaten/3.txt', unpack=True)
plt.plot(Kanal, Teilchen_pro_Kanal[Kanal]/Messzeit_3,'r', linewidth=0.5)
plt.xlabel(r'$\text{Kanalnummer}$')
plt.ylabel(r'$\text{Zählrate} \:/\: \si{\per\second}$')
plt.legend(loc='best')
plt.savefig('build/Messdaten_Teil_4_Rohdaten.pdf')
plt.clf()

# plt.plot(Kanal[267:297], Teilchen_pro_Kanal[Kanal[267:297]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_4_peak1.pdf')
# plt.clf()
Inhalt_peak1 = sum(Teilchen_pro_Kanal[Kanal[270:282]])

# plt.plot(Kanal[907:957], Teilchen_pro_Kanal[Kanal[907:957]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_4_peak2.pdf')
# plt.clf()
Inhalt_peak2 = sum(Teilchen_pro_Kanal[Kanal[927:937]])

# plt.plot(Kanal[997:1037], Teilchen_pro_Kanal[Kanal[997:1037]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_4_peak3.pdf')
# plt.clf()
Inhalt_peak3 = sum(Teilchen_pro_Kanal[Kanal[1013:1027]])

# plt.plot(Kanal[1177:1217], Teilchen_pro_Kanal[Kanal[1177:1217]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_4_peak4.pdf')
# plt.clf()
Inhalt_peak4 = sum(Teilchen_pro_Kanal[Kanal[1187:1204]])

# plt.plot(Kanal[1267:1317], Teilchen_pro_Kanal[Kanal[1267:1317]],'r', linewidth=0.5, label='Messdaten')
# plt.xlabel(r'$Kanalnummer$')
# plt.ylabel(r'$Anzahl der Detektionen$')
# plt.legend(loc='best')
# plt.savefig('build/Messdaten_Teil_4_peak5.pdf')
# plt.clf()
Inhalt_peak5 = sum(Teilchen_pro_Kanal[Kanal[1281:1300]])

Z_4=np.array([Inhalt_peak1,Inhalt_peak2,Inhalt_peak3,Inhalt_peak4,Inhalt_peak5])

Z_4=Z_4/Messzeit_3

Peaksx=np.array([278,931,1020,1198,1291])

Q=Eff(Energie)

write('build/Tabelle_Effizienz_d.tex', make_table([Peaksx,Energie,Z_4,EmW,Q],[0, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_Effizienz_d_texformat.tex', make_full_table(
    caption = 'Werte zur Berechnung der Aktivität der \ce{^{133}_{}Ba-Quelle}.',
    label = 'table:Effizienz_d',
    source_table = 'build/Tabelle_Effizienz_d.tex',
    stacking = [4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
    units = [
    r'$\text{Kanalnummer}$',
    r'$\text{Energie} \:/\: \si{\kilo\electronvolt}$',
    r'$Z  \:/\: \si{\per\second}$',
    r'$W \:/\: \si{\percent}$',
    r'$Q$'],
    replaceNaN = True,                      # default = false
    replaceNaNby = 'not a number'))         # default = '-'


Akt_Barium_Mittelw = sum(Z_4[0:4]/(Raumw*(EmW[0:4]/100)*Q[0:4]))/5
print(Akt_Barium_Mittelw)
write('build/Akt_Barium_Mittelw.tex', make_SI(Akt_Barium_Mittelw, r'\becquerel', figures=1))

# #-------------------- e)
Messzeit_Stein =3048 #sekunden
Peaksx=np.array([265,317,629,816,994,1184,2045,3756,5911])
Teilchen_pro_Kanal = np.genfromtxt('messdaten/Stein.txt', unpack=True)
plt.plot(Kanal, Teilchen_pro_Kanal[Kanal]/Messzeit_Stein,'r', linewidth=0.5, label='Messdaten')
plt.plot(Peaksx, Teilchen_pro_Kanal[Peaksx]/Messzeit_Stein,'bx', linewidth=0.5, label='verwendete Peaks')
plt.xlabel(r'$\text{Kanalnummer}$')
plt.ylabel(r'$\text{Zählrate} \:/\: \si{\per\second}$')
plt.legend(loc='best')
plt.savefig('build/Messdaten_Stein_Rohdaten.pdf')
plt.clf()

E_Stein= Energiefkt(Peaksx)
print(E_Stein)

Energie_Uran,EmW_Uran = np.genfromtxt('messdaten/Uran_zerfallreihe_daten.txt', unpack=True)

write('build/Tabelle_Vergleich_Th.tex', make_table([Energie_Uran[0:2],EmW_Uran[0:2],E_Stein[0:2],(100-(E_Stein[0:2]/Energie_Uran[0:2])*100)*(-1)],[0, 2, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_Vergleich_Th_texformat.tex', make_full_table(
    caption = 'Vergleich mit  \ce{^{234}_{}Th} \cite{skript}.',
    label = 'table:Vegleich_Th',
    source_table = 'build/Tabelle_Vergleich_Th.tex',
    stacking = [2,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
    units = [
    r'$E_{Th} \:/\: \si{\kilo\electronvolt}$',
    r'$W_{Th} \:/\: \si{\percent}$',
    r'$E_{Stein}  \:/\: \si{\kilo\electronvolt}$',
    r'$Abweichung \:/\: \si{\percent}$'],
    replaceNaN = True,                      # default = false
    replaceNaNby = 'not a number'))         # default = '-'

Energie_Uran_extra =np.array([Energie_Uran[2]])
EmW_Uran_extra =np.array([EmW_Uran[2]])
# #E_Stein_extra =unp.uarray(noms(E_Stein[2]),stds(E_Stein[2]))
# #E_Stein_extra = E_Stein[2]
E_Stein_extra = np.array([noms(E_Stein[2])])
E_Stein_extra_stds = np.array([stds(E_Stein[2])])
E_Stein_wahr_n = np.array([noms(a)])
E_Stein_wahr_s = np.array([stds(a)])
a=(100-((E_Stein_extra/Energie_Uran_extra)*100))
write('build/Tabelle_Vergleich_Ra.tex', make_table([Energie_Uran_extra,EmW_Uran_extra,E_Stein_extra,E_Stein_extra_stds,E_Stein_wahr_n,E_Stein_wahr_s],[0, 2, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_Vergleich_Ra_texformat.tex', make_full_table(
    caption = 'Vergleich mit  \ce{^{266}_{}Ra} \cite{skript}.',
    label = 'table:Vegleich_Ra',
    source_table = 'build/Tabelle_Vergleich_Ra.tex',
    stacking = [2,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
    units = [
    r'$E_{Ra} \:/\: \si{\kilo\electronvolt}$',
    r'$W_{Ra} \:/\: \si{\percent}$',
    r'$E_{Stein}  \:/\: \si{\kilo\electronvolt}$',
    r'$Abweichung \:/\: \si{\percent}$'],
    replaceNaN = True,                      # default = false
    replaceNaNby = 'not a number'))         # default = '-'

write('build/Tabelle_Vergleich_Pb.tex', make_table([Energie_Uran[3:6],EmW_Uran[3:6],E_Stein[3:6],(100-(E_Stein[3:6]/Energie_Uran[3:6])*100)*(-1)],[0, 2, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_Vergleich_Pb_texformat.tex', make_full_table(
    caption = 'Vergleich mit  \ce{^{214}_{}Pb} \cite{skript}.',
    label = 'table:Vegleich_Pb',
    source_table = 'build/Tabelle_Vergleich_Pb.tex',
    stacking = [2,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
    units = [
    r'$E_{Pb} \:/\: \si{\kilo\electronvolt}$',
    r'$W_{Pb} \:/\: \si{\percent}$',
    r'$E_{Stein}  \:/\: \si{\kilo\electronvolt}$',
    r'$Abweichung \:/\: \si{\percent}$'],
    replaceNaN = True,                      # default = false
    replaceNaNby = 'not a number'))         # default = '-'

write('build/Tabelle_Vergleich_Bi.tex', make_table([Energie_Uran[6:9],EmW_Uran[6:9],E_Stein[6:9],(100-(E_Stein[6:9]/Energie_Uran[6:9])*100)*(-1)],[0, 2, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_Vergleich_Bi_texformat.tex', make_full_table(
    caption = 'Vergleich mit  \ce{^{214}_{}Bi} \cite{skript}.',
    label = 'table:Vegleich_Bi',
    source_table = 'build/Tabelle_Vergleich_Bi.tex',
    stacking = [2,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern, die Multicolumns sein sollen
    units = [
    r'$E_{Bi} \:/\: \si{\kilo\electronvolt}$',
    r'$W_{Bi} \:/\: \si{\percent}$',
    r'$E_{Stein}  \:/\: \si{\kilo\electronvolt}$',
    r'$Abweichung \:/\: \si{\percent}$'],
    replaceNaN = True,                      # default = false
    replaceNaNby = 'not a number'))         # default = '-'

# a=[b[0] for b in sorted(enumerate(Teilchen_pro_Kanal),key=lambda i:i[1])]
# print(a[8191],Teilchen_pro_Kanal[a[8191]])

# a=[1,4,6,2,3]
# c=[b[0] for b in sorted(enumerate(a),key=lambda i:i[1])]
# print(c)






################################ FREQUENTLY USED CODE ################################
#
########## IMPORT ##########
# t, U, U_err = np.genfromtxt('messdaten/data.txt', unpack=True)
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
# plt.clf()                   # clear actual plot before generating a new one
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
