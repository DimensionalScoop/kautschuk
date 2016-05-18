import os

import numpy as np
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit

import data as d
import plot_helpers as plot
import functions as f
import copy

from table import (
    make_table,
    make_SI,
    write,
)

def gimmeTHATcolumn(array,k):
    """Extracts the k-column of an 2D-array, returns list with those column-elements"""
    helparray = []
    for i in range(len(array)):
        helparray.append(array[i][k])
    return helparray


def saturation_current(array,k):
    helparray1 = gimmeTHATcolumn(array,k)
    maxi = max(helparray1)
    return maxi


saturation_current_1 = saturation_current(d.anode_1,1)
saturation_current_2 = saturation_current(d.anode_2,1)
saturation_current_3 = saturation_current(d.anode_3,1)
saturation_current_4 = saturation_current(d.anode_4,1)
saturation_current_5 = saturation_current(d.anode_5,1)

saturation_current = []
saturation_current.append(saturation_current_1)
saturation_current.append(saturation_current_2)
saturation_current.append(saturation_current_3)
saturation_current.append(saturation_current_4)
saturation_current.append(saturation_current_5)

saturation_tab = copy.deepcopy(saturation_current)
d.make_it_SI2(saturation_tab,6)
#write('../tex-data/saturation.tex',
#      make_table([[1,2,3,4,5],saturation_current], [0, 6]))
write('../tex-data/saturation.tex',
      make_table([[1,2,3,4,5],saturation_tab], [0, 0]))

print("Sättigungsstrom 1:",saturation_current_1)
print("Sättigungsstrom 2:",saturation_current_2)
print("Sättigungsstrom 3:",saturation_current_3)
print("Sättigungsstrom 4:",saturation_current_4)
print("Sättigungsstrom 5:",saturation_current_5)


#calculationg the Langmuir-Schottkysche exponent - should be 1.5

def langmuh_Reg(V, a, b):
    return a*V + b

def make_it_ln(array, k):
    """takes the logarithm"""
    for i in range(len(array)):
        array[i][k] = np.log(array[i][k])
def make_it_ln2(array):
    """takes the logarithm"""
    for i in range(len(array)):
        array[i] = np.log(array[i])


langmuh_anode = make_it_ln(d.anode_langmuh,0)
langmuh_anode = make_it_ln(d.anode_langmuh,1)
langmuh_volt = gimmeTHATcolumn(d.anode_langmuh,0)
langmuh_current = gimmeTHATcolumn(d.anode_langmuh,1)


#linear Curvefit with a = is LS-exponent
raumladung = f.autofit(langmuh_volt,langmuh_current, langmuh_Reg)

print("RAUMLADUNG:",raumladung)
print("Abweichung:", f.abweichung(raumladung[0], 1.5))


#calculating temperature
#print("CURRENT", d.current)
R = 1*10**6
#print(R)
def volt_correction(array):
    U_real = []
    for i in range(len(array)):
        U_real.append(array[i][0] - (array[i][1] * R))
    return U_real

U_real = volt_correction(d.current)
ln_IA = copy.deepcopy(gimmeTHATcolumn(d.current,1))
make_it_ln2(ln_IA)
U_messung = copy.deepcopy(gimmeTHATcolumn(d.current,0))
I_Messung = copy.deepcopy(gimmeTHATcolumn(d.current,1))
d.make_it_SI2(I_Messung,9)

print("CURRENT2", I_Messung)

write('../tex-data/Ureal.tex',
      make_table([I_Messung,U_messung, U_real], [2, 2, 4])) #I in nA
#print(d.current)
#print("log I_A", ln_IA)
#print("U_real",U_real)

#linear Curvefit for temperature
exponent = f.autofit(U_real,ln_IA, f.linearFit)
print("EXPONENTFIT =", exponent)
e_0 = 1.602176e-19
k_B = 1.38064852e-23
#k_B = 8.6173303e-5
#print(e_0,k_B)


Temp = e_0 / (k_B * exponent[0])
print("TEMPERATUR:",Temp)

#Kathodentemperatur aus Kennlinien
I_f = np.array([2.0,1.9,1.8,1.7,1.6])
V_f = np.array([ufloat(4.5,0.1),ufloat(4.5,0.1),ufloat(4.0,0.1),ufloat(3.5,0.1),ufloat(3.0,0.1)])

write('../tex-data/heiz.tex',
      make_table([[1,2,3,4,5],V_f,I_f], [0,1, 1]))

#print(I_f,V_f)

f_diode2 = 0.35e-4 #m²
eta = 0.28
N_WL = 0.95 #Watt
sigma_strahlung = 5.7e-8 #W/m²K⁴

def TempKath(V,I):
    Temp_kath = []
    for i in range(len(V)):
        Temp_kath.append(((V[i]*I[i] - N_WL)/(f_diode2*eta*sigma_strahlung))**(1/4))
    return Temp_kath

#def TempKath2(V,I):
#    Temp_kath2 = []
#    for i in range(len(V)):
#        Temp_kath2.append(unp.sqrt(unp.sqrt(((V[i]*I[i] - #N_WL)/(f_diode2*eta*sigma_strahlung)))))
#    return Temp_kath2

temperature_kathode = TempKath(V_f,I_f)
write('../tex-data/temp.tex',
      make_table([[1,2,3,4,5],temperature_kathode], [0,2]))
#temperature_kathode2 = TempKath2(V_f,I_f)
print("TEMP KATHODE:", Temp - temperature_kathode[4])
#print("TEMP KATHODE2:", temperature_kathode2)

#Austrittsarbeit von Wolfram
h = 6.626070040e-34
m_0 = 9.10938356e-31

def richardson(T,I_S):
    arbeit = []
    for i in range(len(T)):
        arbeit.append(k_B*T[i]* unp.log((4*np.pi*e_0*m_0*f_diode2*(k_B**2) * (T[i]**2))/((h**3)*I_S[i])))
    return arbeit




Austrittsarbeit = richardson(temperature_kathode, saturation_current)
print("Austrittsarbeit", Austrittsarbeit)
arbeitswert,arbeit_err = plot.extract_error(Austrittsarbeit)
print("Mittelwert:",f.mean(arbeitswert), f.abweichung(ufloat(f.mean(arbeitswert),f.stdDevOfMean(arbeitswert)), 4.5e-19))

fehler_arbeit = f.stdDevOfMean(arbeitswert)*10**19
mittel = f.mean(arbeitswert)*10**19
arbeityeah = [ufloat(mittel,fehler_arbeit)]
arbeitbla = copy.deepcopy(Austrittsarbeit)
d.make_it_SI2(arbeitbla,19)



print("ARBEIT:", arbeityeah)

write('../tex-data/arbeit.tex',
      make_table([[1,2,3,4,5],arbeitbla], [0,2]))

def appendstuff(q1,q2):
    quabla = []
    for i in range(len(q1)):
        quabla.append(q1[i])
    for i in range(len(q2)):
        quabla.append(q2[i])
#    print("MAYBE HERE?!", len(quabla))
    return quabla
