# Überführt Messdaten in auslesbare Textdaten (optional)
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)
from table import (
    make_table,
    make_SI,
    write,
)
from uncertainties import ufloat

########## Aufgabenteil 0 (Temperaturen) ##########

T = np.array([26.1, 145.5, 161, 178, 106.6]) # Temperaturen in Celsius
np.savetxt('messdaten/0.txt', np.column_stack([T]), header="T [Celsius]")



########## Aufgabenteil 0 (Fucking Energieverteilung Bra) ##########

# Für T=26.1 C

del_U_a = 0.0394 #Ein Kasten in x-Richtung entspricht in etwa 0.0394V, wähle das als Delta
U_a = np.array([0*0.394, 3*0.394, 6*0.394, 9*0.394, 12*0.394, 15*0.394, 18*0.394 , 19*0.394, 20*0.394,  21*0.394, 24*0.394]) # Messe für jedes 3. Kasten a 0.394 Volt

I_a = np.array([1000-0*6.67, 1000-12*6.67, 1000-23*6.67, 1000-35*6.67, 1000-48*6.67, 1000-67*6.67, 1000-91*6.67, 1000-103*6.67, 1000-122*6.67,  1000-144*6.67, 0]) # Stromwert an i-ter Stelle
I_a_plus_delta = np.array([4*6.67, 4*6.67, 4*6.67, 4.5*6.67, 5.5*6.67, 7*6.67, 11*6.67,  12.5*6.67, 23*6.67, 4.5*6.67, 0      ]) # Stromwert an i-ter Stelle plus ein delta_U_a

# ERLÄUTERUNG: U_a ist das Array mit den Spannungen, an denen gemessen wurde
#               I_a ist der jeweilige Stromwert dort (brauchte ich eigentlich garnicht)
#               I_a_plus_delta ist der Stromwert jeweils ein Delta weiter
#               (Die komischen Multiplikationen folgern aus der Ablesetechnik)

np.savetxt('messdaten/a_1.txt', np.column_stack([U_a, I_a, I_a_plus_delta]), header="U_a [Volt], I_a [nA], I_a_plus_delta [nA]")


# Für T=145.5 C

U_a_2 = np.array([0*0.292, 2*0.292, 4*0.292, 6*0.292, 8*0.292, 9*0.292, 10*0.292, 11*0.292, 12*0.292, 13*0.292, 15*0.292, 17*0.292, 19*0.292, 21*0.292,])
del_U_a_2 = 0.292 # Delta U_a sei 1 Kasten, was 0.292 Volt entspricht
delta_I_a_2 = np.array([17.5*0.0503,16*0.0503, 14*0.0503, 13*0.0503, 10.5*0.0503, 9.5*0.0503, 7*0.0503, 5*0.0503, 3*0.0503, 1*0.0503, 0, 0, 0,0 ])

np.savetxt('messdaten/a_2.txt', np.column_stack([U_a_2, delta_I_a_2]), header="U_a [Volt], I_a [nA]")


########## Aufgabenteil b ##########

# Lage der Maxima
U_max_1 = ([52*0.2313, 74.5*0.2313, 96.5*0.2313, 113.5*0.2313]) # für 161 C
U_max_2 = ([53*0.2332, 74*0.2332, 96.5*0.2332, 119*0.2332]) # für 178 C

np.savetxt('messdaten/b.txt', np.column_stack([U_max_1, U_max_2]), header="U_max_1 [Volt], U_max_2 [Volt]")
