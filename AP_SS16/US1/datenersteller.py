import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

 # use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"python_custom_scripts")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

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

############Echo-Methode#########

h_zylinder_messung = np.array([61.5, 80.55, 102.1, 120.5, 31.1+61.5, 31.3+80.55])
t_zylinder_messung = np.array([44.9, 58.3, 75.0, 87.4, 67.8, 81.1])
np.savetxt('messdaten/a.txt', np.column_stack([h_zylinder_messung, t_zylinder_messung]), header="h [mm], t[µs]")

U_2 = 1.105
U_1 = 1.214
t_1 = 1.3
t_2 = 46.2
write('messdaten/U_1.tex', make_SI(U_1, r'\volt', figures=3))
write('messdaten/U_2.tex', make_SI(U_2, r'\volt', figures=3))
write('messdaten/t_1.tex', make_SI(t_1, r'\micro\second', figures=1))
write('messdaten/t_2.tex', make_SI(t_2, r'\micro\second', figures=1))


###########Durchschallungs-Methode###############

h_zylinder_messung = np.array([31.3,61.5, 80.55,])
t_zylinder_messung = np.array([23.1, 45.6, 60.9])
np.savetxt('messdaten/b.txt', np.column_stack([h_zylinder_messung, t_zylinder_messung]), header="h [mm], t[µs]")

###Auge###

a_1 = 20*(0.3/7)
a_2 = 16.8
a_3 = 23.7
a_4 = 20*(4.4/7)+20
a_5 = 20*(6.7/7)+20
a_6 = 20*(4.7/7)+60
a_messung = np.array([a_1, a_2, a_3, a_4, a_5, a_6])
np.savetxt('messdaten/auge.txt', np.column_stack([a_messung]), header="t [µs]")

#### FFT ####

fft_fre = np.array([1.27, 1.36, 1.44, 1.52, 1.6, 1.69])
np.savetxt('messdaten/fft.txt', np.column_stack([fft_fre]), header="f [MHz]")
