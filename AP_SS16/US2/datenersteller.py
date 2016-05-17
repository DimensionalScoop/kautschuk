# Überführt Messdaten in auslesbare Textdaten (optional)
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

##############A-Scan###################

t_start = 1.6

D_o = np.array([5.5, 0.65, 1.45, 2.265, 3.08, 3.885, 4.625, 5.395, 6.13, 1.76, 1.94])
t_o = np.array([41.3, 5.9, 11.6, 17.5, 23.3, 29.2, 34.6, 40, 45.6, 13.8, 15.1])
t_u = np.array([11.8, 52.6, 46.7, 40.8, 34.9, 29.2, 22.9, 16.7, 10.4, 45.6, 44.4])

t_o = t_o-t_start
t_u = t_u-t_start


np.savetxt('messdaten/a.txt', np.column_stack([D_o, t_o, t_u]), header="D_oben [cm], t_oben[µs], t_unten[µs]")

#############B-Scan#################

t_start = 5.5*0.5

t_o = np.array([40+2*0.5, 5+0.5*0.5, 10+1, 15+0.5*3.5, 20+0.5*5, 25+0.5*7.5, 30+0.5*8, 35+0.5*9, 45, 10+0.5*6, 10+0.5*9.5])
t_u = np.array([10+3*0.5, 50+5*0.5, 45+0.5*2.5, 40+0.5, 30+0.5*9, 25+0.5*7.5, 20+0.5*4.5, 15+0.5*2.5, 10, 45, 40+7*0.5])

t_o = t_o-t_start
t_u = t_u-t_start

np.savetxt('messdaten/b.txt', np.column_stack([t_o, t_u]), header="t_oben[µs], t_unten[µs]")

#############Kack-Herz################

ESD = np.array([75, 77, 77 ,77 ,77]) # µs
th  = np.array([10.5/5, 10.2/5, 10.2/5, 10.3/5]) #s
np.savetxt('messdaten/ESD.txt', np.column_stack([ESD]), header="ESD[µs]")
np.savetxt('messdaten/th.txt', np.column_stack([th]), header="Herzzeit[s]")
