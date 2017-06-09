##################################################### Import system libraries ######################################################
import matplotlib as mpl
mpl.rcdefaults()
mpl.rcParams.update(mpl.rc_params_from_file('meine-matplotlibrc'))
mpl.rcParams['axes.linewidth'] = 0.1 #set the value globally
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.stats
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
from PeakDetect import(
    peakdetect
)
################################################ Finish importing custom libraries #################################################

def plotSpectrum(frequency, amplitude, filename, color, legend, ylimit = 0, aspect = 0.03):
    if (legend==''):
        plt.plot(frequency*1e-3, amplitude, color, lw=0.3)
    else:
        plt.plot(frequency*1e-3, amplitude, color, lw=0.3, label=legend)
        plt.legend(loc='best')
    plt.axes().set_aspect(aspect, 'box')
    if (ylimit!=0):
        plt.ylim(0,ylimit)
    plt.xlabel(r'$f \:/\: \si{\kilo\hertz}$')
    plt.ylabel(r'Amplitude$ \:/\: $a.u.')
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    plt.savefig('build/'+filename,bbox_inches='tight')

def LorentzPeak(omega, f0_values = [], amplitude_values = [], width_values = []):
    result = 0
    for f0,amplitude,width in zip(f0_values, amplitude_values, width_values):
        lamb = width/(2*np.sqrt(3))
        result += amplitude*lamb/np.sqrt((omega-2*np.pi*f0)**2 + lamb**2)
    return result

def calc_delta_f(filename):
    fn = np.genfromtxt(filename, unpack=True)
    delta_f = np.zeros(fn.size-1)
    for i in range(fn.size):
        if (i!=0):
            delta_f[i-1] = fn[i]-fn[i-1]
    if (delta_f.size > 1):
        return ufloat(np.mean(noms(delta_f)), scipy.stats.sem(noms(delta_f)))
    else:
        return ufloat(delta_f,0)

def fillUpListUntilLength (array, length ):
    result = array
    for i in range(len(array), length):
        result = np.append(result,'999')
    return result

########## Kapitel 1 ###########
f, I = np.genfromtxt('messdaten/1--1.dat', unpack=True)
plotSpectrum(f,I,'1_overview.pdf', 'r', '')

plt.clf()
f, I = np.genfromtxt('messdaten/1--2.dat', unpack=True)
plotSpectrum(f,I,'1_2.pdf','r', '2x75mm Tubes, Messdaten')

#Fit 1 from SLC software
f0_values, amplitude_values, width_values, phase = np.genfromtxt('messdaten/1--2Fitparameter_parsed.par', unpack=True)
write('build/1_2_fitParams.tex', make_table([f0_values,amplitude_values, width_values, phase],[0, 0, 0, 3]))
write('build/1_2_fit1.tex', make_full_table(
    caption = 'Fitparameter Fit 1.',
    label = 'table:1_2_fit1',
    source_table = 'build/1_2_fitParams.tex',
    stacking = [],
    units = [r'$f_0 \:/\: \si{\hertz}$',
    r'Amplitude$ \:/\: $a.u.',
    r'Width$ \:/\: \si{\hertz}$',
    r'Phase$ \:/\: \si{\degree}$']))
#Fit2
f0_values, amplitude_values, width_values, phase = np.genfromtxt('messdaten/1--2Fitparameter2_parsed.par', unpack=True)
write('build/1_2_fitParams2.tex', make_table([f0_values,amplitude_values, width_values, phase],[0, 0, 0, 3]))
write('build/1_2_fit2.tex', make_full_table(
    caption = 'Fitparameter Fit 2.',
    label = 'table:1_2_fit2',
    source_table = 'build/1_2_fitParams2.tex',
    stacking = [],
    units = [r'$f_0 \:/\: \si{\hertz}$',
    r'Amplitude$ \:/\: $a.u.',
    r'Width$ \:/\: \si{\hertz}$',
    r'Phase$ \:/\: \si{\degree}$']))

######### Kapitel 2 ###########
plt.clf()
f, I = np.genfromtxt('messdaten/2_1_00.dat', unpack=True)
plotSpectrum(f,I,'2_1_00.pdf', 'r', '', 60)
plt.clf()
f, I = np.genfromtxt('messdaten/2_1_30.dat', unpack=True)
plotSpectrum(f,I,'2_1_30.pdf', 'r', '', 60)
plt.clf()
f, I = np.genfromtxt('messdaten/2_1_60.dat', unpack=True)
plotSpectrum(f,I,'2_1_60.pdf', 'r', '', 60)
plt.clf()
f, I = np.genfromtxt('messdaten/2_1_90.dat', unpack=True)
plotSpectrum(f,I,'2_1_90.pdf', 'r', '', 60)
plt.clf()
f, I = np.genfromtxt('messdaten/2_1_120.dat', unpack=True)
plotSpectrum(f,I,'2_1_120.pdf', 'r', '', 60)
plt.clf()
f, I = np.genfromtxt('messdaten/2_1_150.dat', unpack=True)
plotSpectrum(f,I,'2_1_150.pdf', 'r', '', 60)
plt.clf()
f, I = np.genfromtxt('messdaten/2_1_180.dat', unpack=True)
plotSpectrum(f,I,'2_1_180.pdf', 'r', '')
plt.clf()

f, I = np.genfromtxt('messdaten/2_2_00.dat', unpack=True)
plt.plot(f, I, 'r', lw=0.3, label=r'$\alpha=$0°')
f, I = np.genfromtxt('messdaten/2_2_20.dat', unpack=True)
plt.plot(f, I, 'g', lw=0.3, label=r'$\alpha=$20°')
f, I = np.genfromtxt('messdaten/2_2_40.dat', unpack=True)
plt.plot(f, I, 'b', lw=0.3, label=r'$\alpha=$40°')
plt.xlabel(r'$f \:/\: \si{\hertz}$')
plt.ylabel(r'Amplitude$ \:/\: $a.u.')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/2_2.pdf')

plt.clf()
f, I = np.genfromtxt('messdaten/2_3_overview.dat', unpack=True)
plotSpectrum(f,I,'2_3_overview.pdf', 'r', '')

# Plots für die Schallgeschwindigkeit
plt.clf()
f, I = np.genfromtxt('messdaten/3_1_75mm.dat', unpack=True)
plotSpectrum(f,I,'3_1_75mm.pdf', 'r', '', aspect=0.012)
plt.clf()
f, I = np.genfromtxt('messdaten/3_1_150mm.dat', unpack=True)
plotSpectrum(f,I,'3_1_150mm.pdf', 'r', '', aspect=0.012)
plt.clf()
f, I = np.genfromtxt('messdaten/3_1_225mm.dat', unpack=True)
plotSpectrum(f,I,'3_1_225mm.pdf', 'r', '', aspect=0.012)
plt.clf()
f, I = np.genfromtxt('messdaten/3_1_300mm.dat', unpack=True)
plotSpectrum(f,I,'3_1_300mm.pdf', 'r', '', aspect=0.012)
plt.clf()
f, I = np.genfromtxt('messdaten/3_1_375mm.dat', unpack=True)
plotSpectrum(f,I,'3_1_375mm.pdf', 'r', '', aspect=0.012)
plt.clf()
f, I = np.genfromtxt('messdaten/3_1_450mm.dat', unpack=True)
plotSpectrum(f,I,'3_1_450mm.pdf', 'r', '', aspect=0.012)
plt.clf()
f, I = np.genfromtxt('messdaten/3_1_525mm.dat', unpack=True)
plotSpectrum(f,I,'3_1_525mm.pdf', 'r', '', aspect=0.012)
plt.clf()
f, I = np.genfromtxt('messdaten/3_1_600mm.dat', unpack=True)
plotSpectrum(f,I,'3_1_600mm.pdf', 'r', '', aspect=0.012)
plt.clf()

# Schallgeschwindigkeit
filenames = ['messdaten/3_1_Fitparameter_75mm.par',
'messdaten/3_1_Fitparameter_150mm.par',
'messdaten/3_1_Fitparameter_225mm.par',
'messdaten/3_1_Fitparameter_300mm.par',
'messdaten/3_1_Fitparameter_375mm.par',
'messdaten/3_1_Fitparameter_450mm.par',
'messdaten/3_1_Fitparameter_525mm.par',
'messdaten/3_1_Fitparameter_600mm.par']

delta_f_vector = unp.uarray(np.zeros(8),np.zeros(8))
i = 0
for filename in filenames:
    delta_f_vector[i] = calc_delta_f(filename)
    i+=1

laengen = np.array([75,150,225,300,375,450,525,600])

# Fit
params = ucurve_fit(reg_linear, 1/(laengen*1e-3), delta_f_vector)
a, b = params
write('build/parameter_a.tex', make_SI(a, r'\meter\per\second', figures=1))
write('build/parameter_b.tex', make_SI(b, r'\per\second', figures=1))
write('build/Schallgeschwindigkeit.tex', make_SI(a*2, r'\meter\per\second', figures=1))   # c = 2a
c_lit = 343.6
c_Schall = 2*a
RelFehler_c = (2*a - c_lit) / c_lit
write('build/RelFehler_c.tex', make_SI(RelFehler_c*100, r'\percent', figures=1))

# Tabelle mit L und Delta f
write('build/3_1_deltaF.tex', make_table([laengen, delta_f_vector],[0, 1]))
write('build/3_1_1.tex', make_full_table(
    caption = r'$\Delta f$ in Abhängigkeit der Rohrlänge.',
    label = 'table:3_1_1',
    source_table = 'build/3_1_deltaF.tex',
    stacking = [1],
    units = [r'$L \:/\: \si{\milli\meter}$',
    r'$\Delta f \:/\: \si{\hertz}$']))

# Plot des Fits + Messwerte
plt.clf ()
inverse_laengen = 1/(laengen*1e-3)
## automatically choosing limits
t_plot = np.linspace(np.amin(inverse_laengen), np.amax(inverse_laengen), 100)
plt.xlim(t_plot[0]-1/np.size(inverse_laengen)*(t_plot[-1]-t_plot[0]), t_plot[-1]+1/np.size(inverse_laengen)*(t_plot[-1]-t_plot[0]))
plt.plot(t_plot, reg_linear(t_plot, *noms(params)), 'b-', label='Fit')
plt.errorbar(inverse_laengen, noms(delta_f_vector), fmt='rx', yerr=stds(delta_f_vector), label='Messdaten')
plt.xlabel(r'$1/L \:/\: \si{\per\meter}$')
plt.ylabel(r'$\Delta f \:/\: \si{\hertz}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_fit.pdf')

# 3.2
plt.clf()
f, I = np.genfromtxt('messdaten/3_2_overview.dat', unpack=True)
[maxima, minima] = peakdetect(I, lookahead=10)
Maxima = [list(t) for t in zip(*maxima)]
frequencies = [f[i] for i in Maxima[0]]

plt.plot(f, I, 'r', lw=0.3, label='Messdaten')
plt.plot(frequencies, Maxima[1], 'bx', label='Maxima')
plt.xlabel(r'$f \:/\: \si{\hertz}$')
plt.ylabel(r'Amplitude$ \:/\: $a.u.')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_20.pdf')

f1,f2 = np.array_split(frequencies, 2)  # Array in zwei Hälften teilen
n = np.arange(1, len(frequencies)+1)
kn = n*np.pi/0.6     # k = n*pi/L
k1,k2 = np.array_split(kn, 2)
n1,n2 = np.array_split(n, 2)

write('build/3_2_table.tex', make_table([n1, f1, k1, n2, f2, k2],[0,0,1,0,0,1]))
write('build/3_2.tex', make_full_table(
    caption = r'$\Delta f$ in Abhängigkeit der Rohrlänge.',
    label = 'table:3_2',
    source_table = 'ressources/3_2_table.tex',
    stacking = [],
    units = [r'$n$',
    r'$f_n \:/\: \si{\hertz}$',
    r'$k_n \:/\: \si{\per\meter}$',
    r'$n$',
    r'$f_n \:/\: \si{\hertz}$',
    r'$k_n \:/\: \si{\per\meter}$']))

# Fit
params = ucurve_fit(reg_linear, kn, frequencies)
a, b = params
c_Schall = 2*np.pi*a
write('build/parameter_a2.tex', make_SI(a, r'\meter\per\second', figures=1))
write('build/parameter_b2.tex', make_SI(b, r'\per\second', figures=1))
write('build/Schallgeschwindigkeit2.tex', make_SI(c_Schall, r'\meter\per\second', figures=1))   # c = 2a
c_lit = 343.6
RelFehler_c = (c_Schall - c_lit) / c_lit
write('build/RelFehler_c2.tex', make_SI(RelFehler_c*100, r'\percent', figures=1))

t_plot = np.linspace(np.amin(kn), np.amax(kn), 10)

plt.clf()
plt.plot(kn, frequencies, 'rx', label='Messdaten')
plt.xlim(t_plot[0]-1/np.size(kn)*(t_plot[-1]-t_plot[0]), t_plot[-1]+1/np.size(kn)*(t_plot[-1]-t_plot[0]))
plt.plot(t_plot, reg_linear(t_plot, *noms(params)), 'b-', label='Fit')
plt.ylabel(r'$f \:/\: \si{\hertz}$')
plt.xlabel(r'$k$ \:/\: \si{\per\meter}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_21.pdf')

# 3.3
# 10mm
f, I = np.genfromtxt('messdaten/3_2_10mmIris.dat', unpack=True)
plt.clf()
[maxima10, minima10] = peakdetect(I, lookahead=4, delta=0.2)
Maxima10 = [list(t) for t in zip(*maxima10)]
frequencies10 = [f[i] for i in Maxima10[0]]
plt.plot(f, I, 'r', lw=0.3, label='Messdaten')
plt.plot(frequencies10, Maxima10[1], 'bx', label='Maxima')
plt.xlabel(r'$f \:/\: \si{\hertz}$')
plt.ylabel(r'Amplitude$ \:/\: $a.u.')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_30_10mm.pdf')

# 13mm
f, I = np.genfromtxt('messdaten/3_2_13mmIris.dat', unpack=True)
plt.clf()
[maxima13, minima13] = peakdetect(I, lookahead=4, delta=0.2)
Maxima13 = [list(t) for t in zip(*maxima13)]
frequencies13 = [f[i] for i in Maxima13[0]]
plt.plot(f, I, 'r', lw=0.3, label='Messdaten')
plt.plot(frequencies13, Maxima13[1], 'bx', label='Maxima')
plt.xlabel(r'$f \:/\: \si{\hertz}$')
plt.ylabel(r'Amplitude$ \:/\: $a.u.')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_30_13mm.pdf')

# 16mm
f, I = np.genfromtxt('messdaten/3_2_16mmIris.dat', unpack=True)
plt.clf()
[maxima16, minima16] = peakdetect(I, lookahead=4, delta=0.2)
Maxima16 = [list(t) for t in zip(*maxima16)]
frequencies16 = [f[i] for i in Maxima16[0]]
plt.plot(f, I, 'r', lw=0.3, label='Messdaten')
plt.plot(frequencies16, Maxima16[1], 'bx', label='Maxima')
plt.xlabel(r'$f \:/\: \si{\hertz}$')
plt.ylabel(r'Amplitude$ \:/\: $a.u.')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_30_16mm.pdf')

# manuelles Zerhacken- sorry

frequencies10_band1 = np.array([freq for freq in frequencies10 if freq < 2000])
frequencies10_band1 = np.delete(frequencies10_band1, 0)
Frequencies10_band1 = fillUpListUntilLength(frequencies10_band1, 7)
frequencies10_band2 = np.array([freq for freq in frequencies10 if freq > 2000 and freq < 5000])
Frequencies10_band2 = fillUpListUntilLength(frequencies10_band2, 8)
frequencies10_band3 = np.array([freq for freq in frequencies10 if freq < 8000 and freq > 5000])
Frequencies10_band3 = fillUpListUntilLength(frequencies10_band3, 8)
frequencies10_band4 = np.array([freq for freq in frequencies10 if freq > 8000])
Frequencies10_band4 = fillUpListUntilLength(frequencies10_band4, 8)

frequencies13_band1 = np.array([freq for freq in frequencies13 if freq < 3000])
Frequencies13_band1 = fillUpListUntilLength(frequencies13_band1, 7)
frequencies13_band2 = np.array([freq for freq in frequencies13 if freq > 3000 and freq < 5000])
Frequencies13_band2 = fillUpListUntilLength(frequencies13_band2, 8)
frequencies13_band3 = np.array([freq for freq in frequencies13 if freq < 8000 and freq > 5000])
Frequencies13_band3 = fillUpListUntilLength(frequencies13_band3, 8)
frequencies13_band4 = np.array([freq for freq in frequencies13 if freq > 8000])
Frequencies13_band4 = fillUpListUntilLength(frequencies13_band4, 8)

frequencies16_band1 = np.array([freq for freq in frequencies16 if freq < 3000])
Frequencies16_band1 = fillUpListUntilLength(frequencies16_band1, 7)
frequencies16_band2 = np.array([freq for freq in frequencies16 if freq > 3000 and freq < 5300])
Frequencies16_band2 = fillUpListUntilLength(frequencies16_band2, 8)
frequencies16_band3 = np.array([freq for freq in frequencies16 if freq < 8000 and freq > 5300])
Frequencies16_band3 = fillUpListUntilLength(frequencies16_band3, 8)
frequencies16_band4 = np.array([freq for freq in frequencies16 if freq > 8000])
Frequencies16_band4 = fillUpListUntilLength(frequencies16_band4, 8)

n_1 = np.arange(1, 8)
kn_1 = n_1*np.pi/0.4     # k = n*pi/L
n_2 = np.arange(8, 16)
kn_2 = n_2*np.pi/0.4     # k = n*pi/L
n_3 = np.arange(16, 24)
kn_3 = n_3*np.pi/0.4     # k = n*pi/L
n_4 = np.arange(24, 32)
kn_4 = n_4*np.pi/0.4     # k = n*pi/L

write('build/3_3_table_band1.tex', make_table([n_1, Frequencies10_band1, Frequencies13_band1, Frequencies16_band1, kn_1],[0,0,0,0,1]))
write('build/3_3_table_band2.tex', make_table([n_2, Frequencies10_band2, Frequencies13_band2, Frequencies16_band2, kn_2],[0,0,0,0,1]))
write('build/3_3_table_band3.tex', make_table([n_3, Frequencies10_band3, Frequencies13_band3, Frequencies16_band3, kn_3],[0,0,0,0,1]))
write('build/3_3_table_band4.tex', make_table([n_4, Frequencies10_band4, Frequencies13_band4, Frequencies16_band4, kn_4],[0,0,0,0,1]))
search_replace_within_file('build/3_3_table_band1.tex','999','')
search_replace_within_file('build/3_3_table_band2.tex','999','')
search_replace_within_file('build/3_3_table_band3.tex','999','')
search_replace_within_file('build/3_3_table_band4.tex','999','')

plt.clf()
plt.plot(kn_1, frequencies10_band1, 'bx', label=r'$\SI{10}{\milli\meter}$ Iris')
plt.plot(kn_2[:6], frequencies10_band2, 'bx')
plt.plot(kn_3[:5], frequencies10_band3, 'bx')
plt.plot(kn_4[:4], frequencies10_band4, 'bx')
plt.plot(kn_1, frequencies13_band1, 'r*', label=r'$\SI{13}{\milli\meter}$ Iris')
plt.plot(kn_2[:7], frequencies13_band2, 'r*')
plt.plot(kn_3[:6], frequencies13_band3, 'r*')
plt.plot(kn_4[:5], frequencies13_band4, 'r*')
plt.plot(kn_1, frequencies16_band1, 'g+', label=r'$\SI{16}{\milli\meter}$ Iris')
plt.plot(kn_2, frequencies16_band2, 'g+')
plt.plot(kn_3[:7], frequencies16_band3, 'g+')
plt.plot(kn_4[:7], frequencies16_band4, 'g+')

plt.legend(loc='best')
plt.ylabel(r'$f \:/\: \si{\hertz}$')
plt.xlabel(r'$k$ \:/\: \si{\per\meter}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_31.pdf')

# Zustandsdichte
def density(f):
    dens = np.zeros(len(f)-1)
    for i in range(0, len(f)-1):
        dens[i] = 1/(f[i+1]-f[i])
    return dens

# wir schrauben uns das manuell zusammen.. unschön, ja!
all_frequency_datapoints = np.array(frequencies16_band1)
rho = np.array([])

rho = np.append(rho, density(frequencies16_band1))
rho = np.append(rho, 0)

all_frequency_datapoints = np.append(all_frequency_datapoints, frequencies16_band2[0]-20)
rho = np.append(rho, 0)
all_frequency_datapoints = np.append(all_frequency_datapoints, frequencies16_band2)
rho = np.append(rho, density(frequencies16_band2))
rho = np.append(rho, 0)

all_frequency_datapoints = np.append(all_frequency_datapoints, frequencies16_band3[0]-20)
rho = np.append(rho, 0)
all_frequency_datapoints = np.append(all_frequency_datapoints, frequencies16_band3)
rho = np.append(rho, density(frequencies16_band3))
rho = np.append(rho, 0)

all_frequency_datapoints = np.append(all_frequency_datapoints, frequencies16_band4[0]-20)
rho = np.append(rho, 0)
all_frequency_datapoints = np.append(all_frequency_datapoints, frequencies16_band4)
rho = np.append(rho, density(frequencies16_band4))
rho = np.append(rho, 0)
all_frequency_datapoints = np.append(all_frequency_datapoints, 11500)
rho = np.append(rho, 0)

plt.clf()
plt.plot(all_frequency_datapoints, rho, 'rx')
plt.plot(all_frequency_datapoints, rho, 'b')
plt.ylabel(r'$\rho \:/\: \si{\second}$')
plt.xlabel(r'$f$ \:/\: \si{\hertz}$')
plt.ylim(0,0.012)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_32.pdf')

# 3.4
plt.clf()
f, I = np.genfromtxt('messdaten/3_2_16mmIris10x50.dat', unpack=True)
plotSpectrum(f,I,'3_40.pdf', 'r', '')
plt.clf()
f, I = np.genfromtxt('messdaten/3_2_16mmIris12x50.dat', unpack=True)
plotSpectrum(f,I,'3_41.pdf', 'r', '')
plt.clf()

# 3.5
plt.clf()
f, I = np.genfromtxt('messdaten/3_2_16mmIris8x75.dat', unpack=True)
plotSpectrum(f,I,'3_50.pdf', 'r', '')

# 3.6 + 3.7
plt.clf()

f, I = np.genfromtxt('messdaten/3_b6.dat', unpack=True)
[maxima50, minima50] = peakdetect(I, lookahead=10, delta=0.5)
Maxima50 = [list(t) for t in zip(*maxima50)]
# some manual adjustments
frequencies50 = [f[i] for i in Maxima50[0]]
frequencies50 = np.array(frequencies50)
frequencies50 = np.delete(frequencies50, [0,1,3])
Maxima50 = np.array(Maxima50[1])
Maxima50 = np.delete(Maxima50, [0,1,3])
# fin
plt.plot(f, I, 'r', lw=0.3, label=r'$\SI{50}{\milli\meter}$ Röhre')
plt.plot(frequencies50, Maxima50, 'rx')

f, I = np.genfromtxt('messdaten/3_b7.dat', unpack=True)
[maxima75, minima75] = peakdetect(I, lookahead=10, delta=0.5)
Maxima75 = [list(t) for t in zip(*maxima75)]
# some manual adjustments
frequencies75 = [f[i] for i in Maxima75[0]]
frequencies75 = np.array(frequencies75)
frequencies75 = np.delete(frequencies75, [0,3])
Maxima75 = np.array(Maxima75[1])
Maxima75 = np.delete(Maxima75, [0,3])
# fin
plt.plot(f, I, 'b', lw=0.3, label=r'$\SI{75}{\milli\meter}$ Röhre')
plt.plot(frequencies75, Maxima75, 'bx',)

plt.xlabel(r'$f \:/\: \si{\hertz}$')
plt.ylabel(r'Amplitude$ \:/\: $a.u.')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_60.pdf')

thefile = open('build/3_6_delta_f50.txt', 'w')
for f in frequencies50:
    thefile.write("%s\n" % f)
thefile.close()
thefile = open('build/3_6_delta_f75.txt', 'w')
for f in frequencies75:
    thefile.write("%s\n" % f)
thefile.close()
delta_f50 = calc_delta_f('build/3_6_delta_f50.txt')
delta_f75 = calc_delta_f('build/3_6_delta_f75.txt')
write('build/3_6_delta_f50.tex', make_SI(delta_f50, r'\hertz', figures=2))
write('build/3_6_delta_f75.tex', make_SI(delta_f75, r'\hertz', figures=1))

delta_f50_th = c_Schall/(2*0.05)
delta_f75_th = c_Schall/(2*0.075)
RelFehler_f50 = (delta_f50 - delta_f50_th) / delta_f50_th
RelFehler_f75 = (delta_f75 - delta_f75_th) / delta_f75_th
write('build/3_6_delta_f50_th.tex', make_SI(delta_f50_th, r'\hertz', figures=1))
write('build/3_6_delta_f75_th.tex', make_SI(delta_f75_th, r'\hertz', figures=1))
write('build/3_6_relError50.tex', make_SI(RelFehler_f50*100, r'\percent', figures=1))
write('build/3_6_relError75.tex', make_SI(RelFehler_f75*100, r'\percent', figures=1))

# 3.8
plt.clf()
f, I = np.genfromtxt('messdaten/3_b8_10mm.dat', unpack=True)
plotSpectrum(f,I,'3_80.pdf', 'r', r'$\SI{10}{\milli\meter}$ Iris', aspect=0.06)
f, I = np.genfromtxt('messdaten/3_b8_13mm.dat', unpack=True)
plotSpectrum(f,I,'3_80.pdf', 'b', r'$\SI{13}{\milli\meter}$ Iris', aspect=0.06)
f, I = np.genfromtxt('messdaten/3_b8_16mm.dat', unpack=True)
plotSpectrum(f,I,'3_80.pdf', 'g', r'$\SI{16}{\milli\meter}$ Iris', aspect=0.06)
plt.clf()
f, I = np.genfromtxt('messdaten/3_b8_vergleich.dat', unpack=True)
plotSpectrum(f,I,'3_81.pdf', 'r', '')

# 3.9
plt.clf()
f, I = np.genfromtxt('messdaten/3_b9_3x50_10mmIris.dat', unpack=True)
plotSpectrum(f,I,'3_90.pdf', 'r', r'$\SI{10}{\milli\meter}$ Iris', aspect=0.05)
f, I = np.genfromtxt('messdaten/3_b9_3x50_13mmIris.dat', unpack=True)
plotSpectrum(f,I,'3_90.pdf', 'b', r'$\SI{13}{\milli\meter}$ Iris', aspect=0.05)
f, I = np.genfromtxt('messdaten/3_b9_3x50_16mmIris.dat', unpack=True)
plotSpectrum(f,I,'3_90.pdf', 'g', r'$\SI{16}{\milli\meter}$ Iris', aspect=0.05)
plt.clf()
f, I = np.genfromtxt('messdaten/3_b9_4x50_10mmIris.dat', unpack=True)
plotSpectrum(f,I,'3_91.pdf', 'r', r'$\SI{10}{\milli\meter}$ Iris')
f, I = np.genfromtxt('messdaten/3_b9_4x50_13mmIris.dat', unpack=True)
plotSpectrum(f,I,'3_91.pdf', 'b', r'$\SI{13}{\milli\meter}$ Iris')
f, I = np.genfromtxt('messdaten/3_b9_4x50_16mmIris.dat', unpack=True)
plotSpectrum(f,I,'3_91.pdf', 'g', r'$\SI{16}{\milli\meter}$ Iris')
plt.clf()
f, I = np.genfromtxt('messdaten/3_b9_6x50_10mmIris.dat', unpack=True)
plotSpectrum(f,I,'3_92.pdf', 'r', r'$\SI{10}{\milli\meter}$ Iris')
f, I = np.genfromtxt('messdaten/3_b9_6x50_13mmIris.dat', unpack=True)
plotSpectrum(f,I,'3_92.pdf', 'b', r'$\SI{13}{\milli\meter}$ Iris')
f, I = np.genfromtxt('messdaten/3_b9_6x50_16mmIris.dat', unpack=True)
plotSpectrum(f,I,'3_92.pdf', 'g', r'$\SI{16}{\milli\meter}$ Iris')

# 3.10
plt.clf()
f, I = np.genfromtxt('messdaten/3_b10.dat', unpack=True)
plotSpectrum(f,I,'3_100.pdf', 'r', 'alternierend', aspect=0.06)
f, I = np.genfromtxt('messdaten/3_2_16mmIris12x50.dat', unpack=True)
plotSpectrum(f,I,'3_100.pdf', 'b', r'12x$\SI{50}{\milli\meter}$ Röhre', aspect=0.06)

# band structure... uargh
plt.clf()
# alternierende struktur
f, I = np.genfromtxt('messdaten/3_b10.dat', unpack=True)
[maxima, minima] = peakdetect(I, lookahead=4, delta=0.2)
Maxima = [list(t) for t in zip(*maxima)]
frequencies_1 = [f[i] for i in Maxima[0]]
# plt.plot(f, I, 'r', lw=0.3, label='Messdaten')
# plt.plot(frequencies_1, Maxima[1], 'bx', label='Maxima')
# plt.xlabel(r'$f \:/\: \si{\hertz}$')
# plt.ylabel(r'Amplitude$ \:/\: $a.u.')
# plt.legend(loc='best')
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/onlyForControl_3_101.pdf')
n = np.arange(1, len(frequencies_1)+1)
kn_1 = n*np.pi/0.6     # k = n*pi/L

# 12x50mm mit 16mm blenden
f, I = np.genfromtxt('messdaten/3_2_16mmIris12x50.dat', unpack=True)
[maxima, minima] = peakdetect(I, lookahead=4, delta=0.2)
Maxima = [list(t) for t in zip(*maxima)]
frequencies_2 = [f[i] for i in Maxima[0]]
# plt.clf()
# plt.plot(f, I, 'r', lw=0.3, label='Messdaten')
# plt.plot(frequencies_2, Maxima[1], 'bx', label='Maxima')
# plt.xlabel(r'$f \:/\: \si{\hertz}$')
# plt.ylabel(r'Amplitude$ \:/\: $a.u.')
# plt.legend(loc='best')
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/onlyForControl_3_102.pdf')
n = np.arange(1, len(frequencies_2)+1)
kn_2 = n*np.pi/0.6     # k = n*pi/L

plt.clf()
plt.plot(kn_1, frequencies_1, 'rx', label='alternierend')
plt.plot(kn_2, frequencies_2, 'b+', label=r'12x$\SI{50}{\milli\meter}$ Röhre')
plt.legend(loc='best')
plt.ylabel(r'$f \:/\: \si{\hertz}$')
plt.xlabel(r'$k$ \:/\: \si{\per\meter}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_101.pdf')

# 3.11
plt.clf()
f, I = np.genfromtxt('messdaten/3_b11.dat', unpack=True)
plotSpectrum(f,I,'3_110.pdf', 'r', 'alternierend', aspect=0.06)
f, I = np.genfromtxt('messdaten/3_b8_vergleich.dat', unpack=True)
plotSpectrum(f,I,'3_110.pdf', 'b', r'1x$\SI{50}{\milli\meter}$ Röhre', aspect=0.06)
f, I = np.genfromtxt('messdaten/3_b10_vergleich_75mm.dat', unpack=True)
plotSpectrum(f,I,'3_110.pdf', 'g', r'1x$\SI{75}{\milli\meter}$ Röhre', aspect=0.06)

f, I = np.genfromtxt('messdaten/3_b11.dat', unpack=True)
[maxima, minima] = peakdetect(I, lookahead=4, delta=0.2)
Maxima = [list(t) for t in zip(*maxima)]
frequencies_2 = [f[i] for i in Maxima[0]]
plt.clf()
plt.plot(f, I, 'r', lw=0.3, label='Messdaten')
plt.plot(frequencies_2, Maxima[1], 'bx', label='Maxima')
plt.xlabel(r'$f \:/\: \si{\hertz}$')
plt.ylabel(r'Amplitude$ \:/\: $a.u.')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/onlyForControl_3_102.pdf')
n = np.arange(1, len(frequencies_2)+1)
kn_2 = n*np.pi/0.6     # k = n*pi/L

plt.clf()
plt.plot(kn_2, frequencies_2, 'rx')
plt.legend(loc='best')
plt.ylabel(r'$f \:/\: \si{\hertz}$')
plt.xlabel(r'$k$ \:/\: \si{\per\meter}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_111.pdf')

# 3.12
# 12x50mm mit 16mm blenden
plt.clf()
f, I = np.genfromtxt('messdaten/3_2_16mmIris12x50.dat', unpack=True)
plotSpectrum(f,I,'3_120.pdf', 'r', r'12x$\SI{50}{\milli\meter}$ Röhre', aspect=0.06)
[maxima, minima] = peakdetect(I, lookahead=4, delta=0.2)
Maxima = [list(t) for t in zip(*maxima)]
frequencies_2 = [f[i] for i in Maxima[0]]
n = np.arange(1, len(frequencies_2)+1)
kn_2 = n*np.pi/0.6     # k = n*pi/L

# Defekt an Pos. 3
f, I = np.genfromtxt('messdaten/3_b12_pos3_75mmDefect.dat', unpack=True)
plotSpectrum(f,I,'3_120.pdf', 'b', 'Defekt an Pos. 3', aspect=0.06)
[maxima, minima] = peakdetect(I, lookahead=4, delta=0.2)
Maxima = [list(t) for t in zip(*maxima)]
frequencies_1 = [f[i] for i in Maxima[0]]
n = np.arange(1, len(frequencies_1)+1)
kn_1 = n*np.pi/0.6     # k = n*pi/L

plt.clf()
plt.plot(kn_1, frequencies_1, 'rx', label='Defekt an Pos. 3')
plt.plot(kn_2, frequencies_2, 'b+', label=r'12x$\SI{50}{\milli\meter}$ Röhre')
plt.legend(loc='best')
plt.ylabel(r'$f \:/\: \si{\hertz}$')
plt.xlabel(r'$k$ \:/\: \si{\per\meter}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/3_121.pdf')

plt.clf()
f, I = np.genfromtxt('messdaten/3_b12_pos1_75mmDefect.dat', unpack=True)
plotSpectrum(f,I,'3_122.pdf', 'r', 'Position 1', aspect=0.03)
f, I = np.genfromtxt('messdaten/3_b12_pos3_75mmDefect.dat', unpack=True)
plotSpectrum(f,I,'3_122.pdf', 'b', 'Position 3', aspect=0.03)
f, I = np.genfromtxt('messdaten/3_b12_pos7_75mmDefect.dat', unpack=True)
plotSpectrum(f,I,'3_122.pdf', 'g', 'Position 7', aspect=0.03)
plt.clf()
f, I = np.genfromtxt('messdaten/3_b12_pos3_12mmDefect.dat', unpack=True)
plotSpectrum(f,I,'3_123.pdf', 'r', r'$L_D=\SI{12.5}{\milli\meter}', aspect=0.03)
f, I = np.genfromtxt('messdaten/3_b12_pos3_75mmDefect.dat', unpack=True)
plotSpectrum(f,I,'3_123.pdf', 'b', r'$L_D=\SI{75}{\milli\meter}', aspect=0.03)


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


## convenience file writing for standard make Files
f = open('build/.pysuccess', 'w')
f.write('MarktkraM')
f.close()
