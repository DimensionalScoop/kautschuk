import os

import numpy as np
from uncertainties.unumpy import uarray
import uncertainties

import data as d
import helpers as hel
import plot_helpers as plot
import main as m
import functions as f

import matplotlib as mpl
import matplotlib.pyplot as plt




x_volt_1 = m.gimmeTHATcolumn(d.anode_1,0)
y_current_1 = m.gimmeTHATcolumn(d.anode_1,1)
x_volt_2 = m.gimmeTHATcolumn(d.anode_2,0)
y_current_2 = m.gimmeTHATcolumn(d.anode_2,1)
x_volt_3 = m.gimmeTHATcolumn(d.anode_3,0)
y_current_3 = m.gimmeTHATcolumn(d.anode_3,1)
x_volt_4 = m.gimmeTHATcolumn(d.anode_4,0)
y_current_4 = m.gimmeTHATcolumn(d.anode_4,1)
x_volt_5 = m.gimmeTHATcolumn(d.anode_5,0)
y_current_5 = m.gimmeTHATcolumn(d.anode_5,1)

x_volt_langmuh = m.gimmeTHATcolumn(d.anode_langmuh,0)
y_current_langmuh = m.gimmeTHATcolumn(d.anode_langmuh,1)

print(x_volt_1)
print(y_current_1)

#plot.plot4(x_volt_langmuh,y_current_langmuh, "Anodenspannung in $V$","Anodenstrom in $\mu A$","../plots/anode1.pdf", None)

plot.plot4(x_volt_5,y_current_5, "Anodenspannung in $V$","Anodenstrom in $ A$","../plots/anode1.pdf", None)
plot.plot5(x_volt_4,y_current_4, "Anodenspannung in $V$","Anodenstrom in $ A$","../plots/anode1.pdf", None)
plot.plot3(x_volt_3,y_current_3, "Anodenspannung in $V$","Anodenstrom in $ A$","../plots/anode1.pdf", None)
plot.plot(x_volt_2,y_current_2, "Anodenspannung in $V$","Anodenstrom in $ A$","../plots/anode1.pdf", None)
plot.plot2(x_volt_1,y_current_1, "Anodenspannung in $V$","Anodenstrom in $ A$","../plots/anode1.pdf", None)

for i in range(5):

    plt.axhline(m.saturation_current[i], linewidth=0.2, color='c')

plt.savefig('../plots/anode1.pdf')

plt.clf()
plt.clf()
#plot.plot([0,5,9,12,15,22],m.q_gone_korr, "Teilchennummer","Korrigierte Ladung $q / C$","../plots/ladung2.pdf", None)
plot.log_plot(x_volt_1,y_current_1, "Logarithmierte Anodenspannung $\ln(U)$","Logarithmierter Anodenstrom $\ln(\mu A)$","../plots/raum.pdf", None)

x_flow = np.linspace(1,3,1000)

plt.plot(x_flow, f.linearFit(x_flow, m.raumladung[0].nominal_value,m.raumladung[1].nominal_value), 'b-', label= "Linearer Fit")
plt.tight_layout(pad=0, h_pad=1.20, w_pad=1.20)
plt.legend(loc='best')
plt.savefig('../plots/raum.pdf')

plt.clf()
plt.clf()

plot.plot4(x_volt_langmuh,y_current_langmuh, "$\ln(U_A)$","$\ln(I_A)$","../plots/langmuh.pdf", None)

x_flow = np.linspace(1,3,1000)

plt.plot(x_flow, f.linearFit(x_flow, m.raumladung[0].nominal_value,m.raumladung[1].nominal_value), 'b-', label= "Linearer Fit")
plt.tight_layout(pad=0, h_pad=1.20, w_pad=1.20)
plt.legend(loc='best')
plt.savefig('../plots/langmuh.pdf')



plt.clf()
plt.clf()


x_volt_current = m.gimmeTHATcolumn(d.current,0)
y_current_current = m.gimmeTHATcolumn(d.current,1)
#plot.plot4(m.U_real,y_current_current, "Anodenspannung in $V$","Anodenstrom in $ nA$","../plots/current.pdf", None)
plot.plot(m.U_real,m.ln_IA, "$U_{\mathrm{real}}$ in $V$","$\ln(I_A)$","../plots/current.pdf", None)
x_flow = np.linspace(-1,0,1000)
#print(x_flow)

plt.plot(x_flow, f.linearFit(x_flow, m.exponent[0].nominal_value,m.exponent[1].nominal_value), 'b-', label= "Linearer Fit")
plt.tight_layout(pad=0, h_pad=1.20, w_pad=1.20)
plt.legend(loc='best')
plt.savefig('../plots/current.pdf')
