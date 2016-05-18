import os

import numpy as np
from uncertainties.unumpy import uarray

import data
import helpers as hel
import plot_helpers as plot
import main as m
import functions as f

import matplotlib as mpl
import matplotlib.pyplot as plt

#plot.plot(range(len(m.q)),m.q, "Teilchennummer","Ladung","../plots/ladung.pdf", None)
#plot.plot(range(len(m.q_korr)),m.q_korr, "Teilchennummer","Korrigierte Ladung in C","../plots/ladung2.pdf", None)
plot.plot2(range(len(m.q_new_korr)),m.q_new_korr, "Teilchennummer","Korrigierte Ladung $q / C$","../plots/ladung2.pdf", None)
plot.plot([0,5,9,12,15,22],m.q_gone_korr, "Teilchennummer","Korrigierte Ladung $q / C$","../plots/ladung2.pdf", None)



x,y = plot.extract_error(m.q_korr)
x2,y2 = plot.extract_error(m.q)
x3,y3 = plot.extract_error(m.q_new)
x4,y4 = plot.extract_error(m.q_new_korr)

bla = range(39, 45)
print(bla)

#plt.plot([0,12,22],m.q_gone_korr, 'r.')
plt.plot(np.linspace(0,0.1,len(x4)),x4,'m.')
#plt.plot(range(len(x2)),x2, 'b.') #hahaha korrektur macht zero unterschied -.-
#plt.plot(range(len(x3)),x3, 'm.') #gemittelte Teilchen
#plt.plot(bla,x3, 'm.') #gemittelte Teilchen
#plt.plot(range(len(x3)),x3, 'm.')
#plt.plot(range(len(x4)),x4, 'c.')
#x_flow = np.linspace(-0.2,26,1000)
#plt.plot(x_flow, f.linearFit(x_flow, m.params[0], m.params[1]), 'm-',
#         label='linearer Fit', linewidth=0.8)


for i in range(22):
    c = m.tryit1.nominal_value
#    c = 1.602e-19
    c = 1.46e-19
#    c = m.tryit2.nominal_value
    plt.axhline(y=i*c, linewidth=0.2, color='g')
    a = i*c + c*0.25
    b = i*c - c*0.25
    plt.axhspan(b, a, facecolor='g', alpha=0.1)

plt.savefig('../plots/ladung3.pdf')
