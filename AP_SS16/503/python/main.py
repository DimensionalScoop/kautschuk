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


def charge(E, drops, viscosity):
    q_charge = []
    for i in range(len(drops)):
        charge = 3 * np.pi * viscosity * unp.sqrt(9*viscosity*(drops[i][2]-drops[i][3])/(4*d.g*(d.density_oil-d.density_air))) *(drops[i][2]+drops[i][3])/E[i]
        q_charge.append(charge)
#    print("HEEEEELPP:", len(q_charge))
    return q_charge

def charge_korr(q_0,r):
    q_charge_korr = []
    for i in range(len(q_0)):
        if r[i] != 0:
            charge = q_0[i]**(2/3) * (1+ d.cun / (d.pressure_air * r[i]))
            q_charge_korr.append(charge**(3/2))
        else:
            q_charge_korr.append(0)
#    print("DAFUUUKKKK: ", len(q_charge_korr))
    return q_charge_korr

def radius(drops, viscosity):
    radi = []
    for i  in range(len(drops)):
        radius = unp.sqrt(9*viscosity*(drops[i][2]-drops[i][3])/(2*d.g*(d.density_oil-d.density_air)))
        radi.append(radius)
#    print("UNICORNS 4 EVVAA:", len(radi))
    return radi


def appendstuff(q1,q2):
    quabla = []
    for i in range(len(q1)):
        quabla.append(q1[i])
    for i in range(len(q2)):
        quabla.append(q2[i])
#    print("MAYBE HERE?!", len(quabla))
    return quabla

q1 = charge(d.E, d.drops, d.uncorr_viscosity[1])
q2 = charge(d.E2, d.drops2, d.uncorr_viscosity[0])
r1 = radius(d.drops, d.uncorr_viscosity[1])
r2 = radius(d.drops2, d.uncorr_viscosity[0])

q = appendstuff(q1,q2)
r = appendstuff(r1,r2)

q_korr = charge_korr(q,r)


#mit gemittelten Tröpfchen
q3 = charge(d.E3, d.drops_new, d.uncorr_viscosity[0])
q4 = charge(d.E4, d.drops_new0, d.uncorr_viscosity[1])
r3 = radius(d.drops_new, d.uncorr_viscosity[0])
r4 = radius(d.drops_new0, d.uncorr_viscosity[1])

q_gone = charge(d.E5, d.gone, d.uncorr_viscosity[1])
r_gone = radius(d.gone, d.uncorr_viscosity[1])
q_gone_korr = charge_korr(q_gone, r_gone)

q_new = appendstuff(q4,q3)
r_new = appendstuff(r4,r3)

q_new_korr = charge_korr(q_new,r_new)



#for i in range(len(d.drops)):
#    print("Differenz:", q[i]/1.602e-19 )
#print("Radius:", r)
#print("korrigierte Ladung:", q_korr)
#print("Minimum:",min(q_korr))
#print("Minimum:",0.25*1.602e-19)
#print("Minimum - Elementarladung: ", (min(q_korr)-1.602e-19)/1.602e-19)
#print("Cunningham: ", (1+ d.cun / (d.pressure_air * r[0])))


qui_k,qui_k_err = plot.extract_error(q_korr)
qui,qui_err = plot.extract_error(q)
qui_new, qui_new_err = plot.extract_error(q3)



#Finde den größten gemeinsamen Teiler (GCD) ... und Einhörner existieren wirklich und so...

def GCD(q,maxi):
    #q: Vektor mit reellen Zahlenwerten
    #maxi: maximale Iterationszahl für einen GCD
    gcd = q[0]
    for i in range(1,len(q)):
        n = 0
        while abs(gcd-q[i]) > 1e-19 and n <= maxi :
            if gcd > q[i]:
                gcd = gcd - q[i]
            else:
                q[i] = q[i] - gcd
            n = n+1
    return gcd

test = copy.deepcopy(q_new_korr)
test2 = copy.deepcopy(q_new)

tryit1 = GCD(test,16)
tryit2 = GCD(test2, 16)
e_0 = 1.602e-19

print("Versuch: ",tryit1)
print("Abweichung: ", f.abweichung(tryit1, e_0), f.abweichung(tryit2, e_0))

qui_new, qui_new_err = plot.extract_error(q_new_korr)
#for i in range(len(qui_new)):
#    print("REALLY?!", q_new_korr[i])
#    print("FUK IT:",qui_new[i])

params, covariance = curve_fit(f.linearFit, range(len(qui_new)), qui_new)

print("Steigung m = ", params[0])
print("Achsenabschnitt b = ", params[1])
print("maximum:", max(qui_new))
print("minimum: ", min(qui_new))

quack = []
for i in range(len(q_new)):
    quack.append(q_new[i]*10**(19))

quack2 = []
for i in range(len(q_new_korr)):
    quack2.append(q_new_korr[i]*10**(19))

wtf = []
for i in range(len(r_new)):
    wtf.append(r_new[i]*10**(5))

c = 1.46e-19
e_t = 1.6021766208e-19

N_t = 6.022140857e+23

F = 96485.33289
N = F/c
print("Avogadro:", N)
N = ufloat(N,0)
print("Abweichung Avo:", f.abweichung(N, N_t))
c = ufloat(c,0)
print("Abweichung e_0:", f.abweichung(c, e_t))
#write('../tex-data/r.tex',
#      make_table([r_new], [2]))
print("Länge q:", len(q_new_korr))
print("Länge q:", len(q_korr))

write('../tex-data/q.tex',
      make_table([quack, quack2], [2,2]))
