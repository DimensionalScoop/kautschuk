import numpy as np
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy as unp
import functions as f
from table import (
    make_table,
    make_SI,
    write,
)

g = 9.81

#drops is an array of the oildrops with voltage V, and velocity: v_0 , v_fast and v_slow for each drop.
drops = np.array([[227, 3/21, 3/9, 5/23],
                  [227, 0, 5/11, 4/11],
                  [301, 0, 3/5, 3/7],
                  [301, 0, 5/7, 5/8],
                  [301, 0, 5/6.3, 5/8.8],
                  [301, 0, 5/6.3, 5/9.5],
                  [301, 0, 5/9.6, 5/14.5],
                  [301, 2/38.5, 5/8.3, 5/13],
                  [282, 0, 5/14.8, 5/19.9],
                  [282, 0, 5/10.3, 5/12.8],
                  [282, 0, 5/8.3, 5/15.2],
                  [282, 2/20.6, 5/8.8, 5/11.9],
                  [282, 0, 5/13.7, 5/26.5],
                  [299, 0, 5/12.1, 5/25.2],
                  [299,0, 5/11.4, 5/13.2],
                  [299, 2/26, 5/11.75, 5/12.75],
                  [299, 0, 5/9.2, 5/11,6],
                  [299, 0, 5/8.3, 5/15.5],
                  [299, 0, 5/21.6, 5/41.8],
                  [299, 0, 5/27, 4/34.5],
                  [299, 0, 5/16, 5/16.66],
                  [299, 0, 5/12.4, 5/17.1],
                  [299, 0, 5/15.7, 5/30.7],
                  [299, 1/36.3, 5/21.2, 5/36],
                  [299, 0, 5/9.0, 5/12.5],
                  [299, 0, 5/12.8, 5/22.4],
                  [299, 0, 5/24.6, 5/48.2],
                  [299, 0, 5/21.2, 5/35],
                  [299, 0, 5/21.1, 5/26.8],
                  [299, 0, 5/9.7, 5/11.8],
                  [299, 0, 5/10.2, 5/11.9],
                  [299, 0, 5/7.4, 5/8.8],
                  [299, 0, 5/7.4, 5/8.9],
                  [299, 0, 5/6.3, 5/7.4],
                  [299, 0, 5/6.4, 5/10],
                  [299, 0, 5/9.0, 5/9.0]
                  ])

drops2 = np.array([[299, 0, 5/12.9, 5/16.1],
                  [299, 2/26.8, 5/8.1, 5/8.6],
                  [299, 0, 5/10.4, 5/10.8],
                  [299, 0, 5/9.8, 5/12.5],
                  [299, 0, 5/8.2, 5/9.7],
                  [299, 0, 5/4.7, 5/6.4],
                  [299, 0, 5/9.4, 5/11.1],
                  [299, 0, 5/8.4, 5/11.9],
                  [299, 0, 5/9.2, 5/15.4],
                  [299, 0, 5/6.1, 5/9.3],
                  [299, 0, 5/7.5, 5/10.4],
                  [299, 0, 5/6.5, 5/9.2],
                  [299, 0, 5/7.4, 5/8.3],
                  [301, 0, 5/14, 5/14.5]])

distance = 0.5e-3 #m
def calcdistance(dropsis):
    for i in range(len(dropsis)):
        dropsis[i][1] = dropsis[i][1]*distance
        dropsis[i][2] = dropsis[i][2]*distance
        dropsis[i][3] = dropsis[i][3]*distance
    return dropsis
drops = calcdistance(drops)
drops2 = calcdistance(drops2)

#write('../tex-data/v.tex',
#      make_table([[drops[i][0] for i in range(len(drops))], [drops[i][1] for i #in range(len(drops))], [drops[i][2] for i in range(len(drops))], #[drops[i][3] for i in range(len(drops))]], [0, 1, 1, 1]))

fuck1 = []
fuck2 = []
fuck3 = []
fuck4 = []
for i in range(len(drops)):
    fuck1.append(drops[i][0])
    fuck2.append(drops[i][1]*10**(5))
    fuck3.append(drops[i][2]*10**(5))
    fuck4.append(drops[i][3]*10**(5))

print(fuck3)
write('../tex-data/v.tex',
      make_table([fuck1,fuck2,fuck3,fuck4], [0, 2, 2, 2]))

#print(drops)
#for i in range(len(drops)):
#    diff = drops[i][2]-drops[i][3]
#    print(diff, i)
#    if drops[i][1] != 0:
#        c = 2*drops[i][1] / diff
#        print("Difference:" ,c ,"Stelle:", i)
#
#for i in range(len(drops2)):
#    diff = drops2[i][2]-drops2[i][3]
#    print(diff, i)
#    if drops2[i][1] != 0:
#        c2 = 2*drops2[i][1] / diff
#        print("Difference:" ,c2 ,"Stelle:", i)

#Elecectrical Field E
d = ufloat(7.6250e-3, 0.0051e-3)
E = []
for i in range(len(drops)):
    E.append(drops[i][0]/d)

E2 = []
for i in range(len(drops2)):
    E2.append(drops2[i][0]/d)


#print(E)
#print(d)
#viscosity
temperature = [29, 28] #°C
cun = 6.17e-5 * (101325/760) #Pa*m
density_oil = 886 #kg/m³
density_air = 1.293 #kg/m³
pressure_air = 101325 #Pa
uncorr_viscosity = [1.8665e-5, 1.862e-5] #N*s/m²



#Neuer Versuch mit gemittelten Geschwindigkeiten

drops_01 = np.array([[227, 3/21, 3/9, 5/23]])

drops_02 = np.array([[227, 0, 5/11, 4/11]])

drops_03 = np.array([[301, 0, 3/5, 3/7]])

drops_04 = np.array([[301, 0, 5/7, 5/8]])

drops_05 = np.array([[301, 0, 5/6.3, 5/8.8],
                  [301, 0, 5/6.3, 5/9.5]])

drops_06 = np.array([[301, 0, 5/9.6, 5/14.5],
                  [301, 2/38.5, 5/8.3, 5/13]])

drops_07 = np.array([[282, 0, 5/14.8, 5/19.9]])

drops_08 = np.array([[282, 0, 5/10.3, 5/12.8]])

drops_09 = np.array([[282, 0, 5/8.3, 5/15.2]])

drops_010 = np.array([[282, 2/20.6, 5/8.8, 5/11.9]])

drops_011 = np.array([[282, 0, 5/13.7, 5/26.5]])

drops_012 = np.array([[299, 0, 5/12.1, 5/25.2]])

drops_013 = np.array([[299,0, 5/11.4, 5/13.2],
                  [299, 2/26, 5/11.75, 5/12.75],
                  [299, 0, 5/9.2, 5/11,6]])

drops_014 = np.array([[299, 0, 5/8.3, 5/15.5]])

drops_015 = np.array([[299, 0, 5/21.6, 5/41.8],
                  [299, 0, 5/27, 4/34.5],
                  [299, 0, 5/16, 5/16.66],
                  [299, 0, 5/12.4, 5/17.1],
                  [299, 0, 5/15.7, 5/30.7]])

drops_016 = np.array([[299, 1/36.3, 5/21.2, 5/36]])

drops_017 = np.array([[299, 0, 5/9.0, 5/12.5]])

drops_018 = np.array([[299, 0, 5/12.8, 5/22.4]])

drops_019 = np.array([[299, 0, 5/24.6, 5/48.2],
                  [299, 0, 5/21.2, 5/35]])

drops_020 = np.array([[299, 0, 5/21.1, 5/26.8]])

drops_021 = np.array([[299, 0, 5/9.7, 5/11.8],
                  [299, 0, 5/10.2, 5/11.9]])

drops_022 = np.array([[299, 0, 5/7.4, 5/8.8],
                  [299, 0, 5/7.4, 5/8.9],
                  [299, 0, 5/6.3, 5/7.4],
                  [299, 0, 5/6.4, 5/10],
                  [299, 0, 5/9.0, 5/9.0]
                  ])





drops_1 = np.array([[299, 0, 5/12.9, 5/16.1],
                  [299, 2/26.8, 5/8.1, 5/8.6]])

drops_2 = np.array([[299, 0, 5/10.4, 5/10.8],
                  [299, 0, 5/9.8, 5/12.5],
                  [299, 0, 5/8.2, 5/9.7]])

drops_3 = np.array([[299, 0, 5/4.7, 5/6.4]])

drops_4 = np.array([[299, 0, 5/9.4, 5/11.1],
                  [299, 0, 5/8.4, 5/11.9],
                  [299, 0, 5/9.2, 5/15.4],
                  [299, 0, 5/6.1, 5/9.3]])

drops_5 = np.array([
                  [299, 0, 5/7.5, 5/10.4],
                  [299, 0, 5/6.5, 5/9.2],
                  [299, 0, 5/7.4, 5/8.3]])

drops_6 = np.array([
                  [301, 0, 5/14, 5/14.5]])

def meanDrops(drop):
    help1 = []
    help2 = []
    v0 = 0
    for i in range(len(drop)):

        help1.append(drop[i][2])
        help2.append(drop[i][3])
        if drop[i][1] != 0:
            v0 = drop[i][1]
    if len(help1) < 2:
        v_fast = help1[0]*distance
        v_slow = help2[0]*distance
    else :
        v_fast = ufloat(f.mean(help1) , f.stdDevOfMean(help1) )*distance
        v_slow = ufloat(f.mean(help2) , f.stdDevOfMean(help2) )*distance
    Drop_new = np.array([drop[0][0], v0, v_fast, v_slow])
    del help1[:]
    del help2[:]
    return Drop_new

drops_new = []
drops_new.append(meanDrops(drops_1))       #wrong
drops_new.append(meanDrops(drops_2))
drops_new.append(meanDrops(drops_3))
drops_new.append(meanDrops(drops_4))
drops_new.append(meanDrops(drops_5))
drops_new.append(meanDrops(drops_6))

drops_new0 = []
drops_new0.append(meanDrops(drops_01))     #wrong
drops_new0.append(meanDrops(drops_02))
drops_new0.append(meanDrops(drops_03))
drops_new0.append(meanDrops(drops_04))
drops_new0.append(meanDrops(drops_05))
drops_new0.append(meanDrops(drops_06))
drops_new0.append(meanDrops(drops_07))
drops_new0.append(meanDrops(drops_08))
drops_new0.append(meanDrops(drops_09))
drops_new0.append(meanDrops(drops_010))
drops_new0.append(meanDrops(drops_011))
drops_new0.append(meanDrops(drops_012))
drops_new0.append(meanDrops(drops_013)) #wrong
drops_new0.append(meanDrops(drops_014))
drops_new0.append(meanDrops(drops_015))
drops_new0.append(meanDrops(drops_016))
drops_new0.append(meanDrops(drops_017))
drops_new0.append(meanDrops(drops_018))
drops_new0.append(meanDrops(drops_019))
drops_new0.append(meanDrops(drops_020))
drops_new0.append(meanDrops(drops_021))
drops_new0.append(meanDrops(drops_022))

gone = []
gone.append(meanDrops(drops_01))
gone.append(meanDrops(drops_06))
gone.append(meanDrops(drops_010))
gone.append(meanDrops(drops_013))
gone.append(meanDrops(drops_016))
gone.append(meanDrops(drops_1))

print("Länge gone:",len(gone))

E5 = []
for i in range(len(gone)):
    E5.append(gone[i][0]/d)


E4 = []
E3 = []

for i in range(len(drops_new0)):
    E4.append(drops_new0[i][0]/d)


E3 = []
for i in range(len(drops_new)):
    E3.append(drops_new[i][0]/d)


#print(drops_new0)
#print(drops_new0[5][3])
#print(len(drops_new0))
