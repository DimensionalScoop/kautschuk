from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


Radius = 0.5
alpha_1n = [3.832,7.016,10.174]
alpha_2n = [5.135,8.417,11.620]
alpha_3n = [6.380,9.761,13.015]

def J (z,m):
    container = np.zeros(50)
    z_2 = z/2
    i = 0
    zwischensumme = 0
    for bla in container:
        zwischensumme += (-1)**i / (np.math.factorial(i)*np.math.factorial(i+m)) * (z_2)**(2*i)
        i += 1
    return (z_2)**m * zwischensumme

r = np.linspace(0,Radius,40)
phi = np.linspace(0, np.pi, 40)
R,Phi = np.meshgrid(r,phi)

X, Y = R*np.cos(Phi), R*np.sin(Phi)


Z_11 = J(R*alpha_1n[0]/Radius, 1) * np.sin(1*Phi)
Z_12 = J(R*alpha_1n[1]/Radius, 1) * np.sin(1*Phi)
Z_13 = J(R*alpha_1n[2]/Radius, 1) * np.sin(1*Phi)
Z_21 = J(R*alpha_2n[0]/Radius, 2) * np.sin(2*Phi)
Z_22 = J(R*alpha_2n[1]/Radius, 2) * np.sin(2*Phi)
Z_23 = J(R*alpha_2n[2]/Radius, 2) * np.sin(2*Phi)
Z_31 = J(R*alpha_3n[0]/Radius, 3) * np.sin(3*Phi)
Z_32 = J(R*alpha_3n[1]/Radius, 3) * np.sin(3*Phi)
Z_33 = J(R*alpha_3n[2]/Radius, 3) * np.sin(3*Phi)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z_11, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
plt.savefig('build/u11.pdf')

plt.clf
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z_12, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
plt.savefig('build/u12.pdf')

plt.clf
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z_13, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
plt.savefig('build/u13.pdf')

plt.clf
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z_21, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
plt.savefig('build/u21.pdf')

plt.clf
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z_22, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
plt.savefig('build/u22.pdf')

plt.clf
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z_23, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
plt.savefig('build/u23.pdf')

plt.clf
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z_31, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
plt.savefig('build/u31.pdf')

plt.clf
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z_32, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
plt.savefig('build/u32.pdf')

plt.clf
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z_33, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
plt.savefig('build/u33.pdf')
