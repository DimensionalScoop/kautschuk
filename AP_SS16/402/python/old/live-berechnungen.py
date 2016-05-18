import numpy as np

L = 1.75e-3
C = 22e-9
C2 = 9.4e-9

Eigenfreq = 5058


def Z(omega):
    return np.sqrt(L / C) * 1 / np.sqrt(1 - 1 / 4 * omega**2 * L * C)


def Z_C1C2():
    return np.sqrt(2 * L / (C + C2))

print("Z(0)=", Z(0))
print("Z_C1C2=", Z_C1C2())
print("Z(Eigenfreq)=", Z(Eigenfreq))
