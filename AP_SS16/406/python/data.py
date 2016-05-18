import numpy as np

# please note: the small slit and the big slit are actually the wrong way around.
# in reality, the slide was upside-down, so the big slit measurements should be
# the measurements of the small slit and vice versa. pls kil me.

lambda_ = 635e-9  # wavelanght of the laser used
dark_current = 0.1e-9  # Ampere
screen_distance = 92.8e-2  # distance between screen and slit
analoge_abberation = 0.1


class slit():
    pass

big_slit = slit()
big_slit.spacing = 0.5e-3

a = [np.array([43, 44, 43, 42, 41, 38, 36, 33, 30, 26, 24, 20, 17, 14, 11]) * 1e-7,
     np.array([9, 7]) * 1e-7,
     np.array([56, 40, 26, 16]) * 0.1e-7,
     np.array([10, 5]) * 0.03e-7,
     np.array([25, 14]) * 10e-10,
     np.array([19.5, 28.5]) * 3e-10,
     np.array([38, 52, 66]) * 10e-10]

# gegen den uhrzeigersinn
b = [np.array([42, 40, 39, 34, 32, 28, 25, 22, 18, 14, 12]) * 1e-7,
     np.array([10.5, 7, 6, 4, 2.5]) * 0.3e-7,
     np.array([21.5, 13, 7.5]) * 0.03e-7,
     np.array([50, 40, 44, 56, 75, 94]) * 10e-10,
     np.array([11, 12.5]) * 0.1e-7,
     np.array([12, 12]) * 0.1e-7]

c = np.array([])

for x in range(len(b)):
    c = np.append(c, b[x])
big_slit.midpoint_index = len(c) - 1 + 1  # +1 as midpoint is slightly off to the left

d = np.array([])

for x in range(len(a)):
    d = np.append(d, a[x])

big_slit.currents = np.append(c[::-1], d) - dark_current


medium_slit = slit()

a = [np.array([30, 28, 24, 18, 13, 8, 4]) * 100e-6,
     np.array([59, 26, 8]) * 10e-6,
     np.array([24, 18, 46, 78, 96, 94, 72, 47, 23, 10, 8, 12, 19, 24, 24, 21, 14]) * 1e-6]

# gegen den uhrzeigersinn
b = [np.array([29, 26, 22, 17, 12, 6]) * 100e-6,
     np.array([47, 19]) * 10e-6,
     np.array([68, 17, 32, 74]) * 1e-6,
     np.array([8, 9.5, 8]) * 10e-6,
     np.array([73, 44, 19, 10, 14, 26, 39, 46, 45, 37, 24, 14, 6, 3, 4, 7, 10, 11, 10]) * 1e-6]

c = np.array([])

for x in range(len(b)):
    c = np.append(c, b[x])
medium_slit.midpoint_index = len(c) - 1

d = np.array([])

for x in range(len(a)):
    d = np.append(d, a[x])

medium_slit.currents = np.append(c[::-1], d) - dark_current
medium_slit.spacing = 0.5e-3


small_slit = slit()

a = [np.array([22.5, 15.5, 5]) * 0.3e-3,
     np.array([10, 6.5, 6, 1.7, 1.5, 2]) * 30e-6,
     np.array([12, 12.5, 17.5, 10.5, 5, 6]) * 3e-6,
     np.array([46, 48, 66, 44, 26, 36, 28, 20, 28, 25, 12, 13, 16, 12, 12]) * 1e-6]

# gegen den uhrzeigersinn
b = [np.array([18, 8]) * 0.3e-3,
     np.array([16.5]) * 30e-6,
     np.array([78, 93, 36, 22, 34, 18, 10, 16, 10, 4, 8]) * 10e-6,
     np.array([86, 59, 78, 79, 40, 35, 39, 30, 36, 33, 18, 20, 25]) * 1e-6]

c = np.array([])

for x in range(len(b)):
    c = np.append(c, b[x])
small_slit.midpoint_index = len(c) - 1

d = np.array([])

for x in range(len(a)):
    d = np.append(d, a[x])

small_slit.currents = np.append(c[::-1], d) - dark_current
small_slit.spacing = 0.5e-3


double_slit = slit()

a = [np.array([18, 16.5, 13, 15.5, 17, 14.5, 11, 11.5, 13, 13, 10.5]) * 30e-6,
     np.array([87, 88, 98, 92, 68, 51, 52, 56, 46, 30, 22]) * 10e-6,
     np.array([24.5, 24.5, 28, 10.5, 6.5]) * 3e-6]

# gegen den uhrzeigersinn
b = [np.array([15, 12.5, 13, 15.5, 15.75, 13, 10.25, 10.75, 12.5, 11.75, ]) * 30e-6,
     np.array([94, 74, 78, 85, 74, 52, 40, 42, 43, 33, 20, 15, 15, 12]) * 10e-6,
     np.array([]) * 3e-6,
     np.array([]) * 1e-6]

c = np.array([])

for x in range(len(b)):
    c = np.append(c, b[x])
double_slit.midpoint_index = len(c) - 1

d = np.array([])

for x in range(len(a)):
    d = np.append(d, a[x])

double_slit.currents = np.append(c[::-1], d) - dark_current
double_slit.spacing = 0.25e-3

slits = [big_slit, medium_slit, small_slit, double_slit]
single_slits = [big_slit, medium_slit, small_slit]
