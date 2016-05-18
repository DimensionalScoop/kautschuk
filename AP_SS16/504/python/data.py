import numpy as np
import uncertainties
from uncertainties import unumpy as unp
from uncertainties import ufloat

import functions as f
import copy

from table import make_SI, make_table, write

# Anodenspannung und Anodenstrom. Diode 2
anode_1 = np.array([[0, 0.],
                    [1, 1.],
                    [2, 4.],
                    [3, 8.],
                    [4, 13.],
                    [5, 18.],
                    [6, 25.],
                    [7, 32.],
                    [8, 38.],
                    [9, 44.],
                    [10, 51.],
                    [11, 59.],
                    [12, 65.],
                    [13, 69.],
                    [14, 75.],
                    [15, 79.],
                    [16, 87.],
                    [17, 90.],
                    [18, 97.],
                    [19, 102.],
                    [20, 107.],
                    [21, 110.],
                    [22, 115.],
                    [23, 118.],
                    [24, 121.],
                    [25, 123.],
                    [26, 125.],
                    [27, 127.],
                    [28, 129.],
                    [29, 130.],
                    [30, 131.],
                    [31, 132.],
                    [32, 133.],
                    [33, 134.],
                    [34, 135.],
                    [35, 135.],
                    [36, 136.],
                    [38, 137.],
                    [40, 138.],
                    [42, 138.],
                    [44, 139.],
                    [46, 140.],
                    [48, 140.],
                    [50, 141.],
                    [52, 141.],
                    [54, 142.],
                    [56, 142.],
                    [58, 143.]
                    ])

anode_2 = np.array([[0, 0.],
                    [1, 0.],
                    [2, 2.],
                    [3, 5.],
                    [4, 9.],
                    [5, 13.],
                    [6, 18.],
                    [7, 21.],
                    [8, 24.],
                    [9, 28.],
                    [10, 32.],
                    [11, 35.],
                    [12, 38.],
                    [13, 41.],
                    [14, 44.],
                    [15, 45.],
                    [16, 48.],
                    [17, 49.],
                    [18, 51.],
                    [19, 52.],
                    [20, 53.],
                    [21, 54.],
                    [22, 55.],
                    [23, 56.],
                    [24, 56.],
                    [25, 57.],
                    [26, 57.],
                    [27, 58.],
                    [28, 58.],
                    [30, 59.],
                    [35, 60.],
                    [40, 61.],
                    [49, 62.]
                    ])

anode_3 = np.array([[0, 0.],
                    [1, 0.],
                    [2, 1.],
                    [3, 1.],
                    [4, 3.],
                    [5, 5.],
                    [6, 8.],
                    [7, 9.],
                    [8, 11.],
                    [9, 13.],
                    [10, 14.],
                    [11, 16.],
                    [12, 17.],
                    [13, 18.],
                    [14, 19.],
                    [15, 20.],
                    [16, 20.],
                    [17, 21.],
                    [18, 21.],
                    [19, 21.],
                    [20, 22.],
                    [21, 22.],
                    [22, 22.],
                    [23, 22.],
                    [24, 22.],
                    [25, 23.],
                    [26, 23.],
                    [27, 23.],
                    [28, 23.],
                    [29, 23.],
                    [43, 24.]
                    ])

anode_4 = np.array([[0, 0.],
                    [1, 0.],
                    [2, 0.],
                    [3, 1.],
                    [4, 1.],
                    [5, 2.],
                    [6, 4.],
                    [7, 4.],
                    [8, 5.],
                    [9, 5.],
                    [10, 6.],
                    [11, 6.],
                    [12, 6.],
                    [13, 7.],
                    [14, 7.],
                    [15, 7.],
                    [16, 7.],
                    [17, 8.],
                    [18, 8.],
                    [19, 8.],
                    [20, 8.],
                    [44, 9.]
                    ])

anode_5 = np.array([[0, 0.],
                    [11, 1.],
                    [14, 2.],
                    [22, 3.]
                    ])

AV1 = f.gimmeTHATcolumn(anode_1,0)
AI1 = f.gimmeTHATcolumn(anode_1,1)
AV2 = f.gimmeTHATcolumn(anode_2,0)
AI2 = f.gimmeTHATcolumn(anode_2,1)
AV3 = f.gimmeTHATcolumn(anode_3,0)
AI3 = f.gimmeTHATcolumn(anode_3,1)
AV4 = f.gimmeTHATcolumn(anode_4,0)
AI4 = f.gimmeTHATcolumn(anode_4,1)
AV5 = f.gimmeTHATcolumn(anode_5,0)
AI5 = f.gimmeTHATcolumn(anode_5,1)
write('../tex-data/anode1.tex',
      make_table([AV1,AI1,AV2,AI2,AV3,AI3,AV4,AI4,AV5,AI5], [0, 0,0,0,0,0,0,0,0,0]))



def make_it_SI(array, k, SI):
    """Converts a micro-unit to SI"""
    for i in range(len(array)):
        array[i][k] *= 10**(SI)

def make_it_SI2(array, SI):
    """Converts a micro-unit to SI"""
    for i in range(len(array)):
        array[i] *= 10**(SI)

#print(anode_5)
make_it_SI(anode_5, 1, -6)
make_it_SI(anode_4, 1, -6)
make_it_SI(anode_3, 1,-6)
make_it_SI(anode_2, 1,-6)
make_it_SI(anode_1, 1,-6)
#print(anode_5)

anode_langmuh = copy.deepcopy(anode_1[3:20])
print("LANGMUH:",anode_langmuh)
# Teil2 Spannung und Strom. Diode 1

current = np.array([[0, 16.0],
                    [-0.06, 11.5],
                    [-0.10, 9.0],
                    [-0.12, 8.5],
                    [-0.18, 6.0],
                    [-0.24, 5.0],
                    [-0.30, 3.6],
                    [-0.36, 2.8],
                    [-0.40, 2.2],
                    [-0.42, 2.1],
                    [-0.48, 1.55],
                    [-0.54, 1.15],
                    [-0.60, 0.9],
                    [-0.66, 0.69],
                    [-0.70, 0.52],
                    [-0.72, 0.51],
                    [-0.78, 0.38],
                    [-0.84, 0.28],
                    [-0.90, 0.24],
                    [-0.96, 0.18]
                    ])
#print("Strom: ",current)

make_it_SI(current,1,-9)
#print("Strom: ",current)




# write('../tex-data/v.tex',
#      make_table([[drops[i][0] for i in range(len(drops))], [drops[i][1] for i #in range(len(drops))], [drops[i][2] for i in range(len(drops))], #[drops[i][3] for i in range(len(drops))]], [0, 1, 1, 1]))








# this is the end
