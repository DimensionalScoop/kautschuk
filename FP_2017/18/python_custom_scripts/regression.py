import numpy as np

def reg_linear (x, m, b):
    return m*x + b

def reg_quadratic (x, a, b, c):
    return a*x**2 + b*x + c

def reg_cubic (x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def reg_gauss (x, a, b, c):
    return a*np.exp(-b*(x-c)**2)
