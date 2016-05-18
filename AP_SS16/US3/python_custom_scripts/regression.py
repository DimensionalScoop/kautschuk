import numpy as np

def reg_linear (x, m, b):
    return m*x + b

def reg_quadratic (x, a, x0, c):
    return a*(x-x0)**2 + c

def reg_cubic (x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d
