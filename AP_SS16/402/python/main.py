import numpy as np
import uncertainties.unumpy as unp
from numpy import array
import helpers as hel
import plot
import matplotlib.pyplot as plt
import simplex
from scipy.misc import derivative
from scipy.optimize import newton


# Bad measurements, using default value of 60 degrees
phi = np.deg2rad(60)

# Sorted from red to blue
_omega_r = np.deg2rad(array([122, 122.7, 123, 123.7, 124.3, 124.8, 126, 127.3]))
_omega_l = np.deg2rad(array([160.4, 159.5, 159.1, 158.1, 157.9, 156.9, 156.6, 154.4]))
eta = np.pi - (_omega_r - _omega_l)  # Not sure
print("eta", eta)
n = np.sin((eta + phi) / 2) / np.sin(phi / 2)

# From red to blue, sourced from wiki and NIST
spectral_lines = sorted(array([4047, 4358, 5461, 5782, 4416, 5337, 5378, 6438]) * 1e-10)[::-1]


# Fitting refractive indices

def polynom(X, a1, a2):
    return [np.sqrt(a1 + a2 * x**2) for x in X]


def laurent(lambda_, a0, a2):
    return [np.sqrt(a0 + a2 / x ** 2) for x in lambda_]


# polynom_fit = hel.autofit(spectral_lines, n**2, polynom, p0=(3, 1e-10))
# laurent_fit = hel.autofit(spectral_lines, n**2, laurent)

polynom_fit, polynom_quality = simplex.optimize(spectral_lines, n, polynom, p0=[3, 1e10])
laurent_fit, laurent_quality = simplex.optimize(spectral_lines, n, laurent, p0=[3, 1e10])

plt.clf()

x_messung = spectral_lines
y_messung = n

x_limit = plot.autolimits(x_messung)
x_flow = np.linspace(*x_limit, num=1000)
y_messung = y_messung


plt.plot(x_flow, polynom(x_flow, *polynom_fit), 'g-', label="Fit mit $x^2$")
plt.plot(x_flow, laurent(x_flow, *laurent_fit), 'b-', label="Fit mit $x^{-2}$")
plt.plot(x_messung, y_messung, 'r+', label="Messwerte")

plt.xlabel("Wellenlänge in Metern")
plt.ylabel("Brechungsindex")

plt.legend(loc='best')

plt.xlim(x_limit)
plt.ylim(plot.autolimits(y_messung))
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.show()
plt.savefig('../plots/' + "dispersionskurve.pdf")


def n_fit(lambda_):  # deciding that laurent is the better fit
    return laurent([lambda_], *laurent_fit)[0]

frauenhofer = {'F': 486e-9, 'D': 589e-9, 'C': 656e-9}
abbescheZahl = (n_fit(frauenhofer['D']) - 1) / (n_fit(frauenhofer['F']) - n_fit(frauenhofer['C']))

# prism resolution
b = 3e-2


print("Auswerung Prismadispersion:")
print("Phi:", phi / np.pi * 180, "^o")
print("Brechungsindices: ", n)
print("Spetrallinien: ", spectral_lines)
print("\nPolynomal fit: ", polynom_fit, "Quality", polynom_quality)
print("\nLaurent fit: ", laurent_fit, "Quality", laurent_quality)
print("Laurent Curvefit:", hel.autofit(spectral_lines, n, laurent, p0=(laurent_fit)))
print("Polynomial Curvefit:", hel.autofit(spectral_lines, n, polynom, p0=(polynom_fit)))
print("Abbesche Zahl", abbescheZahl)
print("Auflösungsvermögen: C", b * derivative(n_fit, frauenhofer['C'], dx=1e-20), "F", b * derivative(n_fit, frauenhofer['F'], dx=1e-20))
print("Nächste Absorbtionsstelle:", newton(n_fit, 500e-9, maxiter=5000))
