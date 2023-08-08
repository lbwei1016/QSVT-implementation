from math import cos, exp, pi, sin, sqrt
from cmath import exp as cexp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class InfiniteWell:
    def __init__(self, psi0, width, nbase, nint):
        self.width = width
        self.nbase = nbase
        self.nint = nint
        self.coeffs = self.get_coeffs(psi0)

    def eigenfunction(self, n, x):
        if n % 2:
            return sqrt(2/self.width)*sin((n+1)*pi*x/self.width)
        return sqrt(2/self.width)*cos((n+1)*pi*x/self.width)

    def get_coeffs(self, psi):
        coeffs = []
        for n in range(self.nbase):
            f = lambda x: psi(x)*self.eigenfunction(n, x)
            c = trapezoidal(f, -0.5*self.width, 0.5*self.width, self.nint)
            coeffs.append(c)
        return coeffs

    def psi(self, x, t):
        psit = 0
        for n, c in enumerate(self.coeffs):
            psit = psit + c*cexp(-1j*(n+1)**2*t)*self.eigenfunction(n, x)
        return psit

def trapezoidal(func, a, b, nint):
    delta = (b-a)/nint
    integral = 0.5*(func(a)+func(b))
    for k in range(1, nint):
        integral = integral+func(a+k*delta)
    return delta*integral

def psi0(x):
    sigma = 0.005
    return exp(-x**2/(2*sigma))/(pi*sigma)**0.25

w = InfiniteWell(psi0=psi0, width=2, nbase=100, nint=1000)
x = np.linspace(-0.5*w.width, 0.5*w.width, 500)
ntmax = 1000
z = np.zeros((500, ntmax))
for n in range(ntmax):
    t = 0.25*pi*n/(ntmax-1)
    y = np.array([abs(w.psi(x, t))**2 for x in x])
    z[:, n] = y
z = z/np.max(z)
# plt.rc('text', usetex=True)
# plt.imshow(z, cmap=cm.hot)
# plt.xlabel('$t$', fontsize=20)
# plt.ylabel('$x$', fontsize=20)
# plt.show()