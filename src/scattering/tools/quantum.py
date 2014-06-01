import numpy as np
from numpy import sqrt, exp, sin, cos, pi
from numpy import complex256 as complex
from numpy import float128 as real

import scipy as sp
import scipy.integrate
from scipy.misc import derivative
from scattering.tools import integrate_complex, I


def compute_prob_current(wf, dwf, x: real, ya: real, yb: real):
    """
    Computes the probability current in the cross section [ya, yb] at position x
    :param wf: wavefunction
    :param dwf: partial derivative of the wavefunction wrt x
    :param x:
    """
    return integrate_complex(lambda y: np.conj(wf(x, y)) * dwf(x, y) - wf(x, y) * np.conj(dwf(x, y)), ya, yb)[0]

def compute_prob_current_numerical(wf, x: real, ya: real, yb: real, eps=0.01):
    dwf = lambda x, y: derivative(lambda xx: wf(xx, y), x, eps)
    return compute_prob_current(wf, dwf, x, ya, yb)

# TODO make this a test
if __name__ == '__main__':
    H = 1.0
    m = 1
    ya = 0
    yb = H
    energy = 20.0
    k1 = pi * m / H
    kk = sqrt(energy - k1 ** 2)
    print("kk = {}".format(kk))
    wf = lambda x, y: sqrt(2 / H) * sin(k1 * y) * exp(I * kk * x)
    wfdx = lambda x, y: sqrt(2 / H) * sin(k1 * y) * I * kk * exp(I * kk * x)
    print("Expecting: {}".format(2 * I * kk))
    print("Analytical: {}".format(compute_prob_current(wf, wfdx, 10, ya, yb)))
    print("Numerical: {}".format(compute_prob_current_numerical(wf, 10, ya, yb)))