from collections import namedtuple

import numpy as np
from numpy import sqrt, exp
from numpy import complex256 as complex
from numpy import float128 as real

import scipy as sp
import scipy.integrate

I = complex(1j)

def cnorm2(z: complex):
    """
    Returns the square of the magnitude of a complex number
    :rtype : real
    """
    return (z * np.conj(z)).real

def cnorm(z: complex):
    """
    Returns the magnitude of a complex number
    :rtype : real
    """
    return sqrt(cnorm2(z))


ScatteringResult = namedtuple('ScatteringResult', ['wf', 'T'])

def integrate_complex(f, a: real, b: real, **kwargs):
    """

    :rtype : (complex, complex)
    """
    real_integrand = scipy.integrate.quad(lambda x: sp.real(f(x)), a, b, **kwargs)
    imag_integrand = scipy.integrate.quad(lambda x: sp.imag(f(x)), a, b, **kwargs)
    return real_integrand[0] + I * imag_integrand[0], real_integrand[1] + I * imag_integrand[1]

# TODO
jn_zeros = lambda x, y: scipy.special.jn_zeros(x, int(y))
jn = lambda x, y: scipy.special.jn(x, float(y))