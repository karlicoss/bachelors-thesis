from sys import stderr

import scipy.constants as sc

import numpy as np
from numpy import sqrt, exp, cos
from numpy import complex256 as complex
from numpy import float128 as real
from numpy.testing import assert_almost_equal

# TODO hbar and mass
from scattering.tools import integrate_complex, I
from scattering.tools.tools import inner_product_L2


class FreeParticle:
    def __init__(self):
       pass

    def greens_function_helmholtz(self, energy):
        kk = np.sqrt(np.complex(energy))
        def fun(x, xs):
            return I / (2 * kk) * exp(I * kk * np.abs(x - xs))
        return fun


if __name__ == '__main__':
    free = FreeParticle()
    gf = free.greens_function_helmholtz(10.0)
    for x in np.arange(-10, 10, 1):
        print(gf(x, 0))
