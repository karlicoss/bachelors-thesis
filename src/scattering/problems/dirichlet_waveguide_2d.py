from sys import stderr

import scipy.constants as sc

import numpy as np
from numpy import sqrt, exp, cos
from numpy import complex256 as complex
from numpy import float128 as real
from numpy.testing import assert_almost_equal
from scattering.problems.dirichlet_well_1d import DirichletWell1D
from scattering.problems.free_particle import FreeParticle

from scattering.problems.neumann_well_1d import NeumannWell1D
from scattering.tools import integrate_complex
from scattering.tools.tools import inner_product_L2

class DirichletWaveguide2D:
    def __init__(self, aY: real, bY: real, maxn: int):
        self.aY = aY
        self.bY = bY
        self.maxn = maxn

        self.wellY = DirichletWell1D(aY, bY, maxn)
        self.freeX = FreeParticle()

        self.eigenenergies = None # ???
        self.eigenfunctions = None # ???

    # TODO test
    def greens_function_helmholtz_dy(self, energy):
        def fun(x, y, xs, ys, maxn=self.maxn):
            res = complex(0.0)
            for m in range(1, maxn):
                gf = self.freeX.greens_function_helmholtz(energy - self.wellY.eigenenergies[m])
                # NOTE: no conjugation, wavefunctions are real
                res += self.wellY.eigenfunctions[m](y) * self.wellY.deigenfunctions[m](ys) * gf(x, xs)
            return res
        return fun

    def test(self):
        raise NotImplementedError

if __name__ == '__main__':
    dw = DirichletWaveguide2D(-1, 0, 1000)
    gf_dy = dw.greens_function_helmholtz_dy(100.0)
    print(gf_dy(0.0, -1.0, 0.0, 0.0))
    print(gf_dy(1.0, -1.0, 0.0, 0.0))
    print(gf_dy(-1.0, -1.0, 0.0, 0.0))

    print(gf_dy(0.0, -0.5, 0.0, 0.0))
    print(gf_dy(1.0, -0.5, 0.0, 0.0))
    print(gf_dy(-1.0, -0.5, 0.0, 0.0))

    print(gf_dy(0.0, -0.001, 0.0, 0.0))
    print(gf_dy(1.0, 0.0, 0.0, 0.0))
    print(gf_dy(-1.0, 0.0, 0.0, 0.0))