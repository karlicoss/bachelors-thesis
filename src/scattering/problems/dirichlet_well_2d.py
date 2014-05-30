from sys import stderr

import scipy.constants as sc

import numpy as np
from numpy import sqrt, exp, cos
from numpy import complex256 as complex
from numpy import float128 as real
from numpy.testing import assert_almost_equal
from scattering.problems.dirichlet_well_1d import DirichletWell1D

from scattering.problems.neumann_well_1d import NeumannWell1D
from scattering.tools import integrate_complex
from scattering.tools.tools import inner_product_L2


class DirichletWell2D:
    def __init__(self, aX: real, bX: real, aY: real, bY: real, maxn: int):
        self.aX = aX
        self.bX = bX
        self.aY = aY
        self.bY = bY
        self.maxn = maxn

        self.wellX = DirichletWell1D(aX, bX, maxn)
        self.wellY = DirichletWell1D(aY, bY, maxn)

        self.eigenenergies = None # ???
        self.eigenfunctions = None # ???

    # TODO test
    def greens_function_helmholtz_dy(self, energy, maxn=None):
        if maxn is None:
            maxn = self.maxn
        def fun(x, y, xs, ys):
            res = complex(0.0)
            for n in range(1, maxn): # NOTE 1-indexing
                # stderr.write("{}: {}\n".format(n, str(energy - self.wellY.eigenenergies[n])))
                gf = self.wellY.greens_function_helmholtz_dx(energy - self.wellX.eigenenergies[n])
                # NOTE: no conjugation, wavefunctions are real
                res += self.wellX.eigenfunctions[n](x) * self.wellX.eigenfunctions[n](xs) * gf(y, ys)
            return res
        return fun

    def test(self):
        """
        Checks if the eigenfunctions are orthonormal
        """
        raise NotImplementedError
        # tolerance = 1e-3
        # for i, fi in enumerate(self.eigenfunctions):
        #     for j, fj in enumerate(self.eigenfunctions):
        #         res = inner_product_L2(fi, fj, self.left, self.right)
        #         if i == j:
        #             assert_almost_equal(res, 1.0)
        #         else:
        #             assert_almost_equal(res, 0.0)


# nw = DirichletWell2D(0.0, 1.0, 0.0, 1.0, 10)
# gfdy = nw.greens_function_helmholtz_dy(1.0)
# for x in np.arange(0.0, 1.0, 0.1):
#     for y in np.arange(0.0, 1.0, 0.1):
#         print(gfdy(x, y, 0.5, 0.0))
#


# print(nw.greens_function_helmholtz(20.0)(1.0, 1.0, 1.0, 1.0))

# for maxn in range(100, 2000, 100):
#     nw = NeumannWell2D(0.0, 2.0, 0.0, 2.0, maxn)
#     print(nw.greens_function_helmholtz(20.0)(0.0, 0.0, 0.0, 0.0))
