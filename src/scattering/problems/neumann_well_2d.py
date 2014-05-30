from sys import stderr

import scipy.constants as sc

import numpy as np
from numpy import sqrt, exp, cos
from numpy import complex256 as complex
from numpy import float128 as real
from numpy.testing import assert_almost_equal

# TODO hbar and mass
from scattering.problems.neumann_well_1d import NeumannWell1D
from scattering.tools import integrate_complex
from scattering.tools.tools import inner_product_L2


class NeumannWell2D:
    """
        Boundary conditions:
        psi'[left] = 0
        psi'[right] = 0
    """
    def __init__(self, fromX: real, toX: real, fromY: real, toY: real, maxn: int):
        self.fromX = fromX
        self.toX = toX
        self.fromY = fromY
        self.toY = toY
        self.maxn = maxn

        self.wellX = NeumannWell1D(fromX, toX, maxn)
        self.wellY = NeumannWell1D(fromY, toY, maxn)

        self.eigenenergies = None # ???
        self.eigenfunctions = None # ???

    # TODO test
    def greens_function_helmholtz(self, energy, maxn=None):
        if maxn is None:
            maxn = self.maxn
        def fun(x, y, xs, ys):
            res = complex(0.0)
            for n in range(maxn):
                # stderr.write("{}: {}\n".format(n, str(energy - self.wellY.eigenenergies[n])))
                gf = self.wellY.greens_function_helmholtz(energy - self.wellY.eigenenergies[n])
                res += self.wellX.eigenfunctions[n](x) * np.conj(self.wellX.eigenfunctions[n](xs)) * gf(y, ys)
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

if __name__ == '__main__':
    nw = NeumannWell2D(0.0, 2.0, 0.0, 2.0, 10000)
    print(nw.greens_function_helmholtz(20.0)(1.0, 1.0, 1.0, 1.0))

# for maxn in range(100, 2000, 100):
#     nw = NeumannWell2D(0.0, 2.0, 0.0, 2.0, maxn)
#     print(nw.greens_function_helmholtz(20.0)(0.0, 0.0, 0.0, 0.0))
