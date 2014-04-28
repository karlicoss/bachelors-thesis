from sys import stderr

import scipy.constants as sc

import numpy as np
from numpy import sqrt, exp, cos
from numpy import complex256 as complex
from numpy import float128 as real
from numpy.testing import assert_almost_equal

# TODO hbar and mass
from scattering.tools import integrate_complex
from scattering.tools.tools import inner_product_L2


class NeumannWell1D:
    """
        Boundary conditions:
        psi'[left] = 0
        psi'[right] = 0
    """
    def __init__(self, left: real, right: real, maxn: int):
        width = right - left
        assert width > 0

        self.left = left
        self.right = right
        self.maxn = maxn
        self.eigenenergies = [(sc.pi * n / width) ** 2 for n in range(self.maxn)]
        self.eigenfunctions = []
        for n in range(self.maxn):
            # TODO domain checking??
            if n == 0:
                self.eigenfunctions.append(lambda x: sqrt(1.0 / width))
            else:
                self.eigenfunctions.append(lambda x, n=n: sqrt(2.0 / width) * cos(sc.pi * n / width * (x - self.left)))

    def test(self):
        """
        Checks if the eigenfunctions are orthonormal
        """
        tolerance = 1e-3
        for i, fi in enumerate(self.eigenfunctions):
            for j, fj in enumerate(self.eigenfunctions):
                res = inner_product_L2(fi, fj, self.left, self.right)
                if i == j:
                    assert_almost_equal(res, 1.0)
                else:
                    assert_almost_equal(res, 0.0)
