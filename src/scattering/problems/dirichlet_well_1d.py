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


class DirichletWell1D:
    """
        Boundary conditions:
        psi[left] = 0
        psi[right] = 0
    """
    def __init__(self, a: real, b: real, maxn: int):
        width = b - a
        assert width > 0

        self.a = a
        self.b = b
        self.maxn = maxn

        self.wavevectors = [None]  # to make modes 1-indexed
        self.wavevectors.extend([sc.pi * n / width for n in range(1, self.maxn)])

        self.eigenenergies = [None]  # to make modes 1-indexed
        self.eigenenergies.extend([kk ** 2 for kk in self.wavevectors[1:]])

        self.eigenfunctions = [None]  # to make modes 1-indexed
        self.eigenfunctions.extend([lambda x, kk=kk: sqrt(2.0 / width) * np.sin(kk * (x - self.a)) for kk in self.wavevectors[1:]])

        self.deigenfunctions = [None]  # to make modes 1-indexed
        self.deigenfunctions.extend([lambda x, kk=kk: sqrt(2.0 / width) * kk * np.cos(kk * (x - self.a)) for kk in self.wavevectors[1:]])


    # def greens_function_helmholtz_at_center(self, energy):
    #     k = np.sqrt(complex(energy))
    #     coeff = 1 / (2 * k)
    #     if energy > -10000:
    #         return coeff / np.tan(k * (self.fromm - self.to) / 2)
    #     else:
    #         return coeff * -1 / I

    # # TODO test
    # def greens_function_helmholtz(self, energy, tolerance=1e-5):
    #     k = np.sqrt(complex(energy))
    #     def fun(x, xs):
    #         coeff = k * np.sin(k * (self.fromm - self.to))
    #         if x < xs:
    #             return np.cos(k * (x - self.fromm)) * np.cos(k * (xs - self.to)) / coeff
    #         else:
    #             return np.cos(k * (x - self.to)) * np.cos(k * (xs - self.fromm)) / coeff
    #     return fun

    def greens_function_helmholtz(self, energy):
        k = np.sqrt(complex(energy))
        # TODO normalization
        def fun(x, xs):
            if x < xs:
                return -np.sin(k * (x - self.a) * np.sin(k * (xs - self.b))) / (k * np.sin(k * (self.b - self.a)))
            else:
                return -np.sin(k * (x - self.b) * np.sin(k * (xs - self.a))) / (k * np.sin(k * (self.b - self.a)))
        return fun

    def greens_function_helmholtz_dx(self, energy):
        k = np.sqrt(complex(energy))
        # TODO normalization
        def fun(x, xs):
            if x < xs:
                return -np.sin(k * (x - self.a) * np.cos(k * (xs - self.b))) / np.sin(k * (self.b - self.a))
            else:
                return -np.sin(k * (x - self.b) * np.cos(k * (xs - self.a))) / np.sin(k * (self.b - self.a))
        return fun

    def test(self):
        """
        Checks if the eigenfunctions are orthonormal
        """
        tolerance = 1e-3
        for i, fi in enumerate(self.eigenfunctions[1:]):
            for j, fj in enumerate(self.eigenfunctions[1:]):
                res = inner_product_L2(fi, fj, self.a, self.b)
                if i == j:
                    assert_almost_equal(res, 1.0)
                else:
                    assert_almost_equal(res, 0.0)

    # def test2(self):
    #     energy = -10.0
    #     gf = self.greens_function_helmholtz(energy)
    #     gfc = self.greens_function_helmholtz_at_center(energy)
    #     assert_almost_equal(gfc, gf((self.fromm + self.to) / 2, (self.fromm + self.to) / 2))

if __name__ == '__main__':
    nw = DirichletWell1D(0.0, 2.0, 10)
    nw.test()