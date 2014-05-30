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


class NeumannWell1D:
    """
        Boundary conditions:
        psi'[left] = 0
        psi'[right] = 0
    """
    def __init__(self, fromm: real, to: real, maxn: int):
        width = to - fromm
        assert width > 0

        self.fromm = fromm
        self.to = to
        self.maxn = maxn

        self.eigenenergies = [(sc.pi * n / width) ** 2 for n in range(self.maxn)]

        self.eigenfunctions = []
        for n in range(self.maxn):
            # TODO domain checking??
            if n == 0:
                self.eigenfunctions.append(lambda x: sqrt(1.0 / width))
            else:
                self.eigenfunctions.append(lambda x, n=n: sqrt(2.0 / width) * cos(sc.pi * n / width * (x - self.fromm)))

    def greens_function_helmholtz_at_center(self, energy):
        k = np.sqrt(complex(energy))
        coeff = 1 / (2 * k)
        if energy > -10000:
            return coeff / np.tan(k * (self.fromm - self.to) / 2)
        else:
            return coeff * -1 / I

    # TODO test
    def greens_function_helmholtz(self, energy, tolerance=1e-5):
        k = np.sqrt(complex(energy))
        center = (self.fromm + self.to) / 2
        def fun(x, xs):
            if abs(x - center) < tolerance and abs(xs - center) < tolerance:
                return self.greens_function_helmholtz_at_center(energy)
            else:
                coeff = k * np.sin(k * (self.fromm - self.to))
                if x < xs:
                    return np.cos(k * (x - self.fromm)) * np.cos(k * (xs - self.to)) / coeff
                else:
                    return np.cos(k * (x - self.to)) * np.cos(k * (xs - self.fromm)) / coeff
        return fun



    def test(self):
        """
        Checks if the eigenfunctions are orthonormal
        """
        tolerance = 1e-3
        for i, fi in enumerate(self.eigenfunctions):
            for j, fj in enumerate(self.eigenfunctions):
                res = inner_product_L2(fi, fj, self.fromm, self.to)
                if i == j:
                    assert_almost_equal(res, 1.0)
                else:
                    assert_almost_equal(res, 0.0)

    def test2(self):
        energy = -10.0
        gf = self.greens_function_helmholtz(energy)
        gfc = self.greens_function_helmholtz_at_center(energy)
        assert_almost_equal(gfc, gf((self.fromm + self.to) / 2, (self.fromm + self.to) / 2))

if __name__ == '__main__':
    nw = NeumannWell1D(0.0, 2.0, 100)
    nw.test2()
    energy = -100000
    k = np.sqrt(complex(energy))
    # energy = -32262374
    print(nw.greens_function_helmholtz(energy)(1.0, 1.0))
    print(1 / (2 * k) * -1 / I)
    # print(nw.greens_function_helmholtz_at_center(energy)())