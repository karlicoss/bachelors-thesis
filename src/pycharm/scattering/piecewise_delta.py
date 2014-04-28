from pprint import pprint
import numpy as np
from numpy import linalg
from numpy.core.umath import sqrt, exp
from scipy import constants as sc, constants
from scattering.tools import I, cnorm2, ScatteringResult, jn_zeros, jn

class PiecewiseDeltaCylinderScattering:
    """
        df: [0, RR] -> {0, 1}, returns if there is a delta at point (0, r)
        intf: f1, f2 -> float
        m : int
    """
    def __init__(self, mu, RR, uu, intf, m, maxn):
        self.mu = mu

        self.RR = RR
        self.uu = uu

        self.intf = intf

        self.m = m
        self.maxn = maxn

        self.jzeros = jn_zeros(m, maxn)

        ## !!! n scope interference
        self.phis = [self.get_phi_function(n) for n in range(1, self.maxn + 1)]
        self.phi_energies = [sc.hbar ** 2 / (2 * self.mu) * (self.jzeros[n - 1] / self.RR) ** 2 for n in
                             range(1, self.maxn + 1)]

    # phis are orthonormalized w.r.t. to weight function r
    # n: int
    def get_phi_function(self, n):
        coeff = sqrt(2.0) / (self.RR * jn(abs(self.m) + 1, self.jzeros[n - 1]))

        def fun(r):
            return complex(coeff * jn(self.m, self.jzeros[n - 1] * r / self.RR))

        return fun

    def compute_scattering_full(self, energy):
        res = []
        for i, phiE in enumerate(self.phi_energies):
            if phiE < energy:
                res.append(self.compute_scattering(i + 1, energy))
        T = 0.0
        for r in res:
            T += r.T

        # print("Energy = {} eV, T = {}".format(energy / sc.eV, T))
        return T

    def compute_scattering(self, n, energy, verbose=False):
        assert (energy > self.phi_energies[n - 1])

        if verbose:
            print("-------------------")
            print("Energy: {} eV".format(energy / sc.eV))  # TODO eV
        kks = [sqrt(complex(2 * self.mu * (energy - phiE))) / sc.hbar for phiE in self.phi_energies]

        if verbose:
            print("Wavevectors:")
            print(kks)

        integrals = [[None for j in range(self.maxn)] for i in range(self.maxn)]
        for i in range(self.maxn):
            for j in range(i, self.maxn):
                integrals[i][j] = self.intf(self.phis[i], self.phis[j])
                integrals[j][i] = integrals[i][j]

        if verbose:
            print("Integrals:")
            pprint(integrals)

        A = np.zeros((self.maxn, self.maxn), dtype=np.complex64)
        B = np.zeros(self.maxn, dtype=np.complex64)
        for i in range(1, self.maxn + 1):
            A[i - 1, i - 1] += 2 * kks[i - 1]
            for j in range(1, self.maxn + 1):
                A[i - 1, j - 1] -= 2 * self.mu / sc.hbar ** 2 * self.uu / I * integrals[i - 1][j - 1]
            B[i - 1] = 2 * self.mu / sc.hbar ** 2 * self.uu / I * integrals[n - 1][i - 1]

        Rs = linalg.solve(A, B)
        Ts = np.zeros(self.maxn, dtype=np.complex64)

        for i in range(1, self.maxn + 1):
            if i == n:
                Ts[i - 1] = Rs[i - 1] + 1
            else:
                Ts[i - 1] = Rs[i - 1]

        if verbose:
            print("Rs:")
            print(Rs)
            print("Ts:")
            print(Ts)

        ## calculation of total transmission coefficient
        T = 0.0
        for i, t in enumerate(Ts):
            if self.phi_energies[i] < energy:  # open channel
                T += (kks[i] / kks[n - 1] * cnorm2(t)).real
        if verbose:
            print("Total transmission coefficient:")
            print(T)
        ##


        def psi1(z, r):
            res = complex(0.0)
            res += exp(I * kks[n - 1] * z) * self.phis[n - 1](r)
            for i in range(1, self.maxn + 1):
                res += Rs[i - 1] * exp(-I * kks[i - 1] * z) * self.phis[i - 1](r)
            return res

        def psi2(z, r):
            res = complex(0.0)
            for i in range(1, self.maxn + 1):
                res += Ts[i - 1] * exp(I * kks[i - 1] * z) * self.phis[i - 1](r)
            return res

        def psi(z, r):
            if z < 0:
                return psi1(z, r)
            else:
                return psi2(z, r)

        if verbose:
            print("-------------------")
        return ScatteringResult(wf=psi, T=T)