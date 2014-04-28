from pprint import pprint
import numpy as np
from numpy import arange, eye, linalg
from numpy import sqrt, exp
from numpy import complex256 as complex
from numpy import float128 as real

import scipy as sp
import scipy.constants as sc


from scattering.tools import jn_zeros, jn, I, cnorm2, ScatteringResult


class DoubleDeltaCylinderScattering:
    def __init__(self, mu, RR, u1, u2, a, b, intf1, intf2, m, maxn):
        self.mu = mu
        self.RR = RR

        self.u1 = u1
        self.u2 = u2

        self.a = a
        self.b = b

        self.intf1 = intf1
        self.intf2 = intf2

        self.m = m
        self.maxn = maxn

        self.jzeros = jn_zeros(m, maxn)

        ## !!! n scope interference
        self.phis = [self.get_phi_function(n) for n in range(1, self.maxn + 1)]
        self.phi_energies = [sc.hbar ** 2 / (2 * self.mu) * (self.jzeros[n - 1] / self.RR) ** 2 for n in
                             range(1, self.maxn + 1)]

    # TODO: copy-pasting :(
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

        print("Energy = {} eV, T = {}".format(energy / sc.eV, T))
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

        integrals1 = [[None for j in range(self.maxn)] for i in range(self.maxn)]
        for i in range(self.maxn):
            for j in range(i, self.maxn):
                integrals1[i][j] = self.intf1(self.phis[i], self.phis[j])
                integrals1[j][i] = integrals1[i][j]

        integrals2 = [[None for j in range(self.maxn)] for i in range(self.maxn)]
        for i in range(self.maxn):
            for j in range(i, self.maxn):
                integrals2[i][j] = self.intf2(self.phis[i], self.phis[j])
                integrals2[j][i] = integrals2[i][j]

        if verbose:
            print("Integrals 1:")
            pprint(integrals1)

            print("Integrals 2:")
            pprint(integrals2)

        varR = [0 * self.maxn + i for i in range(self.maxn)]
        varT = [1 * self.maxn + i for i in range(self.maxn)]
        varA = [2 * self.maxn + i for i in range(self.maxn)]
        varB = [3 * self.maxn + i for i in range(self.maxn)]

        A = np.zeros((4 * self.maxn, 4 * self.maxn), dtype=np.complex64)
        B = np.zeros(4 * self.maxn, dtype=np.complex64)

        # Continuity equations for regions 1-2
        shift = 0
        for i in range(0, self.maxn):
            A[shift + i][varR[i]] += exp(-I * kks[i] * self.a)
            A[shift + i][varA[i]] -= exp(- I * kks[i] * self.a)
            A[shift + i][varB[i]] -= exp(I * kks[i] * self.a)
            if i == n - 1:
                B[shift + i] = - exp(I * kks[n - 1] * self.a)
            else:
                B[shift + i] = 0.0

        # Continuity equations for regions 2-3
        shift = self.maxn
        for i in range(0, self.maxn):
            A[shift + i][varA[i]] += exp(-I * kks[i] * self.b)
            A[shift + i][varB[i]] += exp(I * kks[i] * self.b)
            A[shift + i][varT[i]] -= exp(I * kks[i] * self.b)
            B[shift + i] = 0.0

        # Derivative discontinuity for regions 1-2
        shift = 2 * self.maxn
        for i in range(0, self.maxn):
            A[shift + i][varR[i]] += -I * kks[i] * exp(-I * kks[i] * self.a)
            A[shift + i][varA[i]] -= -I * kks[i] * exp(-I * kks[i] * self.a)
            A[shift + i][varB[i]] -= I * kks[i] * exp(I * kks[i] * self.a)

            for j in range(0, self.maxn):
                A[shift + i][varA[j]] += 2 * self.mu / sc.hbar ** 2 * self.u1 * integrals1[i][j] * exp(
                    -I * kks[j] * self.a)
                A[shift + i][varB[j]] += 2 * self.mu / sc.hbar ** 2 * self.u1 * integrals1[i][j] * exp(I * kks[j] * self.a)

            if i == n - 1:
                B[shift + i] = -I * kks[n - 1] * exp(I * kks[n - 1] * self.a)
            else:
                B[shift + i] = 0.0

        # Derivative discontinuity for regions 2-3
        shift = 3 * self.maxn
        for i in range(0, self.maxn):
            A[shift + i][varA[i]] += -I * kks[i] * exp(-I * kks[i] * self.b)
            A[shift + i][varB[i]] += I * kks[i] * exp(I * kks[i] * self.b)
            A[shift + i][varT[i]] -= I * kks[i] * exp(I * kks[i] * self.b)

            for j in range(0, self.maxn):
                A[shift + i][varA[j]] += 2 * self.mu / sc.hbar ** 2 * self.u2 * integrals2[i][j] * exp(
                    -I * kks[j] * self.b)
                A[shift + i][varB[j]] += 2 * self.mu / sc.hbar ** 2 * self.u2 * integrals2[i][j] * exp(I * kks[j] * self.b)

            B[shift + i] = 0.0

        # print(A)
        # print(B)

        Xs = linalg.solve(A, B)
        Rs = Xs[0 * self.maxn: 1 * self.maxn]
        Ts = Xs[1 * self.maxn: 2 * self.maxn]
        As = Xs[2 * self.maxn: 3 * self.maxn]
        Bs = Xs[3 * self.maxn: 4 * self.maxn]

        if verbose:
            print("Rs:")
            print(Rs)
            print("Ts:")
            print(Ts)
            print("As:")
            print(As)
            print("Bs:")
            print(Bs)

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
            for i in range(0, self.maxn):
                res += Rs[i] * exp(-I * kks[i] * z) * self.phis[i](r)
            return res

        def psi2(z, r):
            res = complex(0.0)
            for i in range(0, self.maxn):
                res += As[i] * exp(-I * kks[i] * z) * self.phis[i](r)
                res += Bs[i] * exp(I * kks[i] * z) * self.phis[i](r)
            return res

        def psi3(z, r):
            res = complex(0.0)
            for i in range(0, self.maxn):
                res += Ts[i] * exp(I * kks[i] * z) * self.phis[i](r)
            return res

        def psi(z, r):
            if z < self.a:
                return psi1(z, r)
            elif z < self.b:
                return psi2(z, r)
            else:
                return psi3(z, r)

        if verbose:
            print("-------------------")
        return ScatteringResult(wf=psi, T=T)
