from collections import namedtuple
from pprint import pprint

import numpy as np
from numpy import arange, eye, linalg
from numpy import complex_ as complex  # TODO not sure if that's ok
from numpy import sqrt, exp

import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm, Normalize

import scipy as sp
import scipy.integrate
import scipy.constants as sc
import scipy.special


def cnorm(z):
    return (z * np.conj(z)).real


I = complex(1j)  # TODO not sure if ok

jn_zeros = lambda x, y: scipy.special.jn_zeros(x, int(y))
jn = lambda x, y: scipy.special.jn(x, float(y))

ScatteringResult = namedtuple('ScatteringResult', ['wf', 'T'])


# TODO
# hbar = 1.0
hbar = sc.hbar


def integrate_complex(f, a, b, **kwargs):
    real_integrand = scipy.integrate.quad(lambda x: sp.real(f(x)), a, b, **kwargs)
    imag_integrand = scipy.integrate.quad(lambda x: sp.imag(f(x)), a, b, **kwargs)
    return (real_integrand[0] + I * imag_integrand[0], real_integrand[1] + I * imag_integrand[1])


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
        self.phi_energies = [hbar ** 2 / (2 * self.mu) * (self.jzeros[n - 1] / self.RR) ** 2 for n in
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
        kks = [sqrt(complex(2 * self.mu * (energy - phiE))) / hbar for phiE in self.phi_energies]

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
                A[shift + i][varA[j]] += 2 * self.mu / hbar ** 2 * self.u1 * integrals1[i][j] * exp(
                    -I * kks[j] * self.a)
                A[shift + i][varB[j]] += 2 * self.mu / hbar ** 2 * self.u1 * integrals1[i][j] * exp(I * kks[j] * self.a)

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
                A[shift + i][varA[j]] += 2 * self.mu / hbar ** 2 * self.u2 * integrals2[i][j] * exp(
                    -I * kks[j] * self.b)
                A[shift + i][varB[j]] += 2 * self.mu / hbar ** 2 * self.u2 * integrals2[i][j] * exp(I * kks[j] * self.b)

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
                T += (kks[i] / kks[n - 1] * cnorm(t)).real
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


# df: [0, RR] -> {0, 1}, returns if there is a delta at point (0, r)
# intf: f1, f2 -> float
# m : int
class PiecewiseDeltaCylinderScattering:
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
        self.phi_energies = [hbar ** 2 / (2 * self.mu) * (self.jzeros[n - 1] / self.RR) ** 2 for n in
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
        kks = [sqrt(complex(2 * self.mu * (energy - phiE))) / hbar for phiE in self.phi_energies]

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
                A[i - 1, j - 1] -= 2 * self.mu / hbar ** 2 * self.uu / I * integrals[i - 1][j - 1]
            B[i - 1] = 2 * self.mu / hbar ** 2 * self.uu / I * integrals[n - 1][i - 1]

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
                T += (kks[i] / kks[n - 1] * cnorm(t)).real
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


# df: [0, RR] -> {0, 1}, returns if there is a delta at point (0, r)
# intf: f1, f2 -> float
# m : int
# Lx, Ly, H
class ResonatorRectangular2DScattering:
    def __init__(self, mu, H, Lx, Ly, maxn):
        self.mu = mu

        self.H = H
        self.Lx = Lx
        self.Ly = Ly

        self.x0 = self.Lx / 2
        self.y0 = 0


        self.maxn = maxn

        def well_energy(width, n):
            return hbar ** 2 / (2 * self.mu) * (sc.pi * n / width) ** 2

        def well_neumann_mode(width, n):
            k = sc.pi * n / width
            coeff = sqrt(2 / width)
            def fun(x):
                return coeff * np.cos(k * x)
            return fun


        # Wire
        self.wirey_energies = [well_energy(self.H, n) for n in range(self.maxn)]
        self.wirey_modes = [well_neumann_mode(self.H, n) for n in range(self.maxn)]

        # Resonatos
        self.resx_energies = [well_energy(self.Lx, n) for n in range(self.maxn)]
        self.resy_energies = [well_energy(self.Ly, n) for n in range(self.maxn)]

        self.resx_modes = [well_neumann_mode(self.Lx, n) for n in range(self.maxn)]
        self.resy_modes = [well_neumann_mode(self.Ly, n) for n in range(self.maxn)]

        print([i / sc.eV for i in self.wirey_energies])
        print([i / sc.eV for i in self.resx_energies])
        print([i / sc.eV for i in self.resy_energies])


    def compute_scattering_full(self, energy):
        raise NotImplementedError
        res = []
        for i, phiE in enumerate(self.phi_energies):
            if phiE < energy:
                res.append(self.compute_scattering(i + 1, energy))
        T = 0.0
        for r in res:
            T += r.T

        # print("Energy = {} eV, T = {}".format(energy / sc.eV, T))
        return T

    def compute_greens_function_wirex(self, energy, n):
        kk = sqrt(complex(2 * self.mu * (energy - self.wirey_energies[n]))) / hbar
        coeff = -2 * self.mu / hbar ** 2 * I / (2 * kk) # TODO negate?
        def fun(x, s):
            return coeff * exp(I * kk * abs(x - s))
        return fun

    def compute_greens_function_wire(self, energy):
        greenxs = [self.compute_greens_function_wirex(energy, n) for n in range(self.maxn)]
        def fun(x, y, xs, ys):
            s = complex(0.0)
            for n in range(0, self.maxn):
                s += self.resy_modes[n](y) * np.conj(self.resy_modes[n](ys)) * greenxs[n](x, xs)
            return s
        return fun

    def compute_greens_function_resonator(self, energy):
        def fun(x, y, xs, ys):
            s = complex(0.0)
            for n in range(0, self.maxn):
                for m in range(0, self.maxn):
                    s += (self.resx_modes[n](x) * np.conj(self.resx_modes[n](xs)) *
                          self.resy_modes[m](y) * np.conj(self.resy_modes[m](ys))) /\
                         (self.resx_energies[n] + self.resy_energies[m] - energy)
            return s
        return fun


    # TODO confusion with modes indexing :(
    # This one should be zero-indexed
    def compute_scattering(self, n, energy, verbose=False):
        assert (energy > self.wirey_energies[n])

        if verbose:
            print("-------------------")
            print("Energy: {} eV".format(energy / sc.eV))

        # incoming wavefunction
        kk = sqrt(2 * self.mu * (energy - self.wirey_energies[n])) / hbar
        uwf = lambda x, y: self.wirey_modes[n](y) * exp(I * kk * x)

        delta = 0.01 * sc.nano
        gamma = 0.57721566490153286060
        k0 = I / delta * exp(-gamma) # TODO
        e0 = hbar ** 2 * k0 ** 2 / (2 * self.mu)

        greens_wire = self.compute_greens_function_wire(energy)
        greens_wire0 = self.compute_greens_function_wire(e0)
        greens_resonator = self.compute_greens_function_resonator(energy)
        greens_resonator0 = self.compute_greens_function_resonator(e0)

        AA = greens_wire(self.x0, self.y0, self.x0, self.y0) - greens_wire0(self.x0, self.y0, self.x0, self.y0)
        BB = greens_resonator(self.x0, self.y0, self.x0, self.y0) - greens_resonator0(self.x0, self.y0, self.x0, self.y0)

        a1 = -uwf(self.x0, self.y0) / (AA + BB)
        a2 = -a1

        def psi(x, y):
            if y < -self.H:
                return complex(0.0)
            elif y < 0:
                return uwf(x, y) + a1 * greens_wire(x, y, self.x0, self.y0)
            elif y < self.Ly:
                if x < 0:
                    return complex(0.0)
                elif x < self.Lx:
                    return a2 * greens_resonator(x, y, self.x0, self.y0)
                else:
                    return complex(0.0)
            else:
                return complex(0.0)

        return ScatteringResult(wf=psi, T=-1.0)

def plot_transmission(dcs, left, right, step, maxt = None, fname="transmission.png", info = None):
    if maxt is None:
        maxt = dcs.maxn

    xs = arange(left, right, step)
    ys = list(map(lambda en: dcs.compute_scattering_full(en), xs))

    fig = plt.figure(figsize=(15, 10), dpi=500)
    ax = fig.add_subplot(111)  # , aspect = 'equal'

    ax.vlines(dcs.phi_energies, 0.0, dcs.maxn)

    xticks = arange(left, right, 0.1 * sc.eV)
    xlabels = ["{:.1f}".format(t / sc.eV) for t in xticks]
    ax.set_xlim(left, right)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("E, eV")

    yticks = arange(0.0, maxt, 1.0)
    ylabels = ["{:.1f}".format(t) for t in yticks]
    ax.set_ylim(0.0, maxt)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("T")

    cax = ax.plot(xs, ys)
    ax.set_title("Transmission coefficient, m = {}".format(dcs.m) + ("" if info is None else "\n" + info))

    fig.savefig(fname)
    plt.close(fig)

# TODO MODIFY PLOT_WAVEFUNCTION USAGES
def plot_wavefunction(dcs, n, energy, fx, tx, dx, fy, ty, dy, fname="wavefunction.png", info=None):
    res = dcs.compute_scattering(n, energy, verbose=False)

    T = res.T
    wf = res.wf
    pf = lambda z, r: cnorm(wf(z, r))
    vpf = np.vectorize(pf)

    print("Total transmission: {}".format(T))

    x, y = np.mgrid[slice(fx, tx + dx, dx), slice(fy, ty + dy, dy)]
    z = vpf(x, y)

    z_min, z_max = np.abs(z).min(), np.abs(z).max()

    fig = plt.figure(figsize=(15, 10), dpi=500)
    ax = fig.add_subplot(211)  # , aspect = 'equal'

    ax.set_title("n = {}, E = {:.2f} eV, T = {:.2f}".format(n, energy / sc.eV, T) + (
        "" if info is None else "\n{}".format(info)))

    xticks = arange(fx, tx, 1 * sc.nano)
    xlabels = ["{:.1f}".format(t / sc.nano) for t in xticks]
    ax.set_xlim(fx, tx)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    yticks = arange(fy, ty, sc.nano)
    ylabels = ["{:.1f}".format(t / sc.nano) for t in yticks]
    ax.set_ylim(fy, ty)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    cax = ax.pcolor(x, y, z, cmap='gnuplot', norm=Normalize(z_min, z_max))

    cbar = fig.colorbar(cax)

    bx = fig.add_subplot(212)
    bx.set_title("Wavefunction at x = 0")

    rr = arange(fy, ty, 0.01 * sc.nano)
    ww = list(map(lambda r: pf(0, r), rr))
    bx.plot(rr, ww)

    fig.savefig(fname)
    plt.close(fig)


def test_racec(maxn):
    # Same radius as the host cylinder
    def test_same(maxn):
        print("Same radius as the host cylinder")

        cnt = 0

        RR = 5 * sc.nano
        dleft = -4.0 * sc.nano
        dright = 4.0 * sc.nano
        mu = 0.19 * sc.m_e

        intf = lambda f, g: integrate_complex(lambda r: r * f(r) * g(r), 0, RR)[0]

        for m in [0, 1]:
            for uu_ in [0.0, -0.05, -0.5]:
                print("m = {}, uu_ = {:.2f} eV".format(m, uu_))
                uu = uu_ * sc.eV * (dright - dleft)

                dcs = PiecewiseDeltaCylinderScattering(mu, RR, uu, intf, m, maxn)
                plot_transmission(dcs,
                                  0.0 * sc.eV, 1.0 * sc.eV, 0.001 * sc.eV,
                                  maxt=3.0,
                                  fname="racec/transmission_same{:02d}".format(cnt),
                                  info="Depth {:.2f} eV; m = {}".format(uu_, m))
                cnt += 1

    # Surrounded by the host material
    def test_surrounded(maxn):
        print("Surrounded by the host material")

        cnt = 0

        R = 1 * sc.nano
        RR = 5 * sc.nano
        dleft = -4.0 * sc.nano
        dright = 4.0 * sc.nano
        mu = 0.19 * sc.m_e

        intf = lambda f, g: integrate_complex(lambda r: r * f(r) * g(r), 0, R)[0]

        for m in [0, 1]:
            for uu_ in [-0.15]: # TODO -0.05
                print("m = {}, uu_ = {:.2f} eV".format(m, uu_))
                uu = uu_ * sc.eV * (dright - dleft)

                dcs = PiecewiseDeltaCylinderScattering(mu, RR, uu, intf, m, maxn)
                plot_transmission(dcs,
                                  0.0 * sc.eV, 1.0 * sc.eV, 0.0001 * sc.eV,
                                  maxt=3.0,
                                  fname="racec/transmission_surrounded{:02d}".format(cnt),
                                  info="Depth {:.2f} eV; m = {}".format(uu_, m))
                cnt += 1


    # test_same(maxn)
    test_surrounded(maxn)

def test_resonator():
    Lx = 5 * sc.nano
    Ly = 5 * sc.nano
    H = 5 * sc.nano
    mu = 0.19 * sc.m_e
    maxn = 40
    sp = ResonatorRectangular2DScattering(mu, H, Lx, Ly, maxn)

    energy = 0.5 * sc.eV
    fx = -10.0 * sc.nano
    tx = 15.0 * sc.nano
    dx = 0.1 * sc.nano
    fy = -H
    ty = Ly
    dy = 0.1 * sc.nano

    n = 2

    plot_wavefunction(sp, n, energy,
                      fx, tx, dx,
                      fy, ty, dy)


def test_cylinder(maxn):
    RR = 5.0 * sc.nano
    R = RR
    uu = -0.4 * sc.nano * sc.eV
    m = 0
    mu = 0.19 * sc.m_e # mass

    intf = lambda f, g: integrate_complex(lambda r: r * f(r) * g(r), 0, R)[0]
    dcs = PiecewiseDeltaCylinderScattering(mu, RR, uu, intf, m, maxn)

    print("Transversal mode energies:")
    print([en / sc.eV for en in dcs.phi_energies])

    print(dcs.compute_scattering_full(0.5 * sc.eV))
    # plot_transmission(dcs,
    #                   0.0 * sc.eV, 1.0 * sc.eV, 0.01 * sc.eV,
    #                   maxt=3.0)
    # energy = 0.2179999 * sc.eV
    # energy = 0.244 * sc.eV
    # n = 1
    # plot_wavefunction(dcs, n, energy)


# plot_transmission(dcs, 0.0 * sc.eV, 1.0 * sc.eV, 0.001 * sc.eV)

#######
# n = 1
# for ee in arange(0.05, 1.0, 0.01):
# 	print("Plotting for energy = {:.2f} eV".format(ee))
# 	energy = ee * sc.eV
# 	plot_wavefunction(n, energy, "{:.2f}".format(ee))
# #######

def test_double_delta_slits():
    RR = 5.0 * sc.nano
    # R = 5.0 * sc.nano
    swidth = 0.0000 * sc.nano

    u1 = 10000.0 * sc.nano * sc.eV
    a = -4.0 * sc.nano
    intf1 = lambda f, g: integrate_complex(lambda r: r * f(r) * g(r), swidth, RR)[0]

    u2 = 10000.0 * sc.nano * sc.eV
    b = 4.0 * sc.nano
    intf2 = lambda f, g: integrate_complex(lambda r: r * f(r) * g(r), swidth, RR)[0]

    m = 0
    mu = 0.19 * sc.m_e  # mass
    maxn = 10

    dcs = DoubleDeltaCylinderScattering(mu, RR, u1, u2, a, b, intf1, intf2, m, maxn)
    print("Transversal mode energies:")
    print([en / sc.eV for en in dcs.phi_energies])

    # f = 0.81 * sc.eV
    # t = 0.83 * sc.eV
    # step = 0.0001 * sc.eV
    # plot_transmission(dcs, f, t, step)


    n = 1
    d = 8 * sc.nano
    dz = 0.1 * sc.nano
    dr = 0.1 * sc.nano
    energy = 0.0773102 * sc.eV
    plot_wavefunction(dcs, n, energy, -d, d, dz, dr,
                      info="u1 = {:.2f} nm * eV (at a = {:.2f}) nm\nu2 = {:.2f} nm * eV (at b = {:.2f})\nslit width = {:.2f} nm".format(
                          u1 / sc.nano / sc.eV,
                          a / sc.nano,
                          u2 / sc.nano / sc.eV,
                          b / sc.nano,
                          swidth / sc.nano))


# for energy in arange(0.0773099 * sc.eV, 0.077311 * sc.eV, 0.0000001 * sc.eV):
# 	print("Energy = {:.7f} eV".format(energy / sc.eV))
# 	plot_wavefunction(dcs, n, energy, -d, d, dz, dr,
# 		info = "u1 = {:.2f} nm * eV (at a = {:.2f}) nm\nu2 = {:.2f} nm * eV (at b = {:.2f})\nslit width = {:.2f} nm".format(
# 			u1 / sc.nano / sc.eV,
# 			a / sc.nano,
# 			u2 / sc.nano / sc.eV,
# 			b / sc.nano,
# 			swidth / sc.nano),
# 		fname = "output/{:.2f}.png".format(energy / sc.eV))


def test_double_delta_barrier():
    RR = 5.0 * sc.nano
    R = 5.0 * sc.nano

    u1 = 2.0 * sc.nano * sc.eV
    a = -4.0 * sc.nano
    intf1 = lambda f, g: integrate_complex(lambda r: r * f(r) * g(r), 0, R)[0]

    u2 = 2.0 * sc.nano * sc.eV
    b = 4.0 * sc.nano
    intf2 = lambda f, g: integrate_complex(lambda r: r * f(r) * g(r), 0, R)[0]

    m = 0
    mu = 0.19 * sc.m_e  # mass
    maxn = 5

    dcs = DoubleDeltaCylinderScattering(mu, RR, u1, u2, a, b, intf1, intf2, m, maxn)
    print("Transversal mode energies:")
    print([en / sc.eV for en in dcs.phi_energies])

    plot_transmission(dcs, 0.0 * sc.eV, 1.0 * sc.eV, 0.00005 * sc.eV)


# n = 1
# # energy = 0.05968299 * sc.eV # resonance 11
# # energy = 0.09960 * sc.eV # resonance 12
# d = 8 * sc.nano
# dz = 0.1 * sc.nano
# dr = 0.1 * sc.nano
# plot_wavefunction(dcs, n, energy, -d, d, dz, dr)

def test_slit():
    R = 2.0 * sc.nano
    RR = 3.0 * sc.nano
    uu = -0.1 * sc.nano * sc.eV
    mu = 0.19 * sc.m_e
    m = 0
    maxn = 0

    intf = lambda f, g: integrate_complex(lambda r: r * f(r) * g(r), R, RR)[0]
    dcs = PiecewiseDeltaCylinderScattering(mu, RR, uu, intf, m, maxn)

    print("Transversal mode energies:")
    print([en / sc.eV for en in dcs.phi_energies])

    # plot_transmission(dcs, 0.0 * sc.eV, 1.0 * sc.eV, 0.001 * sc.eV)

    n = 1
    energy = 0.676999 * sc.eV
    plot_wavefunction(dcs, n, energy)


def main():
    test_resonator()
    # test_racec(5)
    # test_cylinder(1)
    # test_double_delta_slits()

# R = 1.0 * sc.nano
# RR = 5.0 * sc.nano
# uu = -0.4 * sc.nano * sc.eV
# m = 0
# mu = 0.19 * sc.m_e # mass
# maxn = 5
# test_cylinder(R, RR, uu, m, mu, maxn)


main()