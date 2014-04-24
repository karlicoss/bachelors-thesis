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
import scipy.optimize as opt
import scipy.special


def cnorm2(z):
    return (z * np.conj(z)).real

def cnorm(z):
    return sqrt(cnorm2(z))

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

def plot_transmission(dcs, left, right, step, maxt = None, fname="transmission.png", info = None, vlines = None):
    if maxt is None:
        maxt = dcs.maxn

    if info is None:
        info = ""

    if vlines is None:
        vlines = []

    xs = arange(left, right, step)
    ys = list(map(lambda en: dcs.compute_scattering_full(en), xs))

    fig = plt.figure(figsize=(15, 10), dpi=500)
    ax = fig.add_subplot(111)  # , aspect = 'equal'


    ax.vlines(vlines, 0.0, maxt)

    # xticks = arange(left, right, 0.1 * sc.eV)
    # xlabels = ["{:.1f}".format(t / sc.eV) for t in xticks]
    ax.set_xlim(left, right)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xlabels)
    ax.set_xlabel("E, eV")

    # yticks = arange(0.0, maxt, 1.0)
    # ylabels = ["{:.1f}".format(t) for t in yticks]
    ax.set_ylim(0.0, maxt)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(ylabels)
    ax.set_ylabel("T")

    cax = ax.plot(xs, ys)
    ax.set_title(info)

    fig.savefig(fname)
    plt.close(fig)

# TODO MODIFY PLOT_WAVEFUNCTION USAGES
def plot_wavefunction(wf, fx, tx, dx, fy, ty, dy, fname="wavefunction.png", title=None):
    pf = lambda z, r: cnorm2(wf(z, r))
    vpf = np.vectorize(pf)

    x, y = np.mgrid[slice(fx, tx + dx, dx), slice(fy, ty + dy, dy)]
    z = vpf(x, y)

    z_min, z_max = np.abs(z).min(), np.abs(z).max()

    fig = plt.figure(figsize=(15, 10), dpi=500)
    ax = fig.add_subplot(111, aspect = 'equal')

    # ax.set_title("n = {}, E = {:.2f} eV, T = {:.2f}".format(n, energy / sc.eV, T) + (
    #     "" if info is None else "\n{}".format(info)))
    if title is not None:
        ax.set_title(title)

    # xticks = arange(fx, tx, sc.nano)
    # xlabels = ["{:.1f}".format(t / sc.nano) for t in xticks]
    xticks = arange(fx, tx, 1.0)
    xlabels = ["{:.1f}".format(t) for t in xticks]
    ax.set_xlim(fx, tx)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    # yticks = arange(fy, ty, sc.nano)
    # ylabels = ["{:.1f}".format(t / sc.nano) for t in yticks]
    yticks = arange(fy, ty, 1.0)
    ylabels = ["{:.1f}".format(t) for t in yticks]
    ax.set_ylim(fy, ty)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    cax = ax.pcolor(x, y, z, cmap='gnuplot', norm=Normalize(z_min, z_max))

    cbar = fig.colorbar(cax)

    # bx = fig.add_subplot(212)
    # bx.set_title("Wavefunction at x = 0")
    #
    # rr = arange(fy, ty, 0.01 * sc.nano)
    # ww = list(map(lambda r: pf(0, r), rr))
    # bx.plot(rr, ww)

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

class ResonatorScattering:
    def __init__(self, H, Lx, Ly, delta, maxn):
        self.H = H
        self.Lx = Lx
        self.Ly = Ly

        self.delta = delta
        self.maxn = maxn

        self.x0 = 0.0
        self.y0 = 0.0


        def well_energy(width, n):
            return (sc.pi * n / width) ** 2

        def well_function(width, shift, n):
            return lambda x: sqrt(2.0 / width) * np.cos(sc.pi * n / width * (x + shift))

        self.res_x_modes = [well_function(self.Lx, self.Lx / 2, n) for n in range(self.maxn)]
        self.res_x_energies = [well_energy(self.Lx, n) for n in range(self.maxn)]
        self.res_y_modes = [well_function(self.Ly, 0, m) for m in range(self.maxn)]
        self.res_y_energies = [well_energy(self.Ly, m) for m in range(self.maxn)]

        self.wire_y_modes = [well_function(self.H, 0, m) for m in range(self.maxn)]
        self.wire_y_energies = [well_energy(self.H, m) for m in range(self.maxn)]

        print("Wire:")
        print(self.wire_y_energies)
        print("Resx:")
        print(self.res_x_energies)
        print("Resy:")
        print(self.res_y_energies)

    def greens_function_resonator(self, energy):
        def fun(x, y, xs, ys):
            res = complex(0.0)
            for n in range(self.maxn):
                for m in range(self.maxn):
                    res += self.res_x_modes[n](x) * self.res_y_modes[m](y) * \
                           np.conj(self.res_x_modes[n](xs) * self.res_y_modes[m](ys)) / \
                           (self.res_x_energies[n] + self.res_y_energies[m] - energy)
                    # print("n = {}, m = {}: {}".format(n, m, self.res_x_energies[n] + self.res_y_energies[m] - energy))
                    # print(self.res_x_modes[n](x) * self.res_y_modes[m](y) * \
                    #        np.conj(self.res_x_modes[n](xs) * self.res_y_modes[m](ys)) / \
                    #        (self.res_x_energies[n] + self.res_y_energies[m] - energy))
            return res
        return fun

    def get_kks(self, energy):
        return [sqrt(complex(energy - self.wire_y_energies[i])) for i in range(self.maxn)]

    def greens_function_wire(self, energy):
        kks = self.get_kks(energy)
        def fun(x, y, xs, ys):
            res = complex(0.0)
            for m in range(self.maxn):
                res += self.wire_y_modes[m](y) * np.conj(self.wire_y_modes[m](ys)) * \
                    I / (2 * kks[m]) * exp(I * kks[m] * np.abs(x - xs)) # TODO sign
            return res
        return fun

    def compute_scattering_full(self, energy):
        res = [self.compute_scattering(1, energy)]
        T = sum([i.T for i in res])
        # print("Energy = {} eV, T = {}".format(energy / sc.eV, T))
        return T

    def compute_scattering(self, m, energy, verbose=False):
        assert (energy > self.wire_y_energies[m])
        kks = self.get_kks(energy)

        # incoming wavefunction
        uwf = lambda x, y: self.wire_y_modes[m](y) * exp(I * kks[m] * x)

        gamma = 0.57721566490153286060
        k0 = I / self.delta * exp(-gamma)
        e0 = k0 ** 2


        greens_wire = self.greens_function_wire(energy)
        greens_wire0 = self.greens_function_wire(e0)
        greens_resonator = self.greens_function_resonator(energy)
        greens_resonator0 = self.greens_function_resonator(e0)


        # ???
        AA = greens_wire(self.x0, self.y0, self.x0, self.y0) - greens_wire0(self.x0, self.y0, self.x0, self.y0)
        BB = greens_resonator(self.x0, self.y0, self.x0, self.y0) - greens_resonator0(self.x0, self.y0, self.x0, self.y0)

        # print(greens_wire(self.x0, self.y0, self.x0, self.y0))
        # if verbose:
        print("|AA| = {}, |BB| = {}".format(cnorm(AA), cnorm(BB)))

        alphaW = -uwf(self.x0, self.y0) / (AA + BB)
        alphaR = -alphaW

        if verbose:
            print("aW = {}, aR = {}".format(alphaW, alphaR))

        jinca = -2 * I * kks[m]
        jtransa = -2 * I * kks[m]
        jtransa += sqrt(2 / self.H) * alphaW # TODO sign
        jtransa -= sqrt(2 / self.H) * np.conj(alphaW) # TODO sign
        for i in range(self.maxn):
            if self.wire_y_energies[i] > energy:
                break
            jtransa += 2 / self.H * cnorm2(alphaW) * -2 * I / (4 * kks[i])

        T = cnorm(jtransa) / cnorm(jinca)
        print("Energy = {}, T = {:.2f}".format(energy, T))

        def psi(x, y):
            if y < -self.H:
                return complex(0.0)
            elif y < 0:
                return uwf(x, y) + alphaW * greens_wire(x, y, self.x0, self.y0)
                # if x < self.x0:
                #     return uwf(x, y)
                # else:
                #     return uwf(x, y) + alphaW * greens_wire(x, y, self.x0, self.y0)
            elif y < self.Ly:
                if x < -self.Lx / 2:
                    return complex(0.0)
                elif x < self.Lx / 2:
                    return alphaR * greens_resonator(x, y, self.x0, self.y0)
                else:
                    return complex(0.0)
            else:
                return complex(0.0)


        return ScatteringResult(wf=psi, T=T)



class OnePointScattering:
    def __init__(self, H, delta, maxn):
        self.H = H

        self.delta = delta
        self.maxn = maxn

        self.x0 = 0.0
        self.y0 = 0.0

        def well_energy(width, n):
            return (sc.pi * n / width) ** 2

        def well_function(width, shift, n):
            return lambda x: sqrt(2.0 / width) * np.cos(sc.pi * n / width * (x + shift))

        self.wire_y_modes = [well_function(self.H, -self.H / 2, m) for m in range(self.maxn)]
        self.wire_y_energies = [well_energy(self.H, m) for m in range(self.maxn)]

        print("Wire:")
        print(self.wire_y_energies)

    def get_kks(self, energy):
        return [sqrt(complex(energy - self.wire_y_energies[i])) for i in range(self.maxn)]

    def greens_function_wire(self, energy):
        kks = self.get_kks(energy)
        def fun(x, y, xs, ys):
            res = complex(0.0)
            for m in range(self.maxn):
                res += self.wire_y_modes[m](y) * np.conj(self.wire_y_modes[m](ys)) * \
                    -I / (2 * kks[m]) * exp(I * kks[m] * np.abs(x - xs))
            return res
        return fun

    def compute_scattering_full(self, energy):
        res = [self.compute_scattering(0, energy)]
        T = sum([i.T for i in res])
        return T

    def compute_scattering(self, m, energy, verbose=False):
        assert (energy > self.wire_y_energies[m])
        kks = self.get_kks(energy)

        # incoming wavefunction
        uwf = lambda x, y: self.wire_y_modes[m](y) * exp(I * kks[m] * x)

        gamma = 0.57721566490153286060
        k0 = I / self.delta * exp(-gamma)
        e0 = k0 ** 2


        greens_wire = self.greens_function_wire(energy)
        greens_wire0 = self.greens_function_wire(e0)

        AA = greens_wire(self.x0, self.y0, self.x0, self.y0) - greens_wire0(self.x0, self.y0, self.x0, self.y0)

        print("AA = {}".format(AA))

        alphaL = -uwf(self.x0, self.y0) / (AA + AA)
        alphaR = -alphaL

        print("aL = {}, aR = {}".format(alphaL, alphaR))

        jinca = -2 * I * kks[m]
        jtransa = complex(0.0)
        for i in range(self.maxn):
            if self.wire_y_energies[i] > energy:
                break
            jtransa += self.wire_y_modes[i](self.y0) ** 2 / (4 * kks[i])
        jtransa *= -2 * I * cnorm2(alphaR)

        T = cnorm(jtransa) / cnorm(jinca)
        print("Energy = {}, T = {:.2f}".format(energy, T))

        def psi(x, y):
            if y < -self.H / 2:
                return complex(0.0)
            elif y < self.H / 2:
                if x < 0:
                    return uwf(x, y) + alphaL * greens_wire(x, y, self.x0, self.y0) # TODO green's function
                else:
                    return alphaR * greens_wire(x, y, self.x0, self.y0) # TODO green's function
            else:
                return complex(0.0)


        return ScatteringResult(wf=psi, T=T)

def test_onepoint():
    H = 1.0
    delta = 0.001
    maxn = 10
    sp = OnePointScattering(H, delta, maxn)

    fx = -5.0
    tx = 6.0
    dx = 0.05
    fy = -H / 2
    ty = H / 2
    dy = 0.02

    plot_transmission(sp,
                  0.1, 10.0, 0.1,
                  maxt=2.0,
                  fname="op_transmission.png",
                  info="Transmission")

def test_resonator():
    Lx = 1.0
    Ly = 1.0
    H = 1.0
    delta = 0.001
    maxn = 100 # TODO
    # sp = ResonatorRectangular2DScattering(H, Lx, Ly, delta, maxn)
    sp = ResonatorScattering(H, Lx, Ly, delta, maxn)

    fx = -5.0
    tx = 6.0
    dx = 0.05
    fy = -H
    ty = Ly
    dy = 0.02

    energy = 10.0
    n = 1


    # res = sp.compute_scattering(n, energy)
    # plot_wavefunction(res.wf,
    #                   fx, tx, dx,
    #                   fy, ty, dy,
    #                   fname="rwavefunction.png",
    #                   title="Wavefunction at energy {:.2f}".format(energy))

    # print(opt.fmin(lambda x: sp.compute_scattering_full(x[0]), 9.0))


    import itertools
    resenergies = [e1 + e2 for e1, e2 in itertools.product(sp.res_x_energies, sp.res_y_energies)]
    plot_transmission(sp,
                      10.0, 100.0, 0.1,
                      maxt=2.0,
                      fname="sq_transmission.png",
                      info="Transmission",
                      vlines=resenergies)

    # for energy in arange(2.5, 30.0, 0.5):
    #     res = sp.compute_scattering(n, energy)
    #     plot_wavefunction(res.wf,
    #                       fx, tx, dx,
    #                       fy, ty, dy,
    #                       fname="wavefunction{:.2f}.png".format(energy),
    #                       title="Wavefunction at energy {:.2f}, T = {:.2f}".format(energy, res.T))


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
                      title="u1 = {:.2f} nm * eV (at a = {:.2f}) nm\nu2 = {:.2f} nm * eV (at b = {:.2f})\nslit width = {:.2f} nm".format(
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
    # test_onepoint()
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