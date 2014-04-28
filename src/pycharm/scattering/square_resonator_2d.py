import numpy as np
from numpy import sqrt, exp
from numpy import complex256 as complex
from numpy import float128 as real

from scipy import constants as sc
from scattering.problems.neumann_well_1d import NeumannWell1D

from scattering.tools import I, cnorm2, cnorm, ScatteringResult

class ResonatorScattering:
    def __init__(self, H: real, Lx: real, Ly: real, delta: real, maxn_params: int, maxn_wavefunction: int = None):
        self.H = H
        self.Lx = Lx
        self.Ly = Ly

        self.delta = delta
        self.maxn_params = maxn_params
        if maxn_wavefunction is None:
            maxn_wavefunction = maxn_params
        self.maxn_wf = maxn_wavefunction

        self.x0 = 0.0
        self.y0 = 0.0

        nw_res_x = NeumannWell1D(-self.Lx / 2, self.Lx / 2, self.maxn_params)
        nw_res_y = NeumannWell1D(0, self.Ly, self.maxn_params)
        nw_wire_y = NeumannWell1D(0, self.H, self.maxn_params) # TODO actually from -self.H to 0

        self.res_x_modes = nw_res_x.eigenfunctions
        self.res_x_energies = nw_res_x.eigenenergies
        self.res_y_modes = nw_res_y.eigenfunctions
        self.res_y_energies = nw_res_y.eigenenergies

        self.wire_y_modes = nw_wire_y.eigenfunctions
        self.wire_y_energies = nw_wire_y.eigenenergies

        print("Wire:")
        print(self.wire_y_energies)
        print("Resx:")
        print(self.res_x_energies)
        print("Resy:")
        print(self.res_y_energies)

    def greens_function_resonator(self, energy, maxn):
        def fun(x, y, xs, ys):
            res = complex(0.0)
            for n in range(maxn):
                for m in range(maxn):
                    res += self.res_x_modes[n](x) * self.res_y_modes[m](y) * \
                           np.conj(self.res_x_modes[n](xs) * self.res_y_modes[m](ys)) / \
                           (self.res_x_energies[n] + self.res_y_energies[m] - energy)
            return res

        return fun

    def get_kks(self, energy):
        return [sqrt(complex(energy - self.wire_y_energies[i])) for i in range(self.maxn_params)]

    def greens_1d_neumann_well_fast(self, a, b, energy):
        k = np.sqrt(complex(energy))
        # stderr.write("k = {}\n".format(k))
        coeff = 1 / (k * np.sin(k * (a - b)))

        def fun(x, s):
            if x < s:
                return coeff * np.cos(k * (x - a)) * np.cos(k * (s - b))
            else:
                return coeff * np.cos(k * (x - b)) * np.cos(k * (s - a))

        return fun

    def greens_1d_neumann_well_slow(self, energy):
        k = np.sqrt(complex(energy))
        def fun(y, ys):
            res = complex(0.0)
            for n in range(self.maxn_params):
                res += self.res_y_modes[n](y) * self.res_y_modes[n](ys) / (self.res_y_energies[n] - energy)
            return res
        return fun

    def greens_function_resonator_fast(self, energy, maxn):
        kks = self.get_kks(energy)  # TODO different kks for resonator and wire in general!!!

        def fun(x, y, xs, ys):
            res = complex(0.0)
            for n in range(maxn):
                gf = self.greens_1d_neumann_well_fast(0, self.Ly, kks[n] ** 2) # TODO
                res += self.res_x_modes[n](x) * np.conj(self.res_x_modes[n](xs)) * gf(y, ys)
            return res

        return fun

    def greens_function_wire(self, energy, maxn):
        kks = self.get_kks(energy)

        def fun(x, y, xs, ys):
            res = complex(0.0)
            for m in range(maxn):
                res += self.wire_y_modes[m](y) * np.conj(self.wire_y_modes[m](ys)) * \
                       I / (2 * kks[m]) * exp(I * kks[m] * np.abs(x - xs))
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

        greens_wire = self.greens_function_wire(energy, self.maxn_params)
        greens_wire0 = self.greens_function_wire(e0, self.maxn_params)
        greens_resonator = self.greens_function_resonator_fast(energy, self.maxn_params)
        greens_resonator0 = self.greens_function_resonator_fast(e0, self.maxn_params)

        greens_wire_wf = self.greens_function_wire(energy, self.maxn_wf)
        greens_resonator_wf = self.greens_function_resonator_fast(energy, self.maxn_wf)

        # ???
        AA = greens_wire(self.x0, self.y0, self.x0, self.y0) - greens_wire0(self.x0, self.y0, self.x0, self.y0)
        BB = greens_resonator(self.x0, self.y0, self.x0, self.y0) - greens_resonator0(self.x0, self.y0, self.x0,
                                                                                      self.y0)

        # stderr.write("|AA| = {}, |BB| = {}\n".format(cnorm(AA), cnorm(BB)))

        alphaW = -uwf(self.x0, self.y0) / (AA + BB)
        alphaR = -alphaW

        if verbose:
            print("aW = {}, aR = {}".format(alphaW, alphaR))

        jinca = -2 * I * kks[m]
        jtransa = -2 * I * kks[m]
        jtransa += sqrt(2 / self.H) * alphaW
        jtransa -= sqrt(2 / self.H) * np.conj(alphaW)
        for i in range(self.maxn_params):
            if self.wire_y_energies[i] > energy:
                break
            if i == 0:
                jtransa += 1 / self.H * cnorm2(alphaW) * -2 * I / (4 * kks[i])
            else:
                jtransa += 2 / self.H * cnorm2(alphaW) * -2 * I / (4 * kks[i])


        T = cnorm(jtransa) / cnorm(jinca)
        print("Energy = {}, T = {:.2f}".format(energy, T))

        def psi(x, y):
            if y < -self.H:
                return complex(0.0)
            elif y < 0:
                return uwf(x, y) + alphaW * greens_wire_wf(x, y, self.x0, self.y0)
            elif y < self.Ly:
                if x < -self.Lx / 2:
                    return complex(0.0)
                elif x < self.Lx / 2:
                    return alphaR * greens_resonator_wf(x, y, self.x0, self.y0)
                else:
                    return complex(0.0)
            else:
                return complex(0.0)


        return ScatteringResult(wf=psi, T=T)