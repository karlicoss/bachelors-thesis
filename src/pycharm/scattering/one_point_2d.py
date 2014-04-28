import numpy as np
from numpy.core.umath import sqrt, exp
from scipy import constants as sc, constants
from scattering.tools import I, cnorm2, cnorm, ScatteringResult

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
                       I / (2 * kks[m]) * (exp(I * kks[m] * np.abs(x - xs)) + exp(
                    I * kks[m] * np.abs(x + xs)))  # taken care of both functions
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

        alphaL = -uwf(self.x0, self.y0) / (AA + AA)
        alphaR = -alphaL

        print("aL = {}, aR = {}".format(alphaL, alphaR))

        jinca = -2 * I * kks[m]
        jtransa = complex(0.0)
        for i in range(self.maxn):
            if self.wire_y_energies[i] > energy:
                break
            jtransa += self.wire_y_modes[i](self.y0) ** 2 / kks[i]
        jtransa *= -2 * I * cnorm2(alphaR)

        T = cnorm(jtransa) / cnorm(jinca)
        print("Energy = {}, T = {:.2f}".format(energy, T))

        def psi(x, y):
            if y < -self.H / 2:
                return complex(0.0)
            elif y < self.H / 2:
                if x < 0:
                    return uwf(x, y) + alphaL * greens_wire(x, y, self.x0, self.y0)
                else:
                    return alphaR * greens_wire(x, y, self.x0, self.y0)
            else:
                return complex(0.0)


        return ScatteringResult(wf=psi, T=T)