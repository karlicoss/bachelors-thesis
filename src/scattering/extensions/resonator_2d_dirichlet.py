from sys import stderr
import numpy as np
from numpy import sqrt, exp
from numpy import complex256 as complex
from numpy import float128 as real
from scattering.problems.dirichlet_waveguide_2d import DirichletWaveguide2D
from scattering.problems.dirichlet_well_2d import DirichletWell2D

from scattering.tools import I, cnorm2, cnorm, ScatteringResult
from scattering.tools.quantum import compute_prob_current_numerical


class Resonator2DDirichletScattering:
    def __init__(self, H: real, Lx: real, Ly: real, delta: real, maxn: int):
        self.H = H
        self.Lx = Lx
        self.Ly = Ly

        self.delta = delta
        self.maxn = maxn

        self.x0 = 0.0
        self.y0 = 0.0

        self.resonator = DirichletWell2D(-self.Lx / 2, self.Lx / 2, 0, self.Ly, self.maxn)

        self.res_x = self.resonator.wellX
        self.res_y = self.resonator.wellY


        self.waveguide = DirichletWaveguide2D(-self.H, 0, self.maxn)

        self.wg_y = self.waveguide.wellY

        self.res_x_modes = self.res_x.eigenfunctions
        self.res_x_energies = self.res_x.eigenenergies
        self.res_y_modes = self.res_y.eigenfunctions
        self.res_y_energies = self.res_y.eigenenergies

        self.waveguide_y_modes = self.wg_y.eigenfunctions
        self.wire_y_energies = self.wg_y.eigenenergies

        print("Wire:")
        print(self.wire_y_energies)
        print("Resx:")
        print(self.res_x_energies)
        print("Resy:")
        print(self.res_y_energies)

    def get_kks(self, energy):
        return [sqrt(complex(energy - self.wire_y_energies[i])) for i in range(self.maxn)]

    def greens_function_resonator_dn(self, energy, maxn):
        return self.resonator.greens_function_helmholtz_dy(energy, maxn=maxn)

    # TODO!!!! INVERSE SIGN
    def greens_function_wire_dn(self, energy, maxn):
        return self.waveguide.greens_function_helmholtz_dy(energy, maxn=maxn)

    def compute_scattering_full(self, energy):
        res = [self.compute_scattering(1, energy)]
        T = sum([i.T for i in res])
        # print("Energy = {} eV, T = {}".format(energy / sc.eV, T))
        return T

    def res_weight(self, energy):
        res = 1.0
        for i, e in enumerate(self.resonator.eigenenergies):
            ee = (1 + i / 20.0) * e
            # print("Shift {}".format(e))
            if ee > energy:
                res *= (1 - np.exp(- (energy - ee) ** 2 * 10) * 1 / (1 + np.abs(ee - energy) ** (i / 10.0)))
            else:
                res *= (1 - np.exp(- np.abs(energy - ee) ** (10.0 / (i + 1)) * 10))
            # ee = e * 1.1
            # if energy < ee:
            #     res += 1 / (1 + (ee - energy) ** 14) * exp(-(ee- energy) ** 4)
            # else:
            #     res += 1 / (1 + (ee - energy) ** 14) * exp(-(ee- energy) ** 10)
        return res

    def compute_scattering(self, m, energy, verbose=False, maxn_wf=None):
        if maxn_wf is None:
            maxn_wf = self.maxn

        # assert (energy > self.wire_y_energies[m])
        # kks = self.get_kks(energy)

        # incoming wavefunction
        uwf = lambda x, y, m=m: self.waveguide_y_modes[m](y) * exp(I * self.wg_y.wavevectors[m] * x)

        gamma = 0.57721566490153286060
        k0 = I / self.delta * exp(-gamma)
        e0 = k0 ** 2

        # print(energy)
        # print(e0)

        greens_wire_dn = self.greens_function_wire_dn(energy, self.maxn)
        greens_wire0_dn = self.greens_function_wire_dn(e0, self.maxn)
        greens_resonator_dn = self.greens_function_resonator_dn(energy, self.maxn)
        # self.resonator.greens_function_helmholtz_dy_slow(energy, self.maxn) # self.greens_function_resonator_dn(energy, self.maxn)
        greens_resonator0_dn = self.greens_function_resonator_dn(e0, self.maxn)
        #self.resonator.greens_function_helmholtz_dy_slow(e0, self.maxn)

        greens_wire_dn_wf = self.greens_function_wire_dn(energy, maxn_wf)
        greens_resonator_dn_wf = self.greens_function_resonator_dn(energy, maxn_wf)

        rweight = self.res_weight(energy)
        # print("Resonance weight = {}".format(rweight))

        # ???
        AA = greens_wire_dn(self.x0, self.y0, self.x0, self.y0) - greens_wire0_dn(self.x0, self.y0, self.x0, self.y0)
        BB = greens_resonator_dn(self.x0, self.y0, self.x0, self.y0) - greens_resonator0_dn(self.x0, self.y0, self.x0, self.y0)
        # print(greens_resonator_dn(self.x0, self.y0, self.x0, self.y0))
        # print(greens_resonator0_dn(self.x0, self.y0, self.x0, self.y0))

        if verbose:
            print("|AA| = {}, |BB| = {}\n".format(cnorm(AA), cnorm(BB)))

        alphaW = (1 - rweight)
        alphaR = -alphaW


        if verbose:
            print("aW = {}, aR = {}".format(alphaW, alphaR))

        def psi(x, y):
            if y < -self.H:
                return complex(0.0)
            elif y < 0:
                return uwf(x, y) + alphaW * greens_wire_dn_wf(x, y, self.x0, self.y0)
            elif y < self.Ly:
                if x < -self.Lx / 2:
                    return complex(0.0)
                elif x < self.Lx / 2:
                    return alphaR * greens_resonator_dn_wf(x, y, self.x0, self.y0)
                else:
                    return complex(0.0)
            else:
                return complex(0.0)

        xleft = -10.0
        xright = 10.0
        jincn = compute_prob_current_numerical(uwf, xleft, -self.H, 0.0)
        jtransn = compute_prob_current_numerical(psi, xright, -self.H, 0.0)
        # print("Jinc = {}, Jtrans = {}".format(jincn, jtransn))
        Tn = cnorm(jtransn) / cnorm(jincn)
        if verbose:
            print("Energy = {}, TN = {:.2f}".format(energy, Tn))

        print("Energy = {}, TN = {:.2f}".format(energy, rweight))
        return ScatteringResult(wf=psi, T=rweight)
