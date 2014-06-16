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

        print("Wire:")
        print(self.wg_y.eigenenergies)
        print("Resx:")
        print(self.res_x.eigenenergies)
        print("Resy:")
        print(self.res_y.eigenenergies)

    # def get_kks(self, energy):
    #     return [sqrt(complex(energy - self.waveguide_y_energies[i])) for i in range(self.maxn)]

    def greens_function_resonator_dn(self, energy):
        return self.resonator.greens_function_helmholtz_dy(energy)

    def greens_function_wire_dn(self, energy):
        pf = self.waveguide.greens_function_helmholtz_dy(energy)
        def fun(x, y, xs, ys, maxn=self.maxn): # kinda ugly
            return -pf(x, y, xs, ys, maxn=maxn)
        return fun

    def compute_scattering_full(self, energy):
        res = [self.compute_scattering(1, energy, verbose=False)]
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

    def compute_scattering(self, m, energy, verbose=False):
        assert (energy > self.wg_y.eigenenergies[m])

        kk = np.sqrt(energy - self.wg_y.eigenenergies[m])
        # incoming wavefunction
        uwf = lambda x, y, m=m: self.wg_y.eigenfunctions[m](y) * exp(I * kk * x)

        gamma = 0.57721566490153286060
        k0 = I / self.delta * exp(-gamma)
        e0 = k0 ** 2

        # print("Energy = {}".format(energy))
        # print("E0 = {}".format(e0))

        greens_wire_dn = self.greens_function_wire_dn(energy)
        greens_wire0_dn = self.greens_function_wire_dn(e0)
        greens_resonator_dn = self.greens_function_resonator_dn(energy)
        greens_resonator0_dn = self.greens_function_resonator_dn(e0)

        # rweight = self.res_weight(energy)
        # print("Resonance weight = {}".format(rweight))

        # ???
        dd = 0.0001
        FW = (greens_wire_dn(self.x0, self.y0 - dd, self.x0, self.y0) - greens_wire0_dn(self.x0, self.y0 - dd, self.x0, self.y0)) / dd
        FR = (greens_resonator_dn(self.x0, self.y0 + dd, self.x0, self.y0) - greens_resonator0_dn(self.x0, self.y0 + dd, self.x0, self.y0)) / dd

        # for dd in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        #     FR = (greens_resonator_dn(self.x0, self.y0 + dd, self.x0, self.y0) - greens_resonator0_dn(self.x0, self.y0 + dd, self.x0, self.y0)) / dd
        #     print("d = {}, FR = {}".format(dd, FR))

        if verbose:
            print("|FW| = {}, |FR| = {}".format(cnorm(FW), cnorm(FR)))

        # alphaW = (1 - rweight)
        # alphaR = -alphaW

        # aR = -aW
        alphaR = -self.wg_y.deigenfunctions[m](self.y0) / (FW + FR) # minus sign due to the direction
        alphaW = -alphaR


        if verbose:
            print("aW = {}, aR = {}".format(alphaW, alphaR))

        def psi_density(x, y, maxn=self.maxn):
            if y < -self.H:
                return complex(0.0)
            elif y < 0:
                return uwf(x, y) + alphaW * greens_wire_dn(x, y, self.x0, self.y0, maxn=maxn)
            elif y < self.Ly:
                if x < -self.Lx / 2:
                    return complex(0.0)
                elif x < self.Lx / 2:
                    return alphaR * greens_resonator_dn(x, y, self.x0, self.y0, maxn=maxn)
                else:
                    return complex(0.0)
            else:
                return complex(0.0)

        xleft = -20.0
        xright = 21.0
        jincn = compute_prob_current_numerical(uwf, xleft, -self.H, 0.0)
        jtransn = compute_prob_current_numerical(psi_density, xright, -self.H, 0.0, eps=0.0001)
        if verbose:
            print("Jinc = {}, Jtrans = {}".format(jincn, jtransn))
        Tn = cnorm(jtransn) / cnorm(jincn)
        if verbose:
            print("Energy = {}, TN = {:.3f}".format(energy, Tn))

        print("Energy = {:.7f}, TN = {:.3f}".format(energy, Tn))


        # print("Energy = {:.7f}?, TN = {:.3f}".format(energy, rweight))
        return ScatteringResult(wf=psi_density, T=Tn)
