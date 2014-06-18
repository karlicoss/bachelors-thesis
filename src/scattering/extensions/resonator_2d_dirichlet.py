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

        print("Wire eigenenergies:")
        print(self.wg_y.eigenenergies)
        print("Resx eigenenergies:")
        print(self.res_x.eigenenergies)
        print("Resy eigenenergies:")
        print(self.res_y.eigenenergies)
        print("Resonator eigenenergies:")
        print(self.resonator.eigenenergies)

    # def get_kks(self, energy):
    # return [sqrt(complex(energy - self.waveguide_y_energies[i])) for i in range(self.maxn)]

    def greens_function_resonator_dn(self, energy):
        return self.resonator.greens_function_helmholtz_dy(energy)

    def greens_function_wire_dn(self, energy):
        pf = self.waveguide.greens_function_helmholtz_dy(energy)

        def fun(x, y, xs, ys, maxn=self.maxn):  # kinda ugly
            return -pf(x, y, xs, ys, maxn=maxn)

        return fun

    def compute_scattering_full(self, energy):
        return self.compute_scattering(1, energy, verbose=False).T

    def compute_scattering(self, mode, energy, verbose=False):
        assert (energy > self.wg_y.eigenenergies[mode])

        kk = np.sqrt(energy - self.wg_y.eigenenergies[mode])
        # incoming wavefunction
        uwf = lambda x, y, mode=mode: self.wg_y.eigenfunctions[mode](y) * exp(I * kk * x)

        gamma = 0.57721566490153286060
        k0 = I / self.delta * exp(-gamma)
        e0 = k0 ** 2

        # print("Energy = {}".format(energy))
        # print("E0 = {}".format(e0))

        greens_wire_dn = self.greens_function_wire_dn(energy)
        greens_wire0_dn = self.greens_function_wire_dn(e0)
        greens_resonator_dn = self.greens_function_resonator_dn(energy)
        greens_resonator0_dn = self.greens_function_resonator_dn(e0)

        # ???
        dd = 0.0001
        FW = (greens_wire_dn(self.x0, self.y0 - dd, self.x0, self.y0) - greens_wire0_dn(self.x0, self.y0 - dd, self.x0,
                                                                                        self.y0)) / dd
        FR = (greens_resonator_dn(self.x0, self.y0 + dd, self.x0, self.y0) - greens_resonator0_dn(self.x0, self.y0 + dd,
                                                                                                  self.x0,
                                                                                                  self.y0)) / dd

        if verbose:
            print("|FW| = {}, |FR| = {}".format(cnorm(FW), cnorm(FR)))

        # aR = -aW
        alphaW = -(-self.wg_y.deigenfunctions[mode](self.y0)) / (FW + FR)  # minus sign due to the direction
        alphaR = -alphaW

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

        # xleft = -20.0
        # xright = 21.0
        # jincn = compute_prob_current_numerical(uwf, xleft, -self.H, 0.0)
        # jtransn = compute_prob_current_numerical(psi_density, xright, -self.H, 0.0, eps=0.0001)

        jinc = 2 * I * kk
        jtrans = 2 * I * kk + (-self.wg_y.deigenfunctions[mode](self.y0)) * (np.conj(alphaW) - alphaW)
        for m in range(1, self.maxn):
            ee = self.wg_y.eigenenergies[m]
            if ee > energy:
                break
            zmm = sqrt(energy - ee)
            jtrans += cnorm2(alphaW) * (-self.wg_y.deigenfunctions[m](self.y0)) ** 2 * I / (2 * zmm)

        if verbose:
            # print("Jincn = {}, Jtransn = {}".format(jincn, jtransn))
            print("Jinc = {}, Jtrans = {}".format(jinc, jtrans))
        # Tn = cnorm(jtransn) / cnorm(jincn)
        T = cnorm(jtrans) / cnorm(jinc)
        T = 1.0 / np.exp(1000 * (1 - T)) # TODO crap

        # print("Energy = {}, Tn = {:.3f}".format(energy, Tn))
        print("Energy = {:.6f}, T = {:.3f}".format(energy, T))


        # print("Energy = {:.7f}?, TN = {:.3f}".format(energy, rweight))
        return ScatteringResult(wf=psi_density, T=T)
