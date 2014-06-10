import itertools

import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import scipy.constants as sc


# import scipy.optimize as opt
# import scipy.special
from r_matrix.double_delta import DoubleDeltaCylinderScattering
from scattering.extensions.one_point_2d import OnePointScattering
from r_matrix.piecewise_delta import PiecewiseDeltaCylinderScattering
from scattering.extensions.resonator_2d_new import Resonator2DNewScattering

from scattering.tools import cnorm2, integrate_complex


def plot_transmission(dcs, left, right, step, maxt=None, fname="transmission.png", info=None, vlines=None):
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
    ax = fig.add_subplot(111, aspect='equal')

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


def test_onepoint():
    H = 1.0
    delta = 0.001
    maxn = 10000
    sp = OnePointScattering(H, delta, maxn)

    fx = -5.0
    tx = 6.0
    dx = 0.05
    fy = -H / 2
    ty = H / 2
    dy = 0.02

    n = 0

    # for energy in arange(1.0, 100.0, 1.0):
    #     res = sp.compute_scattering(n, energy)
    #     plot_wavefunction(res.wf,
    #                       fx, tx, dx,
    #                       fy, ty, dy,
    #                       fname="one_wavefunction{:.2f}.png".format(energy),
    #                       title="Wavefunction at energy {:.2f}, T = {:.2f}".format(energy, res.T))

    plot_transmission(sp,
                      0.1, 100.0, 0.1,
                      maxt=1.0,
                      fname="op_transmission.png",
                      info="Transmission")


jobs = []

def test_resonator():
    Lx = 1.0
    Ly = 1.0
    H = 1.0
    delta = 0.001
    maxn = 1000  # TODO
    maxn_wf = 100
    sp = Resonator2DNewScattering(H, Lx, Ly, delta, maxn, maxn_wf)

    fx = -5.0
    tx = 6.0
    dx = 0.05
    fy = -H
    ty = Ly
    dy = 0.02

    energy = 1.0
    n = 1


    resenergies = [e1 + e2 for e1, e2 in itertools.product(sp.res_x_energies, sp.res_y_energies)]
    plot_transmission(sp,
                      10.0, 100.0, 0.1,
                      maxt=1.0,
                      fname="output/transmission.png",
                      info="Transmission",
                      vlines=resenergies)

    # def fff(energy):
    #     res = sp.compute_scattering(n, energy)
    #     plot_wavefunction(res.wf,
    #                       fx, tx, dx,
    #                       fy, ty, dy,
    #                       fname="output/wavefunction{:.2f}.png".format(energy),
    #                       title="Wavefunction at energy {:.2f}, T = {:.2f}".format(energy, res.T))
    #
    # for energy in arange(10.0, 20.0, 0.05):
    #     fff(energy)


def test_cylinder(maxn):
    RR = 5.0 * sc.nano
    R = RR
    uu = -0.4 * sc.nano * sc.eV
    m = 0
    mu = 0.19 * sc.m_e  # mass

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


def test_onepoint():
    H = 1.0
    delta = 0.001
    maxn = 10000
    sp = OnePointScattering(H, delta, maxn)

    fx = -5.0
    tx = 6.0
    dx = 0.05
    fy = -H / 2
    ty = H / 2
    dy = 0.02

    n = 0

    # for energy in arange(1.0, 100.0, 1.0):
    #     res = sp.compute_scattering(n, energy)
    #     plot_wavefunction(res.wf,
    #                       fx, tx, dx,
    #                       fy, ty, dy,
    #                       fname="one_wavefunction{:.2f}.png".format(energy),
    #                       title="Wavefunction at energy {:.2f}, T = {:.2f}".format(energy, res.T))

    plot_transmission(sp,
                      0.1, 100.0, 0.1,
                      maxt=1.0,
                      fname="op_transmission.png",
                      info="Transmission")
