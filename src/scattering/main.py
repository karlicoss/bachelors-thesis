import itertools
from matplotlib.patches import Ellipse, Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from numpy import complex256 as complex
from numpy import float128 as real


# import scipy.optimize as opt
# import scipy.special
from scipy.optimize import minimize
from scattering.extensions.one_point_2d import OnePointScattering
from scattering.extensions.resonator_2d_dirichlet import Resonator2DDirichletScattering
from scattering.extensions.resonator_2d_domain import Resonator2DDomain
from scattering.extensions.resonator_2d_new import Resonator2DNewScattering

from scattering.tools import cnorm2


def plot_transmission_energy(dcs, left, right, step, fname="transmission.png", vlines=[]):
    maxt = 1.0 + 0.1 # small margin to properly draw T = 1

    xs = arange(left, right, step)
    ys = list(map(lambda en: dcs.compute_scattering_full(en), xs))

    inch = 2.5
    fig = plt.figure(figsize=(18.0 / inch, 8.0 / inch))
    ax = fig.add_subplot(111)  # , aspect = 'equal'

    ax.vlines(vlines, 0.0, maxt,
              linestyles='dashed')

    ax.tick_params(axis='both', which='major', labelsize=8)

    xticks = arange(left, right, 5)
    xlabels = ["{:d}".format(int(tick)) for tick in xticks]
    ax.set_xlim(left, right)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("E")

    yticks = arange(0.0, maxt, 0.1)
    ylabels = ["{:.1f}".format(tick) for tick in yticks]
    ax.set_ylim(0.0, maxt)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("T")

    cax = ax.plot(xs, ys)

    fig.savefig(fname, bbox_inches='tight', dpi=500)
    plt.close(fig)

def plot_transmission_size(domain: Resonator2DDomain,
                           mode: int,
                           energy: real,
                           leftw: real, rightw: real, stepw: real,
                           fname="transmission.png",
                           maxn=None):
    maxt = 1.0 + 0.1 # small margin to properly draw T = 1

    xs = arange(leftw, rightw, stepw)
    ys = []
    for Lx in xs:
        domain.Lx = Lx
        sp = Resonator2DDirichletScattering(domain.H, Lx, domain.Ly, domain.S, maxn)
        res = sp.compute_scattering(mode, energy, verbose=True)
        ys.append(res.T)

    inch = 2.5
    fig = plt.figure(figsize=(18.0 / inch, 8.0 / inch))
    ax = fig.add_subplot(111)  # , aspect = 'equal'
    ax.tick_params(axis='both', which='major', labelsize=8)

    xticks = arange(leftw, rightw, 0.1)
    xlabels = ["{:.1f}".format(tick) for tick in xticks]
    ax.set_xlim(leftw, rightw)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("$L_y$")

    yticks = arange(0.0, maxt, 0.1)
    ylabels = ["{:.1f}".format(tick) for tick in yticks]
    ax.set_ylim(0.0, maxt)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("T")

    cax = ax.plot(xs, ys)

    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


# TODO MODIFY PLOT_WAVEFUNCTION USAGES
def plot_pdensity(domain: Resonator2DDomain, wf, maxn_wf: int,
                      fx: real, tx: real, dx: real,
                      dy: real,
                      fname="wavefunction.png",
                      title=None):
    fy = -domain.H
    ty = domain.Ly

    border = (domain.H + domain.Ly) / 10 # TODO
    fy = fy - border
    ty = ty + border

    pf = lambda x, y: cnorm2(wf(x, y, maxn=maxn_wf))
    pfc = lambda x, y: pf(x, y) if np.abs(x ** 2 + y ** 2) > (domain.S / 2) ** 2 else 0.0 # TODO this is to exclude the singulatiry
    vpf = np.vectorize(pfc)

    x, y = np.mgrid[slice(fx, tx + dx, dx), slice(fy, ty + dy, dy)]
    z = vpf(x, y)
    # z_min, z_max = np.abs(z).min(), np.abs(z).max()

    inch = 2.5
    proportions = (ty - fy) / (tx - fx)# approx height / width
    imgwidth = 25 / inch
    imgheight = imgwidth * proportions
    fig = plt.figure(figsize=(imgwidth, imgheight))

    ax = fig.add_subplot(111, aspect='equal')

    if title is not None:
        ax.set_title(title)

    ax.tick_params(axis='both', which='major', labelsize=8)

    xticks = arange(fx, tx + dx, 0.5)
    xlabels = ["{:.1f}".format(t) for t in xticks]
    ax.set_xlim(fx, tx)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    yticks = arange(-domain.H, domain.Ly + border + dy, 0.5)
    ylabels = ["{:.1f}".format(t) for t in yticks]
    ax.set_ylim(fy, ty)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    # z_max = 100
    pc = ax.pcolor(x, y, z, cmap='gnuplot') #, norm=Normalize(z_min, z_max))

    def draw_wall(a, b, c, d):
        ax.add_patch(Polygon([a, b, c, d], closed=True, fill=True, linewidth=0, hatch='//'))

    wwidth = 5 * dy

    draw_wall([fx, 0 - wwidth / 2], [-domain.Lx / 2, -wwidth / 2], [-domain.Lx / 2, domain.Ly + dy], [fx, domain.Ly + dy])
    draw_wall([tx, 0 - wwidth / 2], [domain.Lx / 2 + dx, -wwidth / 2], [domain.Lx / 2 + dx, domain.Ly + dy], [tx, domain.Ly + dy])
    draw_wall([fx, domain.Ly + dy], [tx, domain.Ly + dy], [tx, domain.Ly + border], [fx, domain.Ly + border])
    draw_wall([fx, -domain.H - border], [tx, -domain.H - border], [tx, -domain.H], [fx, -domain.H])
    draw_wall([-domain.Lx / 2, -wwidth / 2], [-domain.S / 2, -wwidth / 2], [-domain.S / 2, wwidth / 2], [-domain.Lx / 2, wwidth / 2])
    draw_wall([domain.Lx / 2 + dx, -wwidth / 2], [domain.S / 2 + dx, -wwidth / 2], [domain.S / 2 + dx, wwidth / 2], [domain.Lx / 2 + dx, wwidth / 2])

    # from http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#axesdivider
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pc, cax=cax, ticks=[])

    fig.savefig(fname, bbox_inches='tight', dpi=200)
    plt.close(fig)


def test_resonator_dirichlet():
    Lx = 2.5
    Ly = 1.0
    H = 2.0
    S = 0.01
    domain = Resonator2DDomain(H, Lx, Ly, S)

    maxn = 100  # TODO
    maxn_wf = 10
    sp = Resonator2DDirichletScattering(H, Lx, Ly, S, maxn)

    fx = -6.0
    tx = 6.0
    dx = 0.02
    fy = -H
    ty = Ly
    dy = 0.01

    n = 1

    resenergies = sp.resonator.eigenenergies
    plot_transmission_energy(sp,
                            # 11.4584, 11.4586, 0.000001,
                            2.5, 20.0, 0.0001,
                            fname="output2/transmission.png",
                            vlines=resenergies)

    # energy = 11.5536
    # res = sp.compute_scattering(n, energy, verbose=True)
    # plot_pdensity(domain,
    #                   res.wf, maxn_wf,
    #                   fx, tx, dx,
    #                   dy,
    #                   fname="output2/pdensity.png")

    # energy = 19.0
    # plot_transmission_size(domain,
    #                        1, energy,
    #                        1.5, 3.0, 0.005,
    #                        fname="output2/transmission_size.png",
    #                        maxn=maxn
    #                        )

    # for energy in arange(24.05, 24.15, 0.001):
    #     res = sp.compute_scattering(n, energy, verbose=True)
    #     plot_pdensity(domain,
    #                   res.wf, maxn_wf,
    #                   fx, tx, dx,
    #                   dy,
    #                   fname="output2/wavefunction{:.2f}.png".format(energy),
    #                   title="Wavefunction at energy {:.2f}, T = {:.2f}".format(energy, res.T))

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

def test_resonator_neumann():
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
    plot_transmission_energy(sp,
                      25.0, 35.0, 0.001,
                      fname="output2/transmission.png",
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


def test_resonator_sizes():
    H = 1.0
    delta = 0.001
    maxn = 1000  # TODO
    maxn_wf = 100

    left = 1.0
    right = 6.0
    step = 0.001


    xs = arange(left, right, step)
    ys = []
    for dD in arange(left, right, step):
        energy = (5 / H) ** 2
        Lx = 2 * H
        Ly = H * dD
        sp = Resonator2DNewScattering(H, Lx, Ly, delta, maxn, maxn_wf)
        ys.append(sp.compute_scattering_full(energy))

    fname = "output/transmissionxxx.png"
    fig = plt.figure(figsize=(15, 10), dpi=500)
    ax = fig.add_subplot(111)  # , aspect = 'equal'

    # xticks = arange(left, right, 0.1 * sc.eV)
    # xlabels = ["{:.1f}".format(t / sc.eV) for t in xticks]
    ax.set_xlim(left, right)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xlabels)
    ax.set_xlabel("Ly/H")

    # yticks = arange(0.0, maxt, 1.0)
    # ylabels = ["{:.1f}".format(t) for t in yticks]
    ax.set_ylim(0.0, 1.0)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(ylabels)
    ax.set_ylabel("T")

    ax.set_title("Dependency of transmission over the resonator height")

    cax = ax.plot(xs, ys)

    fig.savefig(fname)
    plt.close(fig)


def main():
    # test_onepoint()
    # test_resonator_neumann()
    # test_resonator_sizes()
    test_resonator_dirichlet()



main()