from scipy import constants as sc, constants
from scattering.main import plot_transmission
from scattering.piecewise_delta import PiecewiseDeltaCylinderScattering
from scattering.tools import integrate_complex

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
            for uu_ in [-0.15]:  # TODO -0.05
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