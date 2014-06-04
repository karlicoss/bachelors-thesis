class Resonator2DDomain(object):
    def __init__(self, H, Lx, Ly, S):
        """

        :param H: waveguide height
        :param Lx: resonator width
        :param Ly: resonator height
        :param S: slit width
        """
        self.H = H
        self.Lx = Lx
        self.Ly = Ly
        self.S = S