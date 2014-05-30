from collections import defaultdict
from itertools import product
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
import matplotlib.pyplot as plt


I = np.complex(1j)

class Helmholtz:
    def __init__(self):
        # self.H =
        # self.left = L
        # self.right = L
        pass

class CellType(object):
    DIRICHLET=0
    DOMAIN=1
    RADIATING_FROM_RIGHT=2
    RADIATING_FROM_LEFT=3
    NOTHING=4

# Domain defines the geometry of the problem and the types of boundary conditions
# Same for different energies
class Domain(object):
    def __init__(self, L, H, dx, dy, cells):
        self.L = L
        self.H = H
        # one computational cell is of size dx x dy
        self.dx = dx
        self.dy = dy
        self.cells = cells

        self.mapping = [[None for _ in range(self.H)] for _ in range(self.L)]
        self.variables = 0
        for x in range(self.L):
            for y in range(self.H):
                if self.cells[x][y] == CellType.DOMAIN:
                    self.mapping[x][y] = self.variables
                    self.variables += 1




class ScatteringProblem(object):
    def __init__(self, domain):
        self.domain = domain

    # returns an array of energy-dependent boundary conditions
    def compute_bcs(self, energy):
        raise NotImplementedError()

    def solve_radiating(self, energy):
        data = self.compute_bcs(energy)
        domain = self.domain # shorthand
        dx = domain.dx
        dy = domain.dy

        Ad = defaultdict(np.complex)
        b = np.zeros(domain.variables, dtype=np.complex)

        def processUpper(x, y):
            i = domain.mapping[x][y]
            if domain.cells[x][y + 1] == CellType.DOMAIN:
                ni = domain.mapping[x][y + 1]
                Ad[(i, ni)] += 1 / dy ** 2
            elif domain.cells[x][y + 1] == CellType.DIRICHLET:
                b[i] -= 1 / dy ** 2 * data[x][y + 1]
            else:
                raise RuntimeError("SHH")
            # TODO handle radiating BCs

        def processLower(x, y):
            i = domain.mapping[x][y]
            if domain.cells[x][y - 1] == CellType.DOMAIN:
                ni = domain.mapping[x][y - 1]
                Ad[(i, ni)] += 1 / dy ** 2
            elif domain.cells[x][y - 1] == CellType.DIRICHLET:
                b[i] -= 1 / dy ** 2 * data[x][y - 1]
            else:
                raise RuntimeError("SHH")
            # TODO handle radiating BCs

        def processLeft(x, y):
            i = domain.mapping[x][y]
            ncell = domain.cells[x - 1][y]
            if ncell == CellType.DOMAIN:
                ni = domain.mapping[x - 1][y]
                Ad[(i, ni)] += 1 / dx ** 2
            elif ncell == CellType.RADIATING_FROM_RIGHT:
                # psi = psi_inc + R e^{-i kk1 x}
                # (psi - psi_inc)' = -i kk1 (psi - psi_inc)
                # (psi_{n + 1} - psi_{n - 1}) / 2 dx -psi_inc' = -i kk1 (psi_n - psi_inc)
                # psi_{n - 1} = psi_{n + 1} + 2 dx (-psi_inc' + i kk1 psi_n - i kk1 psi_inc)
                (inc, incdx, kk) = data[x - 1][y]
                ri = domain.mapping[x + 1][y]
                Ad[(i, ri)] += 1 / dx ** 2
                Ad[(i, i)] += 1 / dx ** 2 * (2 * dx * I * kk)
                b[i] -= 1 / dx ** 2 * (2 * dx * (-incdx - I * kk * inc))
            elif ncell == CellType.DIRICHLET:
                b[i] -= 1 / dx ** 2 * data[x - 1][y]
            else:
                raise RuntimeError("SHH")

        def processRight(x, y):
            i = domain.mapping[x][y]
            ncell = domain.cells[x + 1][y]
            if ncell == CellType.DOMAIN:
                ri = domain.mapping[x + 1][y]
                Ad[(i, ri)] += 1 / dx ** 2
            elif ncell == CellType.RADIATING_FROM_LEFT:
                # psi = T e^{i kk1 x}
                # psi' = i kk psi
                # (psi_{n + 1} - psi_{n - 1}) / 2 dx = i kk psi_n
                # psi_{n + 1} = psi_{n - 1} + 2 dx i kk psi_n
                (inc, incdx, kk) = data[x + 1][y]
                li = domain.mapping[x - 1][y]
                Ad[(i, i)] += 1 / dx ** 2 * (2 * dx * I * kk)
                Ad[(i, li)] += 1 / dx ** 2 * 1 # TODO use data
            elif ncell == CellType.DIRICHLET:
                b[i] -= 1 / dx ** 2 * data[x + 1][y]
            else:
                raise RuntimeError("SHH")

        def process(x, y):
            i = domain.mapping[x][y]
            Ad[(i, i)] += -2 / dx ** 2
            Ad[(i, i)] += -2 / dy ** 2
            Ad[(i, i)] += energy

        for x in range(domain.L):
            for y in range(domain.H):
                if domain.cells[x][y] == CellType.DOMAIN:
                    process(x, y)
                    processLeft(x, y)
                    processRight(x, y)
                    processLower(x, y)
                    processUpper(x, y)


        row = [i for (i, _) in Ad.keys()]
        col = [j for (_, j) in Ad.keys()]
        data = list(Ad.values())
        A = csr_matrix((data, (row, col)), dtype=np.complex)
        # print(A.todense())
        # print(b)
        w = spsolve(A, b)
        wf = np.zeros((domain.L, domain.H), dtype=np.complex)
        for x in range(domain.L):
            for y in range(domain.H):
                if domain.cells[x][y] == CellType.DOMAIN:
                    i = domain.mapping[x][y]
                    wf[x][y] = w[i]

        return wf


    def get_pdensity(self, energy):
        wf = self.solve_radiating(energy)
        pf = np.square(np.absolute(wf))
        return pf

def get_wavevector(full_energy, width, mode):
    return np.sqrt(np.complex(full_energy - (np.pi * mode / width) ** 2))

class Tube(ScatteringProblem):
    def __init__(self, width, height, dx, dy, inc_mode):
        self.width = width
        self.height = height
        self.L = int(width // dx)
        self.H = int(height // dy)
        self.inc_mode = inc_mode

        ###
        for i in range(5):
            print("Mode {}: {}".format(i, (np.pi * i / self.height) ** 2))
        ###

        def make_domain():
            cells = [[CellType.NOTHING for _ in range(self.H)] for _ in range(self.L)]
            for x in range(self.L):
                cells[x][0] = CellType.DIRICHLET
            for x in range(self.L):
                cells[x][self.H - 1] = CellType.DIRICHLET
            for y in range(self.H):
                cells[0][y] = CellType.RADIATING_FROM_RIGHT
            for y in range(self.H):
                cells[self.L - 1][y] = CellType.RADIATING_FROM_LEFT
            for x in range(1, self.L - 1):
                for y in range(1, self.H - 1):
                    cells[x][y] = CellType.DOMAIN


            slitsize = int(self.height / 5 // dy)

            for y in range(1, self.H // 2 - slitsize // 2):
                cells[self.L // 3][y] = CellType.DIRICHLET
                cells[self.L // 3 + self.H][y] = CellType.DIRICHLET

            for y in range(self.H // 2 + slitsize // 2, self.H):
                cells[self.L // 3][y] = CellType.DIRICHLET
                cells[self.L // 3 + self.H][y] = CellType.DIRICHLET

            return Domain(self.L, self.H, dx, dy, cells)

        domain = make_domain()
        super().__init__(domain)

    def compute_bcs(self, energy):
        kkinc = get_wavevector(energy, self.height, self.inc_mode)
        data = [[None for _ in range(self.H)] for _ in range(self.L)]
        for x in range(self.L):
            for y in range(self.H):
                if self.domain.cells[x][y] == CellType.DIRICHLET:
                    data[x][y] = 0.0

        for y in range(self.H):
            if self.domain.cells[0][y] == CellType.RADIATING_FROM_RIGHT:
                inc = np.sin(np.pi * self.inc_mode * y / self.H)
                incdx = np.sin(np.pi * self.inc_mode * y / self.H) * I * kkinc
                data[0][y] = (inc, incdx, kkinc)

        for y in range(self.H):
            if self.domain.cells[self.L - 1][y] == CellType.RADIATING_FROM_LEFT:
                data[self.L - 1][y] = (0.0, 0.0, kkinc)
        return data



def main():
    width = 200
    height = 20
    dx = 0.1
    dy = 0.1
    tube = Tube(width, height, dx, dy, 1)


    for energy in np.arange(0.01, 0.4, 0.005):
        pf = tube.get_pdensity(energy)

        pf = pf.transpose()

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        sax = ax.imshow(pf, cmap='gnuplot', origin='lower')
        cbar = fig.colorbar(sax)
        ax.set_title("Energy = {:.3f}".format(energy))
        fig.savefig("output/energy{:.3f}.png".format(energy))
        plt.close(fig)


main()
