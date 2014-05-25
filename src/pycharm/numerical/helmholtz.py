from collections import defaultdict
from itertools import product
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve


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

class Domain(object):
    def __init__(self, cells, data, dx, dy):
        self.cells = cells
        self.data = data
        # one computational cell is of size dx x dy
        self.L = len(self.cells)
        self.H = len(self.cells[0])
        self.mapping = [[None for _ in range(self.H)] for _ in range(self.L)]
        self.variables = 0
        for x in range(self.L):
            for y in range(self.H):
                if self.cells[x][y] == CellType.DOMAIN:
                    self.mapping[x][y] = self.variables
                    self.variables += 1


class Tube:
    def __init__(self, width, height, dx, dy):
        self.width = width
        self.height = height
        self.dx = dx
        self.dy = dy
        self.L = int(self.width // self.dx)
        self.H = int(self.height // self.dy)
        self.inc_mode = 1

        for i in range(1, 6):
            print("Mode {}: {}".format(i, (np.pi * i / self.height) ** 2))

    def prepare_domain(self, energy):
        self.kks = [np.sqrt(np.complex(energy - (np.pi * i / self.height) ** 2)) for i in range(6)]

        cells = [[CellType.NOTHING for _ in range(self.H)] for _ in range(self.L)]
        data = [[None for _ in range(self.H)] for _ in range(self.L)]
        for x in range(self.L):
            cells[x][0] = CellType.DIRICHLET
            data[x][0] = 0.0j
        for x in range(self.L):
            cells[x][self.H - 1] = CellType.DIRICHLET
            data[x][self.H - 1] = 0.0j
        for y in range(self.H):
            cells[0][y] = CellType.RADIATING_FROM_RIGHT
            inc = np.sin(np.pi * self.inc_mode * y / self.H)
            incdx = np.sin(np.pi * self.inc_mode * y / self.H) * I * self.kks[self.inc_mode]
            data[0][y] = (inc, incdx)
        for y in range(self.H):
            cells[self.L - 1][y] = CellType.RADIATING_FROM_LEFT
            # data[self.L - 1][y] = 0.0
        for x in range(1, self.L - 1):
            for y in range(1, self.H - 1):
                cells[x][y] = CellType.DOMAIN

        self.domain = Domain(cells, data, self.dx, self.dy)

    def solve_radiating(self, energy):
        self.prepare_domain(energy)
        domain = self.domain # shorthand

        kk1 = self.kks[1]
        print("kk1 = {}".format(kk1))


        Ad = defaultdict(np.complex)
        b = np.zeros(domain.variables, dtype=np.complex)


        def processUpper(x, y):
            i = domain.mapping[x][y]
            if domain.cells[x][y + 1] == CellType.DOMAIN:
                ni = domain.mapping[x][y + 1]
                Ad[(i, ni)] += 1 / self.dy ** 2
            elif domain.cells[x][y + 1] == CellType.DIRICHLET:
                pass # TODO works only in the case of zero BCs
            else:
                raise RuntimeError("SHH")
            # TODO handle radiating BCs

        def processLower(x, y):
            i = domain.mapping[x][y]
            if domain.cells[x][y - 1] == CellType.DOMAIN:
                ni = domain.mapping[x][y - 1]
                Ad[(i, ni)] += 1 / self.dy ** 2
            elif domain.cells[x][y - 1] == CellType.DIRICHLET:
                pass # TODO works only in the case of zero BCs
            else:
                raise RuntimeError("SHH")
            # TODO handle radiating BCs

        def processLeft(x, y):
            i = domain.mapping[x][y]
            if domain.cells[x - 1][y] == CellType.DOMAIN:
                ni = domain.mapping[x - 1][y]
                Ad[(i, ni)] += 1 / self.dx ** 2
            elif domain.cells[x - 1][y] == CellType.RADIATING_FROM_RIGHT:
                # psi = psi_inc + R e^{-i kk1 x}
                # (psi - psi_inc)' = -i kk1 (psi - psi_inc)
                # (psi_{n + 1} - psi_{n - 1}) / 2 dx -psi_inc' = -i kk1 (psi_n - psi_inc)
                # psi_{n - 1} = psi_{n + 1} + 2 dx (-psi_inc' + i kk1 psi_n - i kk1 psi_inc)
                (inc, incdx) = domain.data[x - 1][y]
                ri = domain.mapping[x + 1][y]
                Ad[(i, ri)] += 1 / self.dx ** 2
                Ad[(i, i)] += 1 / self.dx ** 2 * (2 * self.dx * I * kk1)
                lhs = 1 / self.dx ** 2 * (2 * self.dx * (-incdx - I * kk1 * inc))
                b[i] -= lhs
            else:
                raise RuntimeError("SHH")
             # TODO handle Dirichlet BCs

        def processRight(x, y):
            i = domain.mapping[x][y]
            if domain.cells[x + 1][y] == CellType.DOMAIN:
                ri = domain.mapping[x + 1][y]
                Ad[(i, ri)] += 1 / self.dx ** 2
            elif domain.cells[x + 1][y] == CellType.RADIATING_FROM_LEFT:
                # psi = T e^{i kk1 x}
                # psi' = i kk psi
                # (psi_{n + 1} - psi_{n - 1}) / 2 dx = i kk psi_n
                # psi_{n + 1} = psi_{n - 1} + 2 dx i kk psi_n
                li = domain.mapping[x - 1][y]
                Ad[(i, i)] += 1 / self.dx ** 2 * (2 * self.dx * I * kk1)
                Ad[(i, li)] += 1 / self.dx ** 2 * 1
            else:
                raise RuntimeError("SHH")
             # TODO handle Dirichlet BCs

        def process(x, y):
            i = domain.mapping[x][y]
            Ad[(i, i)] += -2 / self.dx ** 2
            Ad[(i, i)] += -2 / self.dy ** 2
            Ad[(i, i)] += energy

        for x in range(self.L):
            for y in range(self.H):
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
        wf = np.zeros((self.L, self.H), dtype=np.complex)
        for x in range(self.L):
            for y in range(self.H):
                if domain.cells[x][y] == CellType.DOMAIN:
                    i = domain.mapping[x][y]
                    wf[x][y] = w[i]

        return wf


    def solve(self, energy):
        wf = self.solve_radiating(energy)
        pf = np.square(np.absolute(wf))
        return pf

def main():
    width = 100
    height = 20
    dx = 0.1
    dy = 0.1
    tube = Tube(width, height, dx, dy)

    energy = 0.03
    pf = tube.solve(energy)

    pf = pf.transpose()
    import matplotlib.pyplot as plt

    plt.imshow(pf, cmap='gnuplot', origin='lower')
    plt.colorbar()
    plt.show()


main()
