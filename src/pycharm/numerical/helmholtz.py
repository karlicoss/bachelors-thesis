from collections import defaultdict
from itertools import product
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve


class Helmholtz:
    def __init__(self):
        # self.H =
        # self.left = L
        # self.right = L
        pass


class Tube:
    # [-L, L] x [0; H]
    def __init__(self, width, height, dx, dy):
        self.width = width
        self.height = height
        self.dx = dx
        self.dy = dy
        self.L = int(self.width // self.dx)
        self.H = int(self.height // self.dy)


    def solve_sparse(self, energy):
        def convert(x, y):
            return x * self.H + y

        def iconvert(i):
            return (i // self.H, i % self.H)


        kk = np.sqrt(energy - (np.pi / self.height) ** 2)
        print("kk = {}".format(kk))

        dim = self.L * self.H

        # A = np.zeros((dim, dim), dtype=np.complex)
        Ad = defaultdict(np.complex)
        b = np.zeros(dim, dtype=np.complex)
        # nabla^2 psi(x, y) + energy psi(x, y) = 0
        # 1 / dx^2 (psi_{x-1,y} + psi_{x+1,y} + psi_{x,y-1} + psi_{x,y+1}) + (energy - 4 / dx^2) psi(x, y) = 0
        for x in range(self.L):
            for y in range(self.H):
                i = convert(x, y)
                Ad[(i, i)] = energy - 2 / self.dx ** 2 - 2 / self.dy ** 2
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx = x + dx
                    ny = y + dy
                    if nx == -1:
                        # Arbitrary non-zero boundary condition?
                        val = np.sin(np.pi * ny / self.H)
                        b[i] -= val
                    elif nx == self.L:
                        val = np.sin(np.pi * ny / self.H) * np.exp(1j * kk * 1.022 * self.width)
                        b[i] -= val
                    elif ny == -1:
                        # Dirichlet
                        pass
                    elif ny == self.H:
                        # Dirichlet
                        pass
                    else:
                        inb = convert(nx, ny)
                        if dy != 0:
                            Ad[(i, inb)] += 1 / self.dy ** 2
                        else:  # dx != 0
                            Ad[(i, inb)] += 1 / self.dx ** 2
        row = [i for (i, _) in Ad.keys()]
        col = [j for (_, j) in Ad.keys()]
        data = list(Ad.values())
        A = csr_matrix((data, (row, col)))
        w = spsolve(A, b)
        wf = np.zeros((self.L, self.H), dtype=np.complex)
        for i in range(len(w)):
            x, y = iconvert(i)
            wf[x, y] = w[i]
        return wf

    def solve_forward(self, transport_energy):
        kk = np.sqrt(transport_energy)
        full_energy = (np.pi / self.height) ** 2 + transport_energy
        # full_energy = 0
        print("kk = {}".format(kk))
        print("full_energy = {}".format(full_energy))
        wf = np.zeros((self.L, self.H), dtype=np.complex)
        for y in range(self.H):
            wf[0, y] = min(y + 1, (self.H - 1 - y) + 1)  # np.sin(np.pi * y * self.dy / self.height)
            wf[1, y] = min(y + 1, (self.H - 1 - y) + 1)  # wf[0, y] # * np.exp(1j * kk * self.dx)

        def get(i, j):
            if i == -1:
                return 0
            elif j == -1:
                return 0
            elif i == self.L:
                raise RuntimeError("Shouldn't have happened")
            elif j == self.H:
                return 0
            else:
                return wf[i, j]

        # 1/dx^2(psi_(n+1)m + psi_(n-1)m - 2 psi_nm) + 1/dy^2(psi_n(m+1) + psi_n(m-1) - 2 psi_nm)  + E psi_nm = 0
        for x in range(2, self.L):
            for y in range(self.H):
                # wf[x, y] = (4  - self.dd ** 2 * energy) * get(x - 1, y) - get(x - 2, y) - get(x - 1, y - 1) - get(x - 1, y + 1)
                lhs = 0
                lhs += 1 / self.dx ** 2 * (get(x - 2, y) - 2 * get(x - 1, y))
                lhs += 1 / self.dy ** 2 * (get(x - 1, y + 1) + get(x - 1, y - 1) - 2 * get(x - 1, y))
                lhs += full_energy * get(x - 1, y)
                wf[x, y] = -lhs * self.dx ** 2

        return wf

    # energy is the transport energy
    def solve_sparse2(self, transport_energy):
        def convert(x, y):
            if x < 0 or x >= self.L or y < 0 or y >= self.H:
                raise RuntimeError("SHH")
            return x * self.H + y

        def iconvert(i):
            return (i // self.H, i % self.H)

        kk = np.sqrt(transport_energy)
        print("kk = {}".format(kk))
        full_energy = (np.pi / self.height) ** 2 + transport_energy
        full_energy = 0

        dim = self.L * self.H

        # A = np.zeros((dim, dim), dtype=np.complex)
        Ad = defaultdict(np.complex)
        b = np.zeros(dim, dtype=np.complex)
        # nabla^2 psi(x, y) + energy psi(x, y) = 0
        # 1 / dx^2 (psi_{x-1,y} + psi_{x+1,y} + psi_{x,y-1} + psi_{x,y+1}) + (energy - 4 / dx^2) psi(x, y) = 0

        initial = np.zeros((2, self.H), dtype=np.complex)
        for xx in range(2):
            for y in range(self.H):
                # initial[xx, y] = np.sin(np.pi / self.height * y * self.dy)
                initial[xx, y] = 1


        for x in range(0, self.L):
            for y in range(self.H):
                i = convert(x, y)
                lhs = 0j

                # Processing d^2/dx^2
                Ad[(i, i)] = 1 / self.dx ** 2

                if x == 0:
                    lhs += 1 / self.dx ** 2 * (-2 * initial[-1, y])
                else:
                    n1 = convert(x - 1, y)
                    Ad[(i, n1)] = 1 / self.dx ** 2 * (-2)

                if x <= 1:
                    lhs += 1 / self.dx ** 2 * initial[-2 + x, y]
                else:
                    n2 = convert(x - 2, y)
                    Ad[(i, n2)] = 1 / self.dx ** 2


                # Processing d^2/dy^2
                Ad[(i, i)] += 1 / self.dy ** 2 * (-2)

                if y == 0:
                    # lhs += 0
                    pass
                else:
                    n1 = convert(x, y - 1)
                    Ad[(i, n1)] = 1 / self.dy ** 2

                if y == self.H - 1:
                    # lhs += 0
                    pass
                else:
                    n1 = convert(x, y + 1)
                    Ad[(i, n1)] = 1 / self.dy ** 2

                # energy term
                Ad[(i, i)] += full_energy

                b[i] = -lhs


        row = [i for (i, _) in Ad.keys()]
        col = [j for (_, j) in Ad.keys()]
        data = list(Ad.values())
        A = csr_matrix((data, (row, col)), dtype=np.complex)
        print(A.todense())
        print(b)
        w = spsolve(A, b)
        wf = np.zeros((self.L, self.H), dtype=np.complex)
        for i in range(len(w)):
            x, y = iconvert(i)
            wf[x, y] = w[i]
        return wf

    # energy is the transport energy
    def solve_robin(self, transport_energy):
        def convert(x, y):
            if x < 0 or x >= self.L or y < 0 or y >= self.H:
                raise RuntimeError("SHH")
            return x * self.H + y

        def iconvert(i):
            return (i // self.H, i % self.H)

        kk = np.sqrt(transport_energy)
        print("kk = {}".format(kk))
        full_energy = (np.pi / self.height) ** 2 + transport_energy
        # full_energy = 0

        dim = self.L * self.H

        # A = np.zeros((dim, dim), dtype=np.complex)
        Ad = defaultdict(np.complex)
        b = np.zeros(dim, dtype=np.complex)
        # nabla^2 psi(x, y) + energy psi(x, y) = 0
        # 1 / dx^2 (psi_{x-1,y} + psi_{x+1,y} + psi_{x,y-1} + psi_{x,y+1}) + (energy - 4 / dx^2) psi(x, y) = 0


        def processUpper(x, y):
            i = convert(x, y)
            if y == self.H - 1:
                # Dirichlet, do nothing
                pass
            else:
                ni = convert(x, y + 1)
                Ad[(i, ni)] += 1 / self.dy ** 2

        def processLower(x, y):
            i = convert(x, y)
            if y == 0:
                # Dirichlet, do nothing
                pass
            else:
                ni = convert(x, y - 1)
                Ad[(i, ni)] += 1 / self.dy ** 2

        def processLeft(x, y):
            i = convert(x, y)
            if x == 0:
                b[i] -= 1 / self.dx ** 2 * np.sin(np.pi / self.H * y)
                # ri = convert(x + 1, y)
                # Ad[(i, i)] += 1 / self.dx ** 2 * (- kk * 2 * self.dx)
                # Ad[(i, ri)] += 1 / self.dx ** 2 * 1
            else:
                ni = convert(x - 1, y)
                Ad[(i, ni)] += 1 / self.dx ** 2

        def processRight(x, y):
            i = convert(x, y)
            if x == self.L - 1:
                li = convert(x - 1, y)
                Ad[(i, i)] += 1 / self.dx ** 2 * (1j * kk * 2 * self.dx)
                Ad[(i, li)] += 1 / self.dx ** 2 * 1
            else:
                ni = convert(x + 1, y)
                Ad[(i, ni)] += 1 / self.dx ** 2

        def process(x, y):
            i = convert(x, y)
            Ad[(i, i)] += -2 / self.dx ** 2
            Ad[(i, i)] += -2 / self.dy ** 2
            Ad[(i, i)] += full_energy

        for x in range(0, self.L):
            for y in range(self.H):
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
        for i in range(len(w)):
            x, y = iconvert(i)
            wf[x, y] = w[i]
        return wf

    def solve(self, energy):
        # wf = self.solve_forward(energy)

        wf = self.solve_robin(energy)
        # for i in range(10):
        #     print(list(wf[i, :]))
        # # for i in range(self.H):
        # #     print(wf[2, i])
        # # wf = wf[:7, :]
        pf = np.square(np.absolute(wf))
        # # for i in range(self.L):
        #     print(np.max(wf[i,:]))
        # rwf = np.real(wf)
        return pf


def main():
    width = 100
    height = 20
    dx = 0.1
    dy = 0.1
    tube = Tube(width, height, dx, dy)

    energy = 0.3
    pf = tube.solve(energy)

    pf = pf.transpose()
    import matplotlib.pyplot as plt

    plt.imshow(pf, cmap='gnuplot', origin='lower')
    plt.colorbar()
    plt.show()


main()
