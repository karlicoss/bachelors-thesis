# using Parent.Problems
import Parent.Problems: FreeParticle, DirichletWell1D, DirichletWaveguide2D, DirichletWell2D
import Parent.Extensions: Resonator2D

# nw = DirichletWell1D(0.0, 2.0, 10)
# println(nw.greensFunctionHelmholtz(10.0)(0.5, 1.3))


Lx = 2.0
Ly = 1.0
H = 1.0
S = 0.01
maxn = 100

energy = 19.77965
mode = 1

res = Resonator2D(H, Lx, Ly, S, maxn)

for energy in [25.0: 0.01: 45.0]
    res.computeMode(mode, energy, verbose = false)
end