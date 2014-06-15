using Parent.Problems
import Parent.Problems: FreeParticle, DirichletWell1D, DirichletWaveguide2D, DirichletWell2D
import Parent.Extensions: Resonator2D

fp = FreeParticle()
dw = DirichletWell1D(1.0, 2.0, 100)
dwg = DirichletWaveguide2D(1.0, 2.0, 100)
dwell = DirichletWell2D(1.0, 2.0, 1.0, 2.0, 100)

res = Resonator2D(1.0, 1.0, 1.0, 0.01, 100)

println(fp.greensFunctionHelmholtz(complex(1.0))(0.5, 0.0))

println("hello")