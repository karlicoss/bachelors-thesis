include("Problems/FreeParticle.jl")
using FreeParticleM: FreeParticle
include("Problems/DirichletWell1D.jl")
using DirichletWell1DM: DirichletWell1D

include("Problems/DirichletWaveguide2D.jl")
using DirichletWaveguide2DM: DirichletWaveguide2D


fp = FreeParticle()
dw = DirichletWell1D(1.0, 2.0, 100)
dwg = DirichletWaveguide2D(1.0, 2.0, 100)

println(fp.greensFunctionHelmholtz(complex(1.0))(0.5, 0.0))

println("hello")