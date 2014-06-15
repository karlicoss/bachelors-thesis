module DirichletWaveguide2DM

include("FreeParticle.jl")
using .FreeParticleM: FreeParticle
include("DirichletWell1D.jl")
using .DirichletWell1DM: DirichletWell1D

type DirichletWaveguide2D
    aY :: Float64
    bY :: Float64
    maxn :: Int

    freeX :: FreeParticle
    wellY :: DirichletWell1D

    # wavevectors :: Array{Complex{Float64}}
    # eigenenergies :: Array{Float64}
    # eigenstates :: Array{Function}
    # deigenstates :: Array{Function}

    # greensFunctionHelmholtz :: Function
    greensFunctionHelmholtzDys :: Function


    function DirichletWaveguide2D (aY :: Float64, bY :: Float64, maxn :: Int)
        this = new ()
        this.aY = aY
        this.bY = bY
        this.maxn = maxn

        this.freeX = FreeParticle()
        this.wellY = DirichletWell1D(aY, bY, maxn)

        this.greensFunctionHelmholtzDys = function (energy :: Complex{Float64})
            return function (x :: Float64, y :: Float64, xs :: Float64, ys :: Float64)
                res = complex(0.0)
                for m = 1: maxn
                    gf = this.freeX.greensFunctionHelmholtz(complex(energy - this.wellY.eigenenergies[m]))
                    res += this.wellY.eigenstates[m](y) * this.wellY.deigenstates[m](ys) * gf(x, xs)
                end
                return res
            end
        end

        return this
    end

end


end