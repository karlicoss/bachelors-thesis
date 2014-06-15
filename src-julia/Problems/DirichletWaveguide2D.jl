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

        this.greensFunctionHelmholtzDys = function (energy :: Float64)
            function fun(x :: Float64, y :: Float64, xs :: Float64, ys :: Float64; maxn = this.maxn)
                res = 0.0im
                for m = 1: maxn
                    gf = this.freeX.greensFunctionHelmholtz(energy - this.wellY.eigenenergies[m])
                    res += this.wellY.eigenstates[m](y) * this.wellY.deigenstates[m](ys) * gf(x, xs)
                end
                return res
            end
            return fun
        end

        return this
    end

end