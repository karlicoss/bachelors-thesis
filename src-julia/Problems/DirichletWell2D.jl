using Iterators

type DirichletWell2D
    aX :: Float64
    bX :: Float64
    aY :: Float64
    bY :: Float64
    maxn :: Int

    wellX :: DirichletWell1D
    wellY :: DirichletWell1D

    # wavevectors :: Array{Complex{Float64}}
    eigenenergies :: Array{Float64}
    # eigenstates :: Array{Function}
    # deigenstates :: Array{Function}

    # greensFunctionHelmholtz :: Function
    greensFunctionHelmholtzDys :: Function


    function DirichletWell2D (aX :: Float64, bX :: Float64, aY :: Float64, bY :: Float64, maxn :: Int)
        this = new ()
        this.aX = aX
        this.bX = bX
        this.aY = aY
        this.bY = bY
        this.maxn = maxn

        this.wellX = DirichletWell1D(aX, bX, maxn)
        this.wellY = DirichletWell1D(aY, bY, maxn)

        this.eigenenergies = sort([ex + ey for (ex, ey) in Iterators.product(this.wellX.eigenenergies, this.wellY.eigenenergies)])

        this.greensFunctionHelmholtzDys = function (energy :: Complex{Float64})
            function fun(x :: Float64, y :: Float64, xs :: Float64, ys :: Float64; maxn = this.maxn)
                res = complex(0.0)
                for m = 1: maxn
                    gf = this.wellX.greensFunctionHelmholtz(complex(energy - this.wellY.eigenenergies[m]))
                    res += this.wellY.eigenstates[m](y) * this.wellY.deigenstates[m](ys) * gf(x, xs)
                end
                return res
            end
            return fun
        end

        return this
    end

end