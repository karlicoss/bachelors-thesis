import Iterators: product

type DirichletWell2D
    aX :: Float64
    bX :: Float64
    aY :: Float64
    bY :: Float64
    maxn :: Int

    wellX :: DirichletWell1D
    wellY :: DirichletWell1D

    eigenenergies :: Array{Float64}

    eigenenergiess # with state numbers

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

        this.eigenenergiess = sort([(ex + ey, (i, j)) for ((i, ex), (j, ey)) in product(enumerate(this.wellX.eigenenergies), enumerate(this.wellY.eigenenergies))])
        this.eigenenergies = map(x -> x[1], this.eigenenergiess)        

        this.greensFunctionHelmholtzDys = function (energy :: Float64)
            function fun(x :: Float64, y :: Float64, xs :: Float64, ys :: Float64; maxn = this.maxn)
                res = 0.0im
                for m = 1: maxn
                    gf = this.wellX.greensFunctionHelmholtz(energy - this.wellY.eigenenergies[m])
                    gg = gf(x, xs)
                    if isnan(real(gg))
                        # println(energy - this.wellY.eigenenergies[m])
                        gg = 0.0
                    end
                    #println(gg)
                    res += this.wellY.eigenstates[m](y) * this.wellY.deigenstates[m](ys) * gg
                    # for n = 1: maxn
                    #     tt = 1.0
                    #     tt *= this.wellX.eigenstates[n](x)
                    #     tt *= this.wellX.eigenstates[n](xs)
                    #     tt *= this.wellY.eigenstates[m](y)
                    #     tt *= this.wellY.deigenstates[m](ys)
                    #     tt /= (this.wellX.eigenenergies[n] + this.wellY.eigenenergies[m] - energy)
                    #     res += tt
                    # end
                end
                return res
            end
            return fun
        end

        return this
    end

end