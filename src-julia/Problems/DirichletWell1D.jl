type DirichletWell1D
    a :: Float64
    b :: Float64
    maxn :: Int

    wavevectors :: Array{Float64}
    eigenenergies :: Array{Float64}
    eigenstates :: Array{Function}
    deigenstates :: Array{Function}

    greensFunctionHelmholtz :: Function
    greensFunctionHelmholtzDxs :: Function


    function DirichletWell1D(a :: Float64, b :: Float64, maxn :: Int)
        this = new ()
        this.a = a
        this.b = b
        this.maxn = maxn

        width = b - a
        this.wavevectors = [pi * n / width for n in 1: maxn]
        this.eigenenergies = [kk ^ 2 for kk in this.wavevectors]
        this.eigenstates = [x -> sqrt(2 / width) * sin(kk * (x - a)) for kk in this.wavevectors]
        this.deigenstates = [x -> sqrt(2 / width) * kk * cos(kk * (x - a)) for kk in this.wavevectors]

        this.greensFunctionHelmholtz = function (energy :: Float64)
            kk = sqrt(complex(energy))
            return function(x :: Float64, xs :: Float64)
                if x < xs
                    println(sin(kk * (x - a)) * sin(kk * (xs - b)))
                    return -sin(kk * (x - a)) * sin(kk * (xs - b)) / (kk * sin(kk * (b - a)))
                else
                    return -sin(kk * (x - b)) * sin(kk * (xs - a)) / (kk * sin(kk * (b - a)))
                end
            end
        end

        this.greensFunctionHelmholtzDxs = function (energy :: Float64)
            kk = sqrt(complex(energy))
            return function(x :: Float64, xs :: Float64)
                if x < xs
                    return -sin(kk * (x - a)) * cos(kk * (xs - b)) / (sin(kk * (b - a)))
                else
                    return -sin(kk * (x - b)) * cos(kk * (xs - a)) / (sin(kk * (b - a)))
                end
            end
        end

        return this
    end

end