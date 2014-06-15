module FreeParticleM

type FreeParticle
    greensFunctionHelmholtz :: Function

    function FreeParticle()
        this = new ()

        this.greensFunctionHelmholtz = function (energy :: Complex{Float64})
            kk = sqrt(energy)
            return function (x :: Float64, xs :: Float64)
                return 1im / (2 * kk) * exp(1im * kk * abs(x - xs))    
            end
        end

        return this
    end
end

end