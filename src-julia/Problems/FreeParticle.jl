type FreeParticle
    greensFunctionHelmholtz :: Function

    function FreeParticle()
        this = new ()

        this.greensFunctionHelmholtz = function (energy :: Float64)
            kk = sqrt(complex(energy))
            return function (x :: Float64, xs :: Float64)
                return 1im / (2 * kk) * exp(1im * kk * abs(x - xs))    
            end
        end

        return this
    end
end