# using ..Parent.Problems
import ..Parent.Problems: DirichletWell1D, DirichletWaveguide2D, DirichletWell2D
import ..Parent.Tools: computeProbCurrentNumerical

# using Formatting
import Formatting: printfmt

type Resonator2D
    H :: Float64
    Lx :: Float64
    Ly :: Float64

    delta :: Float64
    x0 :: Float64
    y0 :: Float64

    maxn :: Int

    resonator :: DirichletWell2D
    resX :: DirichletWell1D
    resY :: DirichletWell1D

    waveguide :: DirichletWaveguide2D
    wgY :: DirichletWell1D


    greensFunctionResonatorDn :: Function
    greensFunctionWaveguideDn :: Function
    computeMode :: Function
    computeAll :: Function

    function Resonator2D (domain :: Resonator2DDomain, maxn :: Int)
        this = new ()

        this.H = domain.H
        this.Lx = domain.Lx
        this.Ly = domain.Ly
        
        this.delta = domain.S
        this.x0 = 0.0
        this.y0 = 0.0

        this.maxn = maxn

        this.resonator = DirichletWell2D(-this.Lx / 2, this.Lx / 2, 0.0, this.Ly, maxn)
        this.resX = this.resonator.wellX
        this.resY = this.resonator.wellY

        this.waveguide = DirichletWaveguide2D(-this.H, 0.0, maxn)
        this.wgY = this.waveguide.wellY

        println("resX eigenenergies:")
        show(this.resX.eigenenergies[1:10]); println()

        println("resY eigenenergies:")
        show(this.resY.eigenenergies[1:10]); println()

        println("wgY eigenenergies:")
        show(this.wgY.eigenenergies[1:10]); println()

        this.greensFunctionResonatorDn = function (energy :: Float64)
            return this.resonator.greensFunctionHelmholtzDys(energy)
        end

        this.greensFunctionWaveguideDn = function (energy :: Float64)
            pf = this.waveguide.greensFunctionHelmholtzDys(energy)
            function fff(x, y, xs, ys; maxn = this.maxn)
                return -pf(x, y, xs, ys, maxn = maxn)
            end
            return fff
        end

        this.computeAll = function (energy :: Float64)
            # TODO
            return this.computeMode(1, energy)
        end

        this.computeMode = function fff2(mode :: Int, energy :: Float64; verbose = false)
            @assert energy > this.wgY.eigenenergies[mode]
            kk = sqrt(complex(energy - this.wgY.eigenenergies[mode]))

            uwf = (x :: Float64, y :: Float64) -> this.wgY.eigenstates[mode](y) * exp(1im * kk * x)

            gamma = 0.57721566490153286060
            k0 = 1im / this.delta * exp(-gamma)
            e0 = real(k0 ^ 2)

            greensWaveguideDn = this.greensFunctionWaveguideDn(energy)
            greensWaveguide0Dn = this.greensFunctionWaveguideDn(e0)
            greensResonatorDn = this.greensFunctionResonatorDn(energy)
            greensResonator0Dn = this.greensFunctionResonatorDn(e0)

            dd = 0.0001
            FW = (greensWaveguideDn(this.x0, this.y0 - dd, this.x0, this.y0) - greensWaveguide0Dn(this.x0, this.y0 - dd, this.x0, this.y0)) / dd
            FR = (greensResonatorDn(this.x0, this.y0 + dd, this.x0, this.y0) - greensResonator0Dn(this.x0, this.y0 + dd, this.x0, this.y0)) / dd
            if verbose
                printfmt("|FW| = {}, |FR| = {}\n", abs(FW), abs(FR))
            end

            alphaR = -this.wgY.deigenstates[mode](this.y0) / (FW + FR)
            alphaW = -alphaR
            if verbose
                printfmt("AW = {}, AR = {}\n", alphaW, alphaR)
            end

            function wavefunction(x, y; maxn = this.maxn)
                if y < -this.H
                    return 0im
                elseif y < 0
                    return uwf(x, y) + alphaW * greensWaveguideDn(x, y, this.x0, this.y0, maxn = maxn)
                elseif y < this.Ly
                    if x < -this.Lx / 2
                        return 0im
                    elseif x < this.Lx / 2
                        return alphaR * greensResonatorDn(x, y, this.x0, this.y0, maxn = maxn)
                    else
                        return 0im
                    end
                else
                    return 0im
                end
            end

            xleft = -20.0
            xright = 21.0
            jinc = computeProbCurrentNumerical(uwf, xleft, -this.H, 0.0)
            jtrans = computeProbCurrentNumerical(wavefunction, xleft, -this.H, 0.0)
            if verbose
                printfmt("Jinc = {}, Jtrans = {}\n", jinc, jtrans)
            end

            T = abs(jtrans) / abs(jinc)
            @printf("Energy = %.7f, T = %.3f\n", energy, T)

            # printfmt("Energy = {:.7f}, T = {:.3f}\n", energy, T) TODO WTF???
            return (wavefunction, T)
        end

        return this
    end

end