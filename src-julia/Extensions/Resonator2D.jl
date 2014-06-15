using ..Parent.Problems
import Parent.Problems: DirichletWell1D, DirichletWaveguide2D, DirichletWell2D

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

    function Resonator2D (H :: Float64, Lx :: Float64, Ly :: Float64, delta :: Float64, maxn :: Int)
        this = new ()

        this.H = H
        this.Lx = Lx
        this.Ly = Ly
        
        this.delta = delta
        this.x0 = 0.0
        this.y0 = 0.0

        this.maxn = maxn

        this.resonator = DirichletWell2D(-Lx / 2, Lx / 2, 0.0, Ly, maxn)
        this.resX = this.resonator.wellX
        this.resY = this.resonator.wellY

        this.waveguide = DirichletWaveguide2D(-this.H, 0.0, maxn)
        this.wgY = this.waveguide.wellY

        # TODO print eigenenergies

        this.greensFunctionResonatorDn = function (energy :: Complex{Float64})
            return this.resonator.greensFunctionHelmholtzDys(energy)
        end

        return this
    end

end