type Resonator2DDomain
    H :: Float64
    Lx :: Float64
    Ly :: Float64
    S :: Float64

    function Resonator2DDomain(H, Lx, Ly, S)
        this = new ()
        this.H = H
        this.Lx = Lx
        this.Ly = Ly
        this.S = S
        return this
    end
end