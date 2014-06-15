# using Parent.Problems
import Parent.Problems: FreeParticle, DirichletWell1D, DirichletWaveguide2D, DirichletWell2D
import Parent.Extensions: Resonator2D, Resonator2DDomain

import Formatting: fmt

using PyPlot

# nw = DirichletWell1D(0.0, 2.0, 10)
# println(nw.greensFunctionHelmholtz(10.0)(0.5, 1.3))

H = 1.0
Lx = 1.0
Ly = 1.0
S = 0.01

domain = Resonator2DDomain(H, Lx, Ly, S)
maxn = 100

energy = 19.75944
mode = 1

sp = Resonator2D(domain, maxn)

# for energy in [19.7: 0.01: 20.0]
#    res.computeMode(mode, energy, verbose = true)
#    println()
# end

# for energy in [19.75900: 0.000001: 19.75945]
#    res.computeMode(mode, energy, verbose = false)
# end



# TODO HOW TO SET IMAGE SIZE???
function plotTransmissionOverEnergy(
    dcs,
    left :: Float64, right :: Float64, step :: Float64
    ; fname = "transmission.png", lines = [])

    maxt = 1.0 + 0.1

    xs = [left: step: right]
    ys = map(en -> dcs.computeAll(en)[2], xs)

    inch = 2.5
    figure(figsize=(18.0 / inch, 8.0 / inch))

    vlines(lines, 0.0, maxt, linestyles = "dashed")

    tick_params(axis = "both", which = "major", labelsize = 8)

    xticks_ = [left: 5: right]
    xlabels_ = [fmt("d", int(tick)) for tick in xticks_]
    xlim(left, right)
    xticks(xticks_, xlabels_)
    xlabel("E")

    yticks_ = [0.0: 0.1: maxt]
    ylabels_ = [fmt(".1f", tick) for tick in yticks_]
    ylim(0.0, maxt)
    yticks(yticks_, ylabels_)
    ylabel("T")

    plot(xs, ys)

    savefig(fname, bbox_inches = "tight", dpi = 300)
    close()
    return nothing
end

function plotPdensity(
    domain :: Resonator2DDomain,
    wf :: Function,
    maxnWf :: Int,
    fx :: Float64, tx :: Float64, dx :: Float64,
    dy :: Float64;
    fname = "wavefunction.png")

    fy = -domain.H
    ty = domain.Ly

    border = (domain.H + domain.Ly) / 10 # TODO
    fy = fy #  - border
    ty = ty # + border

    pf = (x, y) -> abs2(wf(x, y, maxn = maxnWf))
    pfc = (x, y) -> (x ^ 2 + y ^ 2 > (domain.S / 2) ^ 2) ? pf(x, y) : 0.0 # TODO this is to exclude the singulatiry
    # vpf = np.vectorize(pfc)

    xx = [fx: dx: tx + dx]
    yy = [fy: dy: ty + dy]
    zz = [pfc(x, y) for y in yy, x in xx]

    inch = 2.5
    proportions = (ty - fy) / (tx - fx) # approx height / width
    imgwidth = 25 / inch
    imgheight = imgwidth * proportions
    figure(figsize = (imgwidth, imgheight))
    axis("equal")

    tick_params(axis = "both", which = "major", labelsize = 8)

    xticks_ = [fx: 0.5: tx + dx]
    xlabels_ = [fmt(".1f", tick) for tick in xticks_]
    xlim(fx, tx)
    xticks(xticks_, xlabels_)
    xlabel("x, нм")

    yticks_ = [-domain.H: 0.5: domain.Ly + border + dy]
    ylabels_ = [fmt(".1f", tick) for tick in yticks_]
    ylim(fy, ty)
    yticks(yticks_, ylabels_)
    ylabel("y, нм")

    pc = pcolor(xx, yy, zz, cmap = "gnuplot") #, norm=Normalize(z_min, z_max))

    function drawWall(a, b, c, d)
        fill(
            [a[1], b[1], c[1], d[1]],
            [a[2], b[2], c[2], d[2]],
            closed = true,
            fill = true,
            color = "blue",
            linewidth = 0,
            hatch = "//")
    end

    wwidth = 5 * dy
    drawWall([fx, 0 - wwidth / 2], [-domain.Lx / 2, -wwidth / 2], [-domain.Lx / 2, domain.Ly + dy], [fx, domain.Ly + dy])
    drawWall([tx, 0 - wwidth / 2], [domain.Lx / 2 + dx, -wwidth / 2], [domain.Lx / 2 + dx, domain.Ly + dy], [tx, domain.Ly + dy])
    drawWall([fx, domain.Ly + dy], [tx, domain.Ly + dy], [tx, domain.Ly + 3 * border], [fx, domain.Ly + 3 * border]) # WTF????
    drawWall([fx, -domain.H - 3 * border], [tx, -domain.H - 3 * border], [tx, -domain.H], [fx, -domain.H])
    drawWall([-domain.Lx / 2, -wwidth / 2], [-domain.S / 2, -wwidth / 2], [-domain.S / 2, wwidth / 2], [-domain.Lx / 2, wwidth / 2])
    drawWall([domain.Lx / 2 + dx, -wwidth / 2], [domain.S / 2 + dx, -wwidth / 2], [domain.S / 2 + dx, wwidth / 2], [domain.Lx / 2 + dx, wwidth / 2])

    colorbar(pc, ticks = [])

    savefig(fname, bbox_inches = "tight", dpi = 200)
    close()
end


plotTransmissionOverEnergy(
   res,
   10.0, 21.0, 1.0,
   lines = res.resonator.eigenenergies)

# maxnWf = 2
# fx = -6.0
# tx = 6.0
# dx = 0.02
# dy = 0.01


# energy = 19.75944
# res = sp.computeMode(1, energy, verbose = true)
# plotPdensity(domain,
#     res[1], maxnWf,
#    fx, tx, dx,
#    dy)