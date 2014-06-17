# using Parent.Problems
import Parent.Problems: FreeParticle, DirichletWell1D, DirichletWaveguide2D, DirichletWell2D
import Parent.Extensions: Resonator2D, Resonator2DDomain

import Formatting: fmt, format

using PyPlot

# TODO HOW TO SET IMAGE SIZE???
function plotTransmissionOverEnergy(
    dcs,
    left :: Float64, right :: Float64, step :: Float64; 
    fname = "transmission2.png",
    xticks_ = [],
    xlabels_ = [])

    eigenenergiess = filter(x -> x[1] >= left && x[1] <= right, sp.resonator.eigenenergiess)

    maxt = 1.0 + 0.1

    xs = [left: step: right]
    ys = map(en -> dcs.computeAll(en)[2], xs)

    inch = 2.5
    figure(figsize=(30.0 / inch, 10.0 / inch))

    vlines(map(x -> x[1], eigenenergiess), 0.0, maxt, linestyles = "dashed")

    for i in 1: length(eigenenergiess)
        ee = eigenenergiess[i][1]
        (xn, yn) = eigenenergiess[i][2]
        lbl = format("{},{}", xn, yn)
        text(ee + (right - left) / 1000, i * 1.0 / length(eigenenergiess), lbl,
            color = "red",
            fontsize = "small",
            fontweight = "semibold")
    end

    tick_params(axis = "both", which = "major", labelsize = 8)

    if length(xticks_) == 0
        xticks_ = [0: 5: right]
    end

    if length(xlabels_) == 0        
        xlabels_ = [fmt("d", int(tick)) for tick in xticks_]
    end
    xticks(xticks_, xlabels_)
    xlim(left, right)
    xlabel("E, хартри")

    yticks_ = [0.0: 0.1: maxt]
    ylabels_ = [fmt(".1f", tick) for tick in yticks_]
    yticks(yticks_, ylabels_)
    ylim(0.0, maxt)
    ylabel("T")

    plot(xs, ys, antialiased = true, linewidth = 1.0)

    savefig(fname, bbox_inches = "tight")
    close()
    return nothing
end

function plotTransmissionOverSize(
    domain :: Resonator2DDomain,
    mode :: Int,
    energy :: Float64,
    leftw :: Float64, rightw :: Float64, stepw :: Float64; 
    fname = "transmission.png",
    xticks_ = [],
    xlabels_ = [])

    maxt = 1.0 + 0.1

    function fff(w :: Float64)
        domain.Lx = w
        sp = Resonator2D(domain, maxn)
        return sp.computeMode(mode, energy)[2]
    end

    xs = [leftw: stepw: rightw]
    ys = map(fff, xs)

    inch = 2.5
    figure(figsize=(30.0 / inch, 10.0 / inch))

    tick_params(axis = "both", which = "major", labelsize = 8)

    xticks(xticks_, xlabels_)
    xlim(leftw, rightw)
    xlabel("Ly, б.р.")

    yticks_ = [0.0: 0.1: maxt]
    ylabels_ = [fmt(".1f", tick) for tick in yticks_]
    yticks(yticks_, ylabels_)
    ylim(0.0, maxt)
    ylabel("T")

    plot(xs, ys, antialiased = true, linewidth = 1.0)

    savefig(fname, bbox_inches = "tight")
    close()
    return nothing
end


function plotPdensity(
    domain :: Resonator2DDomain,
    wf :: Function,
    maxnWf :: Int,
    fx :: Float64, tx :: Float64, dx :: Float64,
    dy :: Float64;
    fname = "wavefunction2.png")

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
    # xticks(xticks_, xlabels_)
    xlabel("x, б. р.")

    yticks_ = [-domain.H: 0.5: domain.Ly + border + dy]
    ylabels_ = [fmt(".1f", tick) for tick in yticks_]
    ylim(fy, ty)
    # yticks(yticks_, ylabels_)
    ylabel("y, б. р.")

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

    savefig(fname, bbox_inches = "tight")
    close()
end

H = 100
Lx = 200
Ly = 100
S = 10

domain = Resonator2DDomain(H, Lx, Ly, S)
maxn = 100

sp = Resonator2D(domain, maxn)

mode = 1


# energy = 11.6183
# println(sp.computeMode(mode, energy, verbose = true))

# ff = 0.00315
# tt = 0.00330
# step = 0.0000001
# # step = 0.01
# xticks_ = [ff: (tt - ff) / 10: tt]
# xlabels_ = map(x -> fmt(".5f", x), xticks_)
# plotTransmissionOverEnergy(
#     sp,
#     ff, tt, step,
#     xticks_ = xticks_,
#     xlabels_ = xlabels_)


# maxnWf = 10
# fx = -600.0
# tx = 600.0
# dx = 2.0
# dy = 1.0
# energy = 0.0032288 # 11r
# # energy = 0.0033 # 11nr
# res = sp.computeMode(1, energy, verbose = true)
# plotPdensity(domain,
#     res[1], maxnWf,
#     fx, tx, dx,
#     dy)


energy = 0.0032288
ff = 0.95 * Lx
tt = 1.05 * Lx
stepw = 0.0001 * Lx
xticks_ = [ff: (tt - ff) / 10: tt]
xlabels_ = map(x -> fmt("d", x), xticks_)
plotTransmissionOverSize(
    domain,
    mode,
    energy,
    ff, tt, stepw,
    xticks_ = xticks_,
    xlabels_ = xlabels_)