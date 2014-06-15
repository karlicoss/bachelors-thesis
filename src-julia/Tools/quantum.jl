import Calculus: derivative

function computeProbCurrent(wf :: Function, dwf :: Function, x :: Float64, ya :: Float64, yb :: Float64)
    return integrateComplex(y -> conj(wf(x, y)) * dwf(x, y) - wf(x, y) * conj(dwf(x, y)), ya, yb)
end

# TODO EPSILON??
function computeProbCurrentNumerical(wf :: Function, x :: Float64, ya :: Float64, yb :: Float64; eps = 0.001)
    dwf = (x, y) -> derivative(xx -> wf(xx, y), x)
    return computeProbCurrent(wf, dwf, x, ya, yb)
end