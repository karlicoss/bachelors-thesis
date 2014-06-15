import Calculus: integrate

function integrateComplex(f :: Function, a :: Float64, b :: Float64)
    return integrate(x -> real(f(x)), a, b) + 1im * integrate(x -> imag(f(x)), a, b)
end