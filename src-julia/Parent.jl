# include("Extensions/Resonator2D.jl")
# using Resonator2DM: Resonator2D
module Parent

module Tools
    include("Tools/tools.jl")
    include("Tools/quantum.jl")
end

module Problems
    include("Problems/FreeParticle.jl")
    include("Problems/DirichletWell1D.jl")
    include("Problems/DirichletWaveguide2D.jl")
    include("Problems/DirichletWell2D.jl")
end

module Extensions
    include("Extensions/Resonator2D.jl")
end

end