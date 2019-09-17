################################################################################
#
# WRI dual formulation (frequency domain version)
#
################################################################################



module WRIdual

using LinearAlgebra, SparseArrays
using ImageFiltering
using PyPlot

# const R = Float32
# const C = ComplexF32
const R = Float64
const C = ComplexF64
const RuC = Union{R, C}
export R, C, RuC

include("utils.jl")
include("RestrictionLinAlg.jl")
include("HelmholtzLinAlg.jl")
include("Objectives_FWI.jl")
include("Objectives_WRIdual_eps.jl")
include("Objectives_WRIdual_lambda.jl")
include("PDEWeightFun.jl")

end
