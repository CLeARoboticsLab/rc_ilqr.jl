module rc_ilqr

using Symbolics
using LinearAlgebra
using Revise

include("solve_lqr.jl")
include("solve_ilqr.jl")
include("get_trajectory.jl")

end