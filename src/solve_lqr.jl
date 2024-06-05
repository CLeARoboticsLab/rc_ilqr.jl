#using NLsolve
# using Symbolics
using LinearAlgebra

include("solve_riccati.jl")

"""
OUTPUTS: 

K - Control Feedback Gain -> u(t) = -Kx

S - Defines optimal cost-to-go function -> J* = x' * S * x
"""

function solve_continuous_lqr(A,B,Q,R)
    S = solve_algebraic_riccati(A,B,Q,R)

    K = -inv(R) * B' * S

    return (K,S)
end

"""
"""

function solve_discrete_lqr()

end


"""
x[1] = x[1]
x[2] = x[2]
x[3]
x[4]
"""

function test()
    
    # [
    #     -10 * (x[1] * x[1] + x[2] * x[3]) + 2 * x[1] + 1,
    #     -10 * (x[1] * x[2] + x[2] * x[4]) + 2 * x[2] + x[1],
    #     -10 * (x[1] * x[3] + x[3] * x[4]) + x[1] + 2 * x[3],
    #     -10 * (x[4] * x[4] + x[2] * x[3]) + x[2] + x[3] + 2 * x[4] + 1
    # ]

    A = [1.0 1.0; 0 1.0]
    B = [1.0 0.0;0.0 1.0]
    Q = [1.0 0.0;0.0 1.0]
    R = 0.1 * [1.0 0.0;0.0 1.0]

    K,S = solve_continuous_lqr(A,B,Q,R)

    println("K: ", K)
    println("S: ", S)

end
    
    


