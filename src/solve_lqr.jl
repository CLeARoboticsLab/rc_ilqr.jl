#using NLsolve
# using Symbolics
using LinearAlgebra

include("solve_riccati.jl")

"""
Solves the continuous lqr problem for the given A, B, Q, and R.
Problem defined as:
    minᵤ xᵀ * Q * x + uᵀ * R * u
    such that
    ẋ = A * x + B * u
    u = K * x

Notes:
    It has been proven that solving the Hamilton-Jacobi-Bellman equation for optimal u
    coincides with the optimal u in the corresponding LQR problem. Solving the HJB
    equation can be reduced to solving the riccati equation.

    HJB : for all x, 
        0 = minᵤ xᵀ * Q * x + uᵀ * R * u + ∇ₓJ⋆ * (A * x + B * u)
        J_star is the optimal cost-to-go function (value function). For an LQ
        problem, J_star is known to take the form J⋆(x) = xᵀ * S * x for some S.
        
    Solving the Riccati equation gives S.

    u⋆ = -1 * (inv(R) * Bᵀ * S) * x = -1 * K * x

Parameters:
A = A as defined in problem above
B = B as defined in problem above
Q = Q as defined in problem above
R = R as defined in problem above

Outputs: 
K - Control Feedback Gain -> u(t) = -Kx
S - Defines optimal cost-to-go function -> J* = x' * S * x
"""
function solve_continuous_lqr(A :: Matrix{Float64}, B :: Matrix{Float64},
    Q :: Matrix{Float64}, R :: Matrix{Float64})

    # Solving LQR problem boils down to solving the riccati equation
    S = solve_algebraic_riccati(A, B, Q, R)

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
    
    


