#using NLsolve
# using Symbolics
using LinearAlgebra
using DifferentialEquations

include("solve_riccati.jl")

"""
Solves the continuous, infinite horizon lqr problem for the given A, B, Q, and R.
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
        
    Solving the algebraic Riccati equation (ARE) gives S.

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
function solve_continuous_inf_lqr(A :: Matrix{Float64}, B :: Matrix{Float64},
    Q :: Matrix{Float64}, R :: Matrix{Float64})

    # Solving LQR problem boils down to solving the riccati equation
    S = solve_algebraic_riccati(A, B, Q, R)

    K = -inv(R) * B' * S

    return (K,S)
end

#TODO: fix comments
"""
Solves the continuous, finite horizon lqr problem for the given A, B, Q, and R.
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
        0 = minᵤ xᵀ * Q * x + uᵀ * R * u + ∇ₓJ⋆ * (A * x + B * u) + ∇ₜJ⋆
        J_star is the optimal cost-to-go function (value function). For an LQ
        problem, J_star is known to take the form J⋆(x) = xᵀ * S * x for some S.
        
    Solving the differential Riccati equation gives S. It is difficult (numerically)
    to solve for S, so set S = P * Pᵀ, then re-write as:
        -Ṗ(t) = AᵀP(t) - .5 * S(t) * B * R⁻¹ * Bᵀ * P(t) + .5 * Q * P⁻ᵀ,
            P(t_f) = sqrt(Q_f), Q_f > 0, P(t) invertible

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
function solve_continuous_finite_lqr(A :: Matrix{Float64}, B :: Matrix{Float64},
    Q :: Matrix{Float64}, Q_f :: Matrix{Float64}, R :: Matrix{Float64}, t_0, t_f, step_size)

    S = solve_differential_riccati(A, B, Q, Q_f, R, t_0, t_f, step_size)

    K = Array{Matrix{Float64}}(undef, size(S)[1], 1)

    for i = 1 : size(S)[1]
        K[i] = -1 * inv(R) * B' * S[i]
    end

    return (K, S)
end



"""
"""

function solve_discrete_lqr()

end


"""
"""
function test_solve_continuous_inf_lqr()

    A = [1.0 1.0; 0 1.0]
    B = [1.0 0.0;0.0 1.0]
    Q = [1.0 0.0;0.0 1.0]
    R = 0.1 * [1.0 0.0;0.0 1.0]

    K,S = solve_continuous_info_lqr(A,B,Q,R)

    println("K: ", K)
    println("S: ", S)

end
    
    
function test_solve_continuous_finite_lqr()

    A = [1.0 1.0; 0 1.0]
    B = [1.0 0.0;0.0 1.0]
    Q = [1.0 0.0;0.0 1.0]
    Q_f = copy(Q)
    R = 0.1 * [1.0 0.0;0.0 1.0]

    K,S = solve_continuous_finite_lqr(A,B,Q,Q_f,R, 0, 11, 1)

    println("K: ", K)
    println("S: ", S)

end

