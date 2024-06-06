#using NLsolve
# using Symbolics
using LinearAlgebra
using DifferentialEquations
using GLMakie

include("solve_riccati.jl")
include("get_trajectory.jl")

"""
    Solves the continuous, infinite horizon lqr problem for the given A, B, Q, and R.
    Problem defined as:
        minᵤ xᵀ * Q * x + uᵀ * R * u
        such that
        ẋ = A * x + B * u
        u = K * x

    Notes:
        It has been proven that satisfying the Hamilton-Jacobi-Bellman condition leads to 
        the optimal u in the LQR problem. Solving the resulting equation can be reduced to
        solving the algebraic Riccati equation. Note the equation satisfying the HJB condition is:
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

    K = inv(R) * B' * S

    return (K,S)
end

"""
    Solves the continuous, finite horizon lqr problem for the given A, B, Q, and R.
    Problem defined as:
        minᵤ xᵀ * Q * x + uᵀ * R * u
        such that
        ẋ = A * x + B * u
        u = K * x

    Notes:
        It has been proven that satisfying the Hamilton-Jacobi-Bellman condition leads to 
        the optimal u in the LQR problem. Solving the resulting equation can be reduced to
        solving the differential Riccati equation. Note the equation satisfying the HJB condition is:
            for all x, 
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
        K[i] = inv(R) * B' * S[i]
    end

    return (K, S)
end



"""

"""
function solve_discrete_finite_lqr(A :: Matrix{Float64}, B :: Matrix{Float64},
    Q :: Matrix{Float64}, R :: Matrix{Float64}, N :: Float64)

    S = solve_riccati_difference(A, B, Q, R, N)

    K = Array{Matrix{Float64}}(undef, N, 1)
    for n = 1 : N
        K[n] = inv(R + B' * S[n] * B) * B' * S[n] * A
    end

    return (K, S)
end

