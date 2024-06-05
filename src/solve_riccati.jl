"""
Solving the different Riccati equations.
"""

using LinearAlgebra

include("smith_method.jl")

"""
Solve the Algebraic Riccati equation using Newton-Kleinman with Smith's method
to solve the Lyapunov equation
"""
function solve_algebraic_riccati(A :: Matrix{Float64}, B :: Matrix{Float64},
    Q :: Matrix{Float64}, R :: Matrix{Float64}, riccati_tolerance :: Float64 = 10e-5,
    lyapunov_tolerance :: Float64 = 10e-5, max_riccati_iters :: Float64 = 10e3,
    max_lyapunov_iters :: Int64 = 100)


    # Initialize K, S, P 
    gamma = gerghgorin_gamma(A)
    K = -1 * inv(B' * B) * B' * gamma

    S = A + B * K

    P_new = solve_lyapunov_custom(S, Q, K, R, lyapunov_tolerance, max_lyapunov_iters)
    P_old = Array{Float64}(undef, size(P_new)...)

    iters = 0

    while iters <= max_riccati_iters && norm(P_new - P_old, 2) >= riccati_tolerance
        P_old = copy(P_new)
        iters += 1

        K = -1 * inv(R) * B' * P_old
        S = A + B * K
    
        P_new = solve_lyapunov_custom(S, Q, K, R, lyapunov_tolerance, max_lyapunov_iters)
    end
    return P_new
end


function solve_lyapunov_custom(S :: Matrix{Float64}, Q :: Matrix{Float64}, K :: Matrix{Float64},
    R :: Matrix{Float64}, lyapunov_tolerance :: Float64, max_lyapunov_iters :: Int64 = 50)
    return solve_lyapunov(S, (Q + K' * R * K), lyapunov_tolerance, max_lyapunov_iters, -1.0)
end

function gerghgorin_gamma(A :: Matrix{Float64})
    gamma = 0
    for t = 1 : size(A)[1]
        temp = norm(A[t, :], 1)
        gamma = max(gamma, temp)
    end
    return gamma + 1
end

export solve_algebraic_riccati
