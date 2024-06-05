"""
Solving the different Riccati equations.
"""

using LinearAlgebra

include("smith_method.jl")

"""
Solve the Algebraic Riccati equation using Newton-Kleinman with Smith's method
to solve the Lyapunov equation
"""
function solve_algebraic_riccati(A, B, Q, R, riccati_tolerance = 10e-5,
    lyapunov_tolerance = 10e-5, max_riccati_iters = 10e3,
    max_lyapunov_iters = 10e5)


    # Initialize K, S, P 
    K = Array{Float64}(undef, 0, 0) # TODO: K needs to make A + B * K Hurwitz

    S = A + B * K

    P_old = solve_lyapunov_custom(S, Q, K, R, lyapunov_tolerance, max_lyapunov_iters)
    P_new = Array{Float64}(undef, 0, 0)

    iters = 0

    while iters <= max_riccati_iters && norm(P_new - P_old, 2) >= riccati_tolerance
        P_old = copy(P_new)
        iters += 1

        K = -1 * inv(R) * B' * P_new
        S = A + B * K
        
        P_new = solve_lyapunov_custom(S, Q, K, R, lyapunov_tolerance, max_lyapunov_iters)
    end
    return P_new
end


function solve_lyapunov_custom(S, Q, K, R, lyapunov_tolerance, max_lyapunov_iters = 50)
    return solve_lyapunov(S, q = -1.0, Q + K' * R * K, lyapunov_tolerance, iter_lim = max_lyapunov_iters)
end
