using LinearAlgebra

"""
Uses Smith's method to solve the Lyapunov Equation
"""

function solve_lyapunov(Aₒ :: Matrix{Float64}, q :: Float64 = -1.0, DDᵀ :: Matrix{Float64}, tol :: Float64, iter_lim :: Int64 = 50)

    n = size(Aₒ)[1]

    xₒ = zeros(size(Aₒ))

    Q = q*I(n)

    iter = 0

    x_new = xₒ

    while iter < iter_lim

        x_old = x_new

        U = (Aₒ - Q) * inv(Aₒ + Q)
        V = -2 * q * inv(Aₒ' + Q) * DDᵀ * inv(Aₒ + Q)

        x_new = U' * x_old * U + V

        if LinearAlgebra.norm(x_new - x_old , 2) < tol

            println("Lyapunov Equation Converged")
            return x_new

        end
    end
    println("ERROR: Lyapunov Equation failed to converge.")
    return -1
end




