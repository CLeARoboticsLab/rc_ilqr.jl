using LinearAlgebra

"""
Uses Smith's method to solve the Lyapunov Equation

Notes: 
    Lyapunov equation: Aₒᵀ * X + X * Aₒ = - D * Dᵀ

Parameters:
    Aₒ =  Term being transposed in the left-hand side of the Lyapunov equation
    DDᵀ = Right hand side of the Lyapunov equation
    tol = Tolerance threshold, once the new x values are tol close
        to the old x values we stop iterating
    iter_lim = Maximum number of iterations
    q = A float less than 0 which may contributes to the speed of convergence

Output:
    x_new = A value of X such that the Lyapunov equation holds

"""

function solve_lyapunov(Aₒ :: Matrix{Float64}, DDᵀ :: Matrix{Float64},
    tol :: Float64, iter_lim :: Int64 = 50, q :: Float64 = -1.0)

    # Precondition check
    if q >= 0
        println("q must be a Float less than 0")
    end

    # Initial guess for X
    xₒ = zeros(size(Aₒ))

    # Q is a shorthand for q * I
    Q = q*I(size(Aₒ)[1])

    iter = 0

    # Initialize iteration values
    x_new = xₒ
    x_old = zeros(size(Aₒ))

    while iter < iter_lim

        x_old = x_new

        # Placeholder terms for solving for X
        U = (Aₒ - Q) * inv(Aₒ + Q)
        V = -2 * q * inv(Aₒ' + Q) * DDᵀ * inv(Aₒ + Q)

        # Analytical expression for iterating to a new X
        x_new = U' * x_old * U + V

        # If tolerance threshold reached, stop
        if LinearAlgebra.norm(x_new - x_old , 2) < tol
            # println("Lyapunov Equation Converged")
            return x_new

        end
        iter +=1
    end
    println("ERROR: Lyapunov Equation failed to converge.")
    return nothing
end

export solve_lyapunov


