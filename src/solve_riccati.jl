"""
Solving the different Riccati equations.
"""

using LinearAlgebra

include("smith_method.jl")

"""
Solve the Algebraic Riccati equation using Newton-Kleinman with Smith's method
to solve the Lyapunov equation

Parameters: 

A: State space matrix defining dynamics (ẋ = Ax)
B: State space matrix defining input (ẋ = Ax + Bu)
Q: Matrix defining scaling of state in cost function J (J = ∫ xᵀQx + uᵀRu dt)
R: Matrix defining scaling of input in cost funtion J (J = ∫ xᵀQx + uᵀRu dt)
riccati_tolerance: Convergence tolerance for the Newton-Kleinman algorithm
lyapunov_tolerance: Convergence tolerance for Smith's method
max_riccati_iters: Maximum allowable iterations of the Newton-Kleinman algorithm
max_lyapunov_iters: Maximum allowable iterations of Smith's method

Outputs:

P_new: Matrix minimizing the cost-to-go function J* (J* = xᵀSx)
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

    # Start Newton-Kleinman Algorithm
    while iters <= max_riccati_iters && norm(P_new - P_old, 2) >= riccati_tolerance
        P_old = copy(P_new)
        iters += 1

        K = -1 * inv(R) * B' * P_old
        S = A + B * K
    
        P_new = solve_lyapunov_custom(S, Q, K, R, lyapunov_tolerance, max_lyapunov_iters)
    end
    return P_new
end

"""
Anonymous function to solve the Lyapunov Equation (Aₒx + Aᵀₒx = -DDᵀ) in this
specific problem context (Continuous infinite horizon LQR)

Parameters:

S: (A+BK), Hurwitz matrix with feedback state gain as control input (ẋ = Ax + B(-Kx))
Q: Matrix defining scaling of state in cost function J (J = ∫ xᵀQx + uᵀRu dt)
K: State Feedback Gain Matrix as described above
R: Matrix defining scaling of input in cost funtion J (J = ∫ xᵀQx + uᵀRu dt)
lyapunov_tolerance: Convergence tolerance for Smith's method
max_lyapunov_iters: Maximum allowable iterations of Smith's method

Outputs:

Passes the output of solve_lyapunov, which is the matrix that minimizes the cost-to-go 
function, J* = xᵀSx, given the inputs
"""
function solve_lyapunov_custom(S :: Matrix{Float64}, Q :: Matrix{Float64}, K :: Matrix{Float64},
    R :: Matrix{Float64}, lyapunov_tolerance :: Float64, max_lyapunov_iters :: Int64 = 50)
    return solve_lyapunov(S, (Q + K' * R * K), lyapunov_tolerance, max_lyapunov_iters, -1.0)
end


"""
Implements Gershgorin's Theorem to calculate the needed value to ensure that the inputted
matrix is Hurwitz (Re(λᵢ)<0 ∀ i)

Parameters

A: Matrix to be pushed to be Hurwitz

Outputs:

gamma: parameter that makes A Hurwitz with A = A-gamma*I
"""
function gerghgorin_gamma(A :: Matrix{Float64})
    gamma = 0
    for t = 1 : size(A)[1]
        temp = norm(A[t, :], 1)
        gamma = max(gamma, temp)
    end
    return gamma + 1
end

export solve_algebraic_riccati
