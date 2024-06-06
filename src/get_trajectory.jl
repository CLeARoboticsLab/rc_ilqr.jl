using DifferentialEquations
using LinearAlgebra

"""
    Since the continuous solve methods only return the optimal feedback control gain (and optimal
    cost-to-go S), this function takes the optimal controls and rolls out the system to obtain
    the trajectory.

    Notes:
        System is: ẋ = A * x + B * u

    Parameters:
    A - Defines system dynamics as shown above
    B - Defines system dynamics as shown above
    K - Optimal feedback control gain matrix (or array of matrices for finite horizon)
    x₀ - Starting state, initial state
    step_size - size of each trajectory step
    t_span - time span for which we roll out trajectory

    Outputs:
    x - array of trajectories for the given time span, control, and system dynamics
"""
function get_continuous_LQR_trajectory(A :: Matrix{Float64},B :: Matrix{Float64}, K, xₒ :: Vector{Float64}, 
    step_size :: Float64 = 0.1, t_span :: Vector{Float64} = [-1.0,-1.0])

    function xdot(x,A,B,K)
        return (A - B * K) * x
    end

    # Infinite Horizion
    if t_span == [-1.0, -1.0]

        x = Array{Vector{Float64}}(undef, 1)

        x[1] = xₒ

        i = 0

        iter_lim = 50
        tol = 10e-5

        while i < iter_lim

            i += 1  

            push!(x, xdot(x[i],A,B,K) * step_size + x[i])

            if LinearAlgebra.norm(x[i+1] - x[i] , 2) < tol
                println("Trajectory Solver Converged")
                return x
            end

        end

        println("Trajectory Didn't Converge")

    # Finite Horizion
    else
        x = Array{Vector{Float64}}(undef, 1)

        x[1] = xₒ

        for (i,t) in enumerate(t_span[1]: step_size: t_span[2])

            push!(x, round.(xdot(x[i],A,B,K[i]) * step_size + x[i], digits = 10))

        end

        return x
    end
end

function get_discrete_LQR_trajectory(A :: Matrix{Float64},B :: Matrix{Float64}, K, xₒ :: Vector{Float64}, 
     N :: Int64 = -1)

    function next_x(x,K)
        return (A-B*K) * x
    end

    if N == -1
        x = Array{Vector{Float64}}(undef, 1)

        x[1] = xₒ

        i = 0

        iter_lim = 50
        tol = 10e-5

        while i < iter_lim

            i += 1  

            push!(x, next_x(x[i],K))

            if LinearAlgebra.norm(x[i+1] - x[i] , 2) < tol
                println("Trajectory Solver Converged")
                return x
            end

        end

        println("Trajectory Didn't Converge")


    else
        x = Array{Vector{Float64}}(undef, 1)

        x[1] = xₒ

        for i = 1:N

            push!(x, round.(next_x(x[i], K[i]), digits = 10))

        end

        return x

    end

end

export get_continuous_LQR_trajectory, get_discrete_LQR_trajectory