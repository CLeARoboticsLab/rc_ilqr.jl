using DifferentialEquations
using LinearAlgebra

function get_continuous_LQR_trajectory(A :: Matrix{Float64},B :: Matrix{Float64}, K, xₒ :: Vector{Float64}, 
    step_size :: Float64 = 0.1, t_span :: Vector{Float64} = [-1.0,-1.0])

    function xdot(x,A,B,K)
        return (A - B * K) * x
    end

    # Infinite Horizion
    if t_span == [-1.0,-1.0]

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

        for i = t_span[1]: step_size: t_span[2]

            push!(x, xdot(x[i],A,B,K[i]) * step_size + x[i])
        end

        return x
    end
end

export get_continuous_LQR_trajectory