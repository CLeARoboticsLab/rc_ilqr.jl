# using NLsolve
using Symbolics
using LinearAlgebra

include("solve_lqr.jl")

""" 
- Cost function is sum(x' * Q * x + u' * R * u) for all time + x_f' * Q_f * x_f
Notes:
- Closed form update equations are derived from Hamiltonian
"""
function solve_ilqr(dynamics_function :: Function, cost_function :: Function, Q :: Matrix{Float64},
    Q_T :: Matrix{Float64}, R :: Matrix{Float64}, x_T :: Vector, x_0 :: Vector{Float64},
    T :: Int64,threshold :: Float64, iter_limit :: Int64 = 1000)

    step_size = 0.01

    state_space_degree = size(Q)[1]
    action_space_degree = size(R)[1]
    
    x = Symbolics.variables(:x, 1 : state_space_degree)
    u = Symbolics.variables(:u, 1 : action_space_degree)

    f = dynamics_function(x, u)
    # f = Symbolics.build_function(dynamics_function, x, u)
    dynamics = eval(Symbolics.build_function(f, x, u)[1])

    ∇ₓf = Symbolics.jacobian(f, x)
    ∇ᵤf = Symbolics.jacobian(f, u)

    # First forward pass
    nominal_state_sequence = Array{Vector{Float64}}(undef,1,T)
    nominal_control_sequence = Array{Vector{Float64}}(undef,1,T)

    for i in 1:T
        nominal_control_sequence[i] = 0.1*randn(action_space_degree)
    end

    nominal_state_sequence[1] = x_0

    for k = 2 : T
        nominal_state_sequence[k] = dynamics_function(
            nominal_state_sequence[k - 1],
            nominal_control_sequence[k - 1])
    end

    new_nominal_state_sequence = copy(nominal_state_sequence)
    delta_u = Array{Vector{Float64}}(undef,1,T)
    delta_u[T] = zeros(action_space_degree)

    old_cost = cost_function(Q, R, Q_T, nominal_state_sequence, nominal_control_sequence, x_T, T)

    S_all = Array{Matrix{Float64}}(undef,1,T)
    S_all[T] = Q_T

    v_all = Array{Vector{Float64}}(undef,1,T)
    v_all[T] = Q_T * (nominal_state_sequence[T] - x_T)

    iter = 0
    

    while iter <= iter_limit
    # backwards pass
        for k = T - 1 : -1 : 1
            delta_x = new_nominal_state_sequence - nominal_state_sequence
            A_call = eval(build_function(∇ₓf, nominal_state_sequence[k], 
                nominal_control_sequence[k], expression=Val{false})[1])
                
            B_call = eval(build_function(∇ᵤf, nominal_state_sequence[k], 
                nominal_control_sequence[k], expression=Val{false})[1])

            A = A_call(nominal_state_sequence[k], nominal_control_sequence[k])
            B = B_call(nominal_state_sequence[k], nominal_control_sequence[k])

            K = inv(B' * S_all[k + 1] * B + R) * B' * S_all[k + 1] * A
            Kᵥ = inv(B' * S_all[k + 1] * B + R) * B'
            Kᵤ = inv(B' * S_all[k + 1] * B + R) * R
            S_all[k] = A' * S_all[k + 1] * (A - B * K) + Q
            v_all[k] = (A - B*K)' * v_all[k + 1] - K' * R * nominal_control_sequence[k] 
                + Q * nominal_state_sequence[k]

            delta_u[k] = -K * delta_x[k] - Kᵥ * v_all[k + 1] - Kᵤ * nominal_control_sequence[k]
        end

        # forward pass
        for k = 2 : T
            new_nominal_state_sequence[k] = dynamics_function(
                new_nominal_state_sequence[k - 1],
                nominal_control_sequence[k - 1] + delta_u[k - 1])
        end

        new_cost = cost_function(Q, R, Q_T, new_nominal_state_sequence, nominal_control_sequence + step_size * delta_u, x_T, T)

        delta_cost = old_cost - new_cost
        println("old cost: ", old_cost, " new cost: ", new_cost)
        if delta_cost < 0
            println("ilqr step increased cost")
            println("num iters: ", iter)
            # return [nominal_state_sequence, nominal_control_sequence]
        elseif delta_cost <= threshold
            println("converged, delta cost: ", delta_cost)
            println("num iters: ", iter)
            return [new_nominal_state_sequence, nominal_control_sequence + step_size * delta_u]
        else
            # println("")
            nominal_state_sequence = new_nominal_state_sequence
            nominal_control_sequence += step_size * delta_u
            old_cost = new_cost
        end
        iter += 1
    end
    println("4, didn't converge")
    println("num iters: ", iter)
    println("x: ", nominal_state_sequence)
    println("u: ", nominal_control_sequence)
end

"""
AHHHHHHHHHHHHHHHHHHH
"""


function solve_ilqr_v2(dynamics_function, cost_function, Q, Q_T, R, x_T, xₒ,
    T :: Int64,threshold :: Float64, iter_limit :: Int64 = 1000)

    state_space_degree = size(Q)[1]
    action_space_degree = size(R)[1]

    nominal_state_sequence = Array{Vector{Float64}}(undef, 1,T)
    nominal_control_sequence = Array{Vector{Float64}}(undef, 1,T)

    nominal_state_sequence[1] = xₒ
    nominal_control_sequence[1] = zeros(action_space_degree)
    

    for i in 2:T
        nominal_control_sequence[i] = 0.1*randn(action_space_degree)
        nominal_state_sequence[i] = dynamics_function(nominal_state_sequence[i-1],nominal_control_sequence[i-1])
    end

    old_cost = cost_function(Q, R, Q_T, nominal_state_sequence, nominal_control_sequence, x_T, T)

    S_all = Array{Matrix{Float64}}(undef,1,T)
    S_all[T] = Q_T

    v_all = Array{Vector{Float64}}(undef,1,T)
    v_all[T] = Q_T * (nominal_state_sequence[T] - x_T)


    x = Symbolics.variables(:x, 1 : state_space_degree)
    u = Symbolics.variables(:u, 1 : action_space_degree)

    f = dynamics_function(x, u)
    # f = Symbolics.build_function(dynamics_function, x, u)
    # dynamics = eval(Symbolics.build_function(f, x, u)[1])

    ∇ₓf = Symbolics.jacobian(f, x)
    ∇ᵤf = Symbolics.jacobian(f, u)

    A_call = eval(build_function(∇ₓf,x,u;expression = Val{false})[1])
    B_call = eval(build_function(∇ᵤf,x,u;expression = Val{false})[1])

    delta_u = Array{Vector{Float64}}(undef,1,T)
    delta_x = Array{Vector{Float64}}(undef,1,T)
    delta_x[1] = zeros(state_space_degree)
    delta_u[T] = zeros(action_space_degree)
    
    for i in 1:T-1
        delta_u[i] = zeros(action_space_degree)
        delta_x[i+1] = zeros(state_space_degree)
    end

    iter = 0
    
    while iter <= iter_limit
    # backwards pass
        for k = T - 1 : -1 : 1
            A = A_call(nominal_state_sequence[k],nominal_control_sequence[k])
            B = B_call(nominal_state_sequence[k],nominal_control_sequence[k])

            K = inv(B' * S_all[k + 1] * B + R) * B' * S_all[k + 1] * A
            Kᵥ = inv(B' * S_all[k + 1] * B + R) * B'
            Kᵤ = inv(B' * S_all[k + 1] * B + R) * R
            S_all[k] = A' * S_all[k + 1] * (A - B * K) + Q
            v_all[k] = (A - B*K)' * v_all[k + 1] - K' * R * nominal_control_sequence[k] 
                + Q * nominal_state_sequence[k]

            delta_u[k] = -K * delta_x[k] - Kᵥ * v_all[k + 1] - Kᵤ * nominal_control_sequence[k]
        end

        # forward pass
        nominal_control_sequence = nominal_control_sequence + delta_u

        for k = 2 : T
            nominal_state_sequence[k] = dynamics_function(
                nominal_state_sequence[k - 1],
                nominal_control_sequence[k - 1])
            
            delta_x[k] = A_call(nominal_state_sequence[k-1],nominal_control_sequence[k-1]) * delta_x[k-1] + B_call(nominal_state_sequence[k-1],nominal_control_sequence[k-1]) * delta_u[k-1]
        end

        new_cost = cost_function(Q, R, Q_T, nominal_state_sequence, nominal_control_sequence, x_T, T)

        delta_cost = old_cost - new_cost
        println("old cost: ", old_cost, " new cost: ", new_cost)
        if delta_cost < 0
            println("ilqr step increased cost")
            println("num iters: ", iter)
            return [nominal_state_sequence, nominal_control_sequence]
        elseif delta_cost <= threshold
            println("converged, delta cost: ", delta_cost)
            println("num iters: ", iter)
            return [nominal_state_sequence, nominal_control_sequence]
        else
            # println("")
            old_cost = new_cost
        end
        iter += 1
    end
    println("4, didn't converge")
    println("num iters: ", iter)
    println("x: ", nominal_state_sequence)
    println("u: ", nominal_control_sequence)
end

function solve_ilqr_v3(dynamics_function :: Function, cost_function :: Function, Q :: Matrix{Float64},
    Q_T :: Matrix{Float64}, R :: Matrix{Float64}, x_T :: Vector, x_0 :: Vector{Float64},
    T :: Int64, threshold :: Float64, iter_limit :: Int64 = 100)

    # The reason for this version is to try and use our lqr solvers. Kushagra told me:
    # δuₖ = - K * δxₖ 
    # New trajectory = f( xₖ, uₖ + δuₖ )

    state_space_degree = size(Q)[1]
    action_space_degree = size(R)[1]

    nominal_state_sequence = Array{Vector{Float64}}(undef,1,T)
    nominal_control_sequence = Array{Vector{Float64}}(undef,1,T)
    δu = Array{Vector{Float64}}(undef,1,T)

    nominal_state_sequence[1] = x_0
    nominal_control_sequence[1] = zeros(action_space_degree)
    δu[1] = zeros(action_space_degree)

    # Initialize δu, u (nominal control sequence), trajectory (nominal state sequence)
    for k = 2:T
        nominal_control_sequence[k] = zeros(action_space_degree)
        δu[k] = .1 * ones(action_space_degree)
        nominal_state_sequence[k] = dynamics_function(
            nominal_state_sequence[k - 1],
            nominal_control_sequence[k - 1])
    end

    
    x = Symbolics.variables(:x, 1 : state_space_degree)
    u = Symbolics.variables(:u, 1 : action_space_degree)

    f = dynamics_function(x, u)
    # f = Symbolics.build_function(dynamics_function, x, u)
    dynamics = eval(Symbolics.build_function(f, x, u)[1])

    ∇ₓf = Symbolics.jacobian(f, x)
    ∇ᵤf = Symbolics.jacobian(f, u)

    function find_A_B(state, action)
        A_call = eval(build_function(∇ₓf, state, action, expression=Val{false})[1])
        B_call = eval(build_function(∇ᵤf, state, action, expression=Val{false})[1])

        A = 1.0 * A_call(state, action)
        B = 1.0 * B_call(state, action)

        return (A, B)
    end
        
    function solve_δx(δu)
        x = Array{Vector{Float64}}(undef, 1, T)
        x[1] = zeros(state_space_degree)
        for k = 2 : T
            (A, B) = find_A_B(nominal_state_sequence[k], nominal_control_sequence[k])
            x[k] = A * x[k - 1] + B * δu[k]
        end
        return x
    end
    

    δx = solve_δx(δu)

    iter = 0
    old_cost = cost_function(Q, R, Q_T, nominal_state_sequence, nominal_control_sequence, x_T, T)

    while iter <= iter_limit && norm(δu, 2) > threshold

        # Backward Pass
        for k = T - 1 : -1 : 1
            # TODO: Unsure
            (A, B) = find_A_B(nominal_state_sequence[k], nominal_control_sequence[k])
            K, S = solve_discrete_finite_lqr(A, B, Q, R, T - k)
            δu[k] = - K[1] * δx[k]
        end

        for k = 2 : 1 : T
            # TODO: Ends are weird
            nominal_state_sequence[k] = dynamics_function(
                nominal_state_sequence[k - 1],
                nominal_control_sequence[k - 1] + δu[k - 1])

            (A, B) = find_A_B(nominal_state_sequence[k],
                nominal_control_sequence[k - 1] + δu[k - 1])

            δx[k] = A * δx[k - 1] + B * δu[k - 1]
        end

        new_cost = cost_function(Q, R, Q_T, nominal_state_sequence,
            nominal_control_sequence + δu, x_T, T)
        iter += 1

        if old_cost - new_cost < 0
            println("oops")
            return
        else
            nominal_control_sequence += δu
        end
    end

    if norm(δu, 2) <= threshold
        println("Converged")
        return (nominal_state_sequence, nominal_control_sequence)
    elseif iter >= iter_limit
        println("Reached iter limit")
        return (nominal_state_sequence, nominal_control_sequence)
    else
        println("I don't know what happened please call 911")
        return nothing
    end

end


export solve_ilqr, solve_ilqr_v2, solve_ilqr_v3