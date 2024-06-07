# using NLsolve
using Symbolics
using LinearAlgebra

""" 
- Cost function is sum(x' * Q * x + u' * R * u) for all time + x_f' * Q_f * x_f
Notes:
- Closed form update equations are derived from Hamiltonian
"""
function solve_ilqr(dynamics_function :: Function, Q :: Matrix{Float64},
    Q_T :: Matrix{Float64}, R :: Matrix{Float64}, x_T :: Vector, x_0 :: Vector{Float64},
    T :: Int64, state_space_degree :: Int64, action_space_degree :: Int64, threshold :: Float64,
    iter_limit :: Int64 = 1000)
    
    x = Symbolics.variables(:x, 1 : state_space_degree)
    u = Symbolics.variables(:u, 1 : action_space_degree)

    f = dynamics_function(x, u)
    # f = Symbolics.build_function(dynamics_function, x, u)
    dynamics = eval(Symbolics.build_function(f, x, u)[1])

    ∇ₓf = Symbolics.jacobian(f, x)
    ∇ᵤf = Symbolics.jacobian(f, u)

    # First forward pass
    nominal_state_sequence = zeros((T, action_space_degree))
    nominal_control_sequence = zeros((T - 1, action_space_degree))

    nominal_state_sequence[1,:] = x_0

    for k = 2 : T
        nominal_state_sequence[k,:] = dynamics_function(
            nominal_state_sequence[k - 1, :],
            nominal_control_sequence[k - 1, :])
    end

    new_nominal_state_sequence = copy(nominal_state_sequence)
    delta_u = zeros((T - 1, action_space_degree))

    old_cost = cost(Q, R, Q_T, nominal_state_sequence, nominal_control_sequence, x_T, T)

    S_all = zeros((T, size(Q)...))
    S_all[T, :, :] = Q_T

    v_all = zeros(T, size(Q)[1], 1)
    v_all[T, :, :] = Q_T * (nominal_state_sequence[T, :, :] - x_T)

    iter = 0
    

    while iter <= iter_limit
    # backwards pass
        for k = T - 1 : -1 : 1
            delta_x = new_nominal_state_sequence - nominal_state_sequence
            A_call = eval(build_function(∇ₓf, nominal_state_sequence[k, :, :], 
                nominal_control_sequence[k, :, :], expression=Val{false})[1])
                
            B_call = eval(build_function(∇ᵤf, nominal_state_sequence[k, :, :], 
                nominal_control_sequence[k, :, :], expression=Val{false})[1])

            A = A_call(nominal_state_sequence[k, :, :], nominal_control_sequence[k, :, :])
            B = B_call(nominal_state_sequence[k, :, :], nominal_control_sequence[k, :, :])

            K = inv(B' * S_all[k + 1, :, :] * B + R) * B' * S_all[k + 1, :, :] * A
            Kᵥ = inv(B' * S_all[k + 1, :, :] * B + R) * B'
            Kᵤ = inv(B' * S_all[k + 1, :, :] * B + R) * R
            S_all[k, :, :] = A' * S_all[k + 1, :, :] * (A - B * K) + Q
            v_all[k, :, :] = (A - B*K)' * v_all[k + 1, :, :] - K' * R * nominal_control_sequence[k, :, :] 
                + Q * nominal_state_sequence[k, :, :]

            delta_u[k, :, :] = -K * delta_x[k, :] - Kᵥ * v_all[k + 1, :, :] - Kᵤ * nominal_control_sequence[k, :, :]
        end

        # forward pass
        for k = 2 : T
            new_nominal_state_sequence[k, :, :] = dynamics_function(
                new_nominal_state_sequence[k - 1, :, :],
                nominal_control_sequence[k - 1, :, :] + delta_u[k - 1, :, :])
        end
        new_cost = cost(Q, R, Q_T, new_nominal_state_sequence, nominal_control_sequence + delta_u, x_T, T)

        delta_cost = old_cost - new_cost
        println("old cost: ", old_cost, " new cost: ", new_cost)
        if delta_cost < 0
            println("ilqr step increased cost")
            println("num iters: ", iter)
            return [nominal_state_sequence, nominal_control_sequence]
        elseif delta_cost <= threshold
            println("converged, delta cost: ", delta_cost)
            println("num iters: ", iter)
            return [new_nominal_state_sequence, nominal_control_sequence + delta_u]
        else
            # println("")
            nominal_state_sequence = new_nominal_state_sequence
            nominal_control_sequence += delta_u
            old_cost = new_cost
        end
        iter += 1
    end
    println("4, didn't converge")
    println("num iters: ", iter)
end

function cost(Q, R, Q_T, x̄, ū, x_T, T)
    f = 0
    state_cost = 0
    control_cost = 0
    for t = 1 : T - 1 # Column at t?
        # sc = (x̄[t, :] - x_T)' * Q * (x̄[t, :] - x_T)
        sc = x̄[t, :]' * Q * x̄[t, :]
        cc = ū[t, :]' * R * ū[t, :]

        state_cost += sc
        control_cost += cc

        f += sc + cc
    end
    f += (x̄[T, :] - x_T)' * Q_T * (x̄[T, :] - x_T)
    state_cost += (x̄[T, :] - x_T)' * Q_T * (x̄[T, :] - x_T)

    println("state assoc cost: ", state_cost)
    println("control assoc cost: ", control_cost)
    return f
    # f = 0
    # for t = 1: T - 1
    #     f +=  0.1 * dot((x̄[t, :] - x_T), (x̄[t, :] - x_T)) + 0.01 * dot(ū[t, :], ū[t, :])
    # end
    # f += 0.7 * dot((x̄[T, :] - x_T), (x̄[T, :] - x_T))
    # return f
end

export solve


