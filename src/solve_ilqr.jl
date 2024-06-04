

""" 
- Cost function is sum(x' * Q * x + u' * R * u) for all time + x_f' * Q_f * x_f
Notes:
- Closed form update equations are derived from Hamiltonian
"""
function solve(dynamics_function :: Function, Q :: Vector{Vector},
    Q_T :: Vector{Vector}, R :: Vector{Vector}, x_T :: Vector, x_0 :: Float,
    T :: Float, state_space_degree :: Int, action_space_degree :: Int, threshold :: Float,
    iter_limit :: Int = 10)
    
    x = Symbolics.variables(:x, 1 : state_space_degree)
    u = Symbolics.variables(:u, 1 : action_space_degree)

    dynamics = eval(Symbolics.build_function(dynamics_function(x, u), x, u)[1]);

    ∇ₓf = Symbolics.gradient(dynamics, x)
    ∇ᵤf = Symbolics.gradient(dynamics, u)

    # First forward pass
    nominal_state_sequence = [x_0]
    nominal_control_sequence = zeros((action_space_degree, T - 1))
    for k = 2 : T
        push!(nominal_state_sequence, dynamics_function(
            nominal_state_sequence[k - 1, :, :],
            nominal_control_sequence[k - 1, :, :]))
    end

    new_nominal_state_sequence = copy(nominal_state_sequence)
    delta_u = zeros((action_space_degree, T - 1))

    old_cost = cost(Q, R, Q_T, nominal_state_sequence, full_control_sequence_old, x_T, T)

    S_all = zeros((T, size(Q)...))
    S_all[T, :, :] = Q_T

    v_all = zeros(T, size(Q)[1], 1)
    v_all[T, :, :] = Q_T * (nominal_state_sequence[T, :, :] - x_T)

    iter = 0
    while iter <= iter_limit
    # backwards pass
    for k = T - 1 : -1 : 1
        delta_x = full_trajectory_new - nominal_state_sequence
        A = eval(build_function(∇ₓf, nominal_state_sequence[k, :, :], 
            nominal_control_sequence[k, :, :], expression=Val{false}))
        B = eval(build_function(∇ᵤf, nominal_state_sequence[k, :, :], 
            nominal_control_sequence[k, :, :], expression=Val{false}))

        K = inverse(B' * S_all[k + 1, :, :] * B + R) * B' * S_all[k + 1, :, :] * A
        Kᵥ = inverse(B' * S_all[k + 1, :, :] * B + R) * B'
        Kᵤ = inverse(B' * S_all[k + 1, :, :] * B + R) * R
        S_all[k, :, :] = A' * S_all[k + 1, :, :] * (A - B * K) + Q
        v_all[k, :, :] = (A - B*K)' * v_all[k + 1, :, :] - K' * R * nominal_control_sequence[k, :, :] 
            + Q * nominal_state_sequence[k, :, :]

        delta_u[k, :, :] = -K * delta_x[k, :, :] - Kᵥ * v_all[k + 1, :, :] - Kᵤ * nominal_control_sequence[k, :, :]
    end

    # forward pass
    for k = 2 : T
        new_nominal_state_sequence[k, :, :] = dynamics_function(
            new_nominal_state_sequence[k - 1, :, :],
            nominal_control_sequence[k - 1, :, :] + delta_u[k - 1, :, :])
    end
    new_cost = cost(Q, R, Q_T, new_nominal_state_sequence, nominal_control_sequence + delta_u, x_T, T)

    delta_cost = old_cost - new_cost
    if delta_cost < 0
        return [nominal_state_sequence, nominal_control_sequence]
    else if delta_cost <= threshold 
        return [new_nominal_state_sequence, nominal_control_sequence + delta_u]
    else
        nominal_state_sequence = new_nominal_state_sequence
        nominal_control_sequence += delta_u
        old_cost = new_cost
    end
    iter += 1
    end
end

function cost(Q, R, Q_T, x̄, ū, x_T, T)
    f = 0
    for t = 1 : T - 1 # Column at t?
        f += x̄[:, t]' * Q * x̄[:, t] + ū[:, t]' * R * ū[:, t] 
    end
    f += (x̄[:, T] - x_T)' * Q_T * (x̄[:, T] - x_T)

    return f
end

