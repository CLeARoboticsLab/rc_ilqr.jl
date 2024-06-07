using rc_ilqr


function particle_solve()
    # total number of time steps
    T = 11

    # initial state
    x_0 = [30.0,   10.0]
    x_T = [1.0; 0.0]

    # obj function stuff
    Q = [1 0.0; 0.0 1]
    Q_T = copy(Q)
    R = hcat(0.1)

    # constraints
    equality_constraints = [ 
        (x, u, t) -> t == T ? x - x_T : nothing 
    ]
    inequality_constraints = [
        (x) ->  -x[1] + 10 
    ]

    # system dynamics
    function step_forward(x, u)

        A = [1.0 1.0; 0.0 1.0]
        B = hcat([0.0; 1.0])


        return (A * x + B * u)
    end

    x, u = solve_ilqr_v2(step_forward,cost, Q,
        Q_T, R, x_T, x_0,
        T, 10e-3)

    println(x)
    println("---------")
    println(u)
end

# objective function
function objective_function_particle(x̄, ū, T = T)
    f = 0
    for t = 1 : T - 1
        f += x̄[t]' * Q * x̄[t] + ū[t]' * R * ū[t] 
    end
    f += (x̄[T] - x_T)' * Q_T * (x̄[T] - x_T)

    return f
end


function cost(Q, R, Q_T, x̄, ū, x_T, T)
    f = 0
    state_cost = 0
    control_cost = 0
    for t = 1 : T - 1 # Column at t?
        # sc = (x̄[t, :] - x_T)' * Q * (x̄[t, :] - x_T)
        sc = x̄[t]' * Q * x̄[t]
        cc = ū[t]' * R * ū[t]

        state_cost += sc
        control_cost += cc

        f += sc + cc
    end
    f = 0.5 * f
    f += 0.5 * (x̄[T] - x_T)' * Q_T * (x̄[T] - x_T)
    state_cost += (x̄[T] - x_T)' * Q_T * (x̄[T] - x_T)

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





