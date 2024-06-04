using rc_ilqr


function particle_solve()
    # total number of time steps
    T = 11

    # initial state
    x_0 = [0.0; 0.0]
    x_T = [1.0; 0.0]

    # obj function stuff
    Q = [0.01 0.0; 0.0 0.01]
    Q_T = [.1 0.0; 0.0 .1]
    R = [0.1 0.0; 0.0 0.1]

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
        B = [0.0 1.0; 1.0 0.0]

        return (A * x + B * u)
    end

    x, u = solve(step_forward, Q,
        Q_T, R, x_T, x_0,
        T, 2, 2, 10e-3)

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



