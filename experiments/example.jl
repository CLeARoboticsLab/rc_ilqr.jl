using rc_ilqr

# system dynamics
function step_forward(x :: Vector{Float}, u :: Vector{Float})
    A = [1.0, 1.0; 0.0, 1.0]
    B = [0.0; 1.0]
    return A * x + B * u
end

# total number of time steps
T = 11

# initial state
x_0 = [0.0; 0.0]
x_T = [1.0; 0.0]

# obj function stuff
Q = [0.1, 0.0; 0.0, 0.1]
Q_T = [0.1, 0.0; 0.0, 0.1]
R = [0.1, 0.0; 0.0, 0.1]

# objective function
function objective_function(x̄, ū, T = T)
    f = 0
    for t = 1 : T - 1
        f += x̄[t]' * Q * x̄[t] + ū[t]' * R * ū[t] 
    end
    f += (x̄[T] - x_T)' * Q_T * (x̄[T] - x_T)

    return f
end

# constraints
equality_constraints = [ 
    (x, u, t) -> t == T ? x - x_T : nothing 
]
inequality_constraints = [
    (x) ->  -x[1] + 10 
]

# solve
# TODO: create solver and output

[x, u] = solve(step_forward, Q,
    Q_T, R, x_T, x_0,
    T, 2, 2, 10e-2,
    iter_limit = 10)

println(x)
println("---------")
println(u)

