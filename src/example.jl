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
x_o = [0.0; 0.0]
x_T = [1.0; 0.0]

# objective function
objective_function = (x, u , t) -> t < T ? (x, u) -> 0.1 * dot(x, x) + 0.1 * dot(u, u) : (x, u) -> 0.1 * dot(x, x)

# constraints
equality_constraints = [ 
    (x, u, t) -> t == T ? x - x_T : nothing 
]
inequality_constraints = [
    (x) - >  -x[1] + 10 
]

# solve
# TODO: create solver and output

