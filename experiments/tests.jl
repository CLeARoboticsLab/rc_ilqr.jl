using LinearAlgebra, GLMakie

include("C:/Users/cader/Documents/Research Summer 2024/rc_ilqr.jl/src/solve_lqr.jl")

"""
"""
function test_solve_continuous_inf_lqr()

    A = [1.0 1.0; 0 1.0]
    B = [1.0 0.0;0.0 1.0]
    Q = [1.0 0.0;0.0 1.0]
    R = 0.1 * [1.0 0.0;0.0 1.0]

    K,S = solve_continuous_inf_lqr(A,B,Q,R)

    println("K: ", K)
    println("S: ", S)

    x = get_continuous_LQR_trajectory(A,B,K,[1.0;1.0])

    println("x: ", x)

    xs = [p[1] for p in x]
    ys = [p[2] for p in x]


    fig = Figure()

    ax = Axis(fig[1,1], xlabel = "x₁", ylabel = "x₂")
    lines!(ax, xs,ys)

    fig

end
    
    
function test_solve_continuous_finite_lqr()

    A = [1.0 1.0; 0 1.0]
    B = [1.0 0.0;0.0 1.0]
    Q = [1.0 0.0;0.0 1.0]
    Q_f = copy(Q)
    R = 0.1 * [1.0 0.0;0.0 1.0]

    K,S = solve_continuous_finite_lqr(A,B,Q,Q_f,R, 0, 11, 0.1)

    println("K: ", K)
    println("S: ", S)

    x = get_continuous_LQR_trajectory(A,B,K,[1.0,1.0],0.1,[0.0, 11.0])

    

    println("x: ", x)

    xs = [p[1] for p in x]
    ys = [p[2] for p in x]


    fig = Figure()

    ax = Axis(fig[1,1], xlabel = "x₁", ylabel = "x₂")
    lines!(ax, xs,ys)

    fig

end
