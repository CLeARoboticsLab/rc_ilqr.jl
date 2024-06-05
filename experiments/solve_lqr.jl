using NLsolve
# using Symbolics
using LinearAlgebra

"""
"""

function solve_continuous_lqr()

end

"""
"""

function solve_discrete_lqr()

end


"""
x[1] = x[1]
x[2] = x[2]
x[3]
x[4]
"""

function test()
    function f!(S)
        # [
        #     -10 * (x[1] * x[1] + x[2] * x[3]) + 2 * x[1] + 1,
        #     -10 * (x[1] * x[2] + x[2] * x[4]) + 2 * x[2] + x[1],
        #     -10 * (x[1] * x[3] + x[3] * x[4]) + x[1] + 2 * x[3],
        #     -10 * (x[4] * x[4] + x[2] * x[3]) + x[2] + x[3] + 2 * x[4] + 1
        # ]

        A = [1 1; 0 1]
        B = I(2)
        Q = I(2)
        R = 0.1 * I(2)

        return Q + A'*S + S*A - S*B*inv(R)*B'*S

    end
    
    sol = nlsolve(f!, [0.0 0.0; 0.0 0.0])
    S = sol.zero
    println("found: ", S)
    println("equations come out to: ", f!(S))
    println("found on matlab: ", f!([0.4255 0.635; 0.635 0.4445]))

end
