using NLsolve
# using Symbolics
using LinearAlgebra

"""
"""

function solve_lqr()

end


"""
x[1] = x[1]
x[2] = x[2]
x[3]
x[4]
"""

function test()
    function f(x)
        [
            -10 * (x[1] * x[1] + x[2] * x[3]) + 2 * x[1] + 1,
            -10 * (x[1] * x[2] + x[2] * x[4]) + 2 * x[2] + x[1],
            -10 * (x[1] * x[3] + x[3] * x[4]) + x[1] + 2 * x[3],
            -10 * (x[4] * x[4] + x[2] * x[3]) + x[2] + x[3] + 2 * x[4] + 1
        ]
    end
    
    sol = nlsolve(f, [0.4255 0.635 0.635 0.4445])
    x = sol.zero
    println("found: ", x)
    println("equations come out to: ", f(x))
    println("found on matlab: ", f([0.4255 0.635 0.635 0.4445]))

end
