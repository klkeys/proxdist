source("lc.jl")

p = 50
A = randn(p , p)
A = A'*A
b = randn(p)
(x, y) = linear_complementarity(A, b)
