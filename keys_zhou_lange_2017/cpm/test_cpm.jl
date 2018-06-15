include("cpm.jl")
n = 1000
M = randn(n,n)
M = 0.5*(M+M')

# Hall and Newman example
# n = 5
# M = ones(n,n)
# for j = 2:n
#    M[j,j-1] = -1.0
#    M[j-1,j] = -1.0
# end
# M[1,5] = -1.0
# M[5,1] = -1.0
# for i = 1:n
#    println(M[i,:])
# end
#
#tic(); (iters,x) = copositivity(M); toc()
#loss = 0.5*dot(x,M*x)
#println("iters = ",iters," loss = ",loss," norm = ",norm(x)," min = ",minimum(x))
#tic(); (iters,y) = accelerated_copositivity(M); toc()
#loss = 0.5*dot(y,M*y)
#println("iters = ",iters," loss = ",loss," norm = ",norm(y)," min = ",minimum(y))
## tic(); (iters,z) = accelerated_copositivity2(M); toc()
## loss = 0.5*dot(z,M*z)
## println("iters = ",iters," loss = ",loss," norm = ",norm(z)," min = ",minimum(z))
## println(" norm of difference = ",norm(x-y))

using Convex

# Use Mosek solver
using Mosek
#solver = MosekSolver(LOG = 0, MSK_DPAR_OPTIMIZER_MAX_TIME = 1.0)
solver = MosekSolver(LOG = 0)
set_default_solver(solver)

## Use SCS solver
#using SCS
#solver = SCSSolver(verbose=0)
#set_default_solver(solver)


"""

    CoPositivityBySDP(::AbstractArray{Float64}) -> {Bool}

Check whether M is a copositive matrix by semidefinite programming. M is
copositive if M = A + B, where A is psd and B has nonnegative entries. Note this
is only a sufficient condition. The function returns `true` if such a
decomposition is found and `false` if otherwise.
"""
function CoPositivityBySDP(M::Array{Float64})
  n = size(M, 1)
  A = Convex.Variable(n, n)
  B = Convex.Variable(n, n)
  constraints = (M == A + B)
  constraints += (A in :SDP)
  constraints += B >= 0.0
  problem = satisfy(constraints)
  try solve!(problem)
  catch e
    return "timeout"
  end
  if problem.status == :Optimal
    return "feasible"
  else
    return "infeasible"
  end
end

"""

    horn_matrix(::Int) -> Array{Float64}

Generate an `n`-by-`n` Horn matrix.
"""
function horn_matrix(n::Int)
  M = ones(n, n)
  for j = 2:n
    M[j, j-1] = -1.0
    M[j-1, j] = -1.0
  end
  M[1, n] = -1.0
  M[n, 1] = -1.0
  return M
end

# Horn matrix example
srand(123)
n = 2 .^ (2:6)
n = sort([n; n + 1])
timing = zeros(length(n), 3)
loss = zeros(length(n), 2)
losssdp = Array(AbstractString, length(n))
for i = 1:length(n)
  M = horn_matrix(n[i])
  # proximal distance
  tic(); (iters, y) = copositivity(M); timing[i, 1] = toc();
  loss[i, 1] = 0.5 * dot(y, M * y)
  # accelerated proximal distance
  tic(); (iters, y) = accelerated_copositivity(M); timing[i, 2] = toc();
  loss[i, 2] = 0.5 * dot(y, M * y)
  # semidefinite programming
  tic(); losssdp[i] = CoPositivityBySDP(M); timing[i, 3] = toc();
end

println("\\begin{table}")
println("\t\\centering")
println("\t\\begin{tabular}{rrrrrrr}")
println("\t\t\\multicolumn{1}{c}{Dimensions} & \\multicolumn{3}{c}{Optima} & \\multicolumn{3}{c}{CPU Times} \\\\ ")
println("\t\t\\\cmidrule(r){2-4} \\cmidrule(r){5-7}")
println("\t\t\$n\$ & PD & aPD & Mosek & PD & aPD & Mosek\\\\")
println("\\hline")
for k = 1:length(n)
	@printf("\t\t%d & %3.6f & %3.6f & %s & %3.4f & %3.4f & %3.4f\\\\\n", n[k],
  loss[k, 1], loss[k, 2], losssdp[k], timing[k, 1], timing[k, 2], timing[k, 3])
end
println("\t\t\\bottomrule")
println("\t\\end{tabular}")
println("\t\\caption{CPU times and optima for testing copositivity of Horn matrix. Here \$n\$ is the size of Horn matrix, PD is the proximal distance algorithm, aPD is the accelerated proximal distance algorithm, and Mosek is the Mosek solver.}")
println("\t\\label{tab:copos-horn}")
println("\\end{table}")


# Symmetric matrix example
srand(1234)
n = 2 .^ (2:8)
timing = zeros(length(n), 3)
loss = zeros(length(n), 2)
losssdp = Array(AbstractString, length(n))
for i = 1:length(n)
  M = randn(n[i], n[i])
  M = 0.5 * (M + M')
  # proximal distance
  tic(); (iters, y) = copositivity(M); timing[i, 1] = toc();
  loss[i, 1] = 0.5 * dot(y, M * y)
  # accelerated proximal distance
  tic(); (iters, y) = accelerated_copositivity(M); timing[i, 2] = toc();
  loss[i, 2] = 0.5 * dot(y, M * y)
  # semidefinite programming
  tic(); losssdp[i] = CoPositivityBySDP(M); timing[i, 3] = toc();
end

println("\\begin{table}")
println("\t\\centering")
println("\t\\begin{tabular}{rrrrrrr}")
println("\t\t\\multicolumn{1}{c}{Dimensions} & \\multicolumn{3}{c}{Optima} & \\multicolumn{3}{c}{CPU Times} \\\\ ")
println("\t\t\ \\cmidrule(r){2-4} \\cmidrule(r){5-7}")
println("\t\t\$n\$ & PD & aPD & Mosek & PD & aPD & Mosek\\\\")
println("\\hline")
for k = 1:length(n)
	@printf("\t\t%d & %3.6f & %3.6f & %s & %3.4f & %3.4f & %3.4f\\\\\n", n[k],
  loss[k, 1], loss[k, 2], losssdp[k],
  timing[k, 1], timing[k, 2], timing[k, 3])
end
println("\t\t\\bottomrule")
println("\t\\end{tabular}")
println("\t\\caption{CPU times and optima for testing copositivity of random symmetric matrix. Here \$n\$ is the size of matrix, PD is the proximal distance algorithm, aPD is the accelerated proximal distance algorithm, and Mosek is the Mosek solver.}")
println("\t\\label{tab:copos-sym}")
println("\\end{table}")
