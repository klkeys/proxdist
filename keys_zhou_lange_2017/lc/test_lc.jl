using JuMP
using Gurobi

include("lc.jl")

srand(123)
n = 2 .^ (2:6)
timing = zeros(length(n), 2)
loss = zeros(length(n), 2)

# precompile the routines 
i = 1
A = randn(n[i], n[i])
A = A' * A
b = randn(n[i])
tic(); (iters, x1, y1) = linear_complementarity(A, b); timing[i, 1] = toc();
tic(); (iters, y) = (x2, y2) = linear_complementarity_milp(A, b);

for i = 1:length(n)
  A = randn(n[i], n[i])
  A = A' * A
  b = randn(n[i])

  # proximal distance
  tic(); (iters, x1, y1) = linear_complementarity(A, b); timing[i, 1] = toc();
  loss[i, 1] = 0.5 * sum(abs2, y1 - A*x1 - b)

  # mixed integer programming
  tic(); (iters, y) = (x2, y2) = linear_complementarity_milp(A, b);
  timing[i, 2] = toc();
  loss[i, 2] = 0.5 * sum(abs2, y2 - A*x2 - b)
end

println("\\begin{table}")
println("\t\\centering")
println("\t\\begin{tabular}{rrrrr}")
println("\t\t\\multicolumn{1}{c}{Dimensions} & \\multicolumn{2}{c}{Optima} & \\multicolumn{2}{c}{CPU Times} \\\\ ")
println("\t\t\ \\cmidrule(r){2-3} \\cmidrule(r){4-5}")
println("\t\t\$n\$ & PD & Gurobi & PD & Gurobi\\\\")
println("\\hline")
for k = 1:length(n)
	@printf("\t\t%d & %3.6f & %3.6f & %3.4f & %3.4f\\\\\n", n[k],
  loss[k, 1], loss[k, 2], timing[k, 1], timing[k, 2])
end
println("\t\t\\bottomrule")
println("\t\\end{tabular}")
println("\t\\caption{CPU times and optima for linear complementarity problem with randomly generated data. Here \$n\$ is the size of matrix, PD is the proximal distance algorithm, and Gurobi is the Gurobi solver.}")
println("\t\\label{tab:lincomp}")
println("\\end{table}")
