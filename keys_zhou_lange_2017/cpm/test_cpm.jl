include("cpm.jl")

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
