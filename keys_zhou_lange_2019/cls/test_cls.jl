include("cls.jl")

# print LaTeX-formatted header for table
println("\\begin{table}")
println("\t\\centering")
println("\t\\begin{tabular}{cccccccc}")
println("\t\t\t\\toprule")
println("\t\t\t\\multicolumn{2}{c}{Dimensions} & \\multicolumn{3}{c}{Optima} & \\multicolumn{3}{c}{CPU Times} \\\\")
println("\t\t\t\\cmidrule(r){1-2} \\cmidrule(r){3-5} \\cmidrule(r){6-8}")
println("\t\t\t\\$n\$ & \$p\$ & PD & IPOPT & Gurobi & PD & IPOPT & Gurobi \\\\") 
println("\t\t\t\\hline")

max_dense_dim = 10
for k = 3:14 # 3:14
    # generate data
    srand(k) # seed
    n, p = 2^k, 2^(k - 1)
    if k ≤ max_dense_dim
        X = randn(n, p)
    else
        X = sprandn(n, p, 10 / p) # about 10 non-zero entries per row
    end
    β = abs.(randn(p))
    β ./= sum(β) # true β is in simplex
    y = X * β + randn(n)
    # proximal distance solver
    tic()
    _, obj_pd = cls(X, y, SimplexProjection!)
    time_pd = toq()
    # Gurobi solver
    tic()
    _, obj_gurobi = cls_simplex_mpb(X, y, GurobiSolver(OutputFlag = 0))
    time_gurobi = toq()
    # Ipopt solver
    tic()
    _, obj_ipopt = cls_simplex_mpb(X, y, IpoptSolver(print_level=0))
    time_ipopt = toq()
    # print line of table
    k < 4 && continue # k < 4 cases are for warmup
    @printf("\t\t\t (%d,%d) & %3.4f & %3.4f & %3.4f & %3.4f & %3.4f & %3.4f\\\\\n", 
    n, p, obj_pd, obj_ipopt, obj_gurobi, time_pd, time_ipopt, time_gurobi)
end

# table end matter
println("\t\t\t\\bottomrule")
println("\t\t\\end{tabular}")
println("\t\\caption{CPU times and optima for simplex-constrained least squares. Here \$\\boldsymbol{A} \\in \\mathbb{R}^{n \\times p}\$, PD is the proximal distance algorithm, IPOPT is the Ipopt solver, and Gurobi is the Gurobi solver. After \$n = $(2^max_dense_dim)\$, the predictor matrix \$\\boldsymbol{A}\$ is sparse.}") 
println("\t\\label{tab:nqp}")
println("\\end{table}")
