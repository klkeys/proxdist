using Gurobi
using JuMP
using GLPKMathProgInterface
using MathProgBase
using SCS
using Distances
include("lp.jl")


function test_lp()

    # simulation parameters
    seed       = 2016
    tol        = 1e-4
    quiet      = true
    s          = 0.01
    inc_step   = 200
    inc_step_d = 100
    inc_step_s = 50
    max_iter   = 10000
    rho_inc_d  = 2.0
    rho_inc_s  = 1.5
    rho_max    = 1e30
    nthreads   = 4
    initdim    = 1000 # for precompiling all routines

    # BLAS threads?
    # testing machine has 4 cores, 8 virtual cores with hyperthreading (NUM_CORES = 8)
    # per Julia GitHub issue #6293, BLAS is not tuned for hyperthreading
    # would normally use
    # > blas_set_num_threads(NUM_CORES)
    # but to ensure best BLAS performance use one thread per core
    blas_set_num_threads(nthreads)

    # how many dimensions should we test?
    # N.B. code assumes that max_sparse_dim >= max_dense_dim
    max_dense_dim  = 9
    max_sparse_dim = 13
    max_dense_dim <= max_sparse_dim || throw(ArgumentError("max_dense_dim cannot exceed max_sparse_dim"))

    # SPS parameters
    max_iters = max_iter
    eps       = 1e-4
    verbose   = !quiet

    # Gurobi parameters
    gmethod   = 3
    opttol    = 1e-4 # Gurobi tolerance for optimum
    feastol   = 1e-4 # Gurobi tolerance for feasibility
    goutput   = ifelse(quiet, 0, 1)

    # configure solvers
    scs_solver    = SCSSolver(max_iters=max_iters, eps=eps, verbose=verbose)
    gurobi_solver = GurobiSolver(OptimalityTol=opttol, FeasibilityTol=feastol, OutputFlag=goutput, Method=gmethod, Threads=nthreads)

    # configure dense model and run all routines once
    # this forces precompilation
    m            = initdim
    n            = 2*initdim
    A            = randn(m,n)
    v            = rand(n)
    c            = rand(n)
    b            = A*v
    rho_inc      = rho_inc_d
    output       = lin_prog(A,b,c, inc_step=inc_step, rho_inc=rho_inc, rho_max=rho_max, quiet=quiet, tol=tol, max_iter=max_iter)
#    output       = lin_prog2(A,b,c, inc_step=inc_step, rho_inc=rho_inc, rho_max=rho_max, quiet=quiet, tol=tol, max_iter=max_iter)
    scs_model    = linprog(c, A, '=', b, 0.0, Inf, scs_solver)
    gurobi_model = linprog(c, A, '=', b, 0.0, Inf, gurobi_solver)

    # do same with a sparse model
    A             = sprandn(m,n,s)
    c             = rand(n)
    x             = project_nonneg(sprandn(n,1,s))
    b             = vec(full(A*x))
    rho_inc       = rho_inc_s
    output        = lin_prog(A,b,c, inc_step=inc_step, rho_inc=rho_inc, rho_max=rho_max, quiet=quiet, tol=tol, max_iter=max_iter)
#    output        = lin_prog2(A,b,c, inc_step=inc_step, rho_inc=rho_inc, rho_max=rho_max, quiet=quiet, tol=tol, max_iter=max_iter)
    scs_model     = linprog(c, A, '=', b, 0.0, Inf, scs_solver)
    gurobi_model  = linprog(c, A, '=', b, 0.0, Inf, gurobi_solver)

    # now we perform reproducible tests
    # seed random number generator
    srand(seed)

    # print LaTeX-formatted header for table
    println("\\begin{table}")
    println("\t\\centering")
    println("\t\\begin{tabular}{cccccccc}")
    println("\t\t\\toprule")
    println("\t\t\\multicolumn{2}{c}{Dimensions} & \\multicolumn{3}{c}{Optima} & \\multicolumn{3}{c}{CPU Times} \\\\")
    println("\t\t\\cmidrule(r){1-2} \\cmidrule(r){3-5} \\cmidrule(r){6-8}")
    println("\t\t\$m\$ & \$n\$ & PD & SCS & Gurobi & PD & SCS & Gurobi \\\\")
    println("\t\t\\hline")

    # test all dims
    # after max_dense_dims, switch to sparse problems
    for k = 1:max_sparse_dim

        # problem dimensions
        m = 2^k
        n = 2*m

        # initialize problem variables
        if k <= max_dense_dim
            A       = randn(m,n)
            x       = rand(n)
            c       = rand(n)
            b       = A*x
            rho_inc = rho_inc_d
            inc_step = inc_step_d
        else
            A       = sprandn(m,n,s)
            c       = rand(n)
            x       = rand(n)
            b       = A*x
            rho_inc = rho_inc_s
            inc_step = inc_step_s
        end

        # run using proximal distance algorithm
        tic()
        output = lin_prog(A,b,c, inc_step=inc_step, rho_inc=rho_inc, rho_max=rho_max, quiet=quiet, tol=tol, max_iter=max_iter)
#        output = lin_prog2(A,b,c, inc_step=inc_step, rho_inc=rho_inc, rho_max=rho_max, quiet=quiet, tol=tol, max_iter=max_iter)
        mm_time = toq()

        # run with SCS
        tic()
        scs_model  = linprog(c, A, '=', b, 0.0, Inf, scs_solver)
        scs_time   = toq()

        # run Gurobi solver
        tic()
        gurobi_model  = linprog(c, A, '=', b, 0.0, Inf, gurobi_solver)
        gurobi_time   = toq()

        # print line of table
        @printf("\t\t%d & %d & %3.4f & %3.4f & %3.4f & %3.4f & %3.4f & %3.4f \\\\\n", m, n, output["obj"], scs_model.objval, gurobi_model.objval, mm_time, scs_time, gurobi_time)

    end

    println("\t\t\\bottomrule")
    println("\t\\end{tabular}")
    println("\t\\caption{CPU times and optima for linear programming. Here \$m\$ is the number of constraints, \$n\$ is the number of variables, PD is the proximal distance algorithm, SCS is the Splitting Cone Solver, and Gurobi is the Gurobi solver. After \$m = $(2^max_dense_dim)\$ the constraint matrix \$\\boldsymbol{A}\$ is initialized to be sparse with sparsity level \$s = 0.01\$.}")
    println("\t\\label{tab:lp}")
    println("\\end{table}")

    return nothing
end

test_lp()
