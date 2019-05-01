using Convex
using Gurobi
using MathProgBase
using SCS
using Distances
include("socp.jl")

function test_proj_soc()

    # simulation parameters
    seed       = 2016
    tol        = 1e-4
    quiet      = true 
    s          = 0.01
    inc_step_d = 100
    inc_step_s = 10
    max_iter   = 10000
    rho_inc_d  = 1.5 
#    rho_inc_s  = 5.0 
    rho_inc_s  = 2.5
    rho_max    = 1e30
    nthreads   = 4
    initdim_d  = 50  # for precompiling all routines
    initdim_s  = 500 # for precompiling all routines

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
    m        = initdim_d
    n        = 2*initdim_d
    A        = randn(m,n)
    b        = rand(m)
    c        = ones(n) / n 
    x        = rand(n)
    w        = randn(n)
    d        = norm(A*x + b)
    rho_inc  = rho_inc_d
    inc_step = inc_step_d

    # run routines on dense problem
    pw = proj_soc(w,A,b,c,d, inc_step=inc_step, rho_inc=rho_inc, rho_max=rho_max, quiet=quiet, tol=tol, max_iter=max_iter)
    y = Convex.Variable(n)
    scs_model = minimize(0.5*sumsquares(y - w))
    scs_model.constraints += norm(A*x + b) <= dot(c,x) + d
    solve!(scs_model, scs_solver)
    y = Convex.Variable(n)
    gurobi_model = minimize(0.5*sumsquares(y - w))
    gurobi_model.constraints += norm(A*x + b) <= dot(c,x) + d
    solve!(gurobi_model, gurobi_solver)

    # do same with a sparse model
    m         = initdim_s
    n         = 2*initdim_s
    A         = sprandn(m,n,s)
    b         = vec(full(sprand(m,1,s)))
    c         = vec(full(sprand(n,1,s))) / n
    x         = vec(full(sprand(n,1,s)))
    d         = norm(A*x + b)
    w         = vec(full(sprandn(n,1,s)))
    rho_inc   = rho_inc_s 
    inc_step  = inc_step_s
    output    = proj_soc(w,A,b,c,d, inc_step=inc_step, rho_inc=rho_inc, rho_max=rho_max, quiet=quiet, tol=tol, max_iter=max_iter)
    y         = Convex.Variable(n)
    scs_model = minimize(0.5*sumsquares(y - w))
    scs_model.constraints += norm(A*y + b) <= vecdot(c,y) + d
    solve!(scs_model, scs_solver)
    z            = Convex.Variable(n)
    gurobi_model = minimize(0.5*sumsquares(y - w))
    gurobi_model.constraints += norm(A*z + b) <= vecdot(c,z) + d
    solve!(gurobi_model, gurobi_solver)

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
            A        = randn(m,n)
            b        = rand(m)
            c        = ones(n) / n 
            x        = rand(n)
            w        = randn(n)
            d        = norm(A*x + b)
            rho_inc  = rho_inc_d
            inc_step = inc_step_d
            rho      = 1.0
        else
            A        = sprandn(m,n,s)
            b        = vec(full(sprand(m,1,s)))
            c        = vec(full(sprand(n,1,s))) / n 
            x        = vec(full(sprand(n,1,s)))
            w        = vec(full(sprandn(n,1,s)))
            d        = norm(A*x + b)
            w        = vec(full(sprandn(n,1,s))) 
            rho_inc  = rho_inc_s 
            inc_step = inc_step_s
            rho      = 1e-2
        end

        # run using proximal distance algorithm
        tic()
        pw = proj_soc(w,A,b,c,d, inc_step=inc_step, rho_inc=rho_inc, rho_max=rho_max, quiet=quiet, tol=tol, max_iter=max_iter, p=m, q=n, rho=rho)
        mm_time = toq()

        # run using SCS 
        tic()
        y = Convex.Variable(n)
        scs_model = minimize(0.5*sumsquares(y - w))
        scs_model.constraints += norm(A*y + b) <= dot(c,y) + d
        solve!(scs_model, scs_solver)
        sps_time = toq()

        # ensure optimality!
        scs_model.status == :Optimal || throw(error("SCS failed to solve problem!"))

        # run Gurobi solver
        tic()
        z = Convex.Variable(n)
        gurobi_model = minimize(0.5*sumsquares(z - w))
        gurobi_model.constraints += norm(A*z + b) <= dot(c,z) + d
        solve!(gurobi_model, gurobi_solver)
        gurobi_time = toq()

        # ensure optimality!
        gurobi_model.status == :Optimal || throw(error("Gurobi failed to solve problem!"))


        # print line of table
#        @printf("\t\t%d & %d & %3.7f & %3.7f & %3.7f & %3.4f & %3.4f & %3.4f\\\\\n", m, n, output["obj"], getObjectiveValue(glpk_model), get_objval(gurobi_model), mm_time, glpk_time, gurobi_time)
#        @printf("\t\t%d & %d & %3.7f & %3.7f & %3.7f & %3.4f & %3.4f & %3.4f \\\\\n", m, n, 0.5*sqeuclidean(pw,w), scs_model.optval, gurobi_model.optval, mm_time, sps_time, gurobi_time)
        @printf("\t\t%d & %d & %3.5f & %3.5f & %3.5f & %3.4f & %3.4f & %3.4f \\\\\n", m, n, 0.5*sqeuclidean(pw,w), scs_model.optval, gurobi_model.optval, mm_time, sps_time, gurobi_time)

    end

    println("\t\t\\bottomrule")
    println("\t\\end{tabular}")
    #println("\t\\caption{CPU times and optima for linear programming. Here \$m\$ is the number of constraints, \$n\$ is the number of variables, PD is the proximal distance algorithm, AP is the alternating projection algorithm, and Gurobi is the Gurobi solver.}")
    println("\t\\caption{CPU times and optima for the second order cone projection. Here \$m\$ is the number of constraints, \$n\$ is the number of variables, PD is the proximal distance algorithm, SCS is the Splitting Cone Solver, and Gurobi is the Gurobi solver. After \$m = $(2^max_dense_dim)\$ the constraint matrix \$\\boldsymbol{A}\$ is initialized to be sparse with sparsity level $s.}")
    println("\t\\label{tab:socp}")
    println("\\end{table}")

    return nothing
end

test_proj_soc()
