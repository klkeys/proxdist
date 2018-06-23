using Gurobi
using MathProgBase 
using Ipopt
using Distances
include("nqp.jl")


function test_nqp()

    # simulation parameters
    seed       = 2015
    tol        = 1e-4
    quiet      = true
    inc_step_d = 200
    inc_step_s = 100 
    rho_inc_d  = 1.5 
    rho_inc_s  = 1.5 
    rho_d      = 1e-2
    rho_s      = 1e-4
    rho_max    = 1e30
    pd_tol     = 1e-4 # added to diagonal of A to ensure pos definiteness, i.e. A = AA'*AA + pd_tol*I
    opttol     = 1e-4 # gurobi tolerance for optimum
    feastol    = 1e-4 # Gurobi tolerance for feasibility
    nthreads   = 4
    initdim    = 256  # for precompiling all dense routines
    initdim_s  = 1024 # for precompiling all sparse routines

    # BLAS threads?
    # testing machine has 4 cores, 8 virtual cores with hyperthreading
    # per Julia GitHub issue #6293, BLAS is not tuned for hyperthreading
    # use one thread per core
    BLAS.set_num_threads(nthreads)

    # IPOPT and Gurobi parameters
    ipopt_tol = feastol
    ipopt_lb  = zero(Float64) 
    ipopt_b   = zero(Float64) 

    # configure IPOPT and Gurobi solvers
    gurobi_solver = GurobiSolver(OptimalityTol=opttol, FeasibilityTol=feastol, OutputFlag=0, Threads=nthreads, Method=2) # Method=2 -> barrier method
    ipopt_solver  = IpoptSolver(print_level=0, tol=ipopt_tol, file_print_level=0)

    # run dense model for precompilation
    n          = initdim
    inc_step   = inc_step_d
    A          = eye(n)
    b          = randn(n)
    ipopt_A    = spzeros(0,n)
    pd_out     = nqp(A,b,inc_step=inc_step, rho_inc=rho_inc_d, rho_max=rho_max, quiet=quiet, tol=tol, rho=rho_d, max_iter = 3)
    ipopt_out  = quadprog(b, A, ipopt_A, '=', ipopt_b, ipopt_lb, Inf, ipopt_solver)
    gurobi_out = quadprog(b, A, ipopt_A, '=', ipopt_b, ipopt_lb, Inf, gurobi_solver)

    # now do same for sparse model 
    n          = initdim_s
    s          = log10(n) / n 
    A          = speye(n)
    inc_step   = inc_step_s
    b          = randn(n)
    ipopt_A    = spzeros(0,n)
    pd_out     = nqp(A,b,inc_step=inc_step, rho_inc=rho_inc_s, rho_max=rho_max, quiet=quiet, tol=tol, rho=rho_s, max_iter = 3)
    ipopt_out  = quadprog(b, A, ipopt_A, '=', ipopt_b, ipopt_lb, Inf, ipopt_solver)
    gurobi_out = quadprog(b, A, ipopt_A, '=', ipopt_b, ipopt_lb, Inf, gurobi_solver)

    # how many dimensions should we test?
    max_dense_dim = 9 
    max_sparse_dim = 14

    # seed random number generator
    srand(seed)

    max_dense_dim <= max_sparse_dim || throw(ArgumentError("max_dense_dim cannot exceed max_sparse_dim"))

    # print LaTeX-formatted header for table
    println("\\begin{table}")
    println("\t\\centering")
    println("\t\\begin{tabular}{ccccccc}")
    println("\t\t\t\\toprule")
    println("\t\t\t\\multicolumn{1}{c}{Dimensions} & \\multicolumn{3}{c}{Optima} & \\multicolumn{3}{c}{CPU Times} \\\\")
    println("\t\t\t\\cmidrule(r){1-1} \\cmidrule(r){2-4} \\cmidrule(r){5-7}")
    println("\t\t\t\$n\$ & PD & IPOPT & Gurobi & PD & IPOPT & Gurobi \\\\") 
    println("\t\t\t\\hline")

    # first test dense dims
    for k = 1:max_sparse_dim

        # problem dimensions
        n = 2^k
        m = 2*n

        # initialize problem dimensions
        if k <= max_dense_dim
            AA       = randn(n,n)
            A        = AA' * AA + pd_tol*I
            y        = max.(randn(n),0)
            b        = randn(n)
            AA       = false
            rho_inc  = rho_inc_d
            inc_step = inc_step_d
            rho      = rho_d
        else
            s        = log10(n) / n 
            AA       = sprandn(n,n,s)
            A        = AA' * AA + pd_tol*I
            AA       = false
            b        = randn(n) 
            rho_inc  = rho_inc_s
            inc_step = inc_step_s
            rho      = rho_s
        end

        # run using proximal distance algorithm
        tic()
        output = nqp(A,b,inc_step=inc_step, rho_inc=rho_inc, rho_max=rho_max, tol=tol, rho=rho, nnegtol=feastol, quiet=quiet)
        mm_time = toq()
        x = vec(full(output["x"]))
        mm_obj = 0.5*dot(x, A*x) + dot(b,x)

        # equality parameter for solvers should be zeroes
        ipopt_A = spzeros(0,n)

        # use MathProgBase to specify Ipopt and Gurobi quadratic model
        tic()
        ipopt_output = quadprog(b, A, ipopt_A, '=', ipopt_b, ipopt_lb, Inf, ipopt_solver)
        ipopt_time = toq() 
        ipopt_output.status == :Optimal || throw(error("IPOPT solver failed to find optimum and returned status $(ipopt_output.status)"))

        # Gurobi parameters
        tic()
        gurobi_output = quadprog(b, A, ipopt_A, '=', ipopt_b, ipopt_lb, Inf, gurobi_solver)
        gurobi_time = toq()
        ipopt_output.status == :Optimal || throw(error("Gurobi solver failed to find optimum and returned status $(gurobi_output.status)"))
       
        # print line of table
        @printf("\t\t\t %d & %3.4f & %3.4f & %3.4f & %3.4f & %3.4f & %3.4f\\\\\n", n, mm_obj, ipopt_output.objval, gurobi_output.objval, mm_time, ipopt_time, gurobi_time)

    end

    # table end matter
    println("\t\t\t\\bottomrule")
    println("\t\t\\end{tabular}")
#    println("\t\\caption{CPU times and optima for nonnegative quadratic programming. Here \$n\$ is the number of variables, ``Real\" denotes the true optimum, PD is the proximal distance algorithm, IPOPT is the Ipopt solver, and Gurobi is the Gurobi solver. After \$n = $(2^max_dense_dim)\$, the constraint matrix \$\\boldsymbol{A}\$ is sparse with sparsity level \$2*\\log_{10}(n)/n\$.}")
#    println("\t\\caption{CPU times and optima for nonnegative quadratic programming. Here \$n\$ is the number of variables, PD is the proximal distance algorithm, IPOPT is the Ipopt solver, and Gurobi is the Gurobi solver. After \$n = $(2^max_dense_dim)\$, the constraint matrix \$\\boldsymbol{A}\$ is sparse with sparsity level \$\\log_{10}(n)/n\$.}")
    println("\t\\caption{CPU times and optima for nonnegative quadratic programming. Here \$n\$ is the number of variables, PD is the proximal distance algorithm, IPOPT is the Ipopt solver, and Gurobi is the Gurobi solver. After \$n = $(2^max_dense_dim)\$, the constraint matrix \$\\boldsymbol{A}\$ is sparse.") 
    println("\t\\label{tab:nqp}")
    println("\\end{table}")

    return nothing
end

test_nqp()
