using Gurobi
using MathProgBase 
using Ipopt
using Distances
include("nqp.jl")

## run this at command line with julia --track-allocation=user memory_profile_nqp.jl
## simulation parameters
#seed       = 2015
#tol        = 1e-4
#quiet      = true
#inc_step_d = 200
#inc_step_s = 100 
#rho_inc_d  = 1.5 
#rho_inc_s  = 1.5 
#rho_d      = 1.0
#rho_s      = 1e-4
##    rho_s      = 1.0
#rho_max    = 1e30
#pd_tol     = 1e-6 # added to diagonal of A to ensure pos definiteness, i.e. A = AA'*AA + pd_tol*I
#opttol     = 1e-4 # gurobi tolerance for optimum
#feastol    = 1e-4 # Gurobi tolerance for feasibility
#nthreads   = 4
#initdim    = 256  # for precompiling all dense routines
#initdim_s  = 1024 # for precompiling all sparse routines
#
## BLAS threads?
## testing machine has 4 cores, 8 virtual cores with hyperthreading
## per Julia GitHub issue #6293, BLAS is not tuned for hyperthreading
## use one thread per core
##    blas_set_num_threads(NUM_CORES)
#BLAS.set_num_threads(nthreads)
#
## IPOPT and Gurobi parameters
#ipopt_tol = feastol
#ipopt_lb  = zero(Float64) 
#ipopt_b   = zero(Float64) 
#
## configure IPOPT and Gurobi solvers
#gurobi_solver = GurobiSolver(OptimalityTol=opttol, FeasibilityTol=feastol, OutputFlag=0, Threads=nthreads, Method=2) # Method=2 -> barrier method
#ipopt_solver  = IpoptSolver(print_level=0, tol=ipopt_tol, file_print_level=0)
#
## run dense model for precompilation
#n          = initdim
#inc_step   = inc_step_d
#A          = eye(n)
##    y          = max(randn(n),0)
##    b          = -A*y
#b          = randn(n)
#ipopt_A    = spzeros(0,n)
#pd_out     = nqp(A,b,inc_step=inc_step, rho_inc=rho_inc_d, rho_max=rho_max, quiet=quiet, tol=tol, rho=rho_d)
#ipopt_out  = quadprog(b, A, ipopt_A, '=', ipopt_b, ipopt_lb, Inf, ipopt_solver)
#gurobi_out = quadprog(b, A, ipopt_A, '=', ipopt_b, ipopt_lb, Inf, gurobi_solver)
#
## clear memory
#Profile.clear_malloc_data()
#
## now rerun proxdist 
#for i = 1:10
#    pd_out     = nqp(A,b,inc_step=inc_step, rho_inc=rho_inc_d, rho_max=rho_max, quiet=quiet, tol=tol, rho=rho_d)
#end

    T = Float64
    rho       = one(T)
    rho_inc      = one(T) + one(T)
    rho_max      = 1e15
    max_iter   = 10000
    inc_step   = 100
    tol        = 1e-6
    nnegtol    = 1e-6
    quiet     = true

    # initialize arrays
    n   = length(b)
    x   = zeros(T, n)
    y   = zeros(T, n)
    y2  = zeros(T, n)
    z   = zeros(T, n)
    Ax  = BLAS.gemv('N', one(T), A, x) 

    # need spectral decomposition of A
    (d,V) = eig(A)

    HALF    = convert(T, 0.5)
    loss    = HALF*dot(Ax,x) + dot(b,x)
    loss0   = Inf
    dnonneg = Inf
    ρ_inv   = one(T) / rho
    z_max   = max.(z, 0)

    i = 0
    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max, z)

        # compute distances to constraint sets
        dnonneg = euclidean(z, z_max)

        # print progress of algorithm
        quiet || print_progress(i, loss, dnonneg, rho, quiet, i_interval = 10, inc_step = inc_step)

        # prox dist update y = inv(I + ρ_inv*A)(z_max - ρ_inv*b)
        prox_quad!(y, y2, V, d, b, z_max, ρ_inv)

        # convergence checks
        BLAS.gemv!('N', one(T), A, y, zero(T), Ax) 
        loss        = HALF*dot(Ax,y) + dot(b,y)
        nonneg      = dnonneg < nnegtol
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x,2) + one(T))
        converged   = scaled_norm < tol && nonneg

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho   = min(rho_inc*rho, rho_max)
            ρ_inv = one(T) / rho
            copy!(x,y)
        end
    end
