using Distances
using JuMP
using Gurobi
using MathProgBase
using IterativeSolvers
using LinearMaps
using ILU

# ==============================================================================
# load projection code
# ==============================================================================
include("../projections/project_nonneg.jl")
include("../projections/prox_quad.jl")

# ==============================================================================
# subroutines
# ==============================================================================
include("../common/common.jl")

# create a function handle for CG 
# will pass mulbyA! as an operator into handle 
function mulbyA!(output, v, A, ρ, n)
    A_mul_B!(output, A, v)
    @inbounds for i = 1:n
        output[i] += v[i]*ρ
    end
    return output
end

function nqp_loss(Ax::Vector{T}, b::Vector{T}, y::Vector{T}) where {T <: AbstractFloat}
    a = zero(T)
    h = convert(T, 0.5)
    for i in eachindex(Ax)
        a += (h * Ax[i] + b[i]) * y[i]
    end
    return a
end



# ==============================================================================
# main functions
# ==============================================================================

function nqp(
    A        :: DenseMatrix{T},
    b        :: DenseVector{T};
    rho      :: T    = one(T),
    rho_inc  :: T    = one(T) + one(T),
    rho_max  :: T    = 1e15,
    max_iter :: Int  = 10000,
    inc_step :: Int  = 100,
    tol      :: T    = 1e-6,
    nnegtol  :: T    = 1e-6,
    quiet    :: Bool = true,
) where {T <: AbstractFloat}

    # error checking
    check_conformability(A, b)
    @assert rho >  zero(T) "Argument rho must be positive"

    # initialize arrays
    n   = length(b)
    x   = zeros(T, n)
    y   = zeros(T, n)
    y2  = zeros(T, n)
    z   = zeros(T, n)
    Ax  = BLAS.gemv('N', one(T), A, x) 

    # need spectral decomposition of A
    (d,V) = eig(A)

    #loss    = nqp_loss(Ax, b, x)
    loss    = Inf
    loss0   = Inf
    dnonneg = Inf
    dnonneg0 = Inf
    converged = false
    stuck = false
    ρ_inv   = one(T) / rho
    z_max   = project_nonneg(z) 

    i = 0
    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max, z)

        # compute distances to constraint sets
        dnonneg0 = dnonneg
        dnonneg = euclidean(z, z_max)

        # print progress of algorithm
        quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)

        # prox dist update y = inv(I + ρ_inv*A)(z_max - ρ_inv*b)
        prox_quad!(y, y2, V, d, b, z_max, ρ_inv)

        # convergence checks
        A_mul_B!(Ax, A, y)
        loss        = nqp_loss(Ax, b, y)
        nonneg      = dnonneg < nnegtol
        diffnonneg  = abs(dnonneg - dnonneg0)
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x,2) + one(T))
        converged   = scaled_norm < tol && nonneg
        stuck       = !converged && (scaled_norm < tol) && (diffnonneg < nnegtol)

        # if converged then break, else save loss and continue
        if converged || (stuck && rho >= rho_max)
            quiet || print_progress(i, loss, dnonneg, rho, i_interval = max_iter, inc_step = inc_step)
            break
        end
        loss0 = loss

        if i % inc_step == 0 || diffnonneg < nnegtol || stuck
            rho   = min(rho_inc*rho, rho_max)
            ρ_inv = one(T) / rho
            copy!(x,y)
        end
    end

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => copy(y), "nonneg_dist" => dnonneg, "converged" => converged, "stuck" => stuck)
end


function nqp(
    A        :: SparseMatrixCSC{T,Int},
    b        :: SparseMatrixCSC{T,Int};
    rho      :: T    = one(T),
    rho_inc  :: T    = one(T) + one(T),
    rho_max  :: T    = 1e20,
    max_iter :: Int  = 10000,
    inc_step :: Int  = 100,
    tol      :: T    = 1e-6,
    nnegtol  :: T    = 1e-6,
    quiet    :: Bool = true,
) where {T <: AbstractFloat}

    # error checking
    check_conformability(A, b)
    @assert rho >  zero(T) "Argument rho must be positive"

    # initialize return values
    i       = 0
    loss0   = Inf
    dnonneg = Inf
    dnonneg0 = Inf
    converged = false
    stuck = false
    #HALF    = convert(T, 0.5)

    # initialize arrays
    n  = length(b)
    x  = spzeros(T,n,1)
    y  = spzeros(T,n,1)
    y2 = spzeros(T,n,1)
    z  = spzeros(T,n,1)
    Ax = A*x
    dA = diag(A)
    A0 = A - spdiagm(dA,0)
    b0 = copy(b)

    # compute initial values of loss, projection
    loss  = HALF*dot(Ax,x) + dot(b,x)
    z_max = project_nonneg(z) 

    # set minimum rho to ensure diagonally dominant system 
    # use to set initial value of d
    rho = max(rho, one(T) + maximum(sumabs(A0,1)))
    d   = spdiagm(one(T) ./ (dA + rho), 0)

    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projections onto constraint sets
        # also update b0 = rho*z_max - b
        z_max = project_nonneg(z)
        b0   .= rho*z_max .- b

        # compute distances to constraint sets
        dnonneg = euclidean(z, z_max)

        # print progress of algorithm
        quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)

        # prox dist update y = inv(rho*I + A)(rho*z_max - b)
        # use z as warm start
        y = prox_quad(z,A,b,z_max,rho, d=d, A0=A0, x0=y2, b0=b0)

        # recompute loss
        A_mul_B!(Ax, A, y)
        loss = HALF*vecdot(Ax,y) + vecdot(b,y)

        # check that loss is still finite
        # in contrary case, throw error
        isfinite(loss) || throw(error("Loss is no longer finite, something is wrong...")) 

        # convergence checks
        nonneg      = dnonneg < nnegtol
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x) + one(T))
        converged   = scaled_norm < tol && nonneg

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0 || diffnonneg < nnegtol || stuck
            rho = min(rho_inc*rho, rho_max)
            d   = spdiagm(one(T) ./ (dA + rho), 0)
            x   = y
        end
    end

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => copy(y), "nonneg_dist" => dnonneg, "converged" => converged, "stuck" => stuck)
end


function nqp(
    A        :: SparseMatrixCSC{T,Int},
    b        :: DenseVector{T}; 
    rho      :: T    = one(T),
    rho_inc  :: T    = one(T) + one(T),
    rho_max  :: T    = 1e20,
    max_iter :: Int  = 10000,
    inc_step :: Int  = 100,
    tol      :: T    = 1e-6,
    nnegtol  :: T    = 1e-6,
    quiet    :: Bool = true,
) where{T <: AbstractFloat}

    # error checking
    check_conformability(A, b)
    @assert rho >  zero(T) "Argument rho must be positive"

    # initialize return values
    i         = 0
    loss      = Inf
    loss0     = Inf
    dnonneg   = Inf
    dnonneg0  = Inf
    converged = false
    stuck     = false

    # initialize arrays and algorithm variables
    n     = length(b)
    x     = zeros(T,n)
    y     = zeros(T,n) 
    z     = zeros(T,n)
    z_max = zeros(T,n)
    b0    = zeros(T,n)
    Ax    = zeros(T,n)

    ### various ways to compute update
    ### (1): compute/cache Cholesky factorization, recompute whenever rho changes 
    #A0 = A + rho*I
    Afact = cholfact(A, shift = rho)

    ### (2): use CG with a function handle "Afun" for fast updates
    #f(output, v) = mulbyA!(output, v, A, rho, n)
    #Afun = LinearMap{T}(f, n, ismutating = true)
    #A0 = A + rho*I
    #Pl = crout_ilu(A0, τ=0.01)

    # (3): use LSQR 
    #A0 = A + rho*I
    #Afun = LinearMap{T}(A0)

    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projection onto constraint set
        # z_max = max(z, 0)
        project_nonneg!(z_max, z)

        # also update b0 = rho*z_max - b
        b0 .= rho .* z_max .- b

        # prox dist update y = inv(rho*I + A)(rho*z_max - b)
        # use z as warm start
        #cg!(y, Afun, b0, maxiter=100, tol=1e-6, log = false, verbose = false)                 # CG with no precond
        #cg!(y, Afun, b0, Pl=Pl, maxiter=100, tol=1e-6, log = false, verbose = false)           # CG + precond
        #lsqr!(y, A0, b0, maxiter=200, atol=1e-8, btol=1e-8, log = false, verbose = false)     # LSQR, no damping 
        y .= Afact \ b0                                                                        # Cholesky linear system solve
        #A_ldiv_B!(y, Afact, sparsevec(b0))

        # compute distance to constraint set
        dnonneg0 = dnonneg
        dnonneg = euclidean(y, z_max)

        # recompute loss
        A_mul_B!(Ax, A, y)
        loss = nqp_loss(Ax, b, y)

        # print progress of algorithm
        quiet || print_progress(i, loss, dnonneg, rho, i_interval = 2000, inc_step = inc_step)

        # check that loss is still finite
        # in contrary case, throw error
        @assert isfinite(loss) "Loss is no longer finite after $i iterations, something is wrong..."

        # convergence checks
        nonneg      = dnonneg < nnegtol
        diffnonneg  = abs(dnonneg - dnonneg0) / abs(dnonneg0)
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x) + one(T))
        converged   = (scaled_norm < tol) && nonneg
        stuck       = !converged && (scaled_norm < tol) && (diffnonneg < tol)

        # if converged then break, else save loss and continue
        # also abort if the algo gets stuck
        if converged || (stuck && rho >= rho_max)
            quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)
            break
        end
        loss0 = loss

        if i % inc_step == 0 || diffnonneg < nnegtol || stuck
            rho = min(rho_inc*rho, rho_max)
            #A0  = A + rho*I 
            #Afact = cholfact(A, shift = rho)
            cholfact!(Afact, A, shift = rho)
            #Afun = LinearMap{T}(A0)
            #f(output, v) = mulbyA!(output, v, A, rho, n)
            #Afun = LinearMap{T}(f, n, ismutating = true)
            #A0 = A + rho*I
            #Pl = crout_ilu(A0, τ=0.01)
            copy!(x,y) 
        end
    end

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => copy(y), "nonneg_dist" => dnonneg, "converged" => converged, "stuck" => stuck)
end




# solve an NQP with quadprog() using the Gurobi solver
function nqp_gurobi(
    A       :: Union{DenseMatrix{T}, SparseMatrixCSC{T,Int}},
    b       :: Union{DenseVector{T}, SparseMatrixCSC{T,Int}};
    opttol  :: T = 1e-6,
    feastol :: T = 1e-6,
    quiet   :: Bool    = true,
    nthreads :: Int    = 4,
) where {T <: AbstractFloat}
    n = size(A,1)
    outflag = quiet ? 0 : 1
    gurobi_solver = GurobiSolver(OptimalityTol=opttol, FeasibilityTol=feastol, OutputFlag=outflag, Threads=nthreads)
    tic()
    gurobi_output = quadprog(vec(full(b)), A, zeros(0,n), '=', zero(T), zero(T), Inf, gurobi_solver)
    gurobi_time   = toq()
    z = gurobi_output.sol
#    !quiet && begin
        println("\n==== Gurobi results ====")
        println("Status of model: ", gurobi_output.status)
        println("Optimum: ", gurobi_output.objval) 
        println("Distance to nonnegative set? ", norm(z - max.(z,0)))
        println("\n")
#    end
    return z
end

function test_nqp()

    # set random seed for reproducibility
    seed = 2016
    srand(seed)

    # set number of BLAS threads
    blas_set_num_threads(4)

    # testing parameters
    n        = 2000
    m        = 2*n
    rho      = one(T)
    rho_inc  = 1.5 
    rho_max  = 1e30
    max_iter = 10000 
    inc_step = 200 
    inc_step = 5 
    inc_step = 10          # CAREFUL: for inc_step > 5, code destabilizes 
#    inc_step = 100          # CAREFUL: for inc_step > 5, code destabilizes 
    tol      = 1e-6
    nnegtol  = 1e-6
    quiet    = true 
    quiet    = false
    s        = log10(n)/n   # sparsity roughly scales with dimension 

    ### initialize problem variables

    # A is sparse symmetric positive semidefinite matrix
    AA = sprandn(n,n,s)
    A = AA'*AA
    AA = false
    A = 0.5*(A + A')        # symmetric
    A += 1e-8*I

#    A = sprandn(n,n,s)      # sparse
#    d, = eigs(A, nev=1, ritzvec=false, which=:SR)   # find minimum eigenvalue
#    dmax, = eigs(A, nev=1, ritzvec=false, which=:LR)   # find largest eigenvalue
#    d = abs(d)[1]
#    A = A + (d + 0.001)*I   # enforce PSD by adding just enough of I to ensure positive eigenvalues

    # can initialize different b based on desired result
    y = max.(randn(n), 0)            # feasible starting point 
    b = - A*(y + 0.01*randn(n))    # noisy minimum value 
#    b = - A*y                      # noiseless minimum value
#    b = rand(n)                     # bounds problem below at optimal value 0
#    b = vec(full(sprandn(n,1,s)))   # CAREFUL since b with negative values can unbound problem from below

    # set initial rho
#    rho = max(rho, one(T) + maximum(sumabs(A - spdiagm(diag(A),0),2))) # for Jacobi inversion algorithm 
    rho = 1e-2

    @show countnz(A) / prod(size(A))
    @show cond(full(A)) 

    # precompile @time macro
    @time 1+1

    output = nqp(A,b, n=n, rho=rho, rho_inc=rho_inc, rho_max=rho_max, max_iter=max_iter, inc_step=inc_step, tol=tol, nnegtol=nnegtol, quiet=quiet)
    @time output = nqp(A,b, n=n, rho=rho, rho_inc=rho_inc, rho_max=rho_max, max_iter=max_iter, inc_step=inc_step, tol=tol, nnegtol=nnegtol, quiet=quiet)
    x = copy(output["x"])

    # output for proxdist
    println("\n\n==== Accelerated Prox Dist Results ====")
    println("Iterations: ", output["iter"])
    println("Optimum: ", output["obj"])
    println("Distance to nonnegative set? ", norm(x - max.(x,0)))
    println("\n")

    # compare to Gurobi
    z = nqp_gurobi(A,b,quiet=quiet, opttol=tol, feastol=nnegtol)
    @time z = nqp_gurobi(A,b,quiet=quiet, opttol=tol, feastol=nnegtol)
    threshold!(z, tol)

    println("Distance between nqp, Gurobi opt variables: ", norm(x - z))

    return [x y z] 
end

function profile_sparse_nqp(
    reps     :: Int = 100;
    inc_step :: Int = 100,
    rho_inc  :: T = 2.0,
    rho_max  :: T = 1e30,
) where {T <: AbstractFloat}
    # set random seed for reproducibility
    seed = 2016
    srand(seed)

    # set number of BLAS threads
    blas_set_num_threads(4)

    # testing parameters
    n        = 1000
    rho      = one(T)
    max_iter = 10000 
    tol      = 1e-6
    nnegtol  = 1e-6
    quiet    = true 
    s        = 0.01
    
    # initialize variables 
    AA = sprandn(n,n,s)
    b  = sprandn(n,1,s)
    A  = AA'*AA / n

    # clear buffer before beginning
    Profile.clear()

    # set profiling parameters
    Profile.init(delay = 0.1)

    # profile accelerated LP
    @profile begin 
        for i = 1:reps
            output = nqp(A,b, n=n, rho=rho, rho_inc=rho_inc, rho_max=rho_max, max_iter=max_iter, inc_step=inc_step, tol=tol, nnegtol=nnegtol, quiet=quiet)
        end
    end

    # dump results to console
    println("Profiling results:")
    Profile.print()

    return nothing
end

#test_nqp()
#profile_sparse_nqp()
