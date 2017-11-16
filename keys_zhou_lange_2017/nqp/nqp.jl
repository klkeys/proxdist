using Distances
using RegressionTools
using JuMP
using Gurobi
using MathProgBase
using ProxOpt
using IterativeSolvers

###################
### subroutines ###
###################

# create a function handle for CG 
# will pass mulbyA! as an operator into handle 
function mulbyA!(output, v, A, rho, n)
    A_mul_B!(output, A, v)
    @inbounds for i = 1:n
        output[i] += v[i]*rho
    end
    output
end


######################
### main functions ###
######################

function nqp(
    A        :: DenseMatrix{Float64},
    b        :: DenseVector{Float64};
    n        :: Int     = length(b),
    rho      :: Float64 = one(Float64),
    rho_inc  :: Float64 = 2.0,
    rho_max  :: Float64 = 1e15,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    tol      :: Float64 = 1e-6,
    nnegtol  :: Float64 = 1e-6,
    quiet    :: Bool    = true,
)

    # error checking
    size(A,2) == size(A,1) || throw(DimensionMismatch("Argument A must be a square matrix"))
    n         == length(b) || throw(DimensionMismatch("Nonconformable A and b")) 

    # initialize arrays
    x   = zeros(Float64, n)
    y   = zeros(Float64, n)
    y2  = zeros(Float64, n)
    z   = zeros(Float64, n)
    Ax  = BLAS.gemv('N', one(Float64), A, x) 

    # need spectral decomposition of A
    (d,V) = eig(A)

    iter    = 0
    loss    = 0.5*dot(Ax,x) + dot(b,x)
    loss0   = Inf
    daffine = Inf
    dnonneg = Inf
    invrho  = one(Float64) / rho
    z_max   = max(z, zero(Float64))

    for i = 1:max_iter

        iter += 1

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - one(Float64)) / (i + one(Float64) + one(Float64))
        ky = one(Float64) + kx
        difference!(z,y,x, a=ky, b=kx, n=n)
        copy!(x,y)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max,z,n=n)

        # compute distances to constraint sets
        dnonneg = euclidean(z,z_max)

        # print progress of algorithm
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dnonneg, rho)
        end

        # prox dist update y = inv(I + invrho*A)(z_max - invrho*b)
        prox_quad!(y, V, d, b, z_max, invrho, y2=y2, n=n)

        # convergence checks
        BLAS.gemv!('N', one(Float64), A, y, 0.0, Ax) 
        loss        = 0.5*dot(Ax,y) + dot(b,y)
        nonneg      = dnonneg < nnegtol
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x,2) + one(Float64))
        converged   = scaled_norm < tol && nonneg

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho    = min(rho_inc*rho, rho_max)
            invrho = one(Float64) / rho
            copy!(x,y)
        end
    end

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{ASCIIString, Any}("obj" => loss, "iter" => iter, "x" => copy(y), "nonneg_dist" => dnonneg)
end


function nqp(
    A        :: SparseMatrixCSC{Float64,Int},
    b        :: SparseMatrixCSC{Float64,Int};
    n        :: Int     = b.m,
    rho      :: Float64 = one(Float64),
    rho_inc  :: Float64 = 2.0,
    rho_max  :: Float64 = 1e20,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    tol      :: Float64 = 1e-6,
    nnegtol  :: Float64 = 1e-6,
    quiet    :: Bool    = true,
)

    # error checking
    size(A,2) == size(A,1)     || throw(DimensionMismatch("Argument A must be a square matrix"))
    n         == length(b)     || throw(DimensionMismatch("Nonconformable A and b")) 
    rho       >  zero(Float64) || throw(ArgumentError("rho must be positive"))

    # initialize return values
    i       = 0
    loss    = Inf
    loss0   = Inf
    dnonneg = Inf

    # initialize arrays
    x  = spzeros(Float64,n,1)
    y  = spzeros(Float64,n,1)
    y2 = spzeros(Float64,n,1)
    z  = spzeros(Float64,n,1)
    Ax = A*x
    dA = diag(A)
    A0 = A - spdiagm(dA,0)
    b0 = copy(b)

    # set minimum rho to ensure diagonally dominant system 
    # use to set initial value of d
    rho = max(rho, one(Float64) + maximum(sumabs(A0,1)))
    d   = spdiagm(one(Float64) ./ (dA + rho), 0)

    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - one(Float64)) / (i + one(Float64) + one(Float64))
        ky = one(Float64) + kx

        # z = ky*y - kx*x
        z  = -kx*x
        z += ky*y
        x  = y

        # compute projections onto constraint sets
        # also update b0 = rho*z_max - b
        z_max = project_nonneg(z)
        b0    = rho*z_max
        b0   -= b

        # compute distances to constraint sets
        dnonneg = euclidean(z,z_max)

        # print progress of algorithm
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dnonneg, rho)
        end

        # prox dist update y = inv(rho*I + A)(rho*z_max - b)
        # use z as warm start
        y = prox_quad(z,A,b,z_max,rho, d=d, A0=A0, x0=y2, b0=b0)

        # recompute loss
        Ax   = A*y
        loss = 0.5*vecdot(Ax,y) + vecdot(b,y)

        # check that loss is still finite
        # in contrary case, throw error
        isfinite(loss) || throw(error("Loss is no longer finite, something is wrong...")) 

        # convergence checks
        nonneg      = dnonneg < nnegtol
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x) + one(Float64))
        converged   = scaled_norm < tol && nonneg

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho = min(rho_inc*rho, rho_max)
            d   = spdiagm(one(Float64) ./ (dA + rho), 0)
            x   = y
        end
    end

    # threshold small elements of y before returning
    w = threshold(y,tol)
    return Dict{ASCIIString, Any}("obj" => loss, "iter" => i, "x" => w, "nonneg_dist" => dnonneg)
end


function nqp(
    A        :: SparseMatrixCSC{Float64,Int},
    b        :: DenseVector{Float64}; 
    n        :: Int     = length(b),
    rho      :: Float64 = one(Float64),
    rho_inc  :: Float64 = 2.0,
    rho_max  :: Float64 = 1e20,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    tol      :: Float64 = 1e-6,
    nnegtol  :: Float64 = 1e-6,
    quiet    :: Bool    = true,
)

    # error checking
    size(A,2) == size(A,1)     || throw(DimensionMismatch("Argument A must be a square matrix"))
    n         == length(b)     || throw(DimensionMismatch("Nonconformable A and b")) 
    rho       >  zero(Float64) || throw(ArgumentError("rho must be positive"))

    # initialize return values
    i       = 0
    loss    = Inf
    loss0   = Inf
    dnonneg = Inf

    # initialize arrays
    x     = zeros(Float64,n)
    y     = zeros(Float64,n) 
    z     = zeros(Float64,n)
    z_max = zeros(Float64,n)
    b0    = zeros(Float64,n)
    Ax    = zeros(Float64,n)


    ### various ways to compute update
    ### (1): compute/cache Cholesky factorization, recompute whenever rho changes 
    A0 = A + rho*I
#    Afact = cholfact(A0)

    ### (2): use CG with a function handle "Afun" for fast updates
#    Afun  = MatrixFcn{Float64}(n, n, (output, v) -> mulbyA!(output, v, A, rho, n))
    ### need a preconditioner for cg?
    ### compute a Cholesky factorization of original A as a preconditioner
#    Afact = cholfact(A)

    # (3): use LSQR (reuse A0 from cholfact)
    # need an initialized Convergence History for good memory management
    ch = ConvergenceHistory(false, (0.0,0.0,0.0), 0, Float64[])

    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - 2) / (i + 1)
        ky = one(Float64) + kx
        difference!(z,y,x, a=ky, b=kx, n=n) # z = ky*y - kx*x
        copy!(x,y)

        # compute projection onto constraint set
        # z_max = max(z, 0)
        project_nonneg!(z_max, z, n=n)

        # also update b0 = rho*z_max - b
        difference!(b0, z_max, b, a=rho, n=n)

        # prox dist update y = inv(rho*I + A)(rho*z_max - b)
        # use z as warm start
#        cg!(y, Afun, b0, maxiter=200, tol=1e-8)                # CG with no precond
#        cg!(y, Afun, b0, Afact, maxiter=200, tol=1e-8)         # precond CG
        lsqr!(y, ch, A0, b0, maxiter=200, atol=1e-8, btol=1e-8) # LSQR, no damping 
#        y = Afact \ b0                                         # Cholesky linear system solve

        # compute distance to constraint set
        dnonneg = euclidean(y,z_max)

        # recompute loss
        A_mul_B!(Ax,A,y)
        loss = 0.5*dot(Ax,y) + dot(b,y)

        # print progress of algorithm
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dnonneg, rho)
        end

        # check that loss is still finite
        # in contrary case, throw error
        isfinite(loss) || throw(error("Loss is no longer finite after $i iterations, something is wrong...")) 

        # convergence checks
        nonneg      = dnonneg < nnegtol
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x) + one(Float64))
        converged   = scaled_norm < tol && nonneg

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho = min(rho_inc*rho, rho_max)
            A0  = A + rho*I
#            Afact = cholfact(A0)
#            Afun = MatrixFcn{Float64}(n, n, (output, v) -> mulbyA!(output, v, A, rho, n))
            copy!(x,y) 
        end
    end

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{ASCIIString, Any}("obj" => loss, "iter" => i, "x" => y, "nonneg_dist" => dnonneg)
end




# solve an NQP with quadprog() using the Gurobi solver
function nqp_gurobi(
    A       :: Union{DenseMatrix{Float64}, SparseMatrixCSC{Float64,Int}},
    b       :: Union{DenseVector{Float64}, SparseMatrixCSC{Float64,Int}};
    opttol  :: Float64 = 1e-6,
    feastol :: Float64 = 1e-6,
    quiet   :: Bool    = true,
    nthreads :: Int    = 4,
)
    n = size(A,1)
    outflag = quiet ? 0 : 1
    gurobi_solver = GurobiSolver(OptimalityTol=opttol, FeasibilityTol=feastol, OutputFlag=outflag, Threads=nthreads)
    tic()
    gurobi_output = quadprog(vec(full(b)), A, zeros(0,n), '=', zero(Float64), zero(Float64), Inf, gurobi_solver)
    gurobi_time   = toq()
    z = gurobi_output.sol
#    !quiet && begin
        println("\n==== Gurobi results ====")
        println("Status of model: ", gurobi_output.status)
        println("Optimum: ", gurobi_output.objval) 
        println("Distance to nonnegative set? ", norm(z - max(z,0.0)))
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
    rho      = one(Float64)
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
    y = max(randn(n), 0)            # feasible starting point 
    b = - A*(y + 0.01*randn(n))    # noisy minimum value 
#    b = - A*y                      # noiseless minimum value
#    b = rand(n)                     # bounds problem below at optimal value 0
#    b = vec(full(sprandn(n,1,s)))   # CAREFUL since b wih negative values can unbound problem from below

    # set initial rho
#    rho = max(rho, one(Float64) + maximum(sumabs(A - spdiagm(diag(A),0),2))) # for Jacobi inversion algorithm 
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
    println("Distance to nonnegative set? ", norm(x - max(x,0)))
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
    rho_inc  :: Float64 = 2.0,
    rho_max  :: Float64 = 1e30,
)
    # set random seed for reproducibility
    seed = 2016
    srand(seed)

    # set number of BLAS threads
    blas_set_num_threads(4)

    # testing parameters
    n        = 1000
    rho      = one(Float64)
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
