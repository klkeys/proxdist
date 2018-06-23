using Distances
using JuMP
using Gurobi
using MathProgBase
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

include("../projections/prox_quad.jl")
include("../common/common.jl")
include("../projections/project_nonneg.jl")


######################
### main functions ###
######################

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

	i       = 0
    loss    = dot(Ax,x)/2 + dot(b,x)
    loss0   = Inf
    daffine = Inf
    dnonneg = Inf
    invrho  = one(T) / rho
    z_max   = max(z, zero(T))

    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - one(T)) / (i + one(T) + one(T))
        ky = one(T) + kx
        #difference!(z,y,x, a=ky, b=kx, n=n)
		z .= ky.*y .- kx.*x
        copy!(x,y)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max,z)

        # compute distances to constraint sets
        dnonneg = euclidean(z,z_max)

        # print progress of algorithm
        quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)

        # prox dist update y = inv(I + invrho*A)(z_max - invrho*b)
        prox_quad!(y, y2, V, d, b, z_max, invrho)

        # convergence checks
        A_mul_B!(Ax, A, y)
        loss        = dot(Ax,y)/2 + dot(b,y)
        nonneg      = dnonneg < nnegtol
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x,2) + one(T))
        converged   = scaled_norm < tol && nonneg

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho    = min(rho_inc*rho, rho_max)
            invrho = one(T) / rho
            copy!(x,y)
        end
    end

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => copy(y), "nonneg_dist" => dnonneg)
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
) where {T <: AbstractFloat}

    # error checking
    check_conformability(A, b)
    @assert rho >  zero(T) "Argument rho must be positive"

    # initialize return values
    i       = 0
    loss    = Inf
    loss0   = Inf
    dnonneg = Inf

    # initialize arrays
	n     = length(b)
    x     = zeros(T,n)
    y     = zeros(T,n) 
    z     = zeros(T,n)
    z_max = zeros(T,n)
    b0    = zeros(T,n)
    Ax    = zeros(T,n)


    ### various ways to compute update
    ### (1): compute/cache Cholesky factorization, recompute whenever rho changes 
    A0 = A + rho*I
#    Afact = cholfact(A0)

    ### (2): use CG with a function handle "Afun" for fast updates
#    Afun  = MatrixFcn{T}(n, n, (output, v) -> mulbyA!(output, v, A, rho, n))
    ### need a preconditioner for cg?
    ### compute a Cholesky factorization of original A as a preconditioner
#    Afact = cholfact(A)

    # (3): use LSQR (reuse A0 from cholfact)
    # need an initialized Convergence History for good memory management
    #ch = ConvergenceHistory(false, (0.0,0.0,0.0), 0, T[])

    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - 2) / (i + 1)
        ky = one(T) + kx
        #difference!(z,y,x, a=ky, b=kx, n=n) # z = ky*y - kx*x
		z .= ky.*y .+ kx.*x
        copy!(x,y)

        # compute projection onto constraint set
        # z_max = max(z, 0)
        i > 1 && project_nonneg!(z_max, z)

        # also update b0 = rho*z_max - b
        #difference!(b0, z_max, b, a=rho, n=n)
		b0 .= rho.*z_mac .- b

        # prox dist update y = inv(rho*I + A)(rho*z_max - b)
        # use z as warm start
#        cg!(y, Afun, b0, maxiter=200, tol=1e-8)                # CG with no precond
#        cg!(y, Afun, b0, Afact, maxiter=200, tol=1e-8)         # precond CG
        lsqr!(y, A0, b0, maxiter=200, atol=1e-8, btol=1e-8) # LSQR, no damping 
#        y = Afact \ b0                                         # Cholesky linear system solve

        # compute distance to constraint set
        dnonneg = euclidean(y,z_max)

        # recompute loss
        A_mul_B!(Ax,A,y)
        loss = dot(Ax,y) / 2 + dot(b,y)

        # print progress of algorithm
        quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)

        # check that loss is still finite
        # in contrary case, throw error
        isfinite(loss) || throw(error("Loss is no longer finite after $i iterations, something is wrong...")) 

        # convergence checks
        nonneg      = dnonneg < nnegtol
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x) + one(T))
        converged   = scaled_norm < tol && nonneg

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho = min(rho_inc*rho, rho_max)
            A0  = A + rho*I
#            Afact = cholfact(A0)
#            Afun = MatrixFcn{T}(n, n, (output, v) -> mulbyA!(output, v, A, rho, n))
            copy!(x,y) 
        end
    end

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => y, "nonneg_dist" => dnonneg)
end
