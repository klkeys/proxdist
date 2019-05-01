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
    A        :: Matrix{T},
    b        :: Vector{T};
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
    Ax  = zeros(T, n) 

    # need spectral decomposition of A
    (d,V) = eig(A)

	i         = 0
    loss      = Inf 
    loss0     = Inf
    dnonneg   = Inf
    converged = false
    stuck     = false
    ρ_inv     = one(T) / rho
    z_max     = max.(z, zero(T))

    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - one(T)) / (i + one(T) + one(T))
        ky = one(T) + kx
		z .= ky.*y .- kx.*x
        copy!(x,y)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max,z)

        # compute distances to constraint sets
        dnonneg0 = dnonneg
        dnonneg = euclidean(z,z_max)

        # prox dist update y = inv(I + ρ_inv*A)(z_max - ρ_inv*b)
        prox_quad!(y, y2, V, d, b, z_max, ρ_inv)

        # recompute loss
        A_mul_B!(Ax, A, y)
        loss = nqp_loss(Ax, b, y) 

        # print progress of algorithm
        quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)
        @assert isfinite(loss) "Loss is no longer finite after $i iterations, something is wrong..."

        # convergence checks
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
    b        :: Vector{T}; 
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
    i         = 0
    loss      = Inf
    loss0     = Inf
    dnonneg   = Inf
    dnonneg0  = Inf
    converged = false
    stuck     = false

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
    #A0 = A + rho*I
    Afact = cholfact(A, shift = rho)

    ### (2): use CG with a function handle "Afun" for fast updates
    ### need a preconditioner for cg?
    ### compute a factorization of A or A + rho*I as a preconditioner
#    f(output, v) = mulbyA!(output, v, A, rho, n)
#    Afun = LinearMap{T}(f, n, ismutating = true)
#    A0 = A + rho*I
#    Pl = cholfact(A0)
#    Pl = crout_ilu(A0, τ=0.01)

    # (3): use LSQR
    #A0 = A + rho*I

    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - one(T)) / (i + one(T) + one(T))
        ky = one(T) + kx
		z .= ky.*y .- kx.*x
        copy!(x,y)

        # compute projection onto constraint set
        # z_max = max(z, 0)
        i > 1 && project_nonneg!(z_max, z)

        # also update b0 = rho*z_max - b
		b0 .= rho.*z_max .- b

        # prox dist update y = inv(rho*I + A)(rho*z_max - b)
        # use z as warm start
        #cg!(y, Afun, b0, Pl=Pl, maxiter=200, tol=1e-6, log = false, verbose = false)  # CG with precond
        #lsqr!(y, A0, b0, maxiter=200, atol=1e-8, btol=1e-8) # LSQR, no damping 
        y .= Afact \ b0                                         # Cholesky linear system solve

        # compute distance to constraint set
        dnonneg0 = dnonneg
        dnonneg = euclidean(y, z_max)

        # recompute loss
        A_mul_B!(Ax,A,y)
        loss = nqp_loss(Ax, b, y) 

        # print progress of algorithm
        quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)

        # check that loss is still finite
        # in contrary case, throw error
        @assert isfinite(loss) "Loss is no longer finite after $i iterations, something is wrong..."

        # convergence checks
        nonneg      = dnonneg < nnegtol
        diffnonneg  = abs(dnonneg - dnonneg0) / abs(dnonneg0)
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x) + one(T))
        converged   = scaled_norm < tol && nonneg
        stuck       = !converged && (scaled_norm < tol) && (diffnonneg < tol)

        # if converged then break, else save loss and continue
        if converged || (stuck && rho >= rho_max)
            quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)
            break
        end
        loss0 = loss

        if i % inc_step == 0 || diffnonneg < nnegtol || stuck
            rho = min(rho_inc*rho, rho_max)
            Afact = cholfact(A, shift = rho)
#            f(output, v) = mulbyA!(output, v, A, rho, n)
#            Afun = LinearMap{T}(f, n, ismutating = true)
#            A0 .= A + rho*I
#            Pl = cholfact(A0)
#            Pl = crout_ilu(A0, τ=0.01)
            copy!(x,y) 
        end
    end

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => copy(y), "nonneg_dist" => dnonneg, "converged" => converged, "stuck" => stuck)
end
