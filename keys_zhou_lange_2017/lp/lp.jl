using Distances
using RegressionTools
using JuMP
using Gurobi
#using ProxOpt
using IterativeSolvers
using LinearMaps

# ==============================================================================
# load projection code
# ==============================================================================
include("../projections/project_affine.jl")
include("../projections/project_nonneg.jl")

# ==============================================================================
# subroutines
# ==============================================================================


# function handle to efficiently compute sparse matrix-vector operation in CG
function mulbyA!(output, v, A, At, v2)
    A_mul_B!(v2,At,v)
    A_mul_B!(output, A, v2)
end

# function to compute a Nesterov acceleration step at iteration i
function compute_accelerated_step!(z::DenseVector{T}, x::DenseVector{T}, y::DenseVector{T}, i::Int) where {T <: AbstractFloat}
	kx = (i - one(T)) / (i + one(T) + one(T))
	ky = one(T) + kx

	# z = ky*y - kx*x
	z .= ky .* y .- kx .* x 
	copy!(x,y)
end

# subroutine to check conformability of linear program inputs
# if one of the vectors is nonconformable to the matrix, then throw an error
function check_conformability(A, b, c)
    p = length(b)
    q = length(c)
    @assert (p,q) == size(A) "nonconformable A, b, and c\nsize(A) = $(size(A))\nsize(b) = $(size(b))\nsize(c) = $(size(c))"
end

# subroutine to print algorithm progress
function print_progress(i, loss, daffine, dnonneg, rho, quiet; i_interval::Int = 10, inc_step::Int = 100)
    if (i <= i_interval || i % inc_step == 0) && !quiet
        @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", i, loss, daffine, dnonneg, rho)
    end
end

function print_progress(i, loss, dnonneg, rho, quiet; i_interval::Int = 10, inc_step::Int = 100)
    if (i <= i_interval || i % inc_step == 0) && !quiet
        @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dnonneg, rho)
    end
end



# ==============================================================================
# main functions
# ==============================================================================

"""
    lin_prog(A,b,c)

Solve the optimization problem

    minimize dot(x,c)
    s.t.     A*x == b
             x   >= 0

with an accelerated proximal distance algorithm.
"""
function lin_prog2(
    A        :: DenseMatrix{T},
    b        :: DenseVector{T},
    c        :: DenseVector{T};
    rho      :: T    = one(T),
    rho_inc  :: T    = one(T) + one(T),
    rho_max  :: T    = 1e15,
    max_iter :: Int  = 10000,
    inc_step :: Int  = 100,
    tol      :: T    = 1e-6,
    afftol   :: T    = 1e-6,
    nnegtol  :: T    = 1e-6,
    quiet    :: Bool = true
) where {T <: AbstractFloat}

    # error checking
	check_conformability(A, b, c)

    # declare algorithm variables
    iter    = 0
    loss    = dot(c,x)
    loss0   = Inf
    daffine = Inf
    dnonneg = Inf
    ρ_inv   = one(T) / rho
    HALF    = one(T) / 2
    TWO     = one(T) + one(T)
    
    # initialize temporary arrays
    x = zeros(T, q)
    y = zeros(T, q)
    z = zeros(T, q)

    # apply initial projections
    z_affine, C, d = project_affine(z,A,b)
    z_max = max(z, 0) 

    # main loop
    for i = 1:max_iter

        iter += 1

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max, z)
        i > 1 && project_affine!(z_affine, z, C, d)

        # compute distances to constraint sets
        daffine = euclidean(z, z_affine)
        dnonneg = euclidean(z, z_max)

        # print progress of algorithm
        print_progress(i, loss, daffine, dnonneg, rho, quiet, i_interval = 10, inc_step = inc_step)

        # prox dist update y = 0.5*(z_max + z_affine) - c/rho
        y .= HALF .* z_max .+ HALF .* z_affine .- ρ_inv .* c

        # convergence checks
        loss        = dot(c,y)
        nonneg      = dnonneg < nnegtol
        affine      = daffine < afftol
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x,2) + one(T))
        converged   = scaled_norm < tol && nonneg && affine

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho    = min(rho_inc*rho, rho_max)
            ρ_inv = one(T) / rho
            copy!(x,y)
        end
    end

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{String, Any}("obj" => loss, "iter" => iter, "x" => copy(y), "affine_dist" => daffine, "nonneg_dist" => dnonneg)
end


"""
    lin_prog(A,b,c)

Solve the optimization problem

    minimize dot(x,c) + λ'(Ax - b)
             x   >= 0

with an accelerated proximal distance algorithm. Here the affine constraints of the linear program are moved into the objective function.
The vector `λ` represents the Lagrange multiplier. If we let `y = max(x,0)` and we denote the penalty parameter by `ρ`,
then the iterative update scheme is

    x+ = (I - pinv(A)*A)(y - c/ρ) + pinv(A)*b
"""
function lin_prog(
    A        :: DenseMatrix{T},
    b        :: DenseVector{T},
    c        :: DenseVector{T};
    rho      :: T    = one(T),
    rho_inc  :: T    = 2.0,
    rho_max  :: T    = 1e15,
    max_iter :: Int  = 10000,
    inc_step :: Int  = 100,
    tol      :: T    = 1e-6,
    afftol   :: T    = 1e-6,
    nnegtol  :: T    = 1e-6,
    quiet    :: Bool = true,
) where {T <: AbstractFloat}

    # error checking
	check_conformability(A, b, c)

	# initialize temporary arrays
    q  = size(A,2)
    x  = zeros(T, q)
    y  = zeros(T, q)
    y2 = zeros(T, q)
    z  = zeros(T, q)

    # initialize algorithm parameters
    iter    = 0
    loss    = dot(c,x)
    loss0   = Inf
    daffine = Inf
    dnonneg = Inf
    ρ_inv   = one(T) / rho

	# perform initial affine projection
    pA = pinv(A)
    C  = BLAS.gemm('N', 'N', -one(T), pA, A)
    C  += I.λ
    d  = BLAS.gemv('N', one(T), pA, b)
    z_max = max.(z, zero(T))

    for i = 1:max_iter

        iter += 1

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max, z)

        # compute distances to constraint sets
#        dnonneg = euclidean(z,z_max)
        dnonneg = euclidean(y,z_max)

        # print progress of algorithm
        print_progress(i, loss, daffine, dnonneg, rho, quiet, i_interval = 10, inc_step = inc_step)

        # prox dist update
        copy!(y2, z_max)
        BLAS.axpy!(q, -ρ_inv, c, 1, y2, 1)
        copy!(y,d)
        BLAS.symv!('u', one(T), C, y2, one(T), y)

        # convergence checks
        loss        = dot(c,y)
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

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{String, Any}("obj" => loss, "iter" => iter, "x" => copy(y), "nonneg_dist" => dnonneg)
end

"""
    lin_prog(A::SparseMatrixCSC, b, c)

For sparse matrix `A` and dense vectors `b` and `c`, solve the optimization problem

    minimize dot(x,c) + λ(Ax - b)
             x   >= 0

with an accelerated proximal distance algorithm. Here the affine constraints of the linear program are moved into the objective function.
The vector `λ` represents the Lagrange multiplier. `linprog` factorizes `A` to obtain a suitable proxy for the pseudoinverse of `A`.
"""
function lin_prog2(
    A        :: SparseMatrixCSC{T,Int},
    b        :: DenseVector{T},
    c        :: DenseVector{T};
    rho      :: T = one(T),
    rho_inc  :: T = 2.0,
    rho_max  :: T = 1e15,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    tol      :: T = 1e-6,
    afftol   :: T = 1e-6,
    nnegtol  :: T = 1e-6,
    quiet    :: Bool    = true,
) where {T <: AbstractFloat}

    # error checking
	check_conformability(A, b, c)

    iter    = 0
    loss0   = Inf
    dnonneg = Inf
    ρ_inv  = one(T) / rho
    At      = A'
    AA      = cholfact(A * At)
#    AA      = factorize(A * At)

    x     = zeros(T, q)
    y     = zeros(T, q)
    y2    = zeros(T, q)
    z     = zeros(T, q)
    z_max = max.(y, zero(T))
    C     = full(I - (At * ( AA \ A)))
    d     = vec(full(At * (AA \ b)))

    loss = dot(c,x)

    i = 0
    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max, z)

        # compute distances to constraint sets
        dnonneg = euclidean(z,z_max)

        # print progress of algorithm
        print_progress(i, loss, daffine, dnonneg, rho, quiet, i_interval = 10, inc_step = inc_step)

        isfinite(loss) || throw(error("Loss is not finite after $i iterations, something is wrong..."))

        # prox dist update y = C*(z_max - ρ_inv*c) + d
        copy!(y2,z_max)
        BLAS.axpy!(q, -ρ_inv, c, 1, y2, 1)
        copy!(y,d)
        BLAS.symv!('u', one(T), C, y2, one(T), y)

        # convergence checks
        loss        = dot(c,y)
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

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => sparsevec(y), "nonneg_dist" => dnonneg)
end

"""
    lin_prog(A::SparseMatrix, b, c)

For sparse matrix `A` and dense vectors `b` and `c`, solve the optimization problem

    minimize dot(x,c) + λ(Ax - b)
             x   >= 0

with an accelerated proximal distance algorithm. Here the affine constraints of the linear program are moved into the objective function.
The vector `λ` represents the Lagrange multiplier. `linprog` uses the conjugate gradient method to solve for the update.
"""
function lin_prog(
    A        :: SparseMatrixCSC{T,Int},
    b        :: DenseVector{T},
    c        :: DenseVector{T};
    rho      :: T    = 1e-2,
    rho_inc  :: T    = 2.0,
    rho_max  :: T    = 1e30,
    max_iter :: Int  = 10000,
    inc_step :: Int  = 5,
    tol      :: T    = 1e-6,
    afftol   :: T    = 1e-6,
    nnegtol  :: T    = 1e-6,
    quiet    :: Bool = true,
) where {T <: AbstractFloat}

    # error checking
	check_conformability(A, b, c)

    iter    = 0
    loss0   = Inf
    dnonneg = Inf
    invrho  = one(T) / rho
    At      = A'

    (p,q) = size(A)
    y2    = zeros(T, p)
    yp    = zeros(T, p)
    x     = zeros(T, q)
    y     = zeros(T, q)
    yq    = zeros(T, q)
    z     = zeros(T, q)
    v2    = zeros(T, q)
    shift = zeros(T, q)
    z_max = max.(y, zero(T))

    # compute initial loss function
    loss = dot(c,x)

    # compute the shift: A' * (A * A') \ b using CG
    #Afun = MatrixFcn{T}(p, p, (output, v) -> mulbyA!(output, v, A, At, v2))
    f(output, v) = mulbyA!(output, v, A, At, v2)
    Afun = LinearMap{T}(f, p, p, ismutating = true) 
    cg!(yp, Afun, b, maxiter=200, tol=1e-8)
    shift .= At * yp

    i = 0
    for i = 1:max_iter


        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max, z)

        # print progress of algorithm
        print_progress(i, loss, dnonneg, rho, quiet, i_interval = 10, inc_step = inc_step)

        isfinite(loss) || throw(error("Loss is not finite after $i iterations, something is wrong..."))

        ### LSQR solve ###
        copy!(yq, z_max)
        BLAS.axpy!(-invrho, c, yq)
        lsqr!(yp, At, yq, maxiter=200, atol=1e-8, btol=1e-8)
        A_mul_B!(z, At, yp)
        copy!(y, shift)
        BLAS.axpy!(one(T), yq, y)
        BLAS.axpy!(-one(T), z, y)

        # compute distances to constraint sets
        dnonneg = euclidean(y,z_max)

        # convergence checks
        loss        = dot(c,y)
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
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => sparsevec(y), "nonneg_dist" => dnonneg)
end
