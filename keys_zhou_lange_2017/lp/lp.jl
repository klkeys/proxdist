using Distances
using RegressionTools
using JuMP
using Gurobi
#using ProxOpt
using IterativeSolvers


###################
### subroutines ###
###################


# function handle to efficiently compute sparse matrix-vector operation in CG
function mulbyA!(output, v, A, At, v2)
    A_mul_B!(v2,At,v)
    A_mul_B!(output, A, v2)
end

# function to compute a Nesterov acceleration step at iteration i
function compute_accelerated_step!(z, x, y, i)
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
    if (p,q) != size(A) 
		throw(DimensionMismatch("nonconformable A, b, and c\nsize(A) = $(size(A))\nsize(b) = $(size(b))\nsize(c) = $(size(c))"))
	end
	return()
end

"""
    project_affine!(x,y,C,d)

Perform the affine projection `x = Cy + d`, overwriting `x`.
The matrix `C` and the vector `d` are calculated as `C = pinv(A)(A + I)` and `d = pinv(A)*b`,
where `Ay == b`.

Arguments:

-- `x` is the vector to overwrite in the projection.
-- `y` is the vector to project.
-- `C` is the projection matrix.
-- `d` is the affine offset of the projection.
"""
function project_affine!(
    x :: DenseVector{T},
    y :: DenseVector{T},
    C :: DenseMatrix{T},
    d :: DenseVector{T}
) where {T} <: AbstractFloat
    BLAS.gemv!('N', one(T), C, y, zero(T), x)
    BLAS.axpy!(length(x), one(T), d, 1, x, 1)
end

"""
    project_affine(v,A,b) -> x, C, d

`project_affine`computes the projection of `v` onto the affine constraints `A*v == b` as

    x = v - pinv(A)*(A*v - b)

To facilitate future projections, `project_affine` returns, in addition to `x`, the matrix `C = I - pinv(A)*A` and `d = pinv(A)*b`.
"""
function project_affine(
    v :: DenseVector{T},
    A :: DenseMatrix{T},
    b :: DenseVector{T}
) where {T} <: AbstractFloat # may need BlasFloat?

    # initialize output vector
    x = zeros(T, size(v))

    # x = v - pinv(A)*(A*v - b)
    pA = pinv(A)
    C  = BLAS.gemm('N', 'N', -1.0, pA, A)
    C  += I
    d  = BLAS.gemv('N', 1.0, pA, b)

    # x = C*v + d
    project_affine!(x,v,C,d)

    # return affine projection with cached projection matrix, vector
    return x, C, d
end



"""
    project_affine(v::SparseMatrixCSC, A::SparseMatrixCSC, b::SparseMatrixCSC)

For a sparse matrix `A` and sparse vectors `v` and `b`, `project_affine` computes the projection of `v` onto the affine constraints `A*v == b`.
"""
function project_affine(
    v  :: SparseMatrixCSC{T,Int},
    A  :: SparseMatrixCSC{T,Int},
    b  :: SparseMatrixCSC{T,Int},
    AA :: SparseMatrixCSC{T,Int}
) where {T} <: AbstractFloat
    return v + A' * ( AA \ (b - A*v) )
end

function project_affine(
    v  :: SparseMatrixCSC{T,Int},
    A  :: SparseMatrixCSC{T,Int},
    b  :: SparseMatrixCSC{T,Int}
)
	AA = A*A'
	project_affine(v, A, b, AA)
end

function project_affine(
    v    :: SparseMatrixCSC{T,Int},
    A    :: SparseMatrixCSC{T,Int},
    b    :: SparseMatrixCSC{T,Int},
    AA   :: Base.SparseMatrix.CHOLMOD.Factor{T},
    At   :: SparseMatrixCSC{T,Int},
) where {T} <: AbstractFloat
#    return v + A' * ( AA \ (b - A*v) )
    x = b - A*v
    x = AA \ x
    y = At * x
    y = y + v
    return y
end

function project_affine(
    v  :: SparseMatrixCSC{T,Int},
    A  :: SparseMatrixCSC{T,Int},
    b  :: Union{DenseVector{T}, SparseMatrixCSC{T,Int}},
    AA :: Base.SparseMatrix.CHOLMOD.Factor{T} 
) where {T} <: AbstractFloat
    C = I - (A' * ( AA \ A)) 
    d = A' * (AA \ b)
	# try caching b as last column, computing AA \ [A b], and then recovering d = A'(AA\b) 
    return C*v + d, C, d 
end

function project_nonneg!(x :: AbstractVector{T}, y :: AbstractVector{T}) where {T} <: AbstractFloat
	y .= max.(x, 0)
	return y
end


######################
### main functions ###
######################

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
) where {T} <: AbstractFloat

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
    x = zeros(T,q)
    y = zeros(T,q)
    z = zeros(T,q)

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
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", i, loss, daffine, dnonneg, rho)
        end

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
    return Dict{ASCIIString, Any}("obj" => loss, "iter" => iter, "x" => copy(y), "affine_dist" => daffine, "nonneg_dist" => dnonneg)
end


"""
    lin_prog(A,b,c)

Solve the optimization problem

    minimize dot(x,c) + λ(Ax - b)
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
) where {T} <: AbstractFloat

    # error checking
	check_conformability(A, b, c)

    iter    = 0
    loss    = dot(c,x)
    loss0   = Inf
    daffine = Inf
    dnonneg = Inf
    ρ_inv   = one(T) / rho

	# initialize temporary arrays
    x  = zeros(T,q)
    y  = zeros(T,q)
    y2 = zeros(T,q)
    z  = zeros(T,q)

	# perform initial affine projection
    pA = pinv(A)
    C  = BLAS.gemm('N', 'N', -one(T), pA, A)
    C  += I
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
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dnonneg, rho)
        end

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
    return Dict{ASCIIString, Any}("obj" => loss, "iter" => iter, "x" => copy(y), "nonneg_dist" => dnonneg)
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
)

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
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dnonneg, rho)
        end

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
    return Dict{ASCIIString, Any}("obj" => loss, "iter" => i, "x" => sparsevec(y), "nonneg_dist" => dnonneg)
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
) where {T} <: AbstractFloat

    # error checking
	check_conformability(A, b, c)

    iter    = 0
    loss0   = Inf
    dnonneg = Inf
    invrho  = one(T) / rho
    At      = A'

    x     = zeros(T, q)
    y     = zeros(T, q)
    y2    = zeros(T, p)
    yp    = zeros(T, p)
    yq    = zeros(T, q)
    z     = zeros(T, q)
    v2    = zeros(T, q)
    z_max = max.(y, zero(T))

    # compute initial loss function
    loss = dot(c,x)

    # compute the shift: A' * (A * A') \ b using CG
    Afun = MatrixFcn{T}(p, p, (output, v) -> mulbyA!(output, v, A, At, v2))
    cg!(yp, Afun, b, maxiter=200, tol=1e-8)
    shift .= At * yp

    i = 0
    for i = 1:max_iter


        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max, z)

        # print progress of algorithm
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dnonneg, rho)
        end

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
    return Dict{ASCIIString, Any}("obj" => loss, "iter" => i, "x" => sparsevec(y), "nonneg_dist" => dnonneg)
end


#"""
#    lin_prog(A::SparseMatrixCSC, b::SparseMatrixCSC, c::SparseMatrixCSC)
#
#For sparse matrix `A` and sparse vectors `b` and `c`, solve the optimization problem
#
#    minimize dot(x,c)
#    s.t.     A*x == b
#             x   >= 0
#
#with an accelerated proximal distance algorithm.
#"""
#function lin_prog(
#    A        :: SparseMatrixCSC{Float64,Int},
#    b        :: SparseMatrixCSC{Float64,Int},
#    c        :: SparseMatrixCSC{Float64,Int};
#    p        :: Int     = length(b),
#    q        :: Int     = length(c),
#    rho      :: Float64 = one(Float64),
#    rho_inc  :: Float64 = 2.0,
#    rho_max  :: Float64 = 1e15,
#    max_iter :: Int     = 10000,
#    inc_step :: Int     = 100,
#    tol      :: Float64 = 1e-6,
#    afftol   :: Float64 = 1e-6,
#    nnegtol  :: Float64 = 1e-6,
#    quiet    :: Bool    = true,
#    x        :: SparseMatrixCSC{Float64,Int} = spzeros(Float64,q,1),
#    y        :: SparseMatrixCSC{Float64,Int} = spzeros(Float64,q,1),
#    z        :: SparseMatrixCSC{Float64,Int} = spzeros(Float64,q,1),
#    At       :: SparseMatrixCSC{Float64,Int} = A'
#)
#
#    # error checking
#    (p,q) == size(A) || throw(DimensionMismatch("nonconformable A, b, and c"))
#
#    iter    = 0
#    loss    = vecdot(c,x)
#    loss0   = Inf
#    daffine = Inf
#    dnonneg = Inf
#    invrho  = one(Float64) / rho
#    AA      = cholfact(A * At)
#
#    z_affine = project_affine(z,A,b,AA,At)
#    z_max    = project_nonneg(y)
#
#    for i = 1:max_iter
#
#        iter += 1
#
#        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
#        kx = (i - one(Float64)) / (i + one(Float64) + one(Float64))
#        ky = one(Float64) + kx
##        difference!(z,y,x, a=ky, b=kx, n=q)
#        z = ky*y - kx*x
#        copy!(x,y)
#
#        # compute projections onto constraint sets
#        z_max    = project_nonneg(z)
#        z_affine = project_affine(z,A,b,AA,At)
#
#        # compute distances to constraint sets
#        daffine = norm(z - z_affine)
#        dnonneg = norm(z - z_max)
#
#        # print progress of algorithm
#        if (i <= 10 || i % inc_step == 0) && !quiet
#            @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", i, loss, daffine, dnonneg, rho)
#        end
#
#        # prox dist update y = 0.5*(z_max + z_affine) - c/rho
##        axbycz!(y, 0.5, z_max, 0.5, z_affine, -invrho, c)
#        #y = 0.5*(z_max + z_affine) - invrho*c
#        y .= 0.5.*(z_max .+ z_affine) .- invrho .* c
#
#        # convergence checks
#        loss        = vecdot(c,y)
#        nonneg      = dnonneg < nnegtol
#        affine      = daffine < afftol
#        the_norm    = norm(x - y)
#        scaled_norm = the_norm / (norm(x,2) + one(Float64))
#        converged   = scaled_norm < tol && nonneg && affine
#
#        # if converged then break, else save loss and continue
#        converged && break
#        loss0 = loss
#
#        if i % inc_step == 0
#            rho    = min(rho_inc*rho, rho_max)
#            invrho = one(Float64) / rho
#            copy!(x,y)
#        end
#    end
#
#    # threshold small elements of y before returning
#    w = threshold(y,tol)
#    return Dict{ASCIIString, Any}("obj" => loss, "iter" => iter, "x" => w, "affine_dist" => daffine, "nonneg_dist" => dnonneg)
#end


### DOES NOT WORK
### cannot use SPQR to compute A'*(A*A') \ A
function lin_prog3(
    A        :: SparseMatrixCSC{Float64,Int},
    b        :: DenseVector{Float64},
    c        :: DenseVector{Float64};
    p        :: Int     = length(b),
    q        :: Int     = length(c),
    rho      :: Float64 = one(Float64),
    rho_inc  :: Float64 = 2.0,
    rho_max  :: Float64 = 1e15,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    tol      :: Float64 = 1e-6,
    afftol   :: Float64 = 1e-6,
    nnegtol  :: Float64 = 1e-6,
    quiet    :: Bool    = true,
)

    # error checking
    (p,q) == size(A) || throw(DimensionMismatch("nonconformable A, b, and c"))

    iter    = 0
    loss0   = Inf
    dnonneg = Inf
    invrho  = one(Float64) / rho
    At      = A'
    AA      = qrfact(At)

    x     = zeros(Float64, q)
    y     = zeros(Float64, q)
    y2    = zeros(Float64, q)
    z     = zeros(Float64, q)
    z_max = max(y, zero(Float64))
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
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dnonneg, rho)
        end

        # prox dist update y = C*(z_max - invrho*c) + d
        copy!(y2,z_max)
        BLAS.axpy!(q, -invrho, c, 1, y2, 1)
        copy!(y,d)
        BLAS.symv!('u', one(Float64), C, y2, one(Float64), y)

        # convergence checks
        loss        = dot(c,y)
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
    return Dict{ASCIIString, Any}("obj" => loss, "iter" => i, "x" => sparsevec(y), "nonneg_dist" => dnonneg)
end




# simple JuMP configuration for solving LP with Gurobi solver
function lp_gurobi(
    A       :: DenseMatrix{Float64},
    b       :: DenseVector{Float64},
    c       :: DenseVector{Float64};
    opttol  :: Float64 = 1e-6,
    feastol :: Float64 = 1e-6,
    quiet   :: Bool    = true
)

    # problem dimensions?
    (m,n) = size(A)

    # instantiate a model with a specified solver
    lp = Model(solver = GurobiSolver(OutputFlag=0, OptimalityTol=opttol, FeasibilityTol = feastol))

    # define nonnegative variable
    @variable(lp, x[1:n] >= 0)

    # set objective as minimization of componentwise sum (e.g. dot(c,x))
    @setObjective(lp, Min, sum(c[j]*x[j], j=1:n))

    # must set affine constraints row-by-row
    for i = 1:m
        @addConstraint(lp, sum(A[i,j]*x[j], j=1:n) == b[i])
    end

    # solve model and get variable
    status = solve(lp)
    z      = getValue(x)

    # output
    !quiet && begin
        println("\n==== Gurobi results ====")
        println("Status of model: ", status)
        println("Optimum: ", getObjectiveValue(lp))
        println("Distance to affine set? ", norm(A*z - b))
        println("Distance to nonnegative set? ", norm(z - max(z,0)))
        println("\n")
    end

    return z
end

# direct use of Gurobi solver through Gurobi.jl
function lp_gurobi(
    A       :: SparseMatrixCSC{Float64,Int},
    b       :: DenseVector{Float64},
    c       :: DenseVector{Float64};
    opttol  :: Float64 = 1e-6,
    feastol :: Float64 = 1e-6,
    quiet   :: Bool    = true
)
    # construct Gurobi environment
    env = Gurobi.Env()

    # set parameters
    setparam!(env, "OptimalityTol", opttol)
    setparam!(env, "FeasibilityTol", feastol)
#    quiet && setparam!(env, "OutputFlag", 0)
    setparam!(env, "OutputFlag", 1)
    setparam!(env, "Method", 2)
    setparam!(env, "BarConvTol", opttol)

    # solve model and get variable
#    model = gurobi_model(env, f=c, Aeq=A, beq=b, lb=spzeros(size(c,1),1))
#    model = gurobi_model(env, f=vec(full(c)), Aeq=A, beq=vec(full(b)), lb=zeros(size(c,1)))
#    model = gurobi_model(env, f=c, Aeq=A, beq=b, lb=0.0)
    model = Gurobi.Model(env,"sparse_lp")
    set_sense!(model,:minimize)
    add_cvars!(model, c, 0.0, Inf)
    update_model!(model)
    add_constrs!(model, A, '=', b)
    update_model!(model)
    optimize(model)
    z = get_solution(model)

    # output
    !quiet && begin
        println("\n==== Gurobi results ====")
        println("Status of model: ", get_status(model))
        println("Optimum: ", get_objval(model))
        println("Distance to affine set? ", norm(A*z - b))
        println("Distance to nonnegative set? ", norm(z - max(z,0.0)))
        println("\n")
    end

    return z
end

####################
### testing code ###
####################

function test_dense_lp(;
    quiet :: Bool = true
)
    # set random seed for reproducibility
    seed = 2016
    srand(seed)

    # set BLAS threads to # of cores
    blas_set_num_threads(8)

    # initialize parameters
    m = 1000
    n = 5000
    A = randn(m,n)
    c = rand(n)
    xinitial = max(randn(n),0.0)
    b = A*xinitial + rand(m)

    # run accelerated LP
    output = lin_prog2(A,b,c, inc_step = 100, rho_inc = 2.0, quiet=true)
    @time output = lin_prog2(A,b,c, inc_step = 100, rho_inc = 2.0, quiet=true)
    y = copy(output["x"])
    println("\n\n==== Accelerated Prox Dist Results ====")
    println("Iterations: ", output["iter"])
    println("Optimum: ", output["obj"])
    println("Distance to affine set? ", norm(A*y - b))
    println("Distance to nonnegative set? ", euclidean(y,max(y,0)))
    println("\n")

    # run accelerated LP with affine constraints in objective
    output = lin_prog(A,b,c, inc_step = 100, rho_inc = 2.0, quiet=true)
    @time output = lin_prog(A,b,c, inc_step = 100, rho_inc = 2.0, quiet=true)
    y = copy(output["x"])
    println("\n\n==== Accelerated Affine Prox Dist Results ====")
    println("Iterations: ", output["iter"])
    println("Optimum: ", output["obj"])
    println("Distance to affine set? ", euclidean(A*y,b))
    println("Distance to nonnegative set? ", euclidean(y,max(y,0)))
    println("\n")

    # run unaccelerated LP
    output2 = lp2(A,b,c, inc_step = 100, rho_inc = 2.0, quiet=true)
    @time output2 = lp2(A,b,c, inc_step = 100, rho_inc = 2.0, quiet=true)
    y2 = copy(output2["x"])
    println("\n\n==== Prox Dist Results ====")
    println("Iterations: ", output2["iter"])
    println("Optimum: ", output2["obj"])
    println("Distance to affine set? ", norm(A*y2 - b))
    println("Distance to nonnegative set? ", euclidean(y2,max(y2,0)))
    println("\n")

    # run Gurobi
    z = lp_gurobi(A,b,c)
    @time z = lp_gurobi(A,b,c,quiet=false)

    println("Distance between lin_prog, Gurobi opt variables: ", norm(y - z))
    println("Distance between lp, Gurobi opt variables: ", norm(y2 - z))

    return nothing
end

function test_sparse_lp()

    # set random seed for reproducibility
    seed = 2016
    srand(seed)

    # set number of BLAS threads
    blas_set_num_threads(8)

    # initialize parameters
    quiet    = true
    quiet    = false
    inc_step = 10
    rho_inc  = 1.5
    rho_max  = 1e30
    rho      = 1.0
    m        = 2048
    n        = 2*m
#    s        = 0.01
    s        = 2*log10(m)/m
    A        = sprandn(m,n,s)
    c        = rand(n)
#    xinitial = sprand(n,1,s)
    xinitial = rand(n)
#    b = A*xinitial + sprand(m,1,s)
    b        = vec(A*xinitial)

    println("how sparse is A? ", countnz(A) / prod(size(A)))
    AA = A*A';
    println("how sparse is A*A'? ", countnz(AA) / prod(size(AA)))
    @show cond(full(AA))

    # run accelerated LP with affine objective function
    y = 0
    y2 = 0
    try
        output = lin_prog(A,b,c, inc_step = inc_step, rho_inc = rho_inc, quiet=quiet, rho=rho, rho_max=rho_max)
        @time output = lin_prog(A,b,c, inc_step = inc_step, rho_inc = rho_inc, quiet=quiet, rho=rho, rho_max=rho_max)
        y = copy(output["x"])

        println("\n\n==== Accelerated, Affine, CG ====")
        println("Iterations: ", output["iter"])
        println("Optimum: ", output["obj"])
        println("Distance to affine set? ", norm(A*y - b))
        println("Distance to nonnegative set? ", norm(y - max(y,0)))
        println("\n")

    catch e
        warn("problem with linprog: ", e)
    end

    # run accelerated LP
    try
        output2 = lin_prog2(A,b,c, inc_step = inc_step, rho_inc = rho_inc, quiet=quiet, rho_max=rho_max, rho=rho)
        @time output2 = lin_prog2(A,b,c, inc_step = inc_step, rho_inc = rho_inc, quiet=quiet, rho_max=rho_max, rho=rho)
        y2 = copy(output2["x"])

        println("\n\n==== Accelerated, Affine, Cholesky ====")
        println("Iterations: ", output["iter"])
        println("Optimum: ", output["obj"])
        println("Distance to affine set? ", euclidean(A*y, b))
        println("Distance to nonnegative set? ", euclidean(y, max(y,0)))
        println("\n")

    catch e
        warn("problem with linprog2: ", e)
    end

    # run Gurobi
    z = lp_gurobi(A,b,c,quiet=quiet)
    @time z = lp_gurobi(A,b,c,quiet=false)


    println("Distance between lin_prog2, Gurobi opt variables: ", norm(y2 - z))
    println("Distance between lin_prog, Gurobi opt variables: ", norm(y - z))

    return nothing
end

function profile_sparse_lp(
    reps     :: Int = 1000;
    inc_step :: Int = 100,
    rho_inc  :: Float64 = 2.0,
    rho_max  :: Float64 = 1e15,
)

    # set random seed for reproducibility
    seed = 2016
    srand(seed)

    # initialize parameters
    m = 500
    n = 1000
    s = 0.1
    A = sprandn(m,n,s)
    c = sprand(n,1,s)
    xinitial = project_nonneg(sprandn(n,1,s))
    b = A*xinitial

    # clear buffer before beginning
    Profile.clear()

    # set profiling parameters
    Profile.init(delay = 0.1)

    # profile accelerated LP
    @profile begin
        for i = 1:reps
            output = lin_prog(A,b,c, inc_step = inc_step, rho_inc = rho_inc, quiet=true)
        end
    end

    # dump results to console
    println("Profiling results:")
    Profile.print()

    return nothing
end

#println("--- testing dense model ---")
#test_dense_lp()
#println("--- testing sparse model ---")
#test_sparse_lp()
#println("--- profiling sparse model ---")
#profile_sparse_lp(100)
