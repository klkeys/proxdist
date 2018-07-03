using Distances
using JuMP
using Gurobi
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
include("../common/common.jl")

# function handle to efficiently compute sparse matrix-vector operation in CG
function mulbyA!(output, v, A, At, v2)
    A_mul_B!(v2,At,v)
    A_mul_B!(output, A, v2)
end


# ==============================================================================
# main functions
# ==============================================================================



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
    b        :: Vector{T},
    c        :: Vector{T};
    rho      :: T    = one(T),
    rho_inc  :: T    = one(T) + one(T),
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
    @assert rho >  zero(T) "Argument rho must be positive"

	# initialize temporary arrays
    q  = size(A,2)
    x  = zeros(T, q)
    y  = zeros(T, q)
    y2 = zeros(T, q)
    z  = zeros(T, q)

    # initialize algorithm parameters
    i         = 0
    loss      = dot(c,x)
    loss0     = Inf
    dnonneg   = Inf
    dnonneg0  = Inf
    ρ_inv     = one(T) / rho
    converged = false
    stuck     = false

	# perform initial affine projection
    z_affine, C, d = project_affine(x, A, b)
    z_max = max.(z, 0)

    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max, z)

        # compute distances to constraint sets
        dnonneg = euclidean(y,z_max)

        # print progress of algorithm
        quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)

        # prox dist update
        copy!(y2, z_max)
        BLAS.axpy!(q, -ρ_inv, c, 1, y2, 1)
        copy!(y,d)
        BLAS.symv!('u', one(T), C, y2, one(T), y)

        # convergence checks
        loss        = dot(c,y)
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
    b        :: Vector{T},
    c        :: Vector{T};
    rho      :: T    = 1e-2,
    rho_inc  :: T    = one(T) + one(T),
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
    ρ_inv = one(T) / rho 
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
    # exploit matrix-free operations with a LinearMap
    # requires a (mutating) function of form f(output, input)
    # only the output is mutated in the LinearMap
    f(output, v) = mulbyA!(output, v, A, At, v2)
    Afun = LinearMap{T}(f, p, p, ismutating = true) 
    cg!(yp, Afun, b, maxiter=200, tol=1e-8)
    shift .= At * yp

#    # configure a LinearMap for the inner LSQR solve
#    g(output, v) = A_mul_B!(output, At, v)
#    gt(output, v) = At_mul_B!(output, At, v)
#    Atfun = LinearMap{T}(g, gt, q, p, ismutating = true)
    for i = 1:max_iter


        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

        # compute projections onto constraint sets
        i > 1 && project_nonneg!(z_max, z)

        # print progress of algorithm
        quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)

        # check that loss is still finite
        # in contrary case, throw error
        @assert isfinite(loss) "Loss is no longer finite after $i iterations, something is wrong..."

        ### LSQR solve ###
        copy!(yq, z_max)
        BLAS.axpy!(-ρ_inv,  c, yq)
        lsqr!(yp, At, yq, maxiter=200, atol=1e-8, btol=1e-8)
#        lsqr!(yp, Atfun, yq, maxiter=200, atol=1e-8, btol=1e-8)
        A_mul_B!(z, At, yp)
        copy!(y, shift)
        BLAS.axpy!(one(T), yq, y)
        BLAS.axpy!(-one(T), z, y)

        # recompute loss
        loss = dot(c,y)

        # compute distances to constraint sets
        dnonneg0 = dnonneg
        dnonneg  = euclidean(y,z_max)

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
            rho    = min(rho_inc*rho, rho_max)
            ρ_inv  = one(T) / rho
            copy!(x,y)
        end
    end

    # threshold small elements of y before returning
    threshold!(y,tol)
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => copy(y), "nonneg_dist" => dnonneg, "converged" => converged, "stuck" => stuck)
end



#"""
#    lin_prog(A,b,c)
#
#Solve the optimization problem
#
#    minimize dot(x,c)
#    s.t.     A*x == b
#             x   >= 0
#
#with an accelerated proximal distance algorithm.
#"""
#function lin_prog2(
#    A        :: DenseMatrix{T},
#    b        :: Vector{T},
#    c        :: Vector{T};
#    rho      :: T    = one(T),
#    rho_inc  :: T    = one(T) + one(T),
#    rho_max  :: T    = 1e15,
#    max_iter :: Int  = 10000,
#    inc_step :: Int  = 100,
#    tol      :: T    = 1e-6,
#    afftol   :: T    = 1e-6,
#    nnegtol  :: T    = 1e-6,
#    quiet    :: Bool = true
#) where {T <: AbstractFloat}
#
#    # error checking
#	check_conformability(A, b, c)
#    @assert rho >  zero(T) "Argument rho must be positive"
#
#    # declare algorithm variables
#    loss0     = Inf
#    daffine   = Inf
#    daffine0  = Inf
#    dnonneg   = Inf
#    dnonneg0  = Inf
#    ρ_inv     = one(T) / rho
#    HALF      = one(T) / 2
#    converged = false
#    stuck     = false
#    
#    # initialize temporary arrays
#    p,q = size(A)
#    x = zeros(T, q)
#    y = zeros(T, q)
#    z = zeros(T, q)
#
#    # apply initial projections
#    z_affine, C, d = project_affine(z,A,b)
#    z_max = max.(z, 0) 
#
#    # compute initial loss
#    loss = dot(c,x)
#
#    # main loop
#    i = 0
#    for i = 1:max_iter
#
#        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
#		compute_accelerated_step!(z, x, y, i)
#
#        # compute projections onto constraint sets
#        i > 1 && project_nonneg!(z_max, z)
#        i > 1 && project_affine!(z_affine, z, C, d)
#
#        # compute distances to constraint sets
#        daffine = euclidean(z, z_affine)
#        dnonneg = euclidean(z, z_max)
#
#        # prox dist update y = 0.5*(z_max + z_affine) - c/rho
#        y   .= HALF .* z_max .+ HALF .* z_affine .- ρ_inv .* c
#        loss = dot(c,y)
#
#        # print progress of algorithm
#        quiet || print_progress(i, loss, daffine, dnonneg, rho, i_interval = 10, inc_step = inc_step)
#
#        # convergence checks
#        nonneg      = dnonneg < nnegtol
#        diffnonneg  = abs(dnonneg - dnonneg0)
#        affine      = daffine < afftol
#        diffaffine  = abs(daffine - daffine0)
#        the_norm    = euclidean(x,y)
#        scaled_norm = the_norm / (norm(x,2) + one(T))
#        converged   = scaled_norm < tol && nonneg && affine
#        stuck       = !converged && (scaled_norm < tol) && (diffnonneg < tol) && (diffaffine < tol)
#
#        # if converged then break, else save loss and continue
#        if converged || (stuck && rho >= rho_max)
#            quiet || print_progress(i, loss, dnonneg, rho, i_interval = max_iter, inc_step = inc_step)
#            break
#        end
#        loss0 = loss
#
#        # update penalty constant if necessary
#        if i % inc_step == 0 || diffnonneg < tol || diffaffine < tol || stuck
#            rho    = min(rho_inc*rho, rho_max)
#            ρ_inv = one(T) / rho
#            copy!(x,y)
#        end
#    end
#
#    # threshold small elements of y before returning
#    threshold!(y,tol)
#    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => copy(y), "nonneg_dist" => dnonneg, "affine_dist" => daffine, "converged" => converged, "stuck" => stuck)
#end
#
#"""
#    lin_prog(A::SparseMatrixCSC, b, c)
#
#For sparse matrix `A` and dense vectors `b` and `c`, solve the optimization problem
#
#    minimize dot(x,c) + λ(Ax - b)
#             x   >= 0
#
#with an accelerated proximal distance algorithm. Here the affine constraints of the linear program are moved into the objective function.
#The vector `λ` represents the Lagrange multiplier. `linprog` factorizes `A` to obtain a suitable proxy for the pseudoinverse of `A`.
#"""
#function lin_prog2(
#    A        :: SparseMatrixCSC{T,Int},
#    b        :: Vector{T},
#    c        :: Vector{T};
#    rho      :: T    = one(T),
#    rho_inc  :: T    = one(T) + one(T),
#    rho_max  :: T    = 1e30,
#    max_iter :: Int  = 10000,
#    inc_step :: Int  = 100,
#    tol      :: T    = 1e-6,
#    afftol   :: T    = 1e-6,
#    nnegtol  :: T    = 1e-6,
#    quiet    :: Bool = true,
#) where {T <: AbstractFloat}
#
#    # error checking
#	check_conformability(A, b, c)
#    @assert rho >  zero(T) "Argument rho must be positive"
#
#    # initialize return values 
#    i         = 0
#    loss      = Inf
#    loss0     = Inf
#    dnonneg   = Inf
#    dnonneg0  = Inf
#    converged = false
#    stuck     = false
#
#    # initialize arrays and algorithm variables
#    ρ_inv  = one(T) / rho
#    At     = A'
#    AA     = cholfact(A * At)
##    AA      = factorize(A * At)
#    (p,q) = size(A)
#    x     = zeros(T, q)
#    y     = zeros(T, q)
#    y2    = zeros(T, q)
#    z     = zeros(T, q)
#    z_max = max.(y, 0) 
#    C     = full(I - (At * ( AA \ A)))
#    d     = vec(full(At * (AA \ b)))
#
#    for i = 1:max_iter
#
#        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
#		compute_accelerated_step!(z, x, y, i)
#
#        # compute projections onto constraint sets
#        i > 1 && project_nonneg!(z_max, z)
#
#        # compute distances to constraint sets
#        dnonneg0 = dnonneg
#        dnonneg  = euclidean(z,z_max)
#
#        # print progress of algorithm
#        quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)
#
#        # prox dist update y = C*(z_max - ρ_inv*c) + d
#        copy!(y2,z_max)
#        BLAS.axpy!(q, -ρ_inv, c, 1, y2, 1)
#        copy!(y,d)
#        BLAS.symv!('u', one(T), C, y2, one(T), y)
#        loss        = dot(c,y)
#        @assert isfinite(loss) "Loss is not finite after $i iterations, something is wrong..."
#
#        # convergence checks
#        nonneg      = dnonneg < nnegtol
#        diffnonneg  = abs(dnonneg - dnonneg0)
#        the_norm    = euclidean(x,y)
#        scaled_norm = the_norm / (norm(x,2) + one(T))
#        converged   = scaled_norm < tol && nonneg
#        stuck       = !converged && (scaled_norm < tol) && (diffnonneg < tol)
#
#        # if converged then break, else save loss and continue
#        # also abort if the algo gets stuck
#        if converged || (stuck && rho >= rho_max)
#            quiet || print_progress(i, loss, dnonneg, rho, i_interval = 10, inc_step = inc_step)
#            break
#        end
#        loss0 = loss
#
#        if i % inc_step == 0 || diffnonneg < nnegtol || stuck
#            rho   = min(rho_inc*rho, rho_max)
#            ρ_inv = one(T) / rho
#            copy!(x,y)
#        end
#    end
#
#    # threshold small elements of y before returning
#    threshold!(y,tol)
#    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => copy(y), "nonneg_dist" => dnonneg, "converged" => converged, "stuck" => stuck)
#end


 
"""
    lin_prog3(A, b, c)

For dense matrix `A` and dense vectors `b` and `c`, solve the optimization problem

    minimize dot(x,c)
    s.t.     A*x == b
             x   >= 0

with an accelerated proximal distance algorithm. The nonnegative constraint `x >= 0` is folded into the function domain.
`lin_prog3` enforces the affine constraint `A*x == b` with a standard affine projection.
"""
function lin_prog3(
	A        :: DenseMatrix{T},
	b        :: Vector{T},
	c        :: Vector{T};
    rho      :: T    = one(T),
    rho_inc  :: T    = one(T) + one(T),
    rho_max  :: T    = 1e30,
    max_iter :: Int  = 10000,
    inc_step :: Int  = 100,
    tol      :: T    = 1e-6,
    afftol   :: T    = 1e-6,
    quiet    :: Bool = true,
) where {T <: AbstractFloat} 

    # error checking
	check_conformability(A, b, c)
    @assert rho >  zero(T) "Argument rho must be positive"

    # initialize return values
    i         = 0
    loss      = Inf
    loss0     = Inf
    daffine   = Inf
    daffine   = Inf
    converged = false
    stuck     = false

    # initialize arrays and algorithm variables
    ρ_inv   = one(T) / rho
	(p,q)   = size(A)
	x       = zeros(q)
	y       = zeros(q)
	z       = zeros(q)

	# compute initial affine projection
    z_affine, C, d = project_affine(z,A,b)

	for i = 1:max_iter
		
        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

		# project onto the constraint set
        i > 1 && project_affine!(z_affine, z, C, d)
        daffine0 = daffine
		daffine  = euclidean(z, z_affine) 

		# calculate the proximal distance update
		y .= max.(z_affine .- ρ_inv .* c , 0)
        loss = dot(c,y)

        # print progress of algorithm
        quiet || print_progress(i, loss, daffine, rho, i_interval = 10, inc_step = inc_step)

		# check for convergence
        affine      = daffine < afftol 
        diffaffine  = abs(daffine - daffine0)
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x) + one(T))
        converged   = (scaled_norm < tol) && affine 
        stuck       = !converged && (scaled_norm < tol) && (diffaffine < tol)

        # if converged then break, else save loss and continue
        # also abort if the algo gets stuck
        if converged || (stuck && rho >= rho_max)
            quiet || print_progress(i, loss, daffine, rho, i_interval = max_iter, inc_step = inc_step)
            break
        end
        loss0 = loss

        if i % inc_step == 0 || diffaffine < afftol || stuck
            # update penalty constant
			rho = min(rho_inc * rho, rho_max)
			ρ_inv = one(T) / rho 
			copy!(x,y)
		end 
	end 

    threshold!(y, tol)
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => copy(y), "affine_dist" => daffine, "converged" => converged, "stuck" => stuck)
end


function lin_prog3(
	A        :: SparseMatrixCSC{T,Int},
	b        :: Vector{T},
	c        :: Vector{T};
    rho      :: T    = one(T),
    rho_inc  :: T    = one(T) + one(T),
    rho_max  :: T    = 1e30,
    max_iter :: Int  = 10000,
    inc_step :: Int  = 100,
    tol      :: T    = 1e-6,
    afftol   :: T    = 1e-6,
    quiet    :: Bool = true,
) where {T <: AbstractFloat} 

    # error checking
	check_conformability(A, b, c)
    @assert rho >  zero(T) "Argument rho must be positive"

    # initialize return values
    i         = 0
    loss      = Inf
    loss0     = Inf
    daffine   = Inf
    daffine   = Inf
    converged = false
    stuck     = false

    # initialize arrays and algorithm variables
    ρ_inv   = one(T) / rho
	(p,q)   = size(A)
	x       = zeros(q)
	y       = zeros(q)
	z       = zeros(q)

	# compute initial affine projection
#    #z_affine, C, d = project_affine(z,A,b)
#    At = A'
#    AA = cholfact(A * At)
#    C  = full(I - (At * ( AA \ A)))
#    d  = vec(full(At * (AA \ b)))
#    z_affine = copy(d)  ## z_affine = C*0 + d
    At = A'
    z_affine, C, d = project_affine(z,At,b) ## C is a factorization!

	for i = 1:max_iter
		
        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
		compute_accelerated_step!(z, x, y, i)

		# project onto the constraint set
        #i > 1 && project_affine!(z_affine, z, C, d)
        i > 1 && project_affine!(z_affine, z, At, C, d)
        daffine0 = daffine
		daffine  = euclidean(z, z_affine) 

		# calculate the proximal distance update
		y .= max.(z_affine .- ρ_inv .* c , 0)
        loss = dot(c,y)

        # print progress of algorithm
        quiet || print_progress(i, loss, daffine, rho, i_interval = 10, inc_step = inc_step)

		# check for convergence
        affine      = daffine < afftol 
        diffaffine  = abs(daffine - daffine0)
        the_norm    = euclidean(x,y)
        scaled_norm = the_norm / (norm(x) + one(T))
        converged   = (scaled_norm < tol) && affine 
        stuck       = !converged && (scaled_norm < tol) && (diffaffine < tol)

        # if converged then break, else save loss and continue
        # also abort if the algo gets stuck
        if converged || (stuck && rho >= rho_max)
            quiet || print_progress(i, loss, daffine, rho, i_interval = max_iter, inc_step = inc_step)
            break
        end
        loss0 = loss

        if i % inc_step == 0 || diffaffine < afftol || stuck
            # update penalty constant
			rho = min(rho_inc * rho, rho_max)
			ρ_inv = one(T) / rho 
			copy!(x,y)
		end 
	end 

    threshold!(y, tol)
    return Dict{String, Any}("obj" => loss, "iter" => i, "x" => copy(y), "affine_dist" => daffine, "converged" => converged, "stuck" => stuck)
end
