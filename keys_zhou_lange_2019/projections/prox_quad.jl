"""
    prox_quad(x, A, b, py, rho)

For sparse `A` and `b`, compute the proximal map of a quadratic function `f(x) = 0.5*x'*A*x + b'*x` using the Jacobi method.
The Jacobi method iteratively computes the solution to `Px = q`, where `P = rho*I + A` and `q = rho*py - b`. 
""" 
function prox_quad(
    x   :: SparseMatrixCSC{T,Int},
    A   :: SparseMatrixCSC{T,Int},
    b   :: SparseMatrixCSC{T,Int},
    py  :: SparseMatrixCSC{T,Int},
    rho :: T;
    d   :: SparseMatrixCSC{T,Int} = spdiagm(one(T) ./ (diag(A) + rho), 0), 
    A0  :: SparseMatrixCSC{T,Int} = A - spdiagm(diag(A),0), 
    x0  :: SparseMatrixCSC{T,Int} = spzeros(size(A,1),1),
    b0  :: SparseMatrixCSC{T,Int} = rho*py - b, 
    tol :: T = 1e-8,
    itr :: Int     = 1000,
) where {T <: AbstractFloat}
    y  = copy(x)
    x0 = x
    for i = 1:itr
        # y  = d*(b0 - A0*x0)
        y  = b0
        y -= A0*x0
        y  = d*y

        # convergence check
        nd = norm(x0) + one(T)
        nn = euclidean(y,x0)
        nn / nd < tol && break

        # save previous iterate
        x0 = y
    end
    return y
end

"""
    prox_quad!(y, y2, V, d, b, py, ρ_inv)

Efficiently compute the proximal operator of a quadratic function `f(x) = 0.5*x'*A*x + b'*x` based on a spectral decomposition `A = V*diagm(d)*V'`. The proximal operator of `f` is

    y = V * inv(1 + ρ_inv*d) * V' * (py - ρ_inv*b)

`y2` is a temporary array used to store intermediate results. `py` is the projection `py = max.(y, 0)`.
"""
function prox_quad!(
    y      :: DenseVector{T},
    y2     :: DenseVector{T}, 
    V      :: DenseMatrix{T},
    d      :: DenseVector{T},
    b      :: DenseVector{T},
    py     :: DenseVector{T},
    ρ_inv :: T
) where {T <: AbstractFloat}

    y2 .= py .- ρ_inv .* b
    #copy!(y2,py)
    #BLAS.axpy!(-ρ_inv, b, y2)
        
    # y .= V' * y2 
    At_mul_B!(y, V, y2)

    # y2 .= diagm( 1 / (1 + ρ_inv*d) ) * y
    y2 .= y ./ (one(T) .+ ρ_inv .* d)

    # y = V * y2
    A_mul_B!(y, V, y2)

    return nothing
end

function prox_quad!(
    y      :: DenseVector{T},
    V      :: DenseMatrix{T},
    d      :: DenseVector{T},
    b      :: DenseVector{T},
    py     :: DenseVector{T},
    invrho :: T
) where {T <: AbstractFloat}
    y2 = copy(y)
    prox_quad!(y, y2, V, d, b, py, invrho)
end


"""
    prox_quad(LLt, b, py, invrho)

Compute the proximal operator of a quadratic function `f(x) = 0.5*x'*A*x + b'*x` for sparse `A` and `b` based on a sparse Cholesky factorization `LLt` of `inrho*I + A`. The proximal operator of `f` with sparse `A` and `b` is 

    y = LLt \ (invrho*py - b)
"""
function prox_quad(
    LLt    :: Base.SparseArrays.CHOLMOD.Factor{T},
    b      :: SparseMatrixCSC{T,Int},
    py     :: SparseMatrixCSC{T,Int},
    invrho :: T;
) where {T <: AbstractFloat}
    y2 = invrho*py - b
    return LLt \ y2
end
