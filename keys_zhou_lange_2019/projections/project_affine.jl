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
) where {T <: AbstractFloat}
    BLAS.gemv!('N', one(T), C, y, zero(T), x)
    BLAS.axpy!(one(T), d, x)
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
) where {T <: AbstractFloat} # may need BlasFloat?

    # initialize output vector
    x = zeros(T, size(v))

    # x = v - pinv(A)*(A*v - b)
    pA = pinv(A)
    C  = BLAS.gemm('N', 'N', -one(T), pA, A)
    C  += I
    d  = BLAS.gemv('N', pA, b)

    # x = C*v + d
    project_affine!(x,v,C,d)

    # return affine projection with cached projection matrix, vector
    return x, C, d
end



"""
    project_affine(y::Vector, At::SparseMatrixCSC, b::Vector)

Projection of a point `y` to the set of points satisfying `A*x = b`. 
Input should be `At`, transpose of `A`.

# Input
- `y`: point to project.
- `At`: sparse matrix.
- `b`: rhs of affine constraint `A*x = b`.

# Output
- `p`: projection of `y` to affine constraint `A*x = b`.
- `atqr`: sparse QR factorization of `At`.
- `d`: minimum norm solution `pinv(A)*b` of `A*x = b`.
"""
function project_affine(y::Vector, At::SparseMatrixCSC, b::Vector)

    # sparse QR of A'
    #atqr = qrfact(A')
    atqr = qrfact(At)

    # solve R z = b
    z = SparseArrays.SPQR.solve(SparseArrays.SPQR.RTX_EQUALS_ETB, atqr, SparseArrays.CHOLMOD.Dense(b))

    # compute d1 = Q * z, mininum norm solution to Ax=b
    d = SparseArrays.SPQR.qmult(SparseArrays.SPQR.QX, atqr, z)

    # projection to affine set
    p = y - At * (atqr \ y) + d

    return p[:], atqr, d[:]
end

"""
    project_affine(py::Vector, y::Vector, At::SparseMatrixCSC, Atqr::Factorization, b::Vector)

Overwrite `py` by projection of a point `y` to the set of points satisfying `A*x = b`. 
Input should be `At`, the transpose of `A`.

# Input
- `py`: storage for projection of `y`.
- `y`: point to project.
- `At`: sparse matrix.
- `Atqr`: sparse QR factorization of `At`.
- `b`: rhs of affine constraint `Ax=b`.

# Output
- `py`: projection of `y`.
"""
function project_affine!(py::Vector, y::Vector, At::SparseMatrixCSC, Atqr::Factorization, d::Vector)

    # py = y - At * (Atqr \ y) + d
    A_mul_B!(py, At, Atqr \ y)
    py .= y .- py .+ d 
    return py
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
) where {T <: AbstractFloat}
    return v + A' * ( AA \ (b - A*v) )
end

function project_affine(
    v  :: SparseMatrixCSC{T,Int},
    A  :: SparseMatrixCSC{T,Int},
    b  :: SparseMatrixCSC{T,Int}
) where {T <: AbstractFloat}
	AA = A*A'
	project_affine(v, A, b, AA)
end

function project_affine(
    v    :: SparseMatrixCSC{T,Int},
    A    :: SparseMatrixCSC{T,Int},
    b    :: SparseMatrixCSC{T,Int},
    AA   :: Base.SparseArrays.CHOLMOD.Factor{T},
    At   :: SparseMatrixCSC{T,Int},
) where {T <: AbstractFloat}
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
    AA :: Base.SparseArrays.CHOLMOD.Factor{T} 
) where {T <: AbstractFloat}
    C = I - (A' * ( AA \ A)) 
    d = A' * (AA \ b)
	# try caching b as last column, computing AA \ [A b], and then recovering d = A'(AA\b) 
    return C*v + d, C, d 
end
