"""
    project_affine(v,A,b) -> x, C, d

`project_affine`computes the projection of `v` onto the affine constraints `A*v == b` as

    x = v - pinv(A)*(A*v - b)

To facilitate future projections, `project_affine` returns, in addition to `x`, the matrix `C = I - pinv(A)*A` and `d = pinv(A)*b`.
"""
function project_affine(
    v :: DenseVector{Float64},
    A :: DenseMatrix{Float64},
    b :: DenseVector{Float64}
)
    # initialize output vector
    x = zeros(Float64, size(v))

    # x = v - pinv(A)*(A*v - b)
    pA = pinv(A)
    C  = BLAS.gemm('N', 'N', -1.0, pA, A)
    C  = I + C
    d  = BLAS.gemv('N', 1.0, pA, b)

    # x = C*v + d
    project_affine!(x,v,C,d)

    # return affine projection with cached projection matrix, vector
    return x, C, d
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
    x :: DenseVector{Float64},
    y :: DenseVector{Float64},
    C :: DenseMatrix{Float64},
    d :: DenseVector{Float64}
)
    BLAS.gemv!('N', 1.0, C, y, 0.0, x)
    BLAS.axpy!(length(x), 1.0, d, 1, x, 1)
end




"""
    project_affine(v::SparseMatrixCSC, A::SparseMatrixCSC, b::SparseMatrixCSC)

For a sparse matrix `A` and sparse vectors `v` and `b`, `project_affine` computes the projection of `v` onto the affine constraints `A*v == b`.
"""
function project_affine(
    v  :: SparseMatrixCSC{Float64,Int},
    A  :: SparseMatrixCSC{Float64,Int},
    b  :: SparseMatrixCSC{Float64,Int};
    AA :: SparseMatrixCSC{Float64,Int} = A*A'
)
    return v + A' * ( AA \ (b - A*v) )
end

#function project_affine(
#    v  :: SparseMatrixCSC{Float64,Int},
#    A  :: SparseMatrixCSC{Float64,Int},
#    b  :: SparseMatrixCSC{Float64,Int},
#    AA :: Base.SparseMatrix.CHOLMOD.Factor{Float64} 
#)
#    return v + A' * ( AA \ (b - A*v) )
#end

function project_affine(
    v  :: SparseMatrixCSC{Float64,Int},
    A  :: SparseMatrixCSC{Float64,Int},
    b  :: Union(DenseVector{Float64}, SparseMatrixCSC{Float64,Int}),
    AA :: Base.SparseMatrix.CHOLMOD.Factor{Float64} 
)
    C = I - (A' * ( AA \ A))
    d = A' * (AA \ b)
# try caching b as last column, computing AA \ [A b], and then recovering d = A'(AA\b) 
    return C*v + d, C, d 
end


function project_affine(
    v  :: SparseMatrixCSC{Float64,Int},
    C  :: DenseMatrix{Float64}, 
    d  :: DenseVector{Float64},
)
    return C*v + d
end

#function compare_projections()

    # set random seed
    srand(2016)

    # make sparse vectors
    m = 500
    n = 1000
    s = 0.1
    A = sprandn(m,n,s)
    b = sprandn(m,1,s)
    v = sprandn(n,1,s)

    # cache A * A'
    AA = A * A'

    # cache the sparse Cholesky factorization of AA
    faa = cholfact(AA)

    # time sparse projection with no factorization
    pv = project_affine(v,A,b, AA=AA)
    @time pv = project_affine(v,A,b, AA=AA)

    # time sparse projection with cached factorization
#    pv2 = project_affine(v,A,b,faa)
#    @time pv2 = project_affine(v,A,b,faa)
    
    pv2, C2, d2 = project_affine(v,A,b,faa)
    C2 = full(C2)
    d2 = vec(full(d2))
    pv2 = project_affine(v,C2,d2)
    @time pv2 = project_affine(v,C2,d2)

    # get dense copies
    vfull = vec(full(v))
    Afull = full(A)
    bfull = vec(full(b))

    # time dense projection
    pvfull, C, d = project_affine(vfull, Afull, bfull);
#    @time pvfull, C, d = project_affine(vfull, Afull, bfull)
    project_affine!(pvfull, vfull, C, d) 
    @time project_affine!(pvfull, vfull, C, d) 

    # same answers?
    println("Difference between sparse answers?: ", norm(pv - pv2))
    println("Difference between answers?: ", norm(full(pv) - pvfull))

#    return nothing
#end

#compare_projections()
