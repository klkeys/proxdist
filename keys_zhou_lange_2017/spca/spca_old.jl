"""
    update_col!(z, x, j [, n=size(x,1), p=size(x,2), a=1.0])
This subroutine overwrites an `n`-vector `z` with the `j`th column of a matrix `x` scaled by `a`.
It is more efficient than `z = a*x[:,j]`.
Arguments:
- `z` is the `n`-vector to fill with `x[:,j]`.
- `x` is the `n` x `p` matrix to use in filling `z`.
- `j` indexes the column of `x` to use in filling `z`.
Optional Arguments:
- `n` is the leading dimension of `x`. Defaults to `size(x,1)`.
- `p` is the trailing dimension of `x`. Defaults to `size(x,2)`.
- `a` scales the entries of `z`. Defaults to `1.0` (no scaling).
"""
function update_col!(
    z :: DenseVector{T},
    x :: DenseMatrix{T},
    j :: Int;
    n :: Int = size(x,1),
    p :: Int = size(x,2),
    a :: T = one(T)
) where {T <: AbstractFloat}
    length(z) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))
    j <= p || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!"))
    @inbounds for i = 1:n
        z[i] = a*x[i,j]
    end
    return nothing
end

"""
    update_col!(x, z, j [, n=size(x,1), p=size(x,2), a=1.0])
Fills the `j`th column of an `n` x `p` matrix `x` with the entries in the `n`-vector `z`.
"""
function update_col!(
    x :: DenseMatrix{T},
    z :: DenseVector{T},
    j :: Int;
    n :: Int = size(x,1),
    p :: Int = size(x,2),
    a :: T = one(T)
) where {T <: AbstractFloat}
    length(z) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))
    j <= p || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!"))
    @inbounds for i = 1:n
        x[i,j] = a*z[i]
    end

    return nothing
end


"""
    update_col!(z, x, j, q [, n=size(x,1), p=size(x,2), a=1.0])
Updates the `q`th column of a matrix `z` with the `j`th column of a matrix `x` scaled by `a`. Both `x` and `z` must have the same leading dimension `n`. Both `j` and `q` must not exceed the trailing dimension `p` of `x`.
"""
function update_col!(
    z :: DenseMatrix{T},
    x :: DenseMatrix{T},
    j :: Int,
    q :: Int;
    n :: Int = size(x,1),
    p :: Int = size(x,2),
    a :: T = one(T)
) where {T <: AbstractFloat}
    size(z,1) == n || throw(DimensionMismatch("update_col!: arguments z and X must have same number of rows!"))
    j <= p         || throw(DimensionMismatch("update_col!: index j must not exceed number of columns p!"))
    @inbounds for i = 1:n
        z[i,q] = a*x[i,j]
    end
    return nothing
end



"""
    threshold!(X::Matrix, tol)
Send to zero all values of a matrix `X` below tolerance `tol` in absolute value.
"""
function threshold!(
    x   :: DenseMatrix{T},
    tol :: T
) where {T <: AbstractFloat}
    m,n = size(x)
    @inbounds for j = 1:n
        @inbounds for i = 1:m
            a = x[i,j]
            if abs(a) < tol
                x[i,j] = zero(T)
            end
        end
    end
    return nothing
end

"""
    threshold!(x, a, tol) 
If fed a floating point number `a` in addition to vector `x` and tolerance `tol`,
then `threshold!` will send to zero all components of `x` where `abs(x - a) < tol`.
"""
function threshold!(
    x   :: DenseVector{T},
    a   :: T,
    tol :: T;
) where {T <: AbstractFloat}
    @inbounds for i in eachindex(x) 
        x[i] = abs(x[i] - a) < tol ? a : x[i]
    end
    return nothing
end


"""
    threshold!(x, tol)
This subroutine compares the absolute values of the components of a vector `x`
against a thresholding tolerance `tol`. All elements below `tol` are sent to zero.
Arguments:
- `x` is the vector to threshold.
- `tol` is the thresholding tolerance
"""
function threshold!(
    x   :: DenseVector{T},
    tol :: T;
) where {T <: AbstractFloat}
    @inbounds for i in eachindex(x) 
        x[i] = abs(x[i]) < tol ? zero(T) : x[i]
    end
    return nothing
end

"""
threshold!(idx::BitArray{1}, x, tol)
This subroutine compares the absolute values of the components of a vector `x`
against a thresholding tolerance `tol`. It then fills `idx` with the result of `abs(x) .> tol`. 
Arguments:
- `idx` is the `BitArray` to fill with the thresholding of `x`.
- `x` is the vector to threshold.
- `tol` is the thresholding tolerance
"""
function threshold!(
    idx :: BitArray{1},
    x   :: DenseVector{T},
    tol :: T;
) where {T <: AbstractFloat}
    @assert length(idx) == length(x)
    @inbounds for i in eachindex(x) 
        idx[i] = abs(x[i]) >= tol
    end
    return nothing
end

"""
    vec!(x::Vector, X::Matrix)
Perform `copy!(x, vec(X))` without any intermediate arrays.
"""
function vec!(
    x :: Vector{T},
    X :: Matrix{T};
    k :: Int = length(x),
    n :: Int = size(X,1),
    p :: Int = size(X,2)
) where {T <: AbstractFloat}
    # check for conformable dimensions
    k == n*p || throw(DimensionMismatch("Arguments x and X must have same number of elements"))

    # will copy X into x in column-major order
    @inbounds for j = 1:p
        @inbounds for i = 1:n
            x[n*(j-1) + i] = X[i,j]
        end
    end

    return nothing
end

"""
    project_k!(x, k)
This function projects a vector `x` onto the set S_k = { y in R^p : || y ||_0 <= k }.
It does so by first finding the pivot `a` of the `k` largest components of `x` in magnitude.
`project_k!` then thresholds `x` by `abs(a)`, sending small components to 0. 
Arguments:
- `b` is the vector to project.
- `k` is the number of components of `b` to preserve.
"""
function project_k!(
    x    :: Vector{T},
    k    :: Int;
) where {T <: AbstractFloat}
    a = select(x, k, by = abs, rev = true) :: T
    threshold!(x,abs(a)) 
    return nothing
end


"""
    project_k!(idx::BitArray{1}, x, k)
This function computes the indices that project a vector `x` onto the set S_k = { y in R^p : || y ||_0 <= k }.
It does so by first finding the pivot `a` of the `k` largest components of `x` in magnitude.
`project_k!` then fills `idx` with the result of `abs(x) .> a`.
Arguments:
- `idx` is a `BitArray` to fill with the projection of `x`.
- `x` is the vector to project.
- `k` is the number of components of `b` to preserve.
"""
function project_k!(
    idx  :: BitArray{1},
    x    :: Vector{T},
    k    :: Int;
) where {T <: AbstractFloat}
    a = select(x, k, by = abs, rev = true) :: T
    threshold!(idx,x,abs(a)) 
    return nothing
end



"""
    project_k!(b, bk, perm, k)
This function projects a vector `b` onto the set S_k = { x in R^p : || x ||_0 <= k }.
It does so by first finding the indices of the `k` largest components of `x` in magnitude.
Those components are saved in the array `bk`, and then `b` is filled with zeros.
The nonzero components from `bk` are then returned to their proper indices in `b`.
Arguments:
- `b` is the vector to project.
- `bk` is a vector to store the values of the largest `k` components of `b` in magnitude.
- `perm` is an `Int` array that indexes `b`.
- `k` is the number of components of `b` to preserve.
"""
function project_k!(
    b    :: Vector{T},
    bk   :: Vector{T},
    perm :: Vector{Int},
    k    :: Int;
) where {T <: AbstractFloat}
    kk = k == 1 ? 1 : 1:k
    select!(perm, kk, by = (i)->abs(b[i]), rev = true)
    fill_perm!(bk, b, perm, k=k)    # bk = b[perm[kk]]
    fill!(b,zero(T))
    @inbounds for i = 1:k
        b[perm[i]] = bk[i]
    end
    return nothing
end

"""
    project_k!(X, k [, n=size(X,1), p=size(X,2), x=zeros(n), xk=zeros(k), perm=collect(1:n)])
Apply `project_k!` onto each of the columns of a matrix `X`. This enforces the *same* sparsity level on each column.
"""
function project_k!(
    X    :: Matrix{T},
    k    :: Int;
    n    :: Int = size(X,1),
    p    :: Int = size(X,2),
    x    :: Vector{T}   = zeros(n),
) where {T <: AbstractFloat}
    length(x)    = n || throw(DimensionMismatch("Arguments X and x must have same row dimension"))
    @inbounds for i = 1:p
        update_col!(x, X, i, n=n, p=p, a=one(T))
        project_k!(x, k)
        update_col!(X, x, i, n=n, p=p, a=one(T))
    end
    return nothing
end


"""
    project_k!(X, K [, n=size(X,1), p=size(X,2), x=zeros(n), xk=zeros(k), perm=collect(1:n)])
Apply `project_k!` onto each of the columns of a matrix `X`, where the `i`th column has sparsity level `K[i]`. This function permits a *different* sparsity level for each column.
"""
function project_k!(
    X    :: Matrix{T},
    K    :: Vector{Int};
    n    :: Int = size(X,1),
    p    :: Int = size(X,2),
    x    :: Vector{T} = zeros(eltype(X), n),
) where {T <: AbstractFloat}
    length(x) == n || throw(DimensionMismatch("Arguments X and x must have same row dimension"))
    length(K) == p || throw(DimensionMismatch("Argument K must have one entry per column of x"))
    @inbounds for i = 1:p
        k  = K[i]
        update_col!(x, X, i, n=n, p=p, a=one(T))
        project_k!(x, k)
        update_col!(X, x, i, n=n, p=p, a=one(T))
    end
    return nothing
end

"""
    project_k!(x, X, k [, n=size(X,1), p=size(X,2), xk=zeros(k), perm=collect(1:p*n)])
Apply `project_k!` onto the matrix `X` as if it were a vector. The argument `x` facilitates the projection.
Sparsity is enforced on the matrix *as a whole*, so columns may vary in their sparsity.
"""
function project_k!(
    x    :: Vector{T},
    X    :: Matrix{T},
    k    :: Int;
    n    :: Int = size(X,1),
    p    :: Int = size(X,2),
) where {T <: AbstractFloat}
    length(x) == n*p || throw(DimensionMismatch("Arguments X and x must have same number of elements"))
    vec!(x, X, k=n*p, n=n, p=p)
    a = select(x, k, by=abs, rev=true) :: T
    threshold!(X, abs(a))
    return nothing
end

# proximal distance algorithm with domain constraints
function spca(
    X         :: Matrix{T},
    k         :: Int,
    r         :: Union{Vector{Int}, Int},
    proj_type :: String = "column";
    n         :: Int     = size(X,1),
    p         :: Int     = size(X,2),
    rho       :: T = one(T),
    rho_inc   :: T = 2.0,
    rho_max   :: T = 1e30,
    max_iter  :: Int     = 10000,
    inc_step  :: Int     = 10,
    tol       :: T = 1e-6,
    feastol   :: T = 1e-4,
    quiet     :: Bool    = true,
    U         :: Matrix{T} = eye(T, p, k), 
    Ut        :: Matrix{T} = U', 
    U0        :: Matrix{T} = eye(T, p, k),
    Ur        :: Matrix{T} = eye(T, p, k),
    Y         :: Matrix{T} = eye(T, p, k), 
    XU        :: Matrix{T} = BLAS.gemm('N', 'N', one(T), X, U), 
    XXU       :: Matrix{T} = BLAS.gemm('T', 'N', one(T) / n, X, XU), 
) where {T <: AbstractFloat}

    # error checking
    0 <= k <= min(n,p)                || throw(DimensionMismatch("Number of principal components must be nonnegative and cannot smallest marginal dimension of X"))
    0 <= maximum(r) <= p              || throw(DimensionMismatch("Sparsity level must be nonnegative and cannot exceed column dimension of X"))
    proj_type in ["column", "matrix"] || throw(ArgumentError("Argument proj_type must specify either 'column' sparsity or 'matrix' sparsity"))

    # initialize intermediate and output variables
    loss    = Inf
    dsparse = Inf
    i       = 0
    u       = 0
    v       = 0

    ### STILL NEEDS WORK
    R    = length(r) == 1 ? r[1]*ones(Int,size(U,2)) : r
    sumR = sum(R)

    # each projection needs a temporary array x
    # dimension of x depends on projection type
    # preallocate x before entering main loop
    if proj_type == "matrix"
        x = zeros(T, k*p)
    else
        x = zeros(T, p)
    end

    # main loop
    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - 1) / (i + 2)
        ky = one(T) + kx

        # Y = ky*U - kx*U0
		Y .= ky.*U .- kx.*U0
        copy!(U0,U)

        # compute projection onto constraint set
        copy!(Ur,Y)
        if proj_type == "matrix"
            project_k!(x, Ur, sumR, n=p, p=k)
        else
            project_k!(Ur, R, n=p, p=k, x=x)
        end

        # compute distance to constraint set
        dsparse = euclidean(U,Ur) 

        # print progress of algorithm
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dsparse, rho)
        end

        # SPCA prox dist update is SVD on variable 
        # U+ = X'*X*U +rho*Ur 
        # all singular values of U+ are set to 1,
        # so prox dist update is simply multiplication of singular vectors of U 
		U .= XXU .+ rho.*Ur
        transpose!(Ut,U)
        v,s,u = svd(Ut)
        BLAS.gemm!('N', 'T', one(T), u, v, zero(T), U)

        # update X*U and X'*X*U 
        BLAS.gemm!('N', 'N', one(T), X, U, zero(T), XU)
        BLAS.gemm!('T', 'N', one(T) / n, X, XU, zero(T), XXU)

        # convergence checks
        loss        = -0.5*vecdot(XU,XU)
        feas        = dsparse < feastol
        the_norm    = euclidean(U,U0)
        scaled_norm = the_norm / (vecnorm(U0) + one(T))
        converged   = scaled_norm < tol && feas 

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho    = min(rho_inc*rho, rho_max)
            copy!(U0,U)
        end
    end

    # threshold small elements of X before returning
    threshold!(U,tol)

    return Dict{String, Any}("obj" => loss, "iter" => i, "U" => sparse(U), "dsparse" => dsparse)
end
