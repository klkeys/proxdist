using Distances

include("../common/common.jl")
include("../projections/project_sparse.jl")

function spca_update!(U, Ut, XXU, Ur, rho)
    U .= XXU .+ rho .* Ur
    transpose!(Ut,U)
    F = svdfact!(Ut)
    #BLAS.gemm!('N', 'T', one(eltype(U)), F.U, F.Vt, zero(eltype(U)), U)
    BLAS.gemm!('T', 'N', one(eltype(U)), F.Vt, F.U, zero(eltype(U)), U)
end

# proximal distance algorithm with domain constraints
function spca(
    X         :: Matrix{T},
    k         :: Int,
    r         :: Vector{Int};
    proj_type :: String = "column",
    U         :: Matrix{T} = eye(T, size(X,2), k),
    rho       :: T    = one(T),
    rho_inc   :: T    = one(T) + one(T),
    rho_max   :: T    = 1e15,
    max_iter  :: Int  = 10000,
    inc_step  :: Int  = 10,
    tol       :: T    = 1e-6,
    feastol   :: T    = 1e-4,
    quiet     :: Bool = true
) where {T <: AbstractFloat}

    # error checking
    n,p = size(X)
    @assert 0 <= k <= min(n,p)                "Number of principal components k = $(k) must be nonnegative and cannot smallest marginal dimension of X"
    @assert 0 <= maximum(r) <= p              "Sparsity level r = $(r) must be nonnegative and cannot exceed column dimension p = $(p) of X"
    @assert proj_type in ["column", "matrix"] "Argument proj_type must specify either 'column' sparsity or 'matrix' sparsity"
    @assert rho > 0                           "Argument rho must be positive"
    @assert length(r) == k                    "Vector argument r must have k = $(k) entries"

    # initialize intermediate and output variables
    loss    = Inf
    dsparse = Inf
    dsparse0 = Inf
    i       = 0
    u       = 0
    v       = 0
    HALF    = convert(T, 0.5)
    converged = false
    stuck     = false
    n_inv = one(T) / n

    Ut  = U'
    U0  = copy(U)
    Ur  = eye(T, p, k)
    XU  = BLAS.gemm('N', 'N', one(T), X, U)
    XXU = BLAS.gemm('T', 'N', one(T) / n, X, XU)

    # need this for proj_type == "matrix"
    if proj_type == "matrix"
        sumR = sum(r)
    end

    # each projection needs a temporary array x
    # dimension of x depends on projection type
    # preallocate x before entering main loop
    x = zeros(T, proj_type == "matrix" ? k*p : p)

    # main loop
    for i = 1:max_iter

        # compute accelerated step Ur = U + (i - 1)/(i + 2)*(U - U0)
        # will eventually mutate accelerated step Ur with project_sparse!
        compute_accelerated_step!(Ur, U, U0, i)

        # compute projection onto constraint set
        if proj_type == "matrix"
            project_sparse!(x, Ur, sumR)
        else
            project_sparse!(Ur, r, x)
        end

        # compute distance to constraint set
        dsparse0 = dsparse
        dsparse = euclidean(U,Ur) 

        # print progress of algorithm
        quiet || print_progress(i, loss, dsparse, rho, i_interval = 10, inc_step = inc_step)

## try using svdfact! here
        # SPCA prox dist update is SVD on variable 
        # U+ = X'*X*U +rho*Ur 
        # all singular values of U+ are set to 1,
        # so prox dist update is simply multiplication of singular vectors of U 
        U .= XXU .+ rho.*Ur
        transpose!(Ut,U)
        v,s,u = svd(Ut)
        BLAS.gemm!('N', 'T', one(T), u, v, zero(T), U)
## -------
#        spca_update!(U, Ut, XXU, Ur, rho)


        # update X*U and X'*X*U 
        BLAS.gemm!('N', 'N', one(T), X, U, zero(T), XU)
        BLAS.gemm!('T', 'N', n_inv, X, XU, zero(T), XXU)

        # convergence checks
        loss        = -HALF*vecdot(XU,XU)
        feas        = dsparse < feastol
        difffeas    = abs(dsparse - dsparse0)
        the_norm    = euclidean(U,U0)
        scaled_norm = the_norm / (vecnorm(U0) + one(T))
        converged   = scaled_norm < tol && feas 
        stuck       = !converged && (scaled_norm < tol) && (difffeas < tol)

        # if converged then break, else save loss and continue
        if converged || (stuck && (rho >= rho_max || dsparse0 > dsparse))
            quiet || print_progress(i, loss, dsparse, rho, i_interval = max_iter, inc_step = inc_step)
            break
        end

        if i % inc_step == 0 || difffeas < tol || dsparse > dsparse0 || stuck
            rho    = min(rho_inc*rho, rho_max)
            copy!(U0,U)
        end
    end

    # threshold small elements of X before returning
    threshold!(U,tol)

    #return Dict{String, Any}("obj" => loss, "iter" => i, "U" => sparse(U), "dsparse" => dsparse, "converged" => converged, "stuck" => stuck)
    return Dict{String, Any}("obj" => loss, "iter" => i, "U" => U, "dsparse" => dsparse, "converged" => converged, "stuck" => stuck)
end



#spca(X::DenseMatrix{T}, k::Int, r::Int, proj_type::String = "column"; kwargs...) where {T <: AbstractFloat} = _spca(X, k, r*ones(Int, k), proj_type, kwargs...)
#spca(X::DenseMatrix{T}, k::Int, r::Int, proj_type::String = "column"; kwargs...) where {T <: AbstractFloat} = spca(X, k, r*ones(Int, k), proj_type, kwargs...)

function spca(
    X         :: Matrix{T},
    k         :: Int,
    r         :: Int;
    proj_type :: String = "column",
    U         :: Matrix{T} = eye(T, size(X,2), k),
    rho       :: T    = one(T),
    rho_inc   :: T    = one(T) + one(T),
    rho_max   :: T    = 1e30,
    max_iter  :: Int  = 10000,
    inc_step  :: Int  = 10,
    tol       :: T    = 1e-6,
    feastol   :: T    = 1e-4,
    quiet     :: Bool = true
) where {T <: AbstractFloat}
    spca(X, k, r*ones(Int,k), proj_type=proj_type, U=U, rho=rho, rho_inc=rho_inc, rho_max=rho_max, max_iter=max_iter, inc_step=inc_step, tol=tol, feastol=feastol, quiet=quiet)
end
