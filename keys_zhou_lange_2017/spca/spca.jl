using RegressionTools
using Distances

# proximal distance algorithm with domain constraints
function spca(
    X         :: DenseMatrix{Float64},
    k         :: Int,
    r         :: Union{DenseVector{Int}, Int},
    proj_type :: ASCIIString = "column";
    n         :: Int     = size(X,1),
    p         :: Int     = size(X,2),
    rho       :: Float64 = one(Float64),
    rho_inc   :: Float64 = 2.0,
    rho_max   :: Float64 = 1e30,
    max_iter  :: Int     = 10000,
    inc_step  :: Int     = 10,
    tol       :: Float64 = 1e-6,
    feastol   :: Float64 = 1e-4,
    quiet     :: Bool    = true,
    U         :: DenseMatrix{Float64} = eye(Float64, p, k), 
    Ut        :: DenseMatrix{Float64} = U', 
    U0        :: DenseMatrix{Float64} = eye(Float64, p, k),
    Ur        :: DenseMatrix{Float64} = eye(Float64, p, k),
    Y         :: DenseMatrix{Float64} = eye(Float64, p, k), 
    XU        :: DenseMatrix{Float64} = BLAS.gemm('N', 'N', one(Float64), X, U), 
    XXU       :: DenseMatrix{Float64} = BLAS.gemm('T', 'N', one(Float64) / n, X, XU), 
)

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
        x = zeros(Float64, k*p)
    else
        x = zeros(Float64, p)
    end

    # main loop
    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - 1) / (i + 2)
        ky = one(Float64) + kx

        # Y = ky*U - kx*U0
        difference!(Y,U,U0, a=ky, b=kx)
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
        difference!(U, XXU, Ur, a=one(Float64), b=-rho)
        transpose!(Ut,U)
        v,s,u = svd(Ut)
        BLAS.gemm!('N', 'T', one(Float64), u, v, zero(Float64), U)

        # update X*U and X'*X*U 
        BLAS.gemm!('N', 'N', one(Float64), X, U, zero(Float64), XU)
        BLAS.gemm!('T', 'N', one(Float64) / n, X, XU, zero(Float64), XXU)

        # convergence checks
        loss        = -0.5*vecdot(XU,XU)
        feas        = dsparse < feastol
        the_norm    = euclidean(U,U0)
        scaled_norm = the_norm / (vecnorm(U0) + one(Float64))
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

    return Dict{ASCIIString, Any}("obj" => loss, "iter" => i, "U" => sparse(U), "dsparse" => dsparse)
end
