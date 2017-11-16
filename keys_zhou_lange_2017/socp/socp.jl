using Distances
using RegressionTools
using MathProgBase
using SCS
using Gurobi
using ProxOpt
using Convex
using MathProgBase
using IterativeSolvers

###################
### subroutines ###
###################

# create a function handle for CG 
# will pass mulbyA! as an operator into handle 
#function mulbyA!(output, v, A, c, ρ)
##  output = vecdot(c, v) * c + A' * (A * v) + v / ρ
#  output = ρ*vecdot(c, v) * c + ρ*A' * (A * v) + v
#end

function mulbyA!(output, v, A, c, ρ, v2)
    A_mul_B!(v2,A,v) 
    At_mul_B!(output, A, v2)
    scale!(output, ρ)
    BLAS.axpy!(1.0, v, output)
    BLAS.axpy!(ρ*dot(c,v), c, output)
end



######################
### main functions ###
######################

function proj_soc!(
    xp :: Union{SparseMatrixCSC{Float64,Int}, DenseVector{Float64}},
    x  :: Union{SparseMatrixCSC{Float64,Int}, DenseVector{Float64}},
    r  :: Float64
)
    n = norm(x)
    if n <= -r
        fill!(xp,zero(Float64))
        return zero(Float64)
    end
    if n > abs(r)
        a = (n + r) / (2 * n)
        copy!(xp,x)
        scale!(xp, a)
        return a*n 
    end
    copy!(xp,x)
    return r
end

function proj_soc(
    x :: DenseVector{Float64},
    r :: Float64
)
    xp = zeros(size(x))
    r = proj_soc!(xp,x,r)
    return xp, r
end

function proj_soc(
    x :: SparseMatrixCSC{Float64,Int},
    r :: Float64
)
    xp = spzeros(Float64, length(x), 1)
    r = proj_soc!(xp,x,r)
    return xp, r
end


"""
    proj_soc!(px,x,A,b,c,d)

Use a proximal distance algorithm to compute the second order cone projection

    minimize 0.5*||u - x|| + lambda(Au + b - w) + 0.5*rho*|| (w,r) - P(w,r) ||

with `w = A*u + b` and `r = dot(c,u) + d`.
"""
function proj_soc!(
    xp       :: DenseVector{Float64},
    x        :: DenseVector{Float64},
    A        :: DenseMatrix{Float64},
    b        :: DenseVector{Float64},
    c        :: DenseVector{Float64},
    d        :: Float64;
    rho      :: Float64 = 1 / size(A,2), 
    rho_inc  :: Float64 = 2.0,
    rho_max  :: Float64 = 1e30,
    tol      :: Float64 = 1e-6,
    feastol  :: Float64 = 1e-6,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    quiet    :: Bool    = true,
    p        :: Int     = size(A,1),
    q        :: Int     = size(A,2),
    AA       :: DenseMatrix{Float64} = zeros(Float64, q, q),
    u        :: DenseVector{Float64} = zeros(Float64, q),
    U        :: DenseVector{Float64} = zeros(Float64, q),
    u0       :: DenseVector{Float64} = copy(u),
    w        :: DenseVector{Float64} = zeros(Float64, p),
    W        :: DenseVector{Float64} = copy(w),
    w0       :: DenseVector{Float64} = copy(w),
    pw       :: DenseVector{Float64} = copy(b),
    y2       :: DenseVector{Float64} = zeros(Float64, q),
    z        :: DenseVector{Float64} = zeros(Float64, q),
    z2       :: DenseVector{Float64} = zeros(Float64, p),
    r        :: Float64 = zero(Float64),
    r0       :: Float64 = r,
    R        :: Float64 = r
)

    # error checking
    (p,q) == size(A) || throw(DimensionMismatch("nonconformable A, b, and c"))

    iter    = 0
    loss    = 0.5*sqeuclidean(x,u) 
    loss0   = Inf
    dw      = Inf
    dr      = Inf


    # factorize matrix A'*A + c*c'
    BLAS.gemm!('T', 'N', one(Float64), A, A, zero(Float64), AA)
    BLAS.ger!(one(Float64), c, c, AA)
    ev,V = eig(AA) 

    if !quiet
        @printf("Iter\tLoss\tNorm\tdw\tdr\tRho\n")
        @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", 0, loss, Inf, dw, dr, rho)
    end
    i = 0
    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - one(Float64)) / (i + one(Float64) + one(Float64))
        ky = one(Float64) + kx
        difference!(U,u,u0, a=ky, b=kx, n=q)

        # save previous iterate
        copy!(u0,u)

        # update w, r with new u
        copy!(W,b)
        BLAS.gemv!('N', one(Float64), A, U, one(Float64), W)
        R = dot(c,U) + d
        pr = proj_soc!(pw,W,R)
#        pw, pr = proj_soc(W,R)

        # for prox dist update need to recompute z
        # z = A' * (b - pw) + (d - pr)*c)
        copy!(z2,b)
        BLAS.axpy!(-one(Float64), pw, z2)
        BLAS.gemv!('T', one(Float64), A, z2, zero(Float64), z)
        BLAS.axpy!(d - pr, c, z)

        # prox dist update uses quadratic proximal map 
        # u = (invrho*I + AA) \ (invrho*x + A'*(P(w) - b) + (P(r) - d)c)
        prox_quad!(u, V, ev, z, x, rho, y2=y2, n=q)

        # update w, r with new u
        copy!(w,b)
        BLAS.gemv!('N', one(Float64), A, u, one(Float64), w)
        r = dot(c,u) + d

        # convergence checks
        loss        = 0.5*sqeuclidean(u,x) 
        dw          = euclidean(w,pw)
        dr          = sqrt(abs(r*r - 2*r*pr + pr*pr)) 
        feas        = dw < feastol && dr < feastol
        the_norm    = euclidean(u,u0)
        scaled_norm = the_norm / (norm(u0,2) + one(Float64))
        converged   = scaled_norm < tol && feas 

        # print progress of algorithm
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", i, loss, the_norm, dw, dr, rho)
        end

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho    = min(rho_inc*rho, rho_max)
            copy!(u0,u)
#            copy!(w0,w)
#            r0 = r
        end
    end

    # threshold small elements of u before returning
    threshold!(u,tol)
    copy!(xp,u)
    return nothing
end


function proj_soc(
    x        :: DenseVector{Float64},
    A        :: DenseMatrix{Float64},
    b        :: DenseVector{Float64},
    c        :: DenseVector{Float64},
    d        :: Float64;
    rho      :: Float64 = one(Float64),
    rho_inc  :: Float64 = 2.0,
    rho_max  :: Float64 = 1e15,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    tol      :: Float64 = 1e-6,
    feastol  :: Float64 = 1e-6,
    quiet    :: Bool    = true,
    p        :: Int     = size(A,1),
    q        :: Int     = size(A,2),
    AA       :: DenseMatrix{Float64} = zeros(Float64, q, q),
    u        :: DenseVector{Float64} = zeros(Float64, q),
    U        :: DenseVector{Float64} = zeros(Float64, q),
    u0       :: DenseVector{Float64} = copy(u),
    w        :: DenseVector{Float64} = zeros(Float64, p),
    W        :: DenseVector{Float64} = copy(w),
    w0       :: DenseVector{Float64} = copy(w),
    pw       :: DenseVector{Float64} = copy(b),
    y2       :: DenseVector{Float64} = zeros(Float64, q),
    z        :: DenseVector{Float64} = zeros(Float64, q),
    z2       :: DenseVector{Float64} = zeros(Float64, p),
    r        :: Float64 = zero(Float64),
    r0       :: Float64 = r,
    R        :: Float64 = r
)
    xp = zeros(size(x))
    proj_soc!(xp,x,A,b,c,d,rho=rho,rho_inc=rho_inc,rho_max=rho_max,max_iter=max_iter,inc_step=inc_step,tol=tol,feastol=feastol,quiet=quiet,p=p,q=q,AA=AA,u=u,w=w,y2=y2,z=z,z2=z2,u0=u0,w0=w0,pw=pw,r=r,r0=r0, W=W, R=R)
    return xp
end

#"""
#    proj_soc(x, A::SparseMatrixCSC, b, c, d)
#
#For sparse matrix `A` and sparse or dense vectors `b` and `c`, solve the optimization problem
#
#    minimize dot(x,c) + lambda(Ax - b)
#             x   >= 0
#
#with an accelerated proximal distance algorithm. Here the affine constraints of the linear program are moved into the objective function.
#The vector `lambda` represents the Lagrange multiplier. 
#"""
function proj_soc(
    x        :: SparseMatrixCSC{Float64,Int},
    A        :: SparseMatrixCSC{Float64,Int},
    b        :: SparseMatrixCSC{Float64,Int},
    c        :: SparseMatrixCSC{Float64,Int},
    d        :: Float64;
    rho      :: Float64 = one(Float64),
    rho_inc  :: Float64 = 2.0,
    rho_max  :: Float64 = 1e30,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    tol      :: Float64 = 1e-6,
    feastol  :: Float64 = 1e-6,
    quiet    :: Bool    = true,
    p        :: Int     = length(b), 
    q        :: Int     = length(c), 
    At       :: SparseMatrixCSC{Float64,Int} = A', 
    u        :: SparseMatrixCSC{Float64,Int} = sprandn(q, 1, 0.1),
    v2       :: SparseMatrixCSC{Float64,Int} = zeros(Float64, p), 
    U        :: SparseMatrixCSC{Float64,Int} = copy(u), 
    u0       :: SparseMatrixCSC{Float64,Int} = copy(u),
    w        :: SparseMatrixCSC{Float64,Int} = spzeros(Float64, p, 1),
    W        :: SparseMatrixCSC{Float64,Int} = copy(w), 
    pw       :: SparseMatrixCSC{Float64,Int} = copy(b),
    z        :: SparseMatrixCSC{Float64,Int} = spzeros(Float64, q, 1),
    r        :: Float64 = zero(Float64),
    pr       :: Float64 = r,
    R        :: Float64 = r,
)

    # error checking
    (p,q) == size(A) || throw(DimensionMismatch("nonconformable A, b, and c"))

    loss    = 0.5*sqeuclidean(x,u) 
    loss0   = Inf
    dw      = Inf
    dr      = Inf

    # cache a factorization of I + rho*A'*A + rho*c*c'
    # we must update the factorization every time that we update rho
#    ey = speye(q,q)
#    E  = [sqrt(1/rho)*ey; A; c'] 
#    Efact = qrfact(E)
#    E = [A; c'; ey / sqrt(rho)] # augmented design matrix

    u2 = vec(full(u))
    if issparse(c)
        c = vec(full(c))
    end
    if issparse(b)
        b = vec(full(b))
    end


    # function handle to efficiently compute multiplication by A
#    Afun = MatrixFcn{Float64}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho))
#    Afun = MatrixFcn{Float64}(q, q, (output, v) -> mulbyA!(output, v, A, full(c), rho))
    Afun = MatrixFcn{Float64}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho, v2))

    if !quiet
        @printf("Iter\tLoss\tNorm\tdw\tdr\tRho\n")
        @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", 0, loss, Inf, dw, dr, rho)
    end
    i = 0
    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - one(Float64)) / (i + one(Float64) + one(Float64))
        ky = one(Float64) + kx
        U  = ky*u - kx*u0
        copy!(u0,u)

        # compute projections onto constraint sets
        W  = A*U + b
        R  = vecdot(c,U) + d
        pr = proj_soc!(pw,sparse(W),R)

        # update z for prox dist update
        # update b0 for Jacobi inversion
        z  = At*(b-pw) + (d-pr)*c
#        b0 = [vec(full(x - rho*z)); zeros(p+1)]
        b0 = vec(full(x - rho*z))

        # prox dist update 
#        u2 = Efact \ b0
#        u2 = vec(full(U))
        cg!(u2, Afun, b0, maxiter=200, tol=1e-8)
#        u2, convhist = cg(Afun, b0, tol=1e-8)
        u = sparse(u2)
#        @show length(convhist.residuals)

        # now update w,r
        w = A*u + b
        r = vecdot(c,u) + d

        # convergence checks
        loss        = 0.5*sqeuclidean(u,x) 
        dw          = euclidean(w,pw)
        dr          = sqrt(abs(r*r - 2*r*pr + pr*pr)) 
        feas        = dr < feastol && dw < feastol
        the_norm    = euclidean(u,u0) 
        scaled_norm = the_norm / (norm(u0) + one(Float64))
        converged   = scaled_norm < tol && feas 

        # print progress of algorithm
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", i, loss, the_norm, dw, dr, rho)
        end

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho    = min(rho_inc*rho, rho_max)
#            E = [A; c'; ey / sqrt(rho)] # augmented design matrix
#            Afun = MatrixFcn{Float64}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho))
#            Afun = MatrixFcn{Float64}(q, q, (output, v) -> mulbyA!(output, v, A, full(c), rho))
#            Efact = qrfact(E)
            Afun = MatrixFcn{Float64}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho, v2))
            copy!(u0,u)
        end
    end

    # threshold small elements of y before returning
    threshold!(u.nzval,tol)
    return u
end


function proj_soc(
    x        :: DenseVector{Float64},
    A        :: SparseMatrixCSC{Float64,Int},
    b        :: DenseVector{Float64},
    c        :: DenseVector{Float64},
    d        :: Float64;
    rho      :: Float64 = one(Float64),
    rho_inc  :: Float64 = 2.0,
    rho_max  :: Float64 = 1e30,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    tol      :: Float64 = 1e-6,
    feastol  :: Float64 = 1e-6,
    quiet    :: Bool    = true,
    p        :: Int     = length(b), 
    q        :: Int     = length(c), 
    At       :: SparseMatrixCSC{Float64,Int} = A', 
    u        :: DenseVector{Float64} = zeros(Float64, q), 
    v2       :: DenseVector{Float64} = zeros(Float64, p), 
    U        :: DenseVector{Float64} = zeros(Float64, q), 
    u0       :: DenseVector{Float64} = zeros(Float64, q), 
    w        :: DenseVector{Float64} = zeros(Float64, p), 
    W        :: DenseVector{Float64} = zeros(Float64, p), 
    pw       :: DenseVector{Float64} = zeros(Float64, p), 
    z        :: DenseVector{Float64} = zeros(Float64, q), 
    b0       :: DenseVector{Float64} = zeros(Float64, q), 
    r        :: Float64 = zero(Float64),
    pr       :: Float64 = r,
    R        :: Float64 = r,
)

    # error checking
    (p,q) == size(A) || throw(DimensionMismatch("nonconformable A, b, and c"))

    loss    = 0.5*sqeuclidean(x,u) 
    loss0   = Inf
    dw      = Inf
    dr      = Inf

    # cache a factorization of I + rho*A'*A + rho*c*c'
    # we must update the factorization every time that we update rho
#    ey = speye(q,q)
#    E = [A; c'; ey / sqrt(rho)] # augmented design matrix
#    Et = E'
#    Efact = qrfact(E)

#    u2 = copy(u)

    # function handle to efficiently compute multiplication by A
    Afun = MatrixFcn{Float64}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho, v2))

    if !quiet
        @printf("Iter\tLoss\tNorm\tdw\tdr\tRho\n")
        @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", 0, loss, Inf, dw, dr, rho)
    end
    i = 0
    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - one(Float64)) / (i + one(Float64) + one(Float64))
        ky = one(Float64) + kx
        difference!(U, u, u0, a=ky, b=kx)
        copy!(u0,u)

        # compute projections onto constraint sets
        # W  = A*U + b, R = dot(c,U) + d
        A_mul_B!(W,A,U)
        BLAS.axpy!(one(Float64), b, W)
        R  = dot(c,U) + d
        pr = proj_soc!(pw,W,R)

        # update z for prox dist update
        # z  = At*(b-pw) + (d-pr)*c
        difference!(v2, b, pw)
        A_mul_B!(z, At, v2)
        BLAS.axpy!(d-pr, c, z)

        # update b0 for CG inversion
#        b0 = [vec(full(x - rho*z)); zeros(p+1)]
        difference!(b0, x, z, a=1.0, b=rho)
#        A_mul_B!(v2, E, b0)

        # prox dist update 
#        u2 = Efact \ b0
#        u2 = vec(full(U))
        cg!(u, Afun, b0, maxiter=200, tol=1e-8)
#        lsqr!(u, E, v2, maxiter=200, atol=1e-8, btol=1e-8)

        # now update w,r
        # w = A*u + b, r = dot(c,u) + d 
        A_mul_B!(w, A, u)
        BLAS.axpy!(one(Float64), b, w)
        r = dot(c,u) + d

        # convergence checks
        loss        = 0.5*sqeuclidean(u,x) 
        dw          = euclidean(w,pw)
        dr          = sqrt(abs(r*r - 2*r*pr + pr*pr)) 
        feas        = dr < feastol && dw < feastol
        the_norm    = euclidean(u,u0) 
        scaled_norm = the_norm / (norm(u0) + one(Float64))
        converged   = scaled_norm < tol && feas 

        # print progress of algorithm
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", i, loss, the_norm, dw, dr, rho)
        end

        # if converged then break, else save loss and continue
        converged && break
        loss0 = loss

        if i % inc_step == 0
            rho    = min(rho_inc*rho, rho_max)
#            E  = [ey; sqrt(rho)*A; sqrt(rho)*c'] 
#            E  = [sqrt(1/rho)*ey; A; c'] 
#            E = [A; c'; ey / sqrt(rho)] # augmented design matrix
#            Et = E'
#            Efact = qrfact(E)
#            Afun = MatrixFcn{Float64}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho))
            Afun = MatrixFcn{Float64}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho, v2))
            copy!(u0,u)
        end
    end

    # threshold small elements of y before returning
    threshold!(u,tol)
    return u
end

####################
### testing code ###
####################

function test_dense_socp()

    # set RNG
    seed = 2015
    srand(seed)

    # set algorithm parameters
    max_iter = 10000
    eps      = 1e-4
    quiet    = false 
    verbose  = !quiet
    inc_step = 100
    rho_inc  = 2.0

    # set dimensions 
    m = 1024 
    n = 2*m 

    A = randn(m,n)
    x = rand(n)
    c = ones(n) / n 
    b = rand(m) 
    d = norm(A*x + b) 
    w  = randn(n)
    pw = copy(w)

    println("Problem specs ok?")
    @show norm(A*x + b)
    @show dot(c,x) + d
    @show norm(A*x + b) <= dot(c,x) + d

    println("Before projection:")
    @show norm(A*pw + b)
    @show dot(c,pw) + d
    @show norm(A*pw + b) <= dot(c,pw) + d
    @time proj_soc!(pw,w,A,b,c,d, quiet=quiet, max_iter=max_iter, inc_step=inc_step, rho_inc=rho_inc)
    println("After projection:")
    @show norm(A*pw + b)
    @show dot(c,pw) + d
    @show norm(A*pw + b) <= dot(c,pw) + d

    # now use Convex.jl to find projection using DCP
    y = Convex.Variable(n)

    # configure SCS
    scs_solver = SCSSolver(max_iters=max_iter, eps=eps, verbose=verbose)

    problem = minimize(0.5*sumsquares(y - w))
    problem.constraints += norm(A*y + b) <= vecdot(c,y) + d
    @time solve!(problem, scs_solver)
    @show problem.status
    @show problem.optval
    @show norm(A*y.value + b)
    @show vecdot(c,y.value) + d
    @show norm(A*y.value + b) <= vecdot(c,y.value) + d
    @show norm(pw - y.value)
#    @show [pw y.value]
    @show (norm(pw - w), norm(y.value - w))

    return nothing
end


function test_sparse_socp()

    # set RNG
    seed = 2016
    srand(seed)

    # set algorithm parameters
    max_iter = 10000
    eps      = 1e-3 # tol for SCS 
    tol      = 1e-6
    quiet    = false 
    verbose  = !quiet
    inc_step = 10
    rho_inc  = 5.0 

    # set dimensions 
    m = 1024 
    n = 2056
    s = 0.01
    rho = 1e-2

    m *= 4 
    n *= 4

    A = sprandn(m,n,s)
    x = sprand(n,1,s)
    c = sprand(n,1,s) 
    b = sprand(m,1,s) 
    d = norm(A*x + b) 
#    x = vec(full(sprand(n,1,s)))
#    c = vec(full(sprand(n,1,s))) 
#    b = vec(full(sprand(m,1,s))) 
#    d = norm(A*x + b) 

    println("Problem specs ok?")
    @show norm(A*x + b)
    @show vecdot(c,x) + d
    @show norm(A*x + b) <= vecdot(c,x) + d

    w  = sprandn(n,1,s)
#    w  = vec(full(sprandn(n,1,s)))
#    pw = copy(w)
    println("Before projection:")
    @show norm(A*w + b)
    @show vecdot(c,w) + d
    @show norm(A*w + b) <= vecdot(c,w) + d
    @time pw = proj_soc(w,A,b,c,d, quiet=quiet, max_iter=max_iter, inc_step=inc_step, rho_inc=rho_inc, feastol=eps, tol=tol, rho=rho, feastol=tol)
    @time pw = proj_soc(vec(full(w)),A,vec(full(b)),vec(full(c)),d, quiet=quiet, max_iter=max_iter, inc_step=inc_step, rho_inc=rho_inc, feastol=eps, tol=tol, rho=rho, feastol=tol)
    println("After projection:")
    @show norm(A*pw + b)
    @show vecdot(c,pw) + d
    @show norm(A*pw + b) <= vecdot(c,pw) + d

    # now use Convex.jl to find projection using DCP
    y = Convex.Variable(n)

    # configure SCS
    scs_solver = SCSSolver(max_iters=max_iter, eps=eps, verbose=verbose)

    problem = minimize(0.5*sumsquares(y - w))
    problem.constraints += norm(A*y + b) <= vecdot(c,y) + d
    @time solve!(problem, scs_solver)
    @show problem.status
    @show problem.optval
    @show norm(A*y.value + b)
    @show vecdot(c,y.value) + d
    @show norm(A*y.value + b) <= vecdot(c,y.value) + d
    @show norm(pw - y.value)
#    @show [pw y.value]
    @show (sqeuclidean(pw,w), sqeuclidean(y.value,w))

    return nothing
end


@time 1+1
#test_dense_socp()
#test_sparse_socp()
