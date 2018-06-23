using Distances
using MathProgBase
using SCS
using Gurobi
using Convex
using MathProgBase
using IterativeSolvers
using LinearMaps

###################
### subroutines ###
###################

include("../common/common.jl")
include("../projections/prox_quad.jl")

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
    BLAS.axpy!(one(eltype(v)), v, output)
    BLAS.axpy!(ρ*dot(c,v), c, output)
end



######################
### main functions ###
######################

function proj_soc!(
    xp :: Union{SparseMatrixCSC{T,Int}, Vector{T}},
    x  :: Union{SparseMatrixCSC{T,Int}, Vector{T}},
    r  :: T
) where {T <: AbstractFloat}
    n = norm(x)
    if n <= -r
        fill!(xp,zero(T))
        return zero(T)
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
    x :: Vector{T},
    r :: T
) where {T <: AbstractFloat}
    xp = zeros(size(x))
    r = proj_soc!(xp,x,r)
    return xp, r
end

function proj_soc(
    x :: SparseMatrixCSC{T,Int},
    r :: T
) where {T <: AbstractFloat}
    xp = spzeros(T, length(x), 1)
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
    xp       :: Vector{T},
    x        :: Vector{T},
    A        :: Matrix{T},
    b        :: Vector{T},
    c        :: Vector{T},
    d        :: T;
    rho      :: T    = 1 / size(A,2), 
    rho_inc  :: T    = one(T) + one(T),
    rho_max  :: T    = 1e30,
    tol      :: T    = 1e-6,
    feastol  :: T    = 1e-6,
    max_iter :: Int  = 10000,
    inc_step :: Int  = 100,
    quiet    :: Bool = true,
) where {T <: AbstractFloat}

    # error checking
    check_conformability(A,b,c)

    p,q = size(A)
    AA  = zeros(T, q, q)
    u   = zeros(T, q)
    U   = zeros(T, q)
    u0  = copy(u)
    w   = zeros(T, p)
    W   = copy(w)
    w0  = copy(w)
    pw  = copy(b)
    y2  = zeros(T, q)
    z   = zeros(T, q)
    z2  = zeros(T, p)
    r   = zero(T)
    r0  = r
    R   = r

    loss    = sqeuclidean(x,u) / 2 
    loss0   = Inf
    dw      = Inf
    dr      = Inf

    # factorize matrix A'*A + c*c'
    BLAS.gemm!('T', 'N', one(T), A, A, zero(T), AA)
    BLAS.ger!(one(T), c, c, AA)
    ev,V = eig(AA) 

    if !quiet
        @printf("Iter\tLoss\tNorm\tdw\tdr\tRho\n")
        @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", 0, loss, Inf, dw, dr, rho)
    end
    i = 0
    for i = 1:max_iter

        # compute accelerated step U = u + (i - 1)/(i + 2)*(u - u0)
        kx = (i - one(T)) / (i + one(T) + one(T))
        ky = one(T) + kx
        U .= ky.*u .- kx.*u0
#        compute_accelerated_step!(U, u, u0, i)

        # save previous iterate
        #copy!(u0,u)

        # update w, r with new u
        copy!(W,b)
        BLAS.gemv!('N', one(T), A, U, one(T), W)
        R = dot(c,U) + d
        pr = proj_soc!(pw,W,R)
#        pw, pr = proj_soc(W,R)

        # for prox dist update need to recompute z
        # z = A' * (b - pw) + ((d - pr)*c)
        copy!(z2,b)
        BLAS.axpy!(-one(T), pw, z2)
        BLAS.gemv!('T', one(T), A, z2, zero(T), z)
        BLAS.axpy!(d - pr, c, z)

        # prox dist update uses quadratic proximal map 
        # u = (invrho*I + AA) \ (invrho*x + A'*(P(w) - b) + (P(r) - d)c)
        prox_quad!(u, y2, V, ev, z, x, rho)

        # update w, r with new u
        copy!(w,b)
        BLAS.gemv!('N', one(T), A, u, one(T), w)
        r = dot(c,u) + d

        # convergence checks
        loss        = sqeuclidean(u,x) / 2 
        dw          = euclidean(w,pw)
        dr          = sqrt(abs(r*r - 2*r*pr + pr*pr)) 
        feas        = dw < feastol && dr < feastol
        the_norm    = euclidean(u,u0)
        scaled_norm = the_norm / (norm(u0,2) + one(T))
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
    x        :: Vector{T},
    A        :: Matrix{T},
    b        :: Vector{T},
    c        :: Vector{T},
    d        :: T;
    rho      :: T    = one(T),
    rho_inc  :: T    = one(T) + one(T),
    rho_max  :: T    = 1e15,
    max_iter :: Int  = 10000,
    inc_step :: Int  = 100,
    tol      :: T    = 1e-6,
    feastol  :: T    = 1e-6,
    quiet    :: Bool = true,
) where {T <: AbstractFloat}
    xp = zeros(size(x))
    proj_soc!(xp,x,A,b,c,d,rho=rho,rho_inc=rho_inc,rho_max=rho_max,max_iter=max_iter,inc_step=inc_step,tol=tol,feastol=feastol,quiet=quiet)
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
    x        :: SparseMatrixCSC{T,Int},
    A        :: SparseMatrixCSC{T,Int},
    b        :: SparseMatrixCSC{T,Int},
    c        :: SparseMatrixCSC{T,Int},
    d        :: T;
    rho      :: T = one(T),
    rho_inc  :: T = 2.0,
    rho_max  :: T = 1e30,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    tol      :: T = 1e-6,
    feastol  :: T = 1e-6,
    quiet    :: Bool    = true,
) where {T <: AbstractFloat}

    # error checking
    check_conformability(A,b,c)

    p,q = size(A)
    At = A' 
    u  = sprandn(q 1, 0.1)
    v2 = zeros(T p)
    U  = copy(u) 
    u0 = copy(u)
    w  = spzeros(T p, 1)
    W  = copy(w) 
    pw = copy(b)
    z  = spzeros(T q, 1)
    r  = zero(T)
    pr = r
    R  = r

    loss    = sqeuclidean(x,u) / 2
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
#    Afun = MatrixFcn{T}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho))
#    Afun = MatrixFcn{T}(q, q, (output, v) -> mulbyA!(output, v, A, full(c), rho))
#    Afun = MatrixFcn{T}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho, v2))
    f(output, v) = mulbyA!(output, v, A, c, rho, v2)
    Afun = LinearMap(f, q, q, ismutating = true)

    if !quiet
        @printf("Iter\tLoss\tNorm\tdw\tdr\tRho\n")
        @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", 0, loss, Inf, dw, dr, rho)
    end
    i = 0
    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - one(T)) / (i + one(T) + one(T))
        ky = one(T) + kx
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
        scaled_norm = the_norm / (norm(u0) + one(T))
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
#            Afun = MatrixFcn{T}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho))
#            Afun = MatrixFcn{T}(q, q, (output, v) -> mulbyA!(output, v, A, full(c), rho))
#            Efact = qrfact(E)
#            Afun = MatrixFcn{T}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho, v2))
            f(output, v) = mulbyA!(output, v, A, c, rho, v2)
            Afun = LinearMap(f, q, q, ismutating = true)
            copy!(u0,u)
        end
    end

    # threshold small elements of y before returning
    threshold!(u.nzval,tol)
    return u
end


function proj_soc(
    x        :: Vector{T},
    A        :: SparseMatrixCSC{T,Int},
    b        :: Vector{T},
    c        :: Vector{T},
    d        :: T;
    rho      :: T = one(T),
    rho_inc  :: T = 2.0,
    rho_max  :: T = 1e30,
    max_iter :: Int     = 10000,
    inc_step :: Int     = 100,
    tol      :: T = 1e-6,
    feastol  :: T = 1e-6,
    quiet    :: Bool    = true,
    p        :: Int     = length(b), 
    q        :: Int     = length(c), 
    At       :: SparseMatrixCSC{T,Int} = A', 
    u        :: Vector{T} = zeros(T, q), 
    v2       :: Vector{T} = zeros(T, p), 
    U        :: Vector{T} = zeros(T, q), 
    u0       :: Vector{T} = zeros(T, q), 
    w        :: Vector{T} = zeros(T, p), 
    W        :: Vector{T} = zeros(T, p), 
    pw       :: Vector{T} = zeros(T, p), 
    z        :: Vector{T} = zeros(T, q), 
    b0       :: Vector{T} = zeros(T, q), 
    r        :: T = zero(T),
    pr       :: T = r,
    R        :: T = r,
) where {T <: AbstractFloat}

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
    #Afun = MatrixFcn{T}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho, v2))
            f(output, v) = mulbyA!(output, v, A, c, rho, v2)
            Afun = LinearMap(f, q, q, ismutating = true)

    if !quiet
        @printf("Iter\tLoss\tNorm\tdw\tdr\tRho\n")
        @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", 0, loss, Inf, dw, dr, rho)
    end
    i = 0
    for i = 1:max_iter

        # compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - one(T)) / (i + one(T) + one(T))
        ky = one(T) + kx
        #difference!(U, u, u0, a=ky, b=kx)
        U .= ky.*u .- kx.*u0
        copy!(u0,u)

        # compute projections onto constraint sets
        # W  = A*U + b, R = dot(c,U) + d
        A_mul_B!(W,A,U)
        BLAS.axpy!(one(T), b, W)
        R  = dot(c,U) + d
        pr = proj_soc!(pw,W,R)

        # update z for prox dist update
        # z  = At*(b-pw) + (d-pr)*c
        #difference!(v2, b, pw)
        v2 .= b .- pw
        A_mul_B!(z, At, v2)
        BLAS.axpy!(d-pr, c, z)

        # update b0 for CG inversion
#        b0 = [vec(full(x - rho*z)); zeros(p+1)]
        #difference!(b0, x, z, a=1.0, b=rho)
        b0 .= x .- rho.*z
#        A_mul_B!(v2, E, b0)

        # prox dist update 
#        u2 = Efact \ b0
#        u2 = vec(full(U))
        cg!(u, Afun, b0, maxiter=200, tol=1e-8)
#        lsqr!(u, E, v2, maxiter=200, atol=1e-8, btol=1e-8)

        # now update w,r
        # w = A*u + b, r = dot(c,u) + d 
        A_mul_B!(w, A, u)
        BLAS.axpy!(one(T), b, w)
        r = dot(c,u) + d

        # convergence checks
        loss        = 0.5*sqeuclidean(u,x) 
        dw          = euclidean(w,pw)
        dr          = sqrt(abs(r*r - 2*r*pr + pr*pr)) 
        feas        = dr < feastol && dw < feastol
        the_norm    = euclidean(u,u0) 
        scaled_norm = the_norm / (norm(u0) + one(T))
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
#            Afun = MatrixFcn{T}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho))
            #Afun = MatrixFcn{T}(q, q, (output, v) -> mulbyA!(output, v, A, c, rho, v2))
            f(output, v) = mulbyA!(output, v, A, c, rho, v2)
            Afun = LinearMap(f, q, q, ismutating = true)
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
