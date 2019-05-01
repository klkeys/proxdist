using RegressionTools
using ProxOpt
include("jacobi.jl")

###################
### subroutines ###
###################
function logsum(
    x :: DenseVector{Float64}
)
    s = zero(Float64)
    p = length(x)
    @inbounds for i = 1:p
        s += log(x[i])
    end
    return s
end


######################
### main functions ###
######################


"""
    spm(S, k) -> Matrix

Find the closest sparse representation to a given precision matrix S.
"""
function spm(
    S         :: DenseMatrix{Float64}, 
    k         :: Int;
    Y         :: DenseMatrix{Float64} = copy(S),
    rho       :: Float64 = 1.0,
    rho_inc   :: Float64 = 2.0,
    rho_max   :: Float64 = 1e20,
    tol       :: Float64 = 1e-6,
    sparsetol :: Float64 = 1e-6,
    max_iter  :: Int     = 10000,
    inc_step  :: Int     = 100,
    quiet     :: Bool    = true,
#    jtol      :: Float64 = sqrt(sum(triu(S, 1).^2)) / (4.0 * size(S,1)), 
)
    # get size of S 
    (p,q) = size(S)

    # error checking
#    size(Yk) == (p,q) || throw(DimensionMismatch("Arguments Yk and Y must be of same size"))
    isequal(p,q)      || throw(ArgumentError("S must be square"))

    # initialize temporary arrays 
#    X   = eye(p)
    X   = zeros(p,p)
    Yk  = copy(Y)
    Yk2 = copy(Y)
#    Z   = eye(p)
#    Z   = zeros(p,p)
    d   = zeros(Float64, p)
    ev  = zeros(Float64, p)
#    V   = zeros(Float64, p, p)
#    d2,V = eig(S)
#    d2  = diag(S)
    idx = collect(1:p^2)
#    w   = zeros(Float64, p)
#    bw  = copy(d)
#    zw  = zeros(Float64, p)

    # initialize loss function and feasible distance 
    loss      = Inf 
    next_loss = Inf
    dsparse   = Inf 

    # iteratively compute nearest sparse precision matrix
    i = 0
    for i = 1:max_iter

        # Nesterov acceleration step
#        kx = (i - one(Float64)) / (i + one(Float64) + one(Float64))
#        ky = one(Float64) + kx
#        difference!(Z, Y, X, a=ky, b=kx) 
        copy!(X,Y)

        # project onto the nearest sparse symmetric matrix 
#        project_k!(Yk, Z, k, Yk2=Yk2, m=p, n=p, d=d, idx=idx) 
        project_k!(Yk, Y, k, Yk2=Yk2, m=p, n=p, d=d, idx=idx) 

        # prox dist update
#        i == 1 && prox_spm!(Y, S, Yk, rho, Y0=Z)
#        i > 1  && prox_spm!(Y, V, d2, S, Yk, rho, Y0=Z, bw=bw, zw=zw, tol=jtol)
#        prox_spm!(Y, S, Yk, rho, Y0=Z)
        prox_spm!(Y, S, Yk, rho, Y0=Yk2, ev=ev)

        # convergence checks
#        next_loss   = -logdet(Y) + vecdot(S,Y)
        next_loss   = -2.0*logsum(ev) + vecdot(S,Y)
        dsparse     = vecnorm(Y,Yk)
        amisparse   = dsparse < sparsetol
        the_norm    = vecnorm(X,Y)
        scaled_norm = the_norm / (vecnorm(X) + one(Float64))
        converged   = scaled_norm < tol && amisparse 

        # if converged, then exit
        # in contrary case, save previous loss function 
        converged && break
        loss = next_loss

        # monitor output
        if (i <= 10 || i % inc_step == 0) && !quiet
            @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dsparse, rho)
        end

        if i % inc_step == 0 
            rho = min(rho_inc*rho, rho_max)
            copy!(X,Y)
        end
    end

    copy!(Yk,Y)
    threshold!(Yk,tol)
    return Dict{ASCIIString, Any}("obj" => loss, "iter" => i, "X" => Yk, "sparse_dist" => dsparse)
end

function test_spm()

    # seed RNG
    seed = 2016
    srand(seed)

    # testing parameters
    p        = 8 
    k        = (p-1) + (p-2) + (p-3) 
    tol      = 1e-6
    rho      = 1.0
    rho_inc  = 2.0
    rho_max  = 1e30
    inc_step = 1
    quiet    = false 

    # make banded covariance matrix
    x = randn(p,p)
    u = spdiagm( (diag(x,-3), diag(x,-2), diag(x,-1), diag(x,0)), (-3,-2,-1,0), p, p)
    P = u' * u + 0.01*rand(p,p)
    P = 0.5*(P + P')
    S = inv(P)
    S = 0.5*(S + S')

#    Y = inv(S)

    # run algo twice
    # time second run
#    output = spm(S,k, tol=tol, rho=rho, rho_inc=rho_inc, rho_max=rho_max, inc_step=inc_step, quiet=quiet, Y=Y)
    output = spm(S,k, tol=tol, rho=rho, rho_inc=rho_inc, rho_max=rho_max, inc_step=inc_step, quiet=quiet)
#    @time output = spm(S,k, tol=tol, rho=rho, rho_inc=rho_inc, rho_max=rho_max, inc_step=inc_step, quiet=quiet)

    # to compare recovery performance, find the nonzeroes in x and the original u'*u 
    x   = output["X"]
    utu = full(u'*u)
    xnz = x   .!= 0.0
    unz = utu .!= 0.0

    # report
    println("Original nonzeroes: ", 2*k)
    println("Total nonzeroes in x: ", sum(xnz) - p)
    println("True positives in x: ", sum(xnz & unz) - p)
    println("Euclidean difference: ", vecnorm(utu,x))

    println("x = ")
    display(x)
    println("")
    println("utu = "),
    display(utu)
    println("")

    return nothing 
end

test_spm()
