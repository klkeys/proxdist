using Gurobi, Ipopt, LinearMaps, Mosek, SCS
import Convex, IterativeSolvers, MathProgBase

struct ρscheme
    ρstart::Float64
    ρmax::Float64
    ρincfactor::Float64
    ρincmaxiter::Int
    ρinctol::Float64
    ρmaxiter::Int
    ρstoptol::Float64
end

function ρscheme()
    # default setting for ρscheme
    ρscheme(1.0, 1e14, 1.5, 200, 1e-7, 100, 1e-7)
end

function cls(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    proj::Function,
    β::AbstractVector{T} = zeros(T, size(X, 2)),
    ρs::ρscheme = ρscheme()
    ) where T <: AbstractFloat
    n, p = size(X)
    # pre-allocate working arrays
    βproj, βouter, βinner = similar(β), similar(β), similar(β)
    proj(βproj, β)
    # precompute: eigen-decomposition of X'X, X'y
    Geig = eigfact!(Symmetric(X.'X))
    xty = X.'y
    storagep = similar(β)
    # main loop
    ρ = ρs.ρstart
    for iter in 1:ρs.ρmaxiter
        copy!(βouter, β) # record solution at previous ρ
        for iter_inner in 1:ρs.ρincmaxiter
            copy!(βinner, β)
            # solve β = (X'X + ρ * I) \ (X'y + ρ * βproj)
            β .= xty .+ ρ .* βproj
            At_mul_B!(storagep, Geig.vectors, β)
            storagep .= storagep ./ (Geig.values .+ ρ)
            A_mul_B!(β, Geig.vectors, storagep)
            # project β
            proj(βproj, β)
            # decide to increase ρ or not
            storagep .= β .- βinner
            if vecnorm(storagep) < ρs.ρinctol * (vecnorm(βinner) + 1)
                # println("ρ=", ρ, " iters=", iter_inner)
                break
            end
        end
        # decide to stop or not
        storagep .= β .- βouter
        d1 = vecnorm(storagep)
        storagep .= β .- βproj
        d2 = vecnorm(storagep)
        if d1 < ρs.ρstoptol * (vecnorm(βouter) + 1) && 
            d2 < ρs.ρstoptol * (vecnorm(βproj) + 1)
            break
        else
            ρ = min(ρs.ρincfactor * ρ, ρs.ρmax)
        end
    end
    proj(βproj, β)
    βproj, (1//2) * vecnorm(y - X * βproj)^2
end

function cls_simplex_convex(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    solver
    ) where {T <: AbstractFloat}
    n, p = size(X)
    β = Convex.Variable(p)
    loss = vecnorm(y - X * β)
    # loss = sumsquares(y - X * β) # slower
    eqconstr = sum(β) == 1
    ineqconstr = β ≥ 0
    problem = Convex.minimize(loss, eqconstr, ineqconstr)
    Convex.solve!(problem, solver)
    βproj = similar(β.value, p)
    SimplexProjection!(βproj, β.value[:])
    βproj, (1//2) * vecnorm(y - X * βproj)^2
end

function cls_simplex_mpb(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    solver = IpoptSolver(print_level=0)
    ) where {T <: AbstractFloat}
    n, p = size(X)
    c = - X.'y
    Q = X.'X
    A = ones(T, 1, p)
    sense = '='
    b = [one(T)]
    l = zeros(T, p)
    u = ones(T, p)
    qpsol = MathProgBase.quadprog(c, Q, A, sense, b, l, u, solver)
    βproj = Array{T}(p)
    SimplexProjection!(βproj, qpsol.sol)
    βproj, (1//2) * vecnorm(y - X * βproj)^2
end

function cls(
    X::AbstractSparseMatrix{T},
    y::AbstractVector{T},
    proj::Function,
    β::AbstractVector{T} = zeros(T, size(X, 2)),
    ρs::ρscheme = ρscheme()
    ) where T <: AbstractFloat
    n, p = size(X)
    βproj, βouter, βinner = similar(β), similar(β), similar(β)
    proj(βproj, β)
    ρ = [ρs.ρstart]
    # working arrays
    xty = X.'*y
    storagen = zeros(T, n)
    storagep = similar(β)
    storagecg = IterativeSolvers.CGStateVariables{eltype(β), typeof(β)}(zeros(β), similar(β), similar(β))
    # linear mapping for the augmented least squares system
    A = LinearMap{T}((out, in) -> Gfun!(out, in, X, ρ[1], storagen), p;
        ismutating = true, isposdef = true)
    # main loop
    for iter in 1:ρs.ρmaxiter
        copy!(βouter, β) # record solution at previous ρ
        for iter_inner in 1:ρs.ρincmaxiter
            copy!(βinner, β)
            storagep .= xty .+ ρ[1] .* βproj
            # solve β = (X'X + ρ * I) \ (X'y + ρ * βproj)
            IterativeSolvers.cg!(β, A, storagep, statevars=storagecg, maxiter=20, log=false)
            # β = [X; sqrtρ[1] * speye(p)] \ yaug
            proj(βproj, β)
            # decide to increase ρ or not
            storagep .= β .- βinner
            if vecnorm(storagep) < ρs.ρinctol * (vecnorm(βinner) + 1)
                # println("ρ=", ρ, " iters=", iter_inner)
                break
            end
        end
        # decide to stop or not
        storagep .= β .- βouter
        d1 = vecnorm(storagep)
        storagep .= β .- βproj
        d2 = vecnorm(storagep)
        if d1 < ρs.ρstoptol * (vecnorm(βouter) + 1) && 
            d2 < ρs.ρstoptol * (vecnorm(βproj) + 1)
            break
        else
            ρ[1] = min(ρs.ρincfactor * ρ[1], ρs.ρmax)
        end
    end
    proj(βproj, β)
    βproj, (1//2) * vecnorm(y - X * βproj)^2
end

"""
Calculate `(X'X + ρ I) v`.
"""
function Gfun!(
    out::AbstractVector{T},
    v::AbstractVector{T},
    X::AbstractMatrix{T},
    ρ::T,
    storagen::AbstractVector{T} = Array{T}(size(X, 1))
    ) where {T <: AbstractFloat}
    A_mul_B!(storagen, X, v)
    At_mul_B!(out, X, storagen)
    out .+= ρ .* v
end

"""
SimplexProjection!(yproj, y[, r])

Overwrite `yproj` by projection of `y`` onto the simplex {x|x >= 0, sum(x) = r}.
"""
function SimplexProjection!(
    yproj::Vector{T},
    y::Vector{T}, 
    r = one(T)
    ) where T <: Real
    n = length(y)
    copy!(yproj, y)
    sort!(yproj, rev = true)
    s = λ = zero(T)
    for i in 1:n
        s = s + yproj[i]
        λ = (s - r) / i
        if i < n && λ < yproj[i] && λ ≥ yproj[i + 1]
            break
        end
    end
    for i in 1:n
        yproj[i] = max(y[i] - λ, 0)
    end
    yproj
end
