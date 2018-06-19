using Convex

# Use Mosek solver
using Mosek
#solver = MosekSolver(LOG = 0, MSK_DPAR_OPTIMIZER_MAX_TIME = 1.0)
solver = MosekSolver(LOG = 0)
set_default_solver(solver)

## Use SCS solver
#using SCS
#solver = SCSSolver(verbose=0)
#set_default_solver(solver)

"""
This function projects the point y onto the intersection of the unit sphere and the nonnegative orthant.
"""
function sphere_orthant_projection(y::Vector{T}) where {T <: AbstractFloat}
   i = indmax(y)
   if y[i] <= 0
      x = zeros(y)
      x[i] = one(T)
      return x
   else
      x = deepcopy(y)
      for j = 1:length(x)
         if x[j] < 0
            x[j] = zero(T)
         end
      end
      return x / norm(x)
   end
end

"""
This function solves the equation (rho I + M) * x = y by Jacobi's method.
"""
function jacobi_solve(M::Matrix{T}, y::Vector{T}, rho::T) where {T <: AbstractFloat}
    x = zeros(T, length(y))
    z = zeros(T, length(y))
    for i = 1:100
        z = y - M*x
        if norm(z - rho*x) < 1e-11 break end
            x = z/rho
        end
    return x
end

"""
 This function find the minimum point of the quadratic form
       0.5 x^t M x
 over the set of unit vectors with nonnegative components.
 This version uses the distance squared penalty in the proximal
 distance method.
"""
function copositivity(M::Matrix{T}) where {T <: AbstractFloat}

    # Initialize control constants.
    rho = one(T)
    rho_inc = convert(T, 6/5) 
    rho_max = 2^22
    max_iter = 5000
    iters = 0

    # Find the eigen-decomposition of M.
    (d,V) = eig(M)

    # Initialize parameters.
    n = size(M, 1)
    x = rand(n)
    x = x / norm(x)
    x = sphere_orthant_projection(x)
    loss = dot(x, M * x) / 2

    # Enter the proximal distance iteration loop.
    for n = 1:max_iter
        iters = iters + 1

        # Project onto the constraint set.
        px = sphere_orthant_projection(x)
        dist = norm(x - px)

        # if n < 10 || n%100 == 0
        #    println(n,"  &   ",loss,"   &  ",dist," & ",rho)
        # end

        # Calculate the proximal distance update.
        u = V' * (rho * px)
        u = u ./ (d + rho)
        x = V * u

        # Check for convergence.
        next_loss = dot(x, M * x) / 2
        if abs(next_loss - loss) / (abs(loss) + 1) < 1e-7 && dist < 1e-5
            break
        else
            loss = next_loss
            if n%10 == 0
                rho = min(rho_inc * rho, rho_max)
            end
        end
    end
    return (iters, sphere_orthant_projection(x))
end

"""
 This function find the minimum point of the quadratic form
       0.5 x^t M x
 over the set of unit vectors with nonnegative components.
 This version uses the distance squared penalty in the proximal
 distance method.
"""
function accelerated_copositivity(M::Matrix{T}) where {T <: AbstractFloat}

    # Initialize control constants.
    rho = one(T)
    rho_max = 2^22
    max_iter = 5000
    iters = 0

    # Find the eigen-decomposition of M.
    (d, V) = eig(M)

    # Initialize parameters.
    p = size(M, 1)
    x = rand(p)
    x = x / norm(x)
    x = sphere_orthant_projection(x)
    y = deepcopy(x)
    z = zeros(p)
    loss = dot(x, M * x) / 2

    # Enter the proximal distance iteration loop.
    for n = 1:max_iter
        iters = iters + 1
        z = y + ((n - one(T)) / (n + 2*one(T))) * (y - x)
        x = deepcopy(y)

        # Project onto the constraint set.
        pz = sphere_orthant_projection(z)
        dist = norm(z - pz)

        # if n <= 10 || n%100 == 0
        #    println(n,"  &   ",loss,"  & ",dist,"  &  ",rho)
        # end

        # Calculate the proximal distance update.
        u = V' * (rho * pz)
        u = u ./ (d + rho)
        y = V * u

        # Check for convergence.
        next_loss = dot(y, M * y) / 2
        if abs(next_loss - loss) / (abs(loss) + one(T)) < 1e-7 && dist < 1e-4
            break
        else
            loss = next_loss
        end

        if n%100 == 0
            rho = min(5*rho, rho_max)
            x = deepcopy(y)
        end
    end
    return (iters, sphere_orthant_projection(y))
end

"""

    CoPositivityBySDP(::AbstractArray{T}) -> {Bool}

Check whether M is a copositive matrix by semidefinite programming. M is
copositive if M = A + B, where A is psd and B has nonnegative entries. Note this
is only a sufficient condition. The function returns `true` if such a
decomposition is found and `false` if otherwise.
"""
function CoPositivityBySDP(M::Array{T}) where {T <: AbstractFloat}
  n = size(M, 1)
  A = Convex.Variable(n, n)
  B = Convex.Variable(n, n)
  constraints = (M == A + B)
  constraints += (A in :SDP)
  constraints += B >= 0.0
  problem = satisfy(constraints)
  try solve!(problem)
  catch e
    return "timeout"
  end
  if problem.status == :Optimal
    return "feasible"
  else
    return "infeasible"
  end
end

"""

    horn_matrix(::Int) -> Array{T}

Generate an `n`-by-`n` Horn matrix.
"""
function horn_matrix(T::Type, n::Int)
  M = ones(T, n, n)
  for j = 2:n
    M[j, j-1] = -one(T)
    M[j-1, j] = -one(T)
  end
  M[1, n] = -one(T)
  M[n, 1] = -one(T)
  return M
end

horn_matrix(n::Int) = horn_matrix(Float64, n)
