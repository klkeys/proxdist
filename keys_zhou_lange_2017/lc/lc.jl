"""
    complementarity_projection(s,t)

This function projects the vectors `s` and `t` onto the set where `dot(s,t) = 0` and all components of `s` and `t` are nonnegative.
This projection is of value in solving the linear complementarity problem.
"""
function complementarity_projection(s::Vector{T}, t::Vector{T}) where {T <: AbstractFloat}
    n     = length(s)
    (x,y) = (zeros(T, n), zeros(T, n))

    for i = 1:n
        if s[i] >= 0 && s[i] >= t[i]
            (x[i], y[i]) = (s[i], zero(T))
        elseif t[i] >= 0 && t[i] >= s[i]
            (x[i], y[i]) = (zero(T), t[i])
        #       elseif s[i] < 0 && t[i] < 0
        #          (x[i], y[i]) = (zero(T), zero(T))
        #       else
        #          println("projection error")
        end
    end

    return (x, y)
end

"""
    linear_complementarity(A,b)

This function solves the linear complementarity problem by an accelerated proximal distance algorithm.
"""
function linear_complementarity(A::Matrix{T}, b::Vector{T}) where {T <: AbstractFloat}

    # Initialize control constants.
    ρ        = one(T)
    ρ_max    = 2^22
    max_iter = 5000
    iters    = 0
    quiet    = true

    # Initialize arrays and the loss.
    p = size(A, 1)
    (x, y , xa, ya, xb, yb) = (zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p))

    # Perform the projection.
    (px, py) = complementarity_projection(x, y)
    dist = vecnorm(x - px) + vecnorm(y - py)
    loss = (vecnorm(y - A*x - b)^2) / 2

    # Enter the proximal distance loop.
    for n = 1:max_iter

        iters = iters + 1
        xb = xa + ( (n - one(T)) / (n + 2*one(T)) )*(xa - x)
        yb = ya + ( (n - one(T)) / (n + 2*one(T)) )*(ya - y)
        x = deepcopy(xa)
        y = deepcopy(ya)     

        # Project onto the constraints.
        (px, py) = complementarity_projection(xb, yb)
        dist     = sqrt(vecnorm(xb - px)^2 + vecnorm(yb - py)^2)
        loss     = (vecnorm(yb - A*xb - b)^2) / 2
        obj      = loss + (ρ*dist^2) / 2

        # Output the current iterate.
        if (n <= 10 || n % 100 == 0) && !quiet 
            println(n,"  &   ",loss,"  & ",dist,"  &  ",ρ)
        end

        # Calculate the proximal distance update.
        xa = ( (one(T) + ρ)*eye(p) + A'*A) \ (A'*(py - b) + (one(T) + ρ)*px)
        ya = (A*xa + b) / (one(T) + ρ) + (ρ / (one(T) + ρ))*py

        # Check for convergence.
        next_loss = vecnorm(ya - A*xa - b)^2 / 2
        if ( abs(next_loss - loss)/(abs(loss) + one(T)) < 1e-8 && dist < 1e-5 )
            break
        else
            loss = next_loss
        end

        if n % 100 == 0 
            ρ = min(2*ρ, ρ_max)
            x = deepcopy(xa)
            y = deepcopy(ya)
        end
    end

    return (iters, xa, ya)
end

"""
This function solves the linear complementarity problem by an accelerated proximal distance algorithm.
"""
function linear_complementarity_milp(A::Matrix{T}, b::Vector{T}) where {T <: AbstractFloat}
  n = length(b)

  # use Gurobi solver
  m = Model(solver = GurobiSolver(OutputFlag = 0))

  # define variables
  @variable(m, x[1:n] >= 0.0)
  @variable(m, y[1:n] >= 0.0)

  # equality constraint y = Ax + b
  for i = 1:n
    @constraint(m, y[i] == sum(A[i, j] * x[j] for j = 1:n) + b[i])
  end

  # complimentarity constraint
  for i = 1:n
    addSOS1(m, [x[i], y[i]])
  end

  # no objective function -> feasibility problem
  @objective(m, Min, 0.0)

  # ready, set, go!
  #display(m)
  status = solve(m)

  # Return
  xsol = zeros(n); ysol = zeros(n)
  for i = 1:n
    @inbounds xsol[i] = getvalue(x[i])
    @inbounds ysol[i] = getvalue(y[i])
  end
  return xsol, ysol, m
end
