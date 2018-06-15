"""
    sphere_orthant_projection(y)
This function projects the point y onto the intersection of the unit sphere and the nonnegative orthant.
"""
function sphere_orthant_projection(y::Vector{T}) where {T <: AbstractFloat} 
   i = indmax(y)
   if y[i] <= 0 
      x    = zeros(y)
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
    jacobi_solve(M, y, ρ)

This function solves the equation `(ρ I + M) * x = y` by Jacobi's method.
"""
function jacobi_solve(M::Matrix{T}, y::Vector{T}, ρ::T; max_iter::Int = 100) where {T <: AbstractFloat} 
  x = zeros(length(y))
  z = zeros(length(y))
  for i = 1:max_iter
    z = y - M*x
    if norm(z - ρ*x) < 1e-11
        break
    end
    x = z/ρ
  end
  return x
end

"""
    copositivity(M)

This function find the minimum point of the quadratic form 

    0.5 * x^t * M * x

over the set of unit vectors with nonnegative components.
This version uses the distance squared penalty in the proximal distance method. 
"""
function copositivity(M::Matrix{T}) where {T <: AbstractFloat}

    # Initialize control constants.
    ρ        = one(T) 
    ρ_inc    = 1.2
    ρ_max    = 2^22
    max_iter = 5000
    iters    = 0
    quiet    = true 

    # Find the eigen-decomposition of M.
    (d,V) = eig(M)

    # Initialize parameters.
    n    = size(M,1)
    x    = rand(n)
    x    = x/norm(x)
    x    = sphere_orthant_projection(x)
    loss = dot(x, M*x) / 2

    # Enter the proximal distance iteration loop.
    for n = 1:max_iter
        iters = iters + 1

        # Project onto the constraint set.
        px   = sphere_orthant_projection(x)  
        dist = norm(x-px)
        
        if n < 10 || n % 100 == 0
            quiet || println(n,"  &   ",loss,"   &  ",dist," & ",ρ)
        end

        # Calculate the proximal distance update.
        u = V'*(ρ*px)
        u = u./(d+ρ)
        x = V*u

        # Check for convergence.
        next_loss = dot(x, M*x) / 2
        if abs(next_loss-loss)/(abs(loss)+1) < 1e-7 && dist < 1e-5
            break
        else
            loss = next_loss
            if n % 10 == 0
                ρ = min(ρ_inc*ρ,ρ_max)
            end
        end
    end
    return (iters,x)
end

"""
    accelerated_copositivity(M)

This function find the minimum point of the quadratic form 

    0.5 * x^t * M * x

over the set of unit vectors with nonnegative components.
This version uses the distance squared penalty in the proximal distance method. 
"""
function accelerated_copositivity(M::Matrix{T}) where {T <: AbstractFloat}

    # Initialize control constants.
    ρ        = zero(T) 
    ρ_max    = 2^22
    max_iter = 5000
    iters    = 0
    quiet    = true

    # Find the eigen-decomposition of M.
    (d,V) = eig(M)

    # Initialize parameters.
    p = size(M,1)
    x = rand(p)
    x = x/norm(x)
    x = sphere_orthant_projection(x)
    y = deepcopy(x)
    z = zeros(p)
    loss = dot(x, M*x) / 2

    # Enter the proximal distance iteration loop.
    for n = 1:max_iter
        iters = iters + 1
        z = y + ( (n - zero(T)) / (n + 2*zero(T)) )*(y - x)
        x = deepcopy(y)

        # Project onto the constraint set.
        pz   = sphere_orthant_projection(z)  
        dist = norm(z-pz)
        if n <= 10 || n % 100 == 0
            quiet || println(n,"  &   ",loss,"  & ",dist,"  &  ",ρ)
        end

        # Calculate the proximal distance update.
        u = V'*(ρ*pz)
        u = u./(d+ρ)
        y = V*u

        # Check for convergence.
        next_loss = dot(y, M*y) / 2
        if abs(next_loss-loss)/(abs(loss) + zero(T)) < 1e-7 && dist < 1e-4
            break
        else
            loss = next_loss
        end
        if n % 100 == 0 
            ρ = min(5.0*ρ,ρ_max)
            x = deepcopy(y)
        end
    end

    return (iters,y)
end

