"""
    positive_definite_projection(Y)

This function projects the symmetric matrix `Y` onto the closest positive definite matrix.
"""
function positive_definite_projection(Y::Matrix{Float64})
    (D,V) = eig(Y)
    n = size(Y,1)
    X = zeros(Y)
    for j = 1:n
        if D[j] > 0.0
            X = X + D[j] * V[:,j] * V[:,j]'
        end   
    end
    for i = 1:n
        for j = i+1:n
            X[i,j] = X[j,i]
        end
    end
    return X
end

"""
    nonnegative_correlation_projection(Y)

This function projects a square matrix `Y` onto the set of nonnegative matrices with ones on the diagonal.
"""
function nonnegative_correlation_projection(Y::Matrix{Float64})
   X = copy(Y)
   n = size(Y,1)
   for i = 1:n
      for j = 1:n
         X[i,j] = max(X[i,j],0.0)
      end
      X[i,i] = 1.0
   end
   return X
end

"""
    closest_kinship_matrix(M)

This function finds the closest kinship matrix to the symmetric matrix `M` (actually, twice the kinship matrix) by a proximal distance algorithm.
"""
function closest_kinship_matrix(M::Matrix{Float64})

    # Initialize control constants.
    ρ        = 1.0
    ρ_max    = 2^22
    max_iter = 5000
    iters    = 0

    # Initialize arrays and the loss.
    (p,q) = size(M)
    X = copy(M)
    Y = copy(M)
    Z = similar(M)
    U = nonnegative_correlation_projection(M)
    V = positive_definite_projection(X)
    dist = vecnorm(M - U) + vecnorm(M - V)
    loss = 0.5*vecnorm(M - X)^2

    # Enter the proximal distance loop.
    for n = 1:max_iter
        iters = iters + 1
        Z = Y #+ ((n - 1.0)/(n + 2.0))*(Y - X)
        X = deepcopy(Y)

        # Project onto the constraints.
        U = nonnegative_correlation_projection(Z)
        V = positive_definite_projection(Z)
        dist = vecnorm(Z - U) + vecnorm(Z - V)
        loss = 0.5*vecnorm(M - Y)^2

        # Calculate the proximal distance update.
        Y = (1.0/(1.0 + ρ))*(M + (0.5*ρ)*U + (0.5*ρ)*V)

        # Check for convergence.
        next_loss = 0.5*vecnorm(M - Y)^2
        if abs(next_loss-loss)/(abs(loss)+1.0) < 1e-6 && dist < 1e-4
            break
        else
            loss = next_loss
        end

        if n % 100 == 0
            ρ = min(1.2*ρ,ρ_max)
            X = deepcopy(Y)
        end
    end

    return (iters,Y)
end

"""
    accelerated_closest_kinship_matrix(M)

This function finds the closest kinship matrix to the symmetric matrix `M` (actually, twice the kinship matrix) by an _accelerated_ proximal distance algorithm.
"""
function accelerated_closest_kinship_matrix(M::Matrix{Float64})

    # Initialize control constants.
    ρ        = 1.0
    ρ_max    = 2^22
    max_iter = 5000
    iters    = 0

    # Initialize arrays and the loss.
    (p,q) = size(M)
    X     = copy(M)
    Y     = copy(M)
    Z     = similar(M)
    U     = nonnegative_correlation_projection(M)
    V     = positive_definite_projection(X)
    dist  = vecnorm(M - U) + vecnorm(M - V)
    loss  = 0.5*vecnorm(M - X)^2

    # Enter the proximal distance loop.
    for n = 1:max_iter

        iters = iters + 1
        Z     = Y + ((n - 1.0)/(n + 2.0))*(Y - X)
        X     = deepcopy(Y)

        # Project onto the constraints.
        U    = nonnegative_correlation_projection(Z)
        V    = positive_definite_projection(Z)
        dist = vecnorm(Z - U) + vecnorm(Z - V)
        loss = 0.5*vecnorm(M - Y)^2

        # Calculate the proximal distance update.
        Y = (1.0/(1.0 + ρ))*(M + (0.5*ρ)*U + (0.5*ρ)*V)

        # Check for convergence.
        next_loss = 0.5*vecnorm(M - Y)^2
        if abs(next_loss-loss)/(abs(loss)+1.0) < 1e-6 && dist < 1e-4
            break
        else
            loss = next_loss
        end

        if n % 100 == 0
            ρ = min(5.0*ρ,ρ_max)
            X = deepcopy(Y)
        end
    end
    return (iters,Y)
end

"""
    accelerated_closest_kinship_matrix2(M)

This function finds the closest kinship matrix to the symmetric matrix `M` (actually, twice the kinship matrix) by an accelerated proximal distance algorithm.
In this version, the positive semidefinite constraint is folded into the domain of the loss.
"""
function accelerated_closest_kinship_matrix2(M::Matrix{Float64})

    # Initialize control constants.
    ρ        = 1.0
    ρ_max    = 2^22
    max_iter = 5000
    iters    = 0

    # Initialize arrays and the loss.
    (p,q) = size(M)
    X     = deepcopy(M)
    Y     = deepcopy(M)
    Z     = similar(M)
    U     = similar(M)

    # Enter the proximal distance loop.
    for n = 1:max_iter

        iters = iters + 1
        Z     = Y + ((n - 1.0)/(n + 2.0))*(Y - X)
        X     = deepcopy(Y)

        # Project onto the constraints.
        U    = nonnegative_correlation_projection(Z)
        dist = vecnorm(Z - U)
        loss = 0.5*vecnorm(M - Y)^2

        # Calculate the proximal distance update.
        Y = (1.0/(1.0 + ρ))*(M + ρ*U)
        Y = positive_definite_projection(Y)

        # Check for convergence.
        next_loss = 0.5*vecnorm(M - Y)^2
        if abs(next_loss - loss)/(abs(loss) + 1.0) < 1e-6 && dist < 1e-4
            break
        else
            loss = next_loss
        end

        if n % 100 == 0
            ρ = min(5.0*ρ, ρ_max)
            X = deepcopy(Y)
        end
    end

    return (iters,Y)
end

"""
    dykstra_closest_kinship_matrix(M)

This function finds the closest kinship matrix to the symmetric matrix `M` (actually, twice the kinship matrix) by Dykstra's algorithm.
"""
function dykstra_closest_kinship_matrix(M::Matrix{Float64})

    # Initialize control constants.
    maxiters = 5000
    tol      = 1e-8
    iters    = 0

    # Initialize the primary and companion sequences.
    U = zeros(M)
    V = zeros(M)
    Z = zeros(M)
    X = copy(M)

    # Enter the Dykstra loop.
    for i = 1:maxiters

        iters = iters + 1

        # Project onto the nonnegativity and correlation constraints.
        Z     = X
        X     = nonnegative_correlation_projection(X + U)
        dist1 = norm(Z - X)
        U     = Z + U - X

        # Project onto the positive definite constraint.
        Z     = X
        X     = positive_definite_projection(X + V)
        dist2 = norm(Z - X)
        V     = Z+V-X

        # Check for convergence.
        if dist1 < tol && dist2 < tol
            break
        end
    end
    return (iters,X)
end
