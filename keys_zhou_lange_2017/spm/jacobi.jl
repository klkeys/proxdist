"""
    jacobi_eig!(V, d, A [, max_iter=1000, n=size(a,1), tol])

This function computes the eigenvalues and eigenvectors of a real symmetric matrix `A`,
using Rutishauser's modfications of the classical Jacobi rotation method with threshold pivoting.
It uses a matrix `V` of eigenvectors and a vector `d` of eigenvalues as a warm start.
"""
function jacobi_eig!(
    V        :: DenseMatrix{Float64},
    d        :: DenseVector{Float64},
    A        :: DenseMatrix{Float64};
    max_iter :: Int     = 1000,
    n        :: Int     = size(a,1),
    tol      :: Float64 = sqrt(sum(sum(triu(a, 1).^2))) / (4.0 * n),
    bw       :: DenseVector{Float64} = copy(d),
    zw       :: DenseVector{Float64} = zeros(Float64, n),
)

    # error handling
    tol > zero(Float64) || throw(ArgumentError("tol must be positive"))

    iter    = 0
    rot_num = 0

    for iter = 1:max_iter
        for p = 1:n
            for q = (p + 1):n

                gapq  = 10.0 * abs(A[p,q])
                termp = gapq + abs(d[p])
                termq = gapq + abs(d[q])

                # annihilate tiny offdiagonal elements
                # otherwise, apply a rotation
                if 4 < iter && termp == abs(d[p]) && termq == abs(d[q])
                    A[p,q] = zero(Float64)
                elseif tol <= abs(A[p,q])
                    h    = d[q] - d[p]
                    term = abs(h) + gapq
                    if term == abs(h)
                        t = A[p,q] / h
                    else
                        theta = 0.5 * h / A[p,q]
                        t = 1.0 / (abs(theta) + sqrt(1.0 + theta*theta))
                        if theta < zero(Float64)
                            t = - t
                        end
                    end
                    c   = 1.0 / sqrt(1.0 + t*t)
                    s   = t * c
                    tau = s / (1.0 + c)
                    h   = t * A[p,q]

                    # accumulate corrections to diagonal elements
                    zw[p]  = zw[p] - h
                    zw[q]  = zw[q] + h
                    d[p]   = d[p] - h
                    d[q]   = d[q] + h
                    A[p,q] = zero(Float64)

                    # rotate, only using information from the upper triangle of A
                    for j = 1:(p - 1)
                        g      = A[j,p]
                        h      = A[j,q]
                        A[j,p] = g - s * ( h + g * tau )
                        A[j,q] = h + s * ( g - h * tau )
                    end
                    for j = (p + 1):(q - 1)
                        g      = A[p,j]
                        h      = A[j,q]
                        A[p,j] = g - s * ( h + g * tau )
                        A[j,q] = h + s * ( g - h * tau )
                    end
                    for j = (q + 1):n
                        g      = A[p,j]
                        h      = A[q,j]
                        A[p,j] = g - s * ( h + g * tau )
                        A[q,j] = h + s * ( g - h * tau )
                    end

                    # accumulate information in the eigenvector matrix
                    for j = 1 : n
                        g      = V[j,p]
                        h      = V[j,q]
                        V[j,p] = g - s * ( h + g * tau )
                        V[j,q] = h + s * ( g - h * tau )
                    end

                    # track rotations
                    rot_num += 1
                end
            end
        end

        #bw(1:n,1) = bw(1:n,1) + zw(1:n,1)
        #d(1:n,1) = bw(1:n,1)
        #zw(1:n,1) = 0.0
        bw += zw
        copy!(d,bw)
        fill!(zw,zero(Float64))
    end

    # ascending sort the eigenvalues and eigenvectors
    for k = 1:(n - 1)
        m = k
        for l = (k + 1):n
            if d[l] < d[m]
                m = l
            end
        end

        if m != k
            t    = d[m]
            d[m] = d[k]
            d[k] = t

            # w = V[:,m]
            for i = 1:n
                w[i] = V[i,m]
            end

            # V[:,m] = V[:,k]
            for i = 1:n
                V[i,m] = V[i,k]
            end

            # V[:,k] = w
            for i = 1:n
                V[i,k] = w[i]
            end
        end
    end
    return nothing
end
