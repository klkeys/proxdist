# ==============================================================================
# common subroutines
# ==============================================================================

# function to compute a Nesterov acceleration step at iteration i
function compute_accelerated_step!(z::DenseVector{T}, x::DenseVector{T}, y::DenseVector{T}, i::Int) where {T <: AbstractFloat}
	kx = (i - one(T)) / (i + one(T) + one(T))
	ky = one(T) + kx

	# z = ky*y - kx*x
	z .= ky .* y .- kx .* x 
	copy!(x,y)
end

# subroutine to print algorithm progress
# one variant is for two separate constraint distances
# the other is for one constraint distance
function print_progress(i, loss, daffine, dnonneg, rho; i_interval::Int = 10, inc_step::Int = 100)
    if (i <= i_interval || i % inc_step == 0)
        @printf("%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n", i, loss, daffine, dnonneg, rho)
    end
end

function print_progress(i, loss, dnonneg, rho; i_interval::Int = 10, inc_step::Int = 100)
    if (i <= i_interval || i % inc_step == 0)
        @printf("%d\t%3.7f\t%3.7f\t%3.7f\n", i, loss, dnonneg, rho)
    end
end

# function to set all elements of a vector x with magnitude smaller than threshold ε to a value α 
# function for dense vectors is different from function for sparse ones
function threshold!(x::DenseVector{T}, ε::T, α::T = zero(T)) where {T <: AbstractFloat}
    @assert ε > α >= 0 "Arguments must satisfy ε > α >= 0"
    for i in eachindex(x)
        if abs(x[i]) < ε
            x[i] = α 
        end
    end
    return x
end

function threshold!(x::SparseMatrixCSC{T,Int}, ε::T; trim = true) where {T <: AbstractFloat}
    @assert x.n == 1 "x must be a sparse vector"
    threshold!(x.nzval, ε)
    dropzeros!(x, trim)
    return x
end

# subroutine to check conformability of linear program inputs
# if one of the vectors is nonconformable to the matrix, then throw an error
function check_conformability(A::AbstractMatrix{T}, b::AbstractVector{T}, c::AbstractVector{T}) where {T <: AbstractFloat}
    p = length(b)
    q = length(c)
    @assert (p,q) == size(A) "nonconformable A, b, and c\nsize(A) = $(size(A))\nsize(b) = $(size(b))\nsize(c) = $(size(c))"
end

function check_conformability(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: AbstractFloat}
    n = length(b)
    (m1, m2) = size(A)
    @assert m1 == m2 "Argument A must be a square matrix"
    @assert m1 == n  "Nonconformable A and b\nsize(A) = ($m1,$m2)\nsize(b) = ($n,)\n"
end
