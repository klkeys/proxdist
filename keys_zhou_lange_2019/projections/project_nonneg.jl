function project_nonneg!(
    y :: AbstractArray{T,N},
    x :: AbstractArray{T,N}
) where {T,N}
	y .= max.(x, 0)
	return y
end

function project_nonneg(x :: AbstractArray{T,N}) where {T,N}
    y = zeros(T,size(x))
    project_nonneg!(y,x)
end
