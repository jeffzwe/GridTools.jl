"""
    broadcast(f::Field, b_dims::Tuple)

Sets the broadcast dimension of Field f to b_dims
"""
function Base.broadcast(f::Field, b_dims::D)::Field where D <: Tuple{Vararg{Dimension}}
    @assert issubset(f.dims, b_dims)
    return Field(f.dims, f.data, b_dims)
end

function Base.broadcast(n::Number, b_dims::Union{D, Dimension})::Field where D <: Tuple{Vararg{Dimension}}
    return Field((), fill(n), b_dims)
end


"""
    neighbor_sum(f::Field; axis::Dimension)

Sums along the axis dimension. Outputs a field with dimensions size(f.dims)-1.
"""
function neighbor_sum(field_in::Field; axis::Dimension)::Field
    dim = findall(x -> x == axis, field_in.dims)[1]
    return Field((field_in.dims[1:dim-1]..., field_in.dims[dim+1:end]...), dropdims(sum(field_in.data, dims=dim), dims=dim)) 
end
"""
    max_over(f::Field; axis::Dimension)

Gives the maximum along the axis dimension. Outputs a field with dimensions size(f.dims)-1.
"""
function max_over(field_in::Field; axis::Dimension)::Field
    dim = findall(x -> x == axis, field_in.dims)[1]
    return Field((field_in.dims[1:dim-1]..., field_in.dims[dim+1:end]...), dropdims(maximum(field_in.data, dims=dim), dims=dim)) 
end
"""
    min_over(f::Field; axis::Dimension)

Gives the minimum along the axis dimension. Outputs a field with dimensions size(f.dims)-1.
"""
function min_over(field_in::Field; axis::Dimension)::Field
    dim = findall(x -> x == axis, field_in.dims)[1]
    return Field((field_in.dims[1:dim-1]..., field_in.dims[dim+1:end]...), dropdims(minimum(field_in.data, dims=dim), dims=dim)) 
end


"""
    where(mask::Field, true, false)

The 'where' loops over each entry of the mask and returns values corresponding to the same indexes of either the true or the false branch.

# Arguments
- `mask::Field`: a field with eltype Boolean
- `true`: a tuple, a field, or a scalar
- `false`: a tuple, a field, or a scalar

# Examples
```julia-repl
julia> mask = Field((Cell, K), rand(Bool, (3,3)))
3x3  Field with dims (Cell_(), K_()) and broadcasted_dims (Cell_(), K_()):
 1  0  0
 0  1  0
 1  1  1
julia> a = Field((Cell, K), fill(1.0, (3,3)));
julia> b = Field((Cell, K), fill(2.0, (3,3)));
julia> where(mask, a, b)
3x3  Field with dims (Cell_(), K_()) and broadcasted_dims (Cell_(), K_()):
 1.0  2.0  2.0
 2.0  1.0  2.0
 1.0  1.0  1.0
```

The `where` function builtin also allows for nesting of tuples. In this scenario, it will first perform an unrolling:
`where(mask, ((a, b), (b, a)), ((c, d), (d, c)))` -->  `where(mask, (a, b), (c, d))` and `where(mask, (b, a), (d, c))` and then combine results to match the return type:
"""
where(mask::Field, t1::Tuple, t2::Tuple)::Field = map(x -> where(mask, x[1], x[2]), zip(t1, t2))
@inbounds function where(mask::Field, a::Union{Field, Real}, b::Union{Field, Real})::Field 
    @assert eltype(a) == eltype(b) "The true and false branch of a where statment need to have the same type: Got $(eltype(a)) and $(eltype(b))"
    return ifelse.(mask, a, b)
end