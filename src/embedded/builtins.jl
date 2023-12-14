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

Sums along the axis dimension.
"""
function neighbor_sum(field_in::Field; axis::Dimension)::Field
    return reduction_master(field_in, axis, sum)
end
"""
    max_over(f::Field; axis::Dimension)

Gives the maximum along the axis dimension.
"""
function max_over(field_in::Field; axis::Dimension)::Field
    return reduction_master(field_in, axis, maximum)
end
"""
    min_over(f::Field; axis::Dimension)

Gives the minimum along the axis dimension.
"""
function min_over(field_in::Field; axis::Dimension)::Field
    return reduction_master(field_in, axis, minimum)
end


function reduction_master(field_in::Field, axis::Dimension, f::Function)
    neutral_el = get_neutral(f, eltype(field_in))
    dim = get_dim_ind(field_in.dims, axis)

    conn = OFFSET_PROVIDER[get_dim_name(axis)]
    data = dropdims(f(ifelse.(conn.data .!= -1, field_in.data, neutral_el), dims=dim), dims=dim)
    return Field((field_in.dims[1:dim-1]..., field_in.dims[dim+1:end]...), data)
end

get_neutral(f::typeof(sum), type::DataType) = convert(type, 0)
get_neutral(f::typeof(minimum), type::DataType) = typemax(type)
get_neutral(f::typeof(maximum), type::DataType) = typemin(type)

"""
    where(mask::Field, true, false)

"Where" loops over each entry of the mask and returns values corresponding to the same indices of either the true or the false branch.

# Arguments
- `mask::Field`: a field with eltype Boolean
- `true`: a tuple, a field, or a scalar
- `false`: a tuple, a field, or a scalar

# Examples
```julia-repl
julia> mask = Field((Cell, K), rand(Bool, (3,3)))
3×3 Bool Field with dimensions ("Cell", "K") with indices 1:3×1:3:
 0  0  1
 1  1  1
 1  0  0
julia> a = Field((Cell, K), fill(1.0, (3,3)));
julia> b = Field((Cell, K), fill(2.0, (3,3)));
julia> where(mask, a, b)
3x3  Field with dimensions ("Cell", "K") with indices 1:3×1:3:
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


"""
    concat(f1::Field, f2::Field)

Combine two fields containing NaN values to one single field. Both fields are not allowed to contain overlapping data.

# Examples
```julia-repl
julia> f1
3x3  Field with dimensions ("Cell", "K") with indices 1:3×1:3:
 1.0  1.0  1.0
 1.0  NaN  1.0
 1.0  1.0  1.0

julia> f2
3x3  Field with dimensions ("Cell", "K") with indices 1:3×1:3:
 NaN  NaN  NaN
 NaN  5.0  NaN
 NaN  NaN  NaN

julia> concat(f1, f2)
3x3  Field with dimensions ("Cell", "K") with indices 1:3×1:3:
 1.0  1.0  1.0
 1.0  5.0  1.0
 1.0  1.0  1.0
```
"""
function concat(f1::Field, f2::Field) #TODO check if this implementation alines with the future implementation in gt4py
    @assert all(xor.(isnan.(f1), isnan.(f2))) "The matrices $f1 and $f2 are not combineable in a concat operation due to overlapping values."
    return concat_helper.(f1, f2)
end

function concat_helper(x, y)
    return isnan(x) ? y : x
end