# Broadcast ---------------------------------------------------------------------

# Create custom broadcast style
Base.BroadcastStyle(::Type{<:Field}) = Broadcast.ArrayStyle{Field}()

# Custom similar(): Creates output object
function Base.similar(bc::Broadcasted{ArrayStyle{Field}}, ::Type{ElType}) where ElType
    output_props = axes(bc)
    
    Field(output_props[1], similar(Array{ElType}, output_props[2]), output_props[3])
end

# Custom _broadcast_getindex(): Drops additional dimensions to access array size
@propagate_inbounds _broadcast_getindex(b::Extruded{Field}, i) = b.x[f_newindex(i, b.keeps, b.defaults)]

# TODO: Could use default for index shift. Could do: return (I[1]+ Idefault[1], _newindex(tail(I), tail(keep), tail(Idefault))...)
@inline function f_newindex(I, keep, Idefault)
    if keep[1]
        return (I[1], _newindex(Base.tail(I), Base.tail(keep), Base.tail(Idefault))...)
    else
        return _newindex(Base.tail(I), Base.tail(keep), Base.tail(Idefault))
    end
end


# Custom newindexer(): Builds boolean array of dimensions to keep when iterating over all dimensions
@inline function newindexer(A::Field)
    return f_newindexer(A.dims, A.broadcast_dims, axes(A))
end

# Idefault not used atm. Use for indexshift
@inline f_newindexer(dims::Tuple, b_dims::Tuple{}, ax::Tuple) = (), ()
@inline function f_newindexer(dims::Tuple, b_dims::Tuple, ax::Tuple)
    ind1 = b_dims[1]
    keep, Idefault = f_newindexer(dims, Base.tail(b_dims), ax)
    next_keep = ind1 in dims
    (next_keep, keep...) , (next_keep ? ax[findall(x -> x == ind1, dims)][1][1] : 0, Idefault...)
end

# Custom instantiate(): Dimension check and calculation of output dimension
function Base.Broadcast.instantiate(bc::Broadcasted{ArrayStyle{Field}})
    return Broadcasted{ArrayStyle{Field}}(bc.f, bc.args, axes(bc)[2])
end

# Helper function for combine_axes
@inline get_dim_ind(dims::Tuple{}, b_dims::Tuple) =  ()
@inline get_dim_ind(dims::Tuple, b_dims::Tuple) = [findall(x -> x == dims[1], b_dims)[1] , get_dim_ind(Base.tail(dims), b_dims)...]


# Checks dimension and broadcast combatibility of all fields
@inline Base.axes(bc::Broadcasted{ArrayStyle{Field}}) =               combine_axes(bc.args...)
@inline combine_axes(A::Field, bc::Broadcasted{ArrayStyle{Field}}) = combine_axes((A.dims, axes(A), A.broadcast_dims), axes(bc))
@inline combine_axes(bc::Broadcasted{ArrayStyle{Field}}, B::Field) = combine_axes(axes(bc), (B.dims, axes(B), B.broadcast_dims))
@inline combine_axes(A::Field, B::Field) =                            combine_axes((A.dims, axes(A), A.broadcast_dims), (B.dims, axes(B), B.broadcast_dims))
@inline combine_axes(A::Field, t::Tuple{}) = (A.dims, axes(A), A.broadcast_dims)
@inline combine_axes(t::Tuple{}, A::Field) = (A.dims, axes(A), A.broadcast_dims)
@inline function combine_axes(A::Tuple, B::Tuple)
    length(A[1]) > length(B[1]) ? (A,B) = (B,A) : nothing  # swap A and B       
    # A,B = (dims, axes(data), broadcast_dims)
    @assert issubset(A[1], B[1]) "Dimension Mismatch between the Dimensions of the two Fields"
    @assert issubset(B[1], A[3]) "Dimension Mismatch between the broadcasted Dimensions of the two Fields"
    matching_dims = get_dim_ind(A[1], B[1])
    @assert A[2] == B[2][matching_dims] "Dimension Mismatch between the data Dimensions of the two Fields"

    return B
end