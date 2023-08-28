# Create custom broadcast style
Base.BroadcastStyle(::Type{<:Field}) = Broadcast.ArrayStyle{Field}()

@inline function Base.Broadcast.materialize!(dest, bc::Broadcasted{ArrayStyle{Field}})
    return copyto!(dest, Base.Broadcast.instantiate(bc))
end

# Custom instantiate(): Dimension check and calculation of output dimension
function Base.Broadcast.instantiate(bc::Broadcasted{ArrayStyle{Field}})
    return Broadcasted{ArrayStyle{Field}}(bc.f, bc.args, combine_axes(bc)[2])
end

# -----------------------------------------------------------------------------------------------------------------------------------------

# Checks dimension and broadcast combatibility of all fields
@inline Base.axes(bc::Broadcasted{ArrayStyle{Field}}) =                 _axes(bc, bc.axes)
_axes(::Broadcasted, axes::Tuple) =                                     axes
@inline _axes(bc::Broadcasted, ::Nothing)  =                            combine_axes(bc.args...)

# Helper function for combine_axes
@inline get_dim_ind(dims::Tuple{}, b_dims::Tuple) =  ()
@inline get_dim_ind(dims::Tuple, b_dims::Tuple) = [findall(x -> x == dims[1], b_dims)[1] , get_dim_ind(Base.tail(dims), b_dims)...]
# Helper function for combine_axes
@inline format(A::Field) = (A.dims, axes(A), A.broadcast_dims)
@inline format(bc::Broadcasted{ArrayStyle{Field}}) = axes(bc)
@inline format(x::Number) = nothing
@inline format(t::Tuple) = t

@inline combine_axes(i1, i2, rest...) = combine_axes(combine_axes(format(i1), format(i2)), rest...)
@inline combine_axes(i, n::Nothing) = combine_axes(format(i))
@inline combine_axes(n::Nothing, i) = combine_axes(format(i))
@inline combine_axes(i) = format(i)
@inline function combine_axes(A::Tuple, B::Tuple)
    length(A[1]) > length(B[1]) ? (A,B) = (B,A) : nothing  # swap A and B       
    # A,B are of the form (dims, axes(data), broadcast_dims)
    @assert issubset(A[1], B[1]) "Dimension Mismatch between the Dimensions of the two Fields"
    @assert issubset(B[1], A[3]) "Dimension Mismatch between the broadcasted Dimensions of the two Fields"
    matching_dims = get_dim_ind(A[1], B[1])
    @assert A[2] == B[2][matching_dims] "Dimension Mismatch between the data Dimensions of the two Fields"
    return B
end


# -----------------------------------------------------------------------------------------------------------------------------------------

# Custom similar(): Creates output object
function Base.similar(bc::Broadcasted{ArrayStyle{Field}}, ::Type{ElType}) where ElType
    output_props = combine_axes(bc.args...)
    
    Field(output_props[1], similar(Array{ElType}, output_props[2]), output_props[3])
end


# -----------------------------------------------------------------------------------------------------------------------------------------

# Custom copyto!(): Only needed to maintain Broadcast Style
@inline function Base.Broadcast.copyto!(dest::Field, bc::Broadcasted{ArrayStyle{Field}})
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if axes(dest) == axes(bc) && bc.f === identity && bc.args isa Tuple{AbstractArray} # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end

    bc′ = Base.Broadcast.preprocess(dest, bc)

    # Performance may vary depending on whether `@inbounds` is placed outside the
    # for loop or not. (cf. https://github.com/JuliaLang/julia/issues/38086)
    @inbounds @simd for I in eachindex(dest)
        dest[I] = bc′[I]
    end
    return dest
end


# -----------------------------------------------------------------------------------------------------------------------------------------

# Custom preprocess(): Needed inorder to pass output dimensions to extrude
@inline function Base.Broadcast.preprocess(dest::Field, A::Field)
    return f_extrude(Base.Broadcast.broadcast_unalias(dest, A), dest.dims)
end

@inline function f_extrude(A::Field, b_dims::Tuple)
    return Extruded(A, f_newindexer(A.dims, b_dims, axes(A))...)
end

# Idefault not used atm. Use for indexshift
@inline f_newindexer(dims::Tuple, b_dims::Tuple{}, ax::Tuple) = (), ()
@inline function f_newindexer(dims::Tuple, b_dims::Tuple, ax::Tuple)
    ind1 = b_dims[1]
    keep, Idefault = f_newindexer(dims, Base.tail(b_dims), ax)
    next_keep = ind1 in dims
    (next_keep, keep...) , (next_keep ? ax[findall(x -> x == ind1, dims)][1][1] : 0, Idefault...)
end


# -----------------------------------------------------------------------------------------------------------------------------------------

# Custom getindex(): Drops additional dimensions to access array size
@inline function Base.getindex(bc::Broadcasted{ArrayStyle{Field}}, I::Union{Integer,CartesianIndex})
    @boundscheck Base.Broadcast.checkbounds(bc, I)
    @inbounds f_broadcast_getindex(bc, I)
end

Base.@propagate_inbounds function f_broadcast_getindex(bc::Broadcasted{<:Any,<:Any,<:Any,<:Any}, I)
    args = f_getindex(bc.args, I)
    return Base.Broadcast._broadcast_getindex_evalf(bc.f, args...)
end

# No changes to original: Utilities for f_broadcast_getindex
Base.@propagate_inbounds f_getindex(args::Tuple, I) = (f_broadcast_getindex(args[1], I), f_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds f_getindex(args::Tuple{Any}, I) = (f_broadcast_getindex(args[1], I),)
Base.@propagate_inbounds f_getindex(args::Tuple{}, I) = ()

# No changes to original
Base.@propagate_inbounds f_broadcast_getindex(A::Union{Ref,AbstractArray{<:Any,0},Number}, I) = A[] # Scalar-likes can just ignore all indices
Base.@propagate_inbounds f_broadcast_getindex(::Ref{Type{T}}, I) where {T} = T
# Tuples are statically known to be singleton or vector-like
Base.@propagate_inbounds f_broadcast_getindex(A::Tuple{Any}, I) = A[1]
Base.@propagate_inbounds f_broadcast_getindex(A::Tuple, I) = A[I[1]]
# Everything else falls back to dynamically dropping broadcasted indices based upon its axes
Base.@propagate_inbounds f_broadcast_getindex(A, I) = A[Base.Broadcast.newindex(A, I)]
Base.@propagate_inbounds f_broadcast_getindex(b::Extruded, i) = b.x[f_newindex(i, b.keeps, b.defaults)]


@inline f_newindex(I::CartesianIndex, keep, Idefault) = CartesianIndex(_f_newindex(I.I, keep, Idefault))
@inline f_newindex(i::Integer, keep::Tuple, idefault) = ifelse(keep[1], i, idefault[1])
@inline f_newindex(i::Integer, keep::Tuple{}, idefault) = CartesianIndex(())
@inline _f_newindex(i::Integer, keep::Tuple, idefault) = ifelse(keep[1], i, idefault[1])
@inline _f_newindex(i::Integer, keep::Tuple{}, idefault) = CartesianIndex(())
@inline _f_newindex(I::Tuple{}, keep::Tuple{}, Idefault::Tuple{}) = ()
# TODO: Could use default for index shift. Could do: return (I[1]+ Idefault[1], _newindex(tail(I), tail(keep), tail(Idefault))...)
# Dropping dims
@inline function _f_newindex(I, keep::Tuple, Idefault::Tuple)
    if keep[1]
        return (I[1], _f_newindex(Base.tail(I), Base.tail(keep), Base.tail(Idefault))...)
    else
        return _f_newindex(Base.tail(I), Base.tail(keep), Base.tail(Idefault))
    end
end


