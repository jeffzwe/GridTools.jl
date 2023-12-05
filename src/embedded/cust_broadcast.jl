Base.BroadcastStyle(::Type{<:Field}) = Broadcast.ArrayStyle{Field}()

struct FieldShape
    dims::Tuple{Vararg{Dimension}}
    axes::Tuple{Vararg{AbstractUnitRange{Int64}}}
    broadcast_dims::Tuple{Vararg{Dimension}}
end

function shape(f::Field)
    return FieldShape(f.dims, axes(f), f.broadcast_dims)
end

@inline function Base.Broadcast.materialize!(dest, bc::Broadcasted{ArrayStyle{Field}})
    return copyto!(dest, Base.Broadcast.instantiate(bc))
end

# Custom instantiate(): Dimension check and calculation of output dimension
function Base.Broadcast.instantiate(bc::Broadcasted{ArrayStyle{Field}})
    return Broadcasted{ArrayStyle{Field}}(bc.f, bc.args, axes(bc))
end

# -----------------------------------------------------------------------------------------------------------------------------------------

# Checks dimension and broadcast combatibility of all fields
@inline Base.axes(bc::Broadcasted{ArrayStyle{Field}}) =     f_axes(bc, bc.axes)
f_axes(::Broadcasted, shape::FieldShape) =                  shape.axes
@inline f_axes(bc::Broadcasted, ::Nothing)  =               combine_axes(Val{Symbol(bc.f)}(), bc.args...)

function ordered_subset(A::Tuple{Vararg{Dimension}}, B::Tuple{Vararg{Dimension}})
    if isempty(A) || isempty(B) return false end
    i, j = 1, 1

    while i <= length(A) && j <= length(B)
        if A[i] == B[j]
            i += 1 
        end
        j += 1  
    end

    return i > length(A)
end

# Helper function for combine_axes
@inline function get_size(out_dims::Vector{<:Dimension}, A::FieldShape, B::FieldShape)::Tuple
    out_size = Vector()

    if isempty(A.axes) && isempty(B.axes)
        return Tuple(out_size)
    elseif isempty(A.axes)
        return B.axes
    elseif isempty(B.axes)
        return A.axes
    end

    for dim in out_dims
        ind_A = get_dim_ind(A.dims, dim)
        ind_B = get_dim_ind(B.dims, dim)

        if dim in A.dims && dim in B.dims
            # intersection
            low = max(minimum(A.axes[ind_A]), minimum(B.axes[ind_B]))
            up = min(maximum(A.axes[ind_A]), maximum(B.axes[ind_B]))
            if low == 1                                         
                push!(out_size, Base.OneTo(up))                 
            else
                push!(out_size, IdOffsetRange(1:up-low+1, low-1))          
            end
        elseif dim in A.dims
            push!(out_size, A.axes[ind_A])
        elseif dim in B.dims
            push!(out_size, B.axes[ind_B])
        end
    end

    if eltype(Tuple(out_size)) == AbstractUnitRange{Int64}
        map!(IdOffsetRange, out_size, out_size) 
    end

    return Tuple(out_size)
end
# Helper function for combine_axes
function promote_dims(dims_A::Tuple{Vararg{Dimension}}, dims_B::Tuple{Vararg{Dimension}})::Vector{Dimension}

    dims_list = [dims_A, dims_B]

    graph = Dict{Dimension, Set{Dimension}}()
    for dims in dims_list
        if length(dims) == 0
            continue
        end

        for dim in dims
            if !haskey(graph, dim)
                graph[dim] = Set{Dimension}()
            end
        end
        predecessor = dims[1]
        for dim in Base.tail(dims)
            push!(graph[dim], predecessor)
            predecessor = dim
        end
    end

    topological_sort = Vector{Dimension}()

    in_degree = Dict{Dimension, Integer}()
    for key in keys(graph)
        in_degree[key] = length(graph[key])
    end

    zero_in_degree_vertex_list = [key for (key, value) in in_degree if value == 0]
    while !isempty(zero_in_degree_vertex_list)
        if length(zero_in_degree_vertex_list) != 1
            error_message = "Incompatible Dimensions: Promotion failed. Could not determine order of the following dimensions: "
            error_message *= join((dim for dim in zero_in_degree_vertex_list), ", ")
            throw(AssertionError(error_message))
        end

        v = zero_in_degree_vertex_list[1]
        delete!(in_degree, v)
        push!(topological_sort, v)

        for (key, value) in graph
            if v in value
                in_degree[key] = in_degree[key] - 1
            end
        end
        zero_in_degree_vertex_list = [key for (key, value) in in_degree if value == 0]
    end

    if length(keys(in_degree)) > 0
        error_message = "Dimensions cannot be promoted. The following dimensions appear in contradicting order: "
        error_message *= join(keys(in_degree), ", ")
        throw(AssertionError(error_message))
    end

    return topological_sort
end

function get_size_ifelse(mask::FieldShape, branch::FieldShape)
    out_size = [branch.axes...]
    ind_mask = findall(x -> x in branch.dims, mask.dims)
    ind_out = findall(x -> x in mask.dims, branch.dims)

    out_size[ind_out] .= mask.axes[ind_mask]

    if eltype(Tuple(out_size)) == AbstractUnitRange{Int64}
        map!(IdOffsetRange, out_size, out_size) 
    end
    
    return FieldShape(branch.dims, Tuple(out_size), branch.broadcast_dims)

end


# Helper function for combine_axes
@inline format(A::Field) = shape(A)
@inline format(bc::Broadcasted{ArrayStyle{Field}}) = axes(bc)
@inline format(bc::Broadcasted) = Base.axes(bc)
@inline format(shape::FieldShape) = shape
@inline format(x::Number) = nothing

@inline function combine_axes(f::Val{:ifelse}, mask::Field, true_::Field, false_::Field)
    if issubset(true_.dims, mask.dims) && issubset(false_.dims, mask.dims)
        return format(mask)
    else
        get_size_ifelse(shape(mask), combine_axes(f, format(true_), format(false_)))
    end
end

@inline function combine_axes(f::Val{:ifelse}, mask::Field, true_::Real, false_::Field)
    if issubset(false_.dims, mask.dims)
        return format(mask)
    else
        get_size_ifelse(format(mask), format(false_))
    end
end

@inline function combine_axes(f::Val{:ifelse}, mask::Field, true_::Field, false_::Real)
    if issubset(true_.dims, mask.dims)
        return format(mask)
    else
        return get_size_ifelse(format(mask), format(true_))
    end
end

@inline combine_axes(f::Val{<:Any}, arg1, arg2, rest...)            = combine_axes(f, combine_axes(f, format(arg1), format(arg2)), rest...)
@inline combine_axes(f::Val{<:Any}, arg, n::Nothing)                = combine_axes(f, format(arg))
@inline combine_axes(f::Val{<:Any}, n::Nothing, arg)                = combine_axes(f, format(arg))
@inline combine_axes(f::Val{<:Any}, arg)                            = format(arg)
@inline combine_axes(f::Val{<:Any}, shape::FieldShape, t0::Tuple{}) = combine_axes(f, format(shape))
@inline combine_axes(f::Val{<:Any}, t0::Tuple{}, shape::FieldShape) = combine_axes(f, format(shape))

@inline function combine_axes(f::Val{<:Any}, A::FieldShape, B::FieldShape)::FieldShape

    if ordered_subset(A.dims, B.broadcast_dims)
        out_dims = intersect(B.broadcast_dims, union(A.dims, B.dims))
        broadcast_dims = union(B.broadcast_dims, A.broadcast_dims)
    elseif ordered_subset(B.dims, A.broadcast_dims)
        out_dims = intersect(A.broadcast_dims, union(A.dims, B.dims))
        broadcast_dims = union(A.broadcast_dims, B.broadcast_dims)
    else
        out_dims = promote_dims(A.dims, B.dims)
        broadcast_dims = promote_dims(A.broadcast_dims, B.broadcast_dims)
    end

    out_size = get_size(out_dims, A, B)

    return FieldShape(Tuple(out_dims), out_size, Tuple(broadcast_dims))
end


# -----------------------------------------------------------------------------------------------------------------------------------------

# Custom similar(): Creates output object
function Base.similar(bc::Broadcasted{ArrayStyle{Field}}, ::Type{ElType}) where ElType
    Field(bc.axes.dims, similar(Array{ElType}, axes(bc)), bc.axes.broadcast_dims)
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

    bc′ = Base.Broadcast.preprocess(shape(dest), bc)

    # Performance may vary depending on whether `@inbounds` is placed outside the
    # for loop or not. (cf. https://github.com/JuliaLang/julia/issues/38086)
    @inbounds @simd for I in eachindex(dest)
        dest[I] = bc′[I]
    end
    return dest
end


# -----------------------------------------------------------------------------------------------------------------------------------------

# Custom preprocess(): Inorder to pass output dimensions to extrude
@inline function Base.Broadcast.preprocess(dest::FieldShape, A::Field)
    if ndims(A) == 0 
        return (A[],)
    else
        return f_extrude(Base.Broadcast.broadcast_unalias(dest, A), dest)
    end
end

@inline function f_extrude(A::Field, dest::FieldShape)
    return Extruded(A, f_newindexer(A.dims, dest.broadcast_dims, dest.axes)...)
end

# Idefault not needed... Extruded expects a third argument
@inline f_newindexer(dims::Tuple, b_dims::Tuple{}, ax::Tuple) = (), ()
@inline function f_newindexer(dims::Tuple, b_dims::Tuple, ax::Tuple)
    ind1 = b_dims[1]
    keep, Idefault = f_newindexer(dims, Base.tail(b_dims), ax)
    next_keep = ind1 in dims
    (next_keep, keep...) , (0, Idefault...)
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
Base.@propagate_inbounds function f_broadcast_getindex(b::Extruded, i) 
    ind = f_newindex(i, b.keeps)
    return checkbounds(Bool, b.x, ind) ? b.x[ind] : nothing
end

@inline f_newindex(I::CartesianIndex, keep) = CartesianIndex(_f_newindex(I.I, keep))
@inline f_newindex(i::Integer, keep::Tuple{}) = CartesianIndex(())
@inline f_newindex(i::Integer, keep::Tuple) = i

@inline _f_newindex(i::Integer, keep::Tuple{}) = CartesianIndex(())
@inline _f_newindex(I::Tuple{}, keep::Tuple{}) = ()
@inline _f_newindex(i::Integer, keep::Tuple) = i
# Dropping dims
@inline function _f_newindex(I::Tuple{Vararg{Int64}}, keep::Tuple{Vararg{Bool}})
    if keep[1]
        return (I[1], _f_newindex(Base.tail(I), Base.tail(keep))...)
    else
        return _f_newindex(Base.tail(I), Base.tail(keep))
    end
end
