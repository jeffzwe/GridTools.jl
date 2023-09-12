Base.BroadcastStyle(::Type{<:Field}) = Broadcast.ArrayStyle{Field}()

@inline function Base.Broadcast.materialize!(dest, bc::Broadcasted{ArrayStyle{Field}})
    return copyto!(dest, Base.Broadcast.instantiate(bc))
end

# Custom instantiate(): Dimension check and calculation of output dimension
function Base.Broadcast.instantiate(bc::Broadcasted{ArrayStyle{Field}})
    return Broadcasted{ArrayStyle{Field}}(bc.f, bc.args, axes(bc))
end

# -----------------------------------------------------------------------------------------------------------------------------------------

# Checks dimension and broadcast combatibility of all fields
@inline Base.axes(bc::Broadcasted{ArrayStyle{Field}}) =                 f_axes(bc, bc.axes)
f_axes(::Broadcasted, axes::Tuple) =                                    axes[2]
@inline f_axes(bc::Broadcasted, ::Nothing)  =                           combine_axes(bc.args...)

function ordered_subset(A::Tuple, B::Tuple)
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
@inline function get_size(out_dims::Vector{Dimension}, A::Tuple, B::Tuple)::Tuple
    out_size = Vector()
    for dim in out_dims
        ind_A = findfirst(x -> x == dim, A[1])
        ind_B = findfirst(x -> x == dim, B[1])

        if dim in A[1] && dim in B[1] && !isempty(A[2]) && !isempty(B[2])
            @assert length(A[2][ind_A]) == length(B[2][ind_B])
            push!(out_size, A[2][ind_A])
        elseif dim in A[1] && !isempty(A[2])
            push!(out_size, A[2][ind_A])
        elseif dim in B[1] && !isempty(B[2])
            push!(out_size, B[2][ind_B])
        end
    end
    return Tuple(out_size)
end
# Helper function for combine_axes
function promote_dims(dims_A::Tuple{Vararg{<:Dimension}}, dims_B::Tuple{Vararg{<:Dimension}})::Vector{Dimension}

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
            error_message = "Dimensions cannot be promoted. Could not determine order of the following dimensions: "
            error_message *= join((dim for dim in zero_in_degree_vertex_list), ", ")
            throw(ArgumentError(error_message))
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
        throw(ArgumentError(error_message))
    end

    return topological_sort
end
# Helper function for combine_axes
@inline format(A::Field) = (A.dims, axes(A), A.broadcast_dims)
@inline format(bc::Broadcasted{ArrayStyle{Field}}) = axes(bc)
@inline format(bc::Broadcasted) = Base.axes(bc)
@inline format(x::Number) = nothing
@inline format(t::Tuple) = t

@inline combine_axes(i1, i2, rest...) = combine_axes(combine_axes(format(i1), format(i2)), rest...)
@inline combine_axes(i, n::Nothing) = combine_axes(format(i))
@inline combine_axes(n::Nothing, i) = combine_axes(format(i))
@inline combine_axes(i::Tuple, n::Tuple{}) = combine_axes(format(i))
@inline combine_axes(n::Tuple{}, i::Tuple) = combine_axes(format(i))
@inline combine_axes(i) = format(i)
@inline function combine_axes(A::Tuple, B::Tuple)  
    # A,B are of the form (dims, axes(data), broadcast_dims)

    if ordered_subset(A[1], B[3])
        out_dims = intersect(B[3], union(A[1], B[1]))
        broadcast_dims = union(B[3], A[3])
    elseif ordered_subset(B[1], A[3])
        out_dims = intersect(A[3], union(A[1], B[1]))
        broadcast_dims = union(A[3], B[3])
    else
        out_dims = promote_dims(A[1], B[1])
        broadcast_dims = promote_dims(A[3], B[3])
    end

    out_size = get_size(out_dims, A, B)

    return (Tuple(out_dims), out_size, Tuple(broadcast_dims))
end


# -----------------------------------------------------------------------------------------------------------------------------------------

# Custom similar(): Creates output object
function Base.similar(bc::Broadcasted{ArrayStyle{Field}}, ::Type{ElType}) where ElType
    Field(bc.axes[1], similar(Array{ElType}, bc.axes[2]), bc.axes[3])
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

# Custom preprocess(): Inorder to pass output dimensions to extrude
@inline function Base.Broadcast.preprocess(dest::Field, A::Field)
    if ndims(A) == 0 
        return (A[],)
    else
        return f_extrude(Base.Broadcast.broadcast_unalias(dest, A), dest.dims)
    end
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
