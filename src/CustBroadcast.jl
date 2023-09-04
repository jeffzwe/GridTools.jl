# Create custom broadcast style
Base.BroadcastStyle(::Type{<:Field}) = Broadcast.ArrayStyle{Field}()

@inline function Base.Broadcast.materialize!(dest, bc::Broadcasted{ArrayStyle{Field}})
    return copyto!(dest, Base.Broadcast.instantiate(bc))
end

# Custom instantiate(): Dimension check and calculation of output dimension
function Base.Broadcast.instantiate(bc::Broadcasted{ArrayStyle{Field}})
    dims_list = Vector{Tuple{Tuple{Vararg{<:Dimension}}, Tuple{Vararg{<:Integer}}}}()
    axes(bc)
    out_dims = promote_dims([t[1] for t in dims_list])
    out_size = map(Base.OneTo(combine_axes(dims_list)))
    return Broadcasted{ArrayStyle{Field}}(bc.f, bc.args, )
end

# -----------------------------------------------------------------------------------------------------------------------------------------

# Checks dimension and broadcast combatibility of all fields
@inline Base.axes(bc::Broadcasted{ArrayStyle{Field}}) =                 _axes(bc, bc.axes)
_axes(::Broadcasted, axes::Tuple) =                                     axes
@inline _axes(bc::Broadcasted, ::Nothing)  =                            map(format, bc.args)

# Helper function for combine_axes
@inline format(A::Field) = push!(dims_list, (A.dims, size(A)))
@inline format(bc::Broadcasted{ArrayStyle{Field}}) = axes(bc)
@inline format(any) = nothing

@inline function promote_dims(dims_list::Vector{Tuple{Vararg{<:Dimension}}})

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

    in_degree = Dict{Dimension, Integer}
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

        for pred in graph[v]
            in_degree[pred] -= 1
        end
    end

    if length(keys(in_degree)) > 0
        error_message = "Dimensions cannot be promoted. The following dimensions appear in contradicting order: "
        error_message *= join(keys(in_degree), ", ")
        throw(ArgumentError(error_message))
    end


end

@inline function combine_axes(dims_list::Vector{Tuple{Tuple{Vararg{<:Dimension}}, Tuple{Vararg{<:Integer}}}})
    ref = zeros(Integer, length(out_dims))
    for dims in dims_list
        for (i,ind) in enumerate(indexin(dims[1], out_dims))
            if ref[ind] == 0
                ref[ind] = dims[2][i]
            else
                @assert ref[ind] == dims[2][i] "Dimension Mismatch between the data Dimensions of two Fields"
            end
        end
    end
    return Tuple(ref)
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


