

# where implementation with @generated #####################################################################

function dfs_index(exp::Expr)
    if typeof(exp.args[1]) == Symbol
        return [exp.args[2]]
    else
        return [dfs_index(exp.args[1])..., exp.args[2]]
    end 
end

## Recursive Version
# function fold_tuple(res::Expr, t::Expr, p::Expr, d::Vector, mask::Type{<:Field})::Expr
#     ind = d[1]
#     if length(d) > 1
#         @assert length(res.args) in (ind-1, ind)
#         if length(res.args) == ind-1
#             push!(res.args, Expr(:tuple))
#         end
#         res.args[ind] = fold_tuple(res.args[ind], t, p, d[2:length(d)], mask)
#     else
#         temp = :(ifelse.(mask, $t, $p))
#         push!(res.args, temp)
#     end
#     return res
# end

## Iterative version
function fold_tuple(t::Expr, p::Expr, mask::Type{<:Field})::Expr
   res = Expr(:tuple)
   for tuple_id in 1:length(t.args)             # for all tuples
        path = dfs_index(t.args[tuple_id])       # get the path of a tuple
        current_expr = res
        for path_step in 1:length(path)-1
            @assert length(current_expr.args) in (path[path_step]-1, path[path_step])
            if length(current_expr.args) == path[path_step]-1
                push!(current_expr.args, Expr(:tuple))
            end
            current_expr = current_expr.args[path[path_step]]
        end
        push!(current_expr.args, :(ifelse.(mask, $(t.args[tuple_id]), $(p.args[tuple_id]))))
   end
   return res
end

function unfold_tuple(tt::Type{<:Tuple}, tuple_name::Symbol, path::Vector = [])::Expr
    res = Expr(:tuple)
    for i in 1:length(tt.parameters)
        if tt.parameters[i] <: Tuple
            push!(res.args, unfold_tuple(tt.parameters[i], tuple_name, [path..., i]).args...)
        else
            current_expr = tuple_name
            for current_index in [path..., i]
                current_expr = Expr(:ref, current_expr, current_index)
            end
            push!(res.args, current_expr)
        end
    end
    return res
end

@inbounds @generated function where(mask::Field, tt::Tuple, pp::Tuple)   # TODO: Make @generated. Problem with naming String, vectors used like d::Vector and so on.
    @assert tt == pp
    t = unfold_tuple(tt, :tt)
    p = unfold_tuple(pp, :pp)

    ## Recursive Version
    # res = Expr(:tuple)

    # for i in 1:length(t.args)
    #     d = dfs_index(t.args[i])
    #     res = fold_tuple(res, t.args[i], p.args[i], d, mask)
    # end

    ## Iterative version
    res = fold_tuple(t, p, mask)

    return res
end


# Part of custom broadcast using Idefault-----------------------------------------------------------------------------------------------------------------------------------------

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


@inline f_newindexer(dims::Tuple, b_dims::Tuple{}, ax::Tuple) = (), ()
@inline function f_newindexer(dims::Tuple, b_dims::Tuple, ax::Tuple)
    ind1 = b_dims[1]
    keep, Idefault = f_newindexer(dims, Base.tail(b_dims), ax)
    next_keep = ind1 in dims
    (next_keep, keep...) , (next_keep ? minimum(ax[get_dim_ind(dims, ind1)]) : 0, Idefault...)
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
Base.@propagate_inbounds f_broadcast_getindex(b::Extruded, i) = try b.x[f_newindex(i, b.keeps, b.defaults)] catch nothing end

@inline f_newindex(I::CartesianIndex, keep, Idefault) = CartesianIndex(_f_newindex(I.I, keep, Idefault))
@inline f_newindex(i::Integer, keep::Tuple, idefault) = ifelse(keep[1], i, idefault[1])
@inline f_newindex(i::Integer, keep::Tuple{}, idefault) = CartesianIndex(())

@inline _f_newindex(i::Integer, keep::Tuple, idefault) = ifelse(keep[1], i, idefault[1])
@inline _f_newindex(i::Integer, keep::Tuple{}, idefault) = CartesianIndex(())
@inline _f_newindex(I::Tuple{}, keep::Tuple{}, Idefault::Tuple{}) = ()
# Dropping dims. Defaults does nothing here.
@inline function _f_newindex(I, keep::Tuple, Idefault::Tuple)
    if keep[1]
        return (I[1], _f_newindex(Base.tail(I), Base.tail(keep), Base.tail(Idefault))...)
    else
        return _f_newindex(Base.tail(I), Base.tail(keep), Base.tail(Idefault))
    end
end
