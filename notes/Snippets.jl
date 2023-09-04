
# User code exercises from gt4py intro #####################################################################

struct Cell_ <: Dimension end
struct K_ <: Dimension end
struct Edge_ <: Dimension end
struct E2C_ <: Dimension end
Cell = Cell_()
K = K_()
Edge = Edge_()
E2C = E2C_()

function run_add(a::Field, b::Field)
    temp = a + b
    return b + temp
end

function nearest_cell_to_edge(cell_values::Field, e2c::Connectivity)
    return cell_values(e2c(1))
end


function sum_adjacent_cells(cells::Field, e2c::Connectivity)
    return neighbor_sum(cells(e2c()), axis=E2CDim)
end


a = Field((Cell, K), fill(2.0, (3,3)))
b = Field((Cell, K), fill(3.0, (3,3)))
c = Field((Cell, K), fill(5.0, (3,3)))

result = Field((Cell, K), zeros(3,3))
result = run_add(a, b)

@printf "%f + %f = %f ± %f" 3.0 (2.0 + 3.0) mean(result) std(result)
println()

cell_values = Field((Cell,), [1.0, 1.0, 2.0, 3.0, 5.0, 8.0])
cell_values = Field((Cell, K), hcat([1.0, 1.0, 2.0, 3.0, 5.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

edge_to_cell_table = [
    [1  0];
    [3  0];
    [3  0];
    [4  0];
    [5  0];
    [6  0];
    [1  6];
    [1  2];
    [2  3];
    [2  4];
    [4  5];
    [5  6]
]

cell_to_edge_table = [
    [1   7   8];
    [8   9  10];
    [2   3   9];
    [4  10  11];
    [5  11  12];
    [6   7  12]
]


E2C_offset_provider = Connectivity(edge_to_cell_table, (Cell,), (Edge,), 2)
C2E_offset_provider = Connectivity(cell_to_edge_table, (Edge,), (Cell,), 3)

offset_provider = Dict{String, Connectivity}(
                   "E2C" => E2C_offset_provider,
                   "C2E" => C2E_offset_provider
                )

@field_operator function nearest_cell_to_edge(cell_values::Field)
    return cell_values(E2C())
end

nearest_cell_to_edge(cell_values, offset_provider = offset_provider)
println(edge_values)

edge_values = sum_adjacent_cells(cell_values, E2C_offset_provider)
println(edge_values.data)


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


# Snippets for custom broadcast ###############################################################################

function custom_lt(x::Dimension, y::Dimension)

    type_order = Dict{DataType, Int}(
        Cell_ => 1,
        K_ => 2,
        Edge_ => 3,
        E2C_ => 4
    )
    
    # Compare based on type order
    cmp = type_order[typeof(x)] - type_order[typeof(y)]
    
    # If the types are equal, maintain the original order
    if cmp > 0
        return false
    else
        return true
    end
end


@inline function combine_axes(A::Tuple, B::Tuple)
    length(A[1]) > length(B[1]) ? (A,B) = (B,A) : nothing  # swap A and B       

    if issubset(A[1], B[1])
        # A,B = (dims, axes(data), broadcast_dims)
        @assert issubset(B[1], A[3]) "Dimension Mismatch between the broadcasted Dimensions of the two Fields"
        matching_dims = get_dim_ind(A[1], B[1])
        @assert A[2] == B[2][matching_dims] "Dimension Mismatch between the data Dimensions of the two Fields"
        return B
    elseif #!isempty(intersect(A[1],B[1]))  or other condition to merge existing overlapping dimensions
        matching_dimsA = get_dim_ind(intersect(A[1],B[1]),A[1])
        matching_dimsB = get_dim_ind(intersect(A[1],B[1]),B[1])
        @assert A[2][matching_dimsA] == B[2][matching_dimsB] "Dimension Mismatch between the data Dimensions of the two Fields"

        new_dims = [A[1]..., B[1]...]
        new_axes = [A[2]..., B[2]...]
        combine_tuple = hcat(new_dims, new_axes)
        combine_tuple = unique(combine_tuple[sortperm(combine_tuple[:,1], lt=custom_lt),:],dims=1)
        
        return (tuple(combine_tuple[:,1]...), tuple(combine_tuple[:,2]...), union(A[3],B[3]))
    else
        error("Dimension Mismatch between the Dimensions of the two Fields")
    end
    
end


# PyCall Snippets

config = py"config"

# field_operator that puts offset_provider into local scope of the function

macro field_operator(expr::Expr)

    function unpack_dict(dict::Dict)
        for key in keys(dict)
            @eval $(Symbol(key)) = $dict[$key]
        end
    end

    dict = splitdef(expr)

    push!(dict[:kwargs], :($(Expr(:kw, :(offset_provider::Dict), :(Dict()))))) # version with named offset_provider
    
    new_exp = Expr(:call, unpack_dict, :offset_provider)
    dict[:body].args = [dict[:body].args[1:2]..., new_exp, dict[:body].args[3:end]...]
    
    combinedef(dict)
end

# Connectivity call function

# TODO: Sure that this does the right thing? Add to documentation of Connectivity
function (conn_call::Connectivity)(ind::Integer)::Connectivity
    @assert conn_call.dims >= neighbor
    return Connectivity(conn_call.data[:, neighbor], conn_call.source, (conn_call.target[1],), 1)
end


# Gebastel... gaht ned will de field offset ned im GridTools definiert isch. Und wennis us de funktion usenimm denn gahts nur ned will d expression im GridTools baut wird
# und im GridTools sind halt d field offset ned definiert... choennt mer ev umgah...

macro field_operator(expr::Expr)

    function unpack_dict(dict::Dict)
        for key in keys(dict)
            conn = dict[key]
            f_off = @eval $(Symbol(key))
            @assert conn.source == f_off.source
            @assert conn.target in f_off.target

            #Create new Connectivity or change the existing one...
            @eval $(Symbol(key)) = Connectivity(conn.data, f_off.source, f_off.target, conn.dims)
        end
    end

    dict = splitdef(expr)

    push!(dict[:kwargs], :($(Expr(:kw, :(offset_provider::Dict), :(Dict()))))) # version with named offset_provider
    
    new_exp = Expr(:call, unpack_dict, :offset_provider)
    dict[:body].args = [dict[:body].args[1:2]..., new_exp, dict[:body].args[3:end]...]
    
    combinedef(dict)
end



# Snippets from explicit combine axes

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
