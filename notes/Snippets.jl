
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
    return neighbor_sum(cells(e2c()), axis=E2C)
end


a = Field((Cell, K), fill(2.0, (3,3)))
b = Field((Cell, K), fill(3.0, (3,3)))
c = Field((Cell, K), fill(5.0, (3,3)))

result = Field((Cell, K), zeros(3,3))
result = run_add(a, b)

@printf "%f + %f = %f ± %f" 3.0 (2.0 + 3.0) mean(result) std(result)
println()

cell_values = Field((Cell,), [1.0, 1.0, 2.0, 3.0, 5.0, 8.0])

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

E2C_offset_provider = Connectivity(edge_to_cell_table, (Cell,), (Edge, E2C), 2)

edge_values = nearest_cell_to_edge(cell_values, E2C_offset_provider)
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



# advection test

include("Metric.jl")
using Printf

grid = atlas.StructuredGrid("O32")
mesh = AtlasMesh(grid, num_level = 30)

deg2radd = 2.0 * pi / 360.0
δt = 1800.0  # time step in s
niter = 1000
eps = 1.0e-8

metric = M_from_mesh(mesh)
origin = minimum(mesh.xyarc, dims=1)
extent = maximum(mesh.xyarc, dims=1) .- minimum(mesh.xyarc, dims=1)
xlim = (minimum(mesh.xyarc[:, 1]), maximum(mesh.xyarc[:, 1]))
ylim = (minimum(mesh.xyarc[:, 2]), maximum(mesh.xyarc[:, 2]))

vertex_dim = getfield(mesh, Symbol(DIMENSION_TO_SIZE_ATTR[Vertex]))
k_dim = getfield(mesh, Symbol(DIMENSION_TO_SIZE_ATTR[K]))
edge_dim = getfield(mesh, Symbol(DIMENSION_TO_SIZE_ATTR[Edge]))

level_indices = Field((K,), zeros(k_dim))
level_indices.data .= collect(0.0:29.0)


state = St_from_mesh(mesh)
state_next = St_from_mesh(mesh)

tmp_fields = Dict{String, Field}()
for i in 1:6
    tmp_fields[@sprintf("tmp_vertex_%d",i)] = Field((Vertex, K), zeros(vertex_dim, k_dim))
end
for j in 1:3
    tmp_fields[@sprintf("tmp_edge_%d",j)] = Field((Edge, K), zeros(edge_dim, k_dim))
end

@field_operator function initial_rho(
    mesh_radius::AbstractFloat,
    mesh_xydeg_x:: Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    mesh_xydeg_y:: Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    mesh_vertex_ghost_mask:: Field{Bool, 1, Tuple{Vertex_}, <:Tuple}
    )::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}

    lonc = 0.5 * pi
    latc = 0.0
    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x .* deg2radd, mesh_xydeg_y .* deg2radd
    rsina, rcosa = sin.(mesh_xyrad_y), cos.(mesh_xyrad_y)
    
    zdist = mesh_radius .* acos.(sin(latc) .* rsina .+ cos(latc) .* rcosa .* cos.(mesh_xyrad_x .- lonc))
  
    rpr = (zdist ./ (mesh_radius / 2.0)) .^ 2.0
   
    rpr = min.(1.0, rpr)

    return GridTools.broadcast(where(mesh_vertex_ghost_mask, 0.0, 0.5 .* (1.0 .+ cos.(pi .* rpr))), (Vertex, K))
end

function initial_rho(
    mesh_radius::AbstractFloat,
    mesh_xydeg_x:: Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    mesh_xydeg_y:: Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    mesh_vertex_ghost_mask:: Field{Bool, 1, Tuple{Vertex_}, <:Tuple};
    offset_provider::Dict
    )::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}

    unpack_dict(offset_provider)

    lonc = 0.5 * pi
    latc = 0.0
    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x .* deg2radd, mesh_xydeg_y .* deg2radd
    rsina, rcosa = sin.(mesh_xyrad_y), cos.(mesh_xyrad_y)
    
    zdist = mesh_radius .* acos.(sin(latc) .* rsina .+ cos(latc) .* rcosa .* cos.(mesh_xyrad_x .- lonc))
  
    rpr = (zdist ./ (mesh_radius / 2.0)) .^ 2.0
   
    rpr = min.(1.0, rpr)

    return GridTools.broadcast(where(mesh_vertex_ghost_mask, 0.0, 0.5 .* (1.0 .+ cos.(pi .* rpr))), (Vertex, K))
end

state.rho .= GridTools.initial_rho(
    mesh.radius,
    mesh.xydeg_x,
    mesh.xydeg_y,
    mesh.vertex_ghost_mask,
    offset_provider=mesh.offset_provider,
)


@field_operator function initial_velocity(
    mesh_xydeg_x:: Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    mesh_xydeg_y:: Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    metric_gac:: Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    metric_g11:: Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    metric_g22:: Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}
    )::Tuple{Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}, Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}, Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}}
    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x .* deg2radd, mesh_xydeg_y .* deg2radd
    u0 = 22.238985328911745
    flow_angle = 0.0 * deg2radd  # radians

    rsina, rcosa = sin.(mesh_xyrad_y), cos.(mesh_xyrad_y)
    cosb, sinb = cos(flow_angle), sin(flow_angle)
    uvel_x = u0 .* (cosb .* rcosa .+ rsina .* cos.(mesh_xyrad_x) .* sinb)
    uvel_y = -u0 .* sin.(mesh_xyrad_x) .* sinb

    vel_x = broadcast(uvel_x .* metric_g11 .* metric_gac, (Vertex, K))
    vel_y = broadcast(uvel_y .* metric_g22 .* metric_gac, (Vertex, K))
    vel_z = broadcast(0., (Vertex, K))
    return vel_x, vel_y, vel_z
end

out = GridTools.initial_velocity(
    mesh.xydeg_x,
    mesh.xydeg_y,
    metric.gac,
    metric.g11,
    metric.g22,
    offset_provider=mesh.offset_provider,
)

state.vel[1] .= out[1]
state.vel[2] .= out[2]
state.vel[3] .= out[3]

state_next.vel = state.vel

tmp_fields["tmp_vertex_1"] .= transpose(collect(0.:mesh.num_level-1))
tmp_fields["tmp_vertex_2"] .= nabla_z(tmp_fields["tmp_vertex_1"], level_indices, mesh.num_level, offset_provider = mesh.offset_provider)

state_next.rho = upwind_scheme(
    state.rho,
    δt,
    mesh.vol,
    metric.gac,
    state.vel[0],
    state.vel[1],
    state.vel[2],
    mesh.pole_edge_mask,
    mesh.dual_face_orientation,
    mesh.dual_face_normal_weighted_x,
    mesh.dual_face_normal_weighted_y,
    offset_provider=mesh.offset_provider
)