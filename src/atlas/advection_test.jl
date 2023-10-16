# advection test
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

level_indices = Field(K, zeros(Int, k_dim))
level_indices.data .= collect(0:29)


state = SC_from_mesh(mesh)
state_next = SC_from_mesh(mesh)

tmp_fields = Dict{String, Field}()
for i in 1:6
    tmp_fields[@sprintf("tmp_vertex_%d",i)] = Field((Vertex, K), zeros(vertex_dim, k_dim))
end
for j in 1:3
    tmp_fields[@sprintf("tmp_edge_%d",j)] = Field((Edge, K), zeros(edge_dim, k_dim))
end

@field_operator function initial_rho(
    mesh_radius::AbstractFloat,
    mesh_xydeg_x::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    mesh_xydeg_y::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    mesh_vertex_ghost_mask::Field{Bool, 1, Tuple{Vertex_}, <:Tuple}
    )::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}

    lonc = 0.5 * pi
    latc = 0.0
    deg2radd = 2.0 * pi / 360.0

    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x .* deg2radd, mesh_xydeg_y .* deg2radd
    rsina, rcosa = sin.(mesh_xyrad_y), cos.(mesh_xyrad_y)
    
    zdist = mesh_radius .* acos.(sin(latc) .* rsina .+ cos(latc) .* rcosa .* cos.(mesh_xyrad_x .- lonc))
  
    rpr = (zdist ./ (mesh_radius / 2.0)) .^ 2.0
   
    rpr = min.(1.0, rpr)

    return GridTools.broadcast(where(mesh_vertex_ghost_mask, 0.0, 0.5 .* (1.0 .+ cos.(pi .* rpr))), (Vertex, K))
end

state.rho .= initial_rho(
    mesh.radius,
    mesh.xydeg_x,
    mesh.xydeg_y,
    mesh.vertex_ghost_mask,
    offset_provider=mesh.offset_provider,
)


@field_operator function initial_velocity(
    mesh_xydeg_x::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    mesh_xydeg_y::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    metric_gac::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    metric_g11::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    metric_g22::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}
    )::Tuple{Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}, Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}, Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}}
    deg2radd = 2.0 * pi / 360.0
    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x .* deg2radd, mesh_xydeg_y .* deg2radd
    u0 = 22.238985328911745
    flow_angle = 0.0 * deg2radd  # radians

    rsina, rcosa = sin.(mesh_xyrad_y), cos.(mesh_xyrad_y)
    cosb, sinb = cos(flow_angle), sin(flow_angle)
    uvel_x = u0 .* (cosb .* rcosa .+ rsina .* cos.(mesh_xyrad_x) .* sinb)
    uvel_y = -u0 .* sin.(mesh_xyrad_x) .* sinb

    vel_x = GridTools.broadcast(uvel_x .* metric_g11 .* metric_gac, (Vertex, K))
    vel_y = GridTools.broadcast(uvel_y .* metric_g22 .* metric_gac, (Vertex, K))
    vel_z = GridTools.broadcast(0., (Vertex, K))
    return vel_x, vel_y, vel_z
end

out = initial_velocity(
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

# state_next.vel = state.vel   #TODO: immutable object!

tmp_fields["tmp_vertex_1"] .= transpose(collect(0.:mesh.num_level-1))
# tmp_fields["tmp_vertex_2"] .= nabla_z(tmp_fields["tmp_vertex_1"], level_indices, mesh.num_level, offset_provider = mesh.offset_provider)

state_next.rho .= upwind_scheme(
    state.rho,
    δt,
    mesh.vol,
    metric.gac,
    state.vel[1],
    state.vel[2],
    state.vel[3],
    mesh.pole_edge_mask,
    mesh.dual_face_orientation,
    mesh.dual_face_normal_weighted_x,
    mesh.dual_face_normal_weighted_y,
    offset_provider = mesh.offset_provider
)

