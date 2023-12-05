# advection test
using Printf
using Debugger
using Statistics
using GridTools

Cell_ = Dimension{:Cell_, HORIZONTAL}
Edge_ = Dimension{:Edge_, HORIZONTAL}
Vertex_ = Dimension{:Vertex_, HORIZONTAL}
K_ = Dimension{:K_, VERTICAL}
V2VDim_ = Dimension{:V2V_, LOCAL}
V2EDim_ = Dimension{:V2E_, LOCAL} 
E2VDim_ = Dimension{:E2V_, LOCAL} 
Cell = Cell_()
K = K_()
Edge = Edge_()
Vertex = Vertex_()
V2VDim = V2VDim_()
V2EDim = V2EDim_()
E2VDim = E2VDim_()

V2V = FieldOffset("V2V", source=Vertex, target=(Vertex, V2VDim))
E2V = FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
Koff = FieldOffset("Koff", source=K, target=K)

include("atlas_mesh.jl")
include("state_container.jl")
include("metric.jl")
include("advection.jl")


grid = atlas.StructuredGrid("O32")
mesh = AtlasMesh(grid, num_level = 30)

δt = 1800.0  # time step in s
niter = 20
eps = 1.0e-8

metric = m_from_mesh(mesh)
origin = minimum(mesh.xyarc, dims=1)
extent = maximum(mesh.xyarc, dims=1) .- minimum(mesh.xyarc, dims=1)
xlim = (minimum(mesh.xyarc[:, 1]), maximum(mesh.xyarc[:, 1]))
ylim = (minimum(mesh.xyarc[:, 2]), maximum(mesh.xyarc[:, 2]))

vertex_dim = getproperty(mesh, DIMENSION_TO_SIZE_ATTR[Vertex])
k_dim = getproperty(mesh, DIMENSION_TO_SIZE_ATTR[K])
edge_dim = getproperty(mesh, DIMENSION_TO_SIZE_ATTR[Edge])

level_indices = Field(K, zeros(Int, k_dim))
level_indices .= collect(0:mesh.num_level-1)


state = sc_from_mesh(mesh)
state_next = sc_from_mesh(mesh)

tmp_fields = Dict{String, Field}()
for i in 1:6
    tmp_fields[@sprintf("tmp_vertex_%d",i)] = Field((Vertex, K), zeros(vertex_dim, k_dim))
end
for j in 1:3
    tmp_fields[@sprintf("tmp_edge_%d",j)] = Field((Edge, K), zeros(edge_dim, k_dim))
end

@field_operator function initial_rho(
    mesh_radius::Float64,
    mesh_xydeg_x::Field{Float64, 1, Tuple{Vertex_}},
    mesh_xydeg_y::Field{Float64, 1, Tuple{Vertex_}},
    mesh_vertex_ghost_mask::Field{Bool, 1, Tuple{Vertex_}}
    )::Field{Float64, 1, Tuple{Vertex_, K_}}

    lonc = 0.5 * pi
    latc = 0.0
    _deg2rad = 2.0 * pi / 360.0

    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x .* _deg2rad, mesh_xydeg_y .* _deg2rad
    rsina, rcosa = sin.(mesh_xyrad_y), cos.(mesh_xyrad_y)
    
    zdist = mesh_radius .* acos.(sin(latc) .* rsina .+ cos(latc) .* rcosa .* cos.(mesh_xyrad_x .- lonc))
    
    rpr = (zdist ./ (mesh_radius / 2.0)) .^ 2.0
    rpr = min.(1.0, rpr)

    return broadcast(where(mesh_vertex_ghost_mask, 0.0, 0.5 .* (1.0 .+ cos.(pi .* rpr))), (Vertex, K))
end


initial_rho(
    mesh.radius,
    mesh.xydeg_x,
    mesh.xydeg_y,
    mesh.vertex_ghost_mask,
    out = state.rho,
    offset_provider=mesh.offset_provider
)

@field_operator function initial_velocity(
    mesh_xydeg_x::Field{Float64, 1, Tuple{Vertex_}},
    mesh_xydeg_y::Field{Float64, 1, Tuple{Vertex_}},
    metric_gac::Field{Float64, 1, Tuple{Vertex_}},
    metric_g11::Field{Float64, 1, Tuple{Vertex_}},
    metric_g22::Field{Float64, 1, Tuple{Vertex_}}
    )::Tuple{Field{Float64, 1, Tuple{Vertex_, K_}}, Field{Float64, 1, Tuple{Vertex_, K_}}, Field{Float64, 0, Tuple{Vertex_, K_}}}
    _deg2rad = 2.0 * pi / 360.0
    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x .* _deg2rad, mesh_xydeg_y .* _deg2rad
    u0 = 22.238985328911745
    flow_angle = 0.0 * _deg2rad  # radians

    rsina, rcosa = sin.(mesh_xyrad_y), cos.(mesh_xyrad_y)
    cosb, sinb = cos(flow_angle), sin(flow_angle)
    uvel_x = u0 .* (cosb .* rcosa .+ rsina .* cos.(mesh_xyrad_x) .* sinb)
    uvel_y = -u0 .* sin.(mesh_xyrad_x) .* sinb

    vel_x = broadcast(uvel_x .* metric_g11 .* metric_gac, (Vertex, K))
    vel_y = broadcast(uvel_y .* metric_g22 .* metric_gac, (Vertex, K))
    vel_z = broadcast(0., (Vertex, K))
    return vel_x, vel_y, vel_z
end

initial_velocity(
    mesh.xydeg_x,
    mesh.xydeg_y,
    metric.gac,
    metric.g11,
    metric.g22,
    out = state.vel,
    offset_provider=mesh.offset_provider,
)

copyfield!(state_next.vel, state.vel)

println("min max avg of initial rho = $(minimum(state.rho.data)) , $(maximum(state.rho.data)) , $(mean(state.rho.data))")

tmp_fields["tmp_vertex_1"] .= reshape(collect(0.:mesh.num_level-1), (1, mesh.num_level))
nabla_z(tmp_fields["tmp_vertex_1"], level_indices, mesh.num_level, out=tmp_fields["tmp_vertex_2"], offset_provider = mesh.offset_provider)


for i in 1:niter

    upwind_scheme(
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
        out = state_next.rho,
        offset_provider = mesh.offset_provider,
        backend = "py"
    )

    println("Timestep $i")

    temp = state
    global state = state_next
    global state_next = temp

    update_periodic_layers(mesh, state.rho)
end

println("min max sum of final rho = $(minimum(state.rho.data)) , $(maximum(state.rho.data)) , $(sum(state.rho.data))")
println("Final Vel0 sum after $niter iterations: $(sum(state.vel[1].data))")
println("Final Vel1 sum after $niter iterations: $(sum(state.vel[2].data))")