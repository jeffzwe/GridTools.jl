include("GridUser.jl")






function with_boundary_values(
    lower::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    interior::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    upper::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    level_indices::Field{<:Integer, 1, Tuple{K_}, <:Tuple},
    num_level::Integer
    )::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}

    return where(level_indices .== 0, lower, where(level_indices .== num_level-1, upper, interior))
end

function nabla_z(
    psi::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}, 
    level_indices::Field{<:Integer, 1, Tuple{K_}, <:Tuple}, num_level::Integer,
    offset_provider::Dict{String, Union{Connectivity, Dimension}}
    )
    return with_boundary_values(
        psi(offset_provider["Koff"](2)) - psi(offset_provider["Koff"](1)),
        psi(offset_provider["Koff"](2)) - psi(offset_provider["Koff"]()),
        psi(offset_provider["Koff"](1)) - psi(offset_provider["Koff"](1)),
        level_indices, num_level
    )
end

function advector_in_edges(
    vel_x::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    vel_y::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    pole_edge_mask::Field{Bool, 1, Tuple{Edge_}, <:Tuple},
    offset_provider::Dict{String, Union{Connectivity, Dimension}}
    )::Tuple{Field, Field}
    pole_bc = where(pole_edge_mask, -1.0, 1.0)
    vel_edges_x = 0.5 * (vel_x(offset_provider["E2V"](1)) + pole_bc * vel_x(offset_provider["E2V"][1]))
    vel_edges_y = 0.5 * (vel_y(offset_provider["E2V"](1)) + pole_bc * vel_y(offset_provider["E2V"][1]))
    return vel_edges_x, where(pole_edge_mask, 0.0, vel_edges_y)
end



