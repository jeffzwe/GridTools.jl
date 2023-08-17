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

function nabla_z(psi::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}, level_indices::Field{<:Integer, 1, Tuple{K_}, <:Tuple}, num_level::Integer)
    return ??? # What does Koff do?
end

