
struct StateContainer
    rho::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}}
    vel::Tuple{
        Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}},
        Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}},
        Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}},
    }
end

function sc_from_mesh(mesh::AtlasMesh)
    vertex_dim = getproperty(mesh, DIMENSION_TO_SIZE_ATTR[Vertex])
    k_dim = getproperty(mesh, DIMENSION_TO_SIZE_ATTR[K])
    return StateContainer(
        Field((Vertex, K), zeros((vertex_dim, k_dim))),
        (
            Field((Vertex, K), zeros((vertex_dim, k_dim))),
            Field((Vertex, K), zeros((vertex_dim, k_dim))),
            Field((Vertex, K), zeros((vertex_dim, k_dim)))
        )
    )
end
