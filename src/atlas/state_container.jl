
struct StateContainer
    rho::Field{Tuple{Vertex_, K_}, <:AbstractFloat}
    vel::Tuple{
        Field{Tuple{Vertex_, K_}, <:AbstractFloat},
        Field{Tuple{Vertex_, K_}, <:AbstractFloat},
        Field{Tuple{Vertex_, K_}, <:AbstractFloat},
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
