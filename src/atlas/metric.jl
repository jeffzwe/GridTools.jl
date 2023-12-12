

struct Metric
    g11::Field{Tuple{Vertex_}, <:AbstractFloat}
    g22::Field{Tuple{Vertex_}, <:AbstractFloat}
    gac::Field{Tuple{Vertex_}, <:AbstractFloat}
end

function m_from_mesh(mesh::AtlasMesh)
    rsina = sin.(mesh.xyrad[:, 2])
    rcosa = cos.(mesh.xyrad[:, 2])

    g11 = Field(Vertex, 1.0 ./ rcosa)
    g22 = Field(Vertex, 1.0 .* ones(mesh.num_vertices))
    gac = Field(Vertex, rcosa)

    return Metric(g11, g22, gac)
end
