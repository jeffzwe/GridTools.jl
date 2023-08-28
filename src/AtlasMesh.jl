using PyCall

include("GridTools.jl")
using .GridTools

# import .GridTools: Dimension, Field, Connectivity

atlas = pyimport("atlas4py")
py"""
from atlas4py import (
    StructuredGrid,
    Topology,
    Config,
    StructuredMeshGenerator,
    functionspace,
    build_edges,
    build_node_to_edge_connectivity,
    build_node_to_cell_connectivity,
    build_element_to_edge_connectivity,
    build_median_dual_mesh,
    build_periodic_boundaries,
    build_halo,
    build_parallel_fields,
    BlockConnectivity,
)
"""



const rpi = 2.0 * asin(1.0)
const _deg2rad = 2.0 * rpi / 360.0

DIMENSION_TO_SIZE_ATTR = Dict{Dimension, String}(
        Vertex => "num_vertices",
        Edge => "num_edges",
        Cell => "num_cells",
        K => "num_level"
)

function _atlas_connectivity_to_array(atlas_conn; out = nothing, skip_neighbor_indicator = -1)
    if py"isinstance"(atlas_conn, atlas.BlockConnectivity)
        shape = (atlas_conn.rows, atlas_conn.cols)
        out === nothing ? out = zeros(Integer,shape) : out
        @assert size(out) == shape

        for i in 1:atlas_conn.rows
            for nb in 1:atlas_conn.cols
                out[i, nb] = atlas_conn[i, nb]
            end
        end

        return out
    end

    shape = (atlas_conn.rows, atlas_conn.maxcols)
    out === nothing ? out = zeros(Integer,shape) : out
    @assert size(out) == shape

    for i in 1:atlas_conn.rows
        cols = atlas_conn.cols(i-1)
        for nb in 1:cols
            out[i, nb] = atlas_conn[i, nb]
        end
        out[i, (cols+1):end] .= skip_neighbor_indicator
    end

    return out
end

py"""
def _build_atlas_mesh(config, grid, periodic_halos=None):
    # compare https://github.com/ckuehnlein/FVM_BESPOKE_NWP/blob/bespoke_nwp/src/fvm/core/datastruct_module.F90#L353
    mesh = StructuredMeshGenerator(config).generate(grid)

    # note: regularly the following calls are done implicitly using
    functionspace.EdgeColumns(mesh, halo=periodic_halos)
    functionspace.NodeColumns(mesh, halo=periodic_halos)
    # if periodic_halos:
    #    build_parallel_fields(mesh)
    #    build_periodic_boundaries(mesh)
    #    build_halo(mesh, periodic_halos)

    # build_edges(mesh, config)
    build_node_to_edge_connectivity(mesh)
    build_node_to_cell_connectivity(mesh)
    build_median_dual_mesh(mesh)

    return mesh
"""

struct AtlasMesh
    num_vertices::Integer
    num_edges::Integer
    num_cells::Integer
    num_level::Integer
    num_pole_edges::Integer
    nb_vertices_ghost::Integer
    nb_vertices_noghost::Integer

    # connectivities
    c2v::Connectivity
    c2e::Connectivity
    v2e::Connectivity
    v2c::Connectivity
    v2v::Connectivity
    e2v::Connectivity
    e2c::Connectivity

    c2v_np::Array
    c2e_np::Array
    v2e_np::Array
    v2c_np::Array
    v2v_np::Array
    e2v_np::Array
    e2c_np::Array

    # poles
    pole_edges::Array # list of all pole edges
    pole_edge_mask::Field{Bool, 1, Tuple{Edge_}, <:Tuple}
    pole_edge_mask_np::Array

    # remote indices: for each geometric entity it's remote index
    remote_indices::Dict{Dimension, Array}

    # flags
    vertex_flags::Array
    edge_flags::Array
    cell_flags::Array

    vertex_ghost_mask::Field{Bool, 1, Tuple{Vertex_}, <:Tuple}

    # geometry
    radius::Float64
    xydeg_x::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}
    xydeg_y::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}
    xydeg_np::Array
    xyrad::Array
    xyarc::Array
    xyz::Array

    vol::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}
    vol_np::Array

    dual_face_normal_weighted_x::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple}
    dual_face_normal_weighted_y::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple}
    dual_face_normal_weighted_np::Array

    dual_face_orientation::Field{<:AbstractFloat, 2, Tuple{Vertex_, V2EDim_}, <:Tuple}

    dual_face_orientation_np::Array

    offset_provider::Dict{String, Union{Connectivity, Dimension}}

    grid_description::String  # string representation of the atlas grid instance

    _atlas_mesh::Any  # for debugging


    function AtlasMesh(grid; num_level::Integer, radius=6371.22e03, config=nothing)::AtlasMesh
        if config == nothing
            py"""
            config = Config()
            config["triangulate"] = False
            config["angle"] = -1.0
            config["pole_edges"] = True
            config["ghost_at_end"] = True 
            """
            config = py"config"
        end
        # generate mesh from grid points    
        mesh = py"_build_atlas_mesh"(config, grid, periodic_halos=10)

        num_cells = mesh.cells.size
        num_edges = mesh.edges.size
        num_vertices = mesh.nodes.size

        # flags
        vertex_flags = mesh.nodes.flags()
        edge_flags = mesh.edges.flags()
        cell_flags = mesh.cells.flags()

        vertex_ghost_mask = Field((Vertex,), [x != 0 for x in (vertex_flags .&  atlas.Topology.GHOST)])
        nb_vertices_ghost = sum(vertex_ghost_mask)
        nb_vertices_noghost = num_vertices - nb_vertices_ghost
        @assert nb_vertices_noghost == sum(grid.nx)

        # connectivities
        v2e_np = _atlas_connectivity_to_array(mesh.nodes.edge_connectivity)
        v2c_np = _atlas_connectivity_to_array(mesh.nodes.cell_connectivity)
        v2v_np = zeros(Integer, size(v2e_np))  # initialized further below
        e2v_np = _atlas_connectivity_to_array(mesh.edges.node_connectivity)
        e2c_np = _atlas_connectivity_to_array(mesh.edges.cell_connectivity)
        c2v_np = _atlas_connectivity_to_array(mesh.cells.node_connectivity)
        c2e_np = _atlas_connectivity_to_array(mesh.cells.edge_connectivity)

        v2e_np .+= 1
        v2c_np .+= 1 
        v2v_np = ones(Integer, size(v2e_np))  # initialized further below
        e2v_np .+= 1
        e2c_np .+= 1
        c2v_np .+= 1
        c2e_np .+= 1

        @assert size(v2e_np)[1] == num_vertices
        @assert size(v2c_np)[1] == num_vertices
        @assert size(e2v_np)[1] == num_edges
        @assert size(e2c_np)[1] == num_edges
        @assert size(c2v_np)[1] == num_cells
        @assert size(c2e_np)[1] == num_cells

        v2e = Connectivity(v2e_np, (Vertex,), (Edge,), size(v2e_np)[1])
        v2c = Connectivity(v2c_np, (Vertex,), (Cell,), size(v2c_np)[1])
        v2v = Connectivity(v2v_np, (Vertex,), (Vertex,), size(v2v_np)[1])
        e2v = Connectivity(e2v_np, (Edge,), (Vertex,), size(e2v_np)[1])
        e2c = Connectivity(e2c_np, (Edge,), (Cell,), size(e2c_np)[1])
        c2v = Connectivity(c2v_np, (Cell,), (Vertex,), size(c2v_np)[1])
        c2e = Connectivity(c2e_np, (Cell,), (Edge,), size(c2e_np)[1])

        vertex_remote_indices = mesh.nodes.field("remote_idx")
        edge_remote_indices = mesh.edges.field("remote_idx")
        cell_remote_indices = mesh.cells.field("remote_idx")

        # geometrical properties
        xydeg_np = mesh.nodes.lonlat
        xydeg_x = Field((Vertex,),xydeg_np[:, 1])
        xydeg_y = Field((Vertex,),xydeg_np[:, 2])
        xyrad = xydeg_np .* _deg2rad
        xyarc = xydeg_np .* _deg2rad .* radius
        phi, theta = xyrad[:, 2], xyrad[:, 1]
        xyz = hcat(cos.(phi) .* cos.(theta), cos.(phi) .* sin.(theta), sin.(phi))

        # face orientation
        edges_per_node = size(v2e_np)[2]
        dual_face_orientation_np = zeros((num_vertices, edges_per_node))  # formerly known as "sign field"

        @inline is_pole_edge(e::Integer, edge_flags::Array)::Bool = atlas.Topology.check(edge_flags[e], atlas.Topology.POLE)

        num_pole_edges = 0
        pole_edge_mask_np = zeros(Bool, num_edges)
        for e in 1:num_edges
            if is_pole_edge(e, edge_flags)
                num_pole_edges += 1
                pole_edge_mask_np[e] = true
            end
        end
        pole_edge_mask = Field((Edge,), pole_edge_mask_np)

        pole_edges = zeros(Integer, num_pole_edges)
        inum_pole_edge = 0
        for e in 1:num_edges
            if is_pole_edge(e, edge_flags)
                inum_pole_edge += 1
                pole_edges[inum_pole_edge] = e
            end
        end

        for v in 1:num_vertices
            for e_nb in 1:edges_per_node
                e = v2e_np[v, e_nb]
                if e != 0
                    if v == e2v_np[e, 1]
                        dual_face_orientation_np[v, e_nb] = 1.0
                        v2v_np[v, e_nb] = e2v_np[e,2]
                    else
                        dual_face_orientation_np[v, e_nb] = -1.0
                        v2v_np[v, e_nb] = e2v_np[e,1]
                        if is_pole_edge(e, edge_flags)
                            dual_face_orientation_np[v, e_nb] = 1.0
                        end
                    end
                else
                    dual_face_orientation_np[v, e_nb] = NaN
                    v2v_np[v, e_nb] = 0
                end
            end
        end
        dual_face_orientation = Field((Vertex, V2EDim), dual_face_orientation_np)

        # dual normal
        dual_face_normal_weighted_np = mesh.edges.field("dual_normals") .* radius .* _deg2rad
        dual_face_normal_weighted_x = Field((Edge,), dual_face_normal_weighted_np[:,1])
        dual_face_normal_weighted_y = Field((Edge,), dual_face_normal_weighted_np[:,2])

        # dual volume
        vol_np = mesh.nodes.field("dual_volumes") .* _deg2rad^2 .* radius^2
        vol = Field((Vertex,), vol_np)

        # offset_provider
        offset_provider = Dict{String, Union{Connectivity, Dimension}}(
            "V2V" => v2v,
            "V2E" => v2e,
            "E2V" => e2v,
            "Koff" => K  # TODO(tehrengruber): using K here gives a terrible compilation error. Improve in GT4Py!
        )

        remote_indices = Dict{Dimension, Array}(
            Vertex => vertex_remote_indices,
            Edge => edge_remote_indices,
            Cell => cell_remote_indices
        )

        grid_description = string(grid)

        return new(
            num_vertices,
            num_edges,
            num_cells,
            num_level,
            num_pole_edges,
            nb_vertices_ghost,
            nb_vertices_noghost,

            # connectivities
            c2v,
            c2e,
            v2e,
            v2c,
            v2v,
            e2v,
            e2c,

            c2v_np,
            c2e_np,
            v2e_np,
            v2c_np,
            v2v_np,
            e2v_np,
            e2c_np,

            # poles
            pole_edges, # list of all pole edges
            pole_edge_mask,
            pole_edge_mask_np,

            # remote indices: for each geometric entity it's remote index
            remote_indices,

            # flags
            vertex_flags,
            edge_flags,
            cell_flags,
            vertex_ghost_mask,

            # geometry
            radius,
            xydeg_x,
            xydeg_y,
            xydeg_np,
            xyrad,
            xyarc,
            xyz,

            vol,
            vol_np,

            dual_face_normal_weighted_x,
            dual_face_normal_weighted_y,
            dual_face_normal_weighted_np,

            dual_face_orientation,

            dual_face_orientation_np,

            offset_provider,

            grid_description,  # string representation of the atlas grid instance

            mesh  # for debugging
        )
    end
end

function info(atlas_mesh::AtlasMesh)
    n = ceil(log10(atlas_mesh.num_edges + 1))

    return """
    Atlas mesh
      grid:     $(atlas_mesh.grid_description)
      vertices: $(rpad(string(atlas_mesh.num_vertices), n))  (ghost: $(rpad(string(atlas_mesh.nb_vertices_ghost), n)), noghost: $(rpad(string(atlas_mesh.nb_vertices_noghost), n)))
      edges:    $(rpad(string(atlas_mesh.num_edges), n))
      cells:    $(rpad(string(atlas_mesh.num_cells), n))
      level:    $(rpad(string(atlas_mesh.num_level), n))
    """
end

