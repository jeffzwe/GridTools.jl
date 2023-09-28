
include("GridTools.jl")
using .GridTools
# include("Jast_to_Foast.jl")

using OffsetArrays

@create_dim Cell_
@create_dim K_ 
@create_dim Edge_ 
@create_dim Vertex_ 
@create_dim V2VDim_ 
@create_dim V2EDim_ 
@create_dim E2VDim_ 
@create_dim E2CDim_ 
@create_dim C2EDim_
Cell = Cell_()
K = K_()
Edge = Edge_()
Vertex = Vertex_()
V2VDim = V2VDim_()
V2EDim = V2EDim_()
E2VDim = E2VDim_()
E2CDim = E2CDim_()
C2EDim = C2EDim_()

V2V = FieldOffset("V2V", source=Vertex, target=(Vertex, V2VDim))
E2V = FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2C = FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
C2E = FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))
Koff = FieldOffset("Koff", source=K, target=K)


a = Field((Vertex, K), reshape(collect(-3.0:2.0), (3, 2)))
b = Field((K, Edge), reshape(collect(1.0:6.0), (2, 3)))

A = Field((Vertex, K), OffsetArray(reshape(collect(1.:15.), 3, 5), -1:1, 0:4))
B = Field((K, Edge), OffsetArray(reshape(ones(6), 3, 2), 1:3, 1:2))

mask_b = cat([true true false true true ; true false false false true ;true true true true true], [true false true false true ; true false false false true ;true true true true true], dims=3)

mask = Field((Vertex, K, Edge), OffsetArray(mask_b, -1:1, 0:4, 1:2))

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


E2C_offset_provider = Connectivity(edge_to_cell_table, Cell, Edge, 2)
C2E_offset_provider = Connectivity(cell_to_edge_table, Edge, Cell, 3)

offset_provider = Dict{String, Connectivity}(
                   "E2C" => E2C_offset_provider,
                   "C2E" => C2E_offset_provider
                )

expr = :(function hello(x::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple}, z::Tuple{Integer, String}; y::Field, yy::Integer)

    a = x[2]
    field = Field((Vertex, K), matrix)

    return x .+ sum(E2C(1)[1]), x
end)



# include("AtlasMesh.jl")
# include("Advection.jl")
# include("StateContainer.jl")
# include("Metric.jl")
# include("AdvectionTest.jl")