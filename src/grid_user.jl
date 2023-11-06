using OffsetArrays
using Debugger

include("GridTools.jl")
using .GridTools

Cell_ = Dimension{:Cell_, HORIZONTAL}
K_ = Dimension{:K_, HORIZONTAL}
Edge_ = Dimension{:Edge_, HORIZONTAL}
Vertex_ = Dimension{:Vertex_, HORIZONTAL}
V2VDim_ = Dimension{:V2VDim_, LOCAL}
V2EDim_ = Dimension{:V2EDim_, LOCAL} 
E2VDim_ = Dimension{:E2VDim_, LOCAL} 
E2CDim_ = Dimension{:E2CDim_, LOCAL}
C2EDim_ = Dimension{:C2EDim_, LOCAL}
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


a = Field((Cell, K), reshape(collect(-3.0:8.0), (6, 2)))
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

a = Field((Cell, K), reshape(collect(1.:12.), (6, 2)))
out = Field((Cell, K), zeros(Integer, (6, 2)))

@field_operator function fo_astype(a::Field{Float64, 2, Tuple{Cell_, K_}})::Field{Int64, 2, Tuple{Cell_, K_}}
        return convert(Int64, a)
end

fo_astype(a, backend = "py", out = out)