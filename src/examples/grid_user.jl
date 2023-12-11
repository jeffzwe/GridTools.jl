using Debugger
using GridTools

# When creating a local Dimension as V2VDim than the name of the Dimension must match the FieldOffset it will be used in. I.e. V2VDim_ = Dimension{:V2V_, LOCAL}

Cell_ = Dimension{:Cell_, HORIZONTAL}
K_ = Dimension{:K_, HORIZONTAL}
Edge_ = Dimension{:Edge_, HORIZONTAL}
Vertex_ = Dimension{:Vertex_, HORIZONTAL}
V2VDim_ = Dimension{:V2V_, LOCAL}
V2EDim_ = Dimension{:V2E_, LOCAL} 
E2VDim_ = Dimension{:E2V_, LOCAL} 
E2CDim_ = Dimension{:E2C_, LOCAL}
C2EDim_ = Dimension{:C2E_, LOCAL}
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

a = Field((Vertex, K), reshape(collect(-3.0:8.0), (6, 2)))
b = Field((K, Edge), reshape(collect(1.0:6.0), (2, 3)))

A = Field((Vertex, K), reshape(collect(1.:15.), 3, 5), origin = Dict(Vertex => -2, K => -1))
B = Field((K, Edge), reshape(ones(6), 3, 2))

mask_b = cat([true true false true true ; true false false false true ;true true true true true], [true false true false true ; true false false false true ;true true true true true], dims=3)

mask = Field((Vertex, K, Edge), mask_b, origin = Dict(Vertex => -2, K => -1))

edge_to_cell_table = [
    [1  -1];
    [3  -1];
    [3  -1];
    [4  -1];
    [5  -1];
    [6  -1];
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

cell_values = Field(Cell, [1.0, 1.0, 2.0, 3.0, 5.0, 8.0])

E2C_offset_provider = Connectivity(edge_to_cell_table, Cell, Edge, 2)
C2E_offset_provider = Connectivity(cell_to_edge_table, Edge, Cell, 3)

offset_provider = Dict{String, Union{Connectivity, Dimension}}(
                   "E2C" => E2C_offset_provider,
                   "C2E" => C2E_offset_provider
                )

out = Field(Edge, zeros(Float64, 12))

@field_operator function fo_neighbor_sum(a::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Edge_}}
    return neighbor_sum(a(E2C), axis=E2CDim)
end

fo_neighbor_sum(cell_values, offset_provider=offset_provider, out = out)






