using GridTools
using Debugger

# Simple Add example ----------------------------------------------------------

# # Creation of Dimensions 
Cell_ = Dimension{:Cell_, HORIZONTAL}
K_ = Dimension{:K_, HORIZONTAL}
Cell = Cell_()
K = K_()

# Creation of a field_operator
@field_operator function simple_add(a::Field{Tuple{Cell_, K_}, Float64}, b::Field{Tuple{Cell_, K_}, Float64})::Field{Tuple{Cell_, K_}, Float64}
    return a .+ b
end

# Creation of Fields
a = Field((Cell, K), reshape(collect(1.:6.), (2,3)))
b = Field((Cell, K), reshape(collect(2.:7.), (2,3)))
out = Field((Cell, K), zeros(2, 3))

simple_add(a, b, out = out)

println("Simple_add result: $out")

# Neighbor Sum example --------------------------------------------------------

Edge_ = Dimension{:Edge_, HORIZONTAL}
E2CDim_ = Dimension{:E2C_, LOCAL}
Edge = Edge_()
E2CDim = E2CDim_()
E2C = FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))

cell_values = Field(Cell, [1.0, 1.0, 2.0, 3.0, 5.0, 8.0])
out = Field(Edge, zeros(Float64, 12))
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

E2C_offset_provider = Connectivity(edge_to_cell_table, Cell, Edge, 2)
offset_provider = Dict{String, Union{Connectivity, Dimension}}(
    "E2C" => E2C_offset_provider,
)

@field_operator function fo_neighbor_sum(a::Field{Tuple{Cell_}, Float64})::Field{Tuple{Edge_}, Float64}
    return neighbor_sum(a(E2C), axis=E2CDim)
end

fo_neighbor_sum(cell_values, offset_provider=offset_provider, out = out)

println("Neighborsum result: $out")


