using OffsetArrays

include("embedded/grid_tools.jl")
using .GridTools

using Debugger

include("gt2py/gt2py.jl")

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

expr = :(function hello(f::Field{Int32, 1, Tuple{Cell_}}, g::Field{Int32, 1, Tuple{Cell_}})::Field{Int32, 1, Tuple{Cell_}}
                tmp = f
                if 1. .< 10.0
                    tmp = f .+ 1
                    if 30 > 5
                        tmp = tmp .+ 20
                        tmp = tmp ./ 10
                    elseif 40 < 4
                        tmp = 4 == 5 ? tmp : tmp .- 100
                    else 
                        tmp = tmp .* 5
                    end
                    tmp = tmp .+ 10
                elseif 10 < 20
                    tmp = f .- 1
                else
                    tmp = tmp .* 10
                    tmp = tmp .+ 10
                    tmp = tmp .+ 100
                end
                return tmp
            end)

expr = :(function addition(f::Field{Int32, 1, Tuple{Cell_}}, g::Field{Int32, 1, Tuple{Cell_}})::Field{Int32, 1, Tuple{Cell_}}
            return f .+ g
        end)

# ------------------------------------------------

# a = Field(Cell, [5., 6., 7., 8., 3., 4., 5., 7., 4., 3., 2., 4., 6., 7., 5., 3., 2., 2., 5.])
# out = Field(Cell, zeros(Float64, 19))

# @field_operator function arithmetic_test(a::Field{Float64, 1, Tuple{Cell_}})
#     return a .+ 10. ./ 2.
#     end

# res = arithmetic_test(b, backend = "py", out = out)

# # ------------------------------------------------

# a = Field(Cell, Int32[1, 2, 3, 4])
# b = Field(Cell, Int32[5, 6, 7, 8])
# out = Field(Cell, zeros(Int32, 4))

# @field_operator function arithmetic_test(a::Field{Int32, 1, Tuple{Cell_}}, b::Field{Int32, 1, Tuple{Cell_}})
#     return a .+ b
#     end

# res = arithmetic_test(a, b, backend = "py", out = out)

# # ------------------------------------------------

# a = Field(Cell, [5., 6., 7., 8., 3., 4., 5., 7., 4., 3., 2., 4., 6., 7., 5., 3., 2., 2., 5.])
# out = Field(Edge, zeros(Float64, 12))

# @field_operator function remapping_test(b::Field{Float64, 1, Tuple{Cell_}})
#     return b(E2C[1])
#     end

# res = remapping_test(b, offset_provider=offset_provider, backend = "py", out = out)

# # ------------------------------------------------

# out = Field(Edge, zeros(Float64, 12))
# b = Field(Cell, [5., 6., 7., 8., 3., 4., 5., 7., 4., 3., 2., 4., 6., 7., 5., 3., 2., 2., 5.])
    
# @field_operator function neighbor_sum_test(b::Field{Float64, 1, Tuple{Cell_}})
#     return neighbor_sum(b(E2C), axis=E2CDim)
#     end

# res = neighbor_sum_test(b, offset_provider=offset_provider, backend = "py", out = out)

# ------------------------------------------------

# a = Field((Cell, K), reshape(collect(-3.0:8.0), (6, 2)))
# b = Field((Cell, K), fill(-10., (6, 2)))
# mask = Field((Cell, K), rand(Bool, (6, 2)))
# out = Field((Cell, K), zeros(6, 2))

# @field_operator function where_test(mask::Field{Bool, 2, Tuple{Cell_, K_}}, a::Field{Float64, 2, Tuple{Cell_, K_}}, b::Field{Float64, 2, Tuple{Cell_, K_}})
#         return where(mask, a, b)
#         end

# res = where_test(mask, a, b, backend = "py", out = out)


# ------------------------------------------------

