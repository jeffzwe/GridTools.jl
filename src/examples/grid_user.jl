using Debugger
using GridTools
using Test

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

cell_values = Field(Cell, [1.0, 1.0, 2.0, 3.0, 5.0, 8.0])
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
C2E_offset_provider = Connectivity(edge_to_cell_table, Edge, Cell, 2)

offset_provider = Dict{String, Union{Connectivity, Dimension}}(
                   "E2C" => E2C_offset_provider,
                   "C2E" => C2E_offset_provider
                )

x = Field((Cell, K), reshape(collect(-3.0:2.0), (3, 2)))
y = Field((K, Edge), reshape(collect(1.0:6.0), (2, 3)))

x_off = Field((Cell, K), reshape(collect(-3.0:2.0), (3, 2)), origin = Dict(Cell => -2, K => 1))
y_off = Field((K, Edge), reshape(collect(1.0:6.0), (2, 3)), origin = Dict(K => 1, Edge => 2))

X = Field((Vertex, K), reshape(collect(1.:15.), 3, 5), origin = Dict(Vertex => -2, K => -1))
Y = Field((K, Edge), reshape(ones(6), 3, 2))

mask = cat([true true false true true ; true false false false true ;true true true true true], [true false true false true ; true false false false true ;true true true true true], dims=3)
mask_offset = Field((Vertex, K, Edge), mask, origin = Dict(Vertex => -2, K => -1))

# Tests ------------------------------------------------------------------------------------------------------------------------------

@testset "Testset arithmetic broadcast" begin

	result_data = zeros(Float64, (3,2,3))

    result_data[:, :, 1] = [-2.0  2.0;
                            -1.0  3.0;
                             0.0  4.0]
    
    result_data[:, :, 2] = [0.0  4.0;
                            1.0  5.0;
                            2.0  6.0]

    result_data[:, :, 3] = [2.0  6.0;
                            3.0  7.0;
                            4.0  8.0]

    # No Offsets -------------------------------

    result_addition = Field((Cell, K, Edge), result_data)
    @test x .+ y == result_addition

    # With Offsets -------------------------------
    
    result_addition_off = Field((Cell, K, Edge), result_data, origin = Dict(Cell => -2, K => 1, Edge => 2))
    @test x_off .+ y_off == result_addition_off

end

@testset "Testset Field-Offset call" begin

    result_offset_call_data = [
            1.0  0.0;
            2.0  0.0;
            2.0  0.0;
            3.0  0.0;
            5.0  0.0;
            8.0  0.0;
            1.0  8.0;
            1.0  1.0;
            1.0  2.0;
            1.0  3.0;
            3.0  5.0;
            5.0  8.0
    ]

    result_offset_call = Field((Edge, E2CDim), result_offset_call_data)

    out = Field((Edge, E2CDim), zeros(Float64, (12, 2)))

    @field_operator function fo_remapping(a::Field{Tuple{Cell_}, Float64})::Field{Tuple{Edge_, E2CDim_}, Float64}
        return a(E2C)
    end

    fo_remapping(cell_values, offset_provider=offset_provider, out = out)
    @test  out == result_offset_call
end


@testset "Testset built_ins" begin

    # Neighbor_sum -------------------------

    result_neighbor_sum_data = [
        1.0,
        2.0,
        2.0,
        3.0,
        5.0,
        8.0,
        9.0,
        2.0,
        3.0,
        4.0,
        8.0,
       13.0
    ]

    result_neighbor_sum = Field(Edge, result_neighbor_sum_data)
    out = Field(Edge, zeros(Float64, 12))

    @field_operator function fo_neighbor_sum(a::Field{Tuple{Cell_}, Float64})::Field{Tuple{Edge_}, Float64}
        return neighbor_sum(a(E2C), axis=E2CDim)
    end

    fo_neighbor_sum(cell_values, offset_provider=offset_provider, out = out)
    @test out == result_neighbor_sum

    # Broadcast -------------------------

    @test typeof(broadcast(cell_values, (Cell, K))) == Field{Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}, Float64, 1, Tuple{Dimension{:Cell_, HORIZONTAL}}, Vector{Float64}}
    @test typeof(broadcast(5.0, (Cell, K))) == Field{Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}, Float64, 0, Tuple{}, Array{Float64, 0}}

    # Where -----------------------------------------

    #TODO

    # Where with offsets-----------------------------
    result_data = zeros(Float64, 3, 5, 2)
    result_data[:, :, 1] .= [1.0 4.0 1.0 10.0 13.0;
                             2.0 1.0 1.0  1.0 14.0;
                             3.0 6.0 9.0 12.0 15.0]
    
    result_data[:, :, 2] .= [1.0 1.0 7.0 1.0 13.0;
                             2.0 1.0 1.0 1.0 14.0;
                             3.0 6.0 9.0 12.0 15.0;]
                                          
    result_where = Field((Vertex, K, Edge), result_data, origin = Dict(Vertex => -2, K => -1))

    @test where(mask_offset, X, Y) == result_where 

    # Convert ----------------------------

    @test typeof(convert(Int64, cell_values)) == Field{ Tuple{Dimension{:Cell_, HORIZONTAL}}, Int64, 1, Tuple{Dimension{:Cell_, HORIZONTAL}}, Vector{Int64}}

    # Concat -----------------------------
    a_data = [1.0 1.0 1.0;
              1.0 NaN 1.0;
              1.0 1.0 1.0]
    a = Field((Cell, K), a_data)
    b_data = [NaN NaN NaN;
              NaN 100.0 NaN;
              NaN NaN NaN]
    b = Field((Cell, K), b_data)

    result_data = [1.0 1.0 1.0;
              1.0 100.0 1.0;
              1.0 1.0 1.0]
    result = Field((Cell, K), result_data)

    @test concat(a, b) == result
end