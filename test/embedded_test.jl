using Statistics

# Setup ------------------------------------------------------------------------------------------------------------------------------

num_cells = 5
num_layers = 6
grid_shape = (num_cells, num_layers)
a_value = 2.0
b_value = 3.0

a = Field((Cell, K), fill(a_value, grid_shape))
b = Field((Cell, K), fill(b_value, grid_shape))
e = Field(Edge, fill(b_value, 4))


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

offset_provider = Dict{String, Connectivity}(
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
mask_offset = Field((Vertex, K, Edge), mask_b, origin = Dict(Vertex => -2, K => -1))
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

    result_offset_call = [
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

    out = Field((Edge, E2CDim), zeros(Float64, (12, 2)))

    @field_operator function fo_remapping(a::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 2, Tuple{Edge_, E2CDim_}}
        return a(E2C)
    end

    fo_remapping(cell_values, offset_provider=offset_provider, out = out)
    @test  out .== result_offset_call
end


@testset "Testset built_ins" begin

    # Neighbor_sum -------------------------

    result_neighbor_sum = [
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
    out = Field(Edge, zeros(Float64, 12))

    @field_operator function fo_neighbor_sum(a::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Edge_}}
        return neighbor_sum(a(E2C), axis=E2CDim)
    end

    fo_neighbor_sum(a, offset_provider=offset_provider, out = out)
    @test out .== result_neighbor_sum

    # Broadcast -------------------------

    @test typeof(broadcast(cell_values, (Cell, K))) == Field{Float64, 1, Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}, Tuple{Dimension{:Cell_, HORIZONTAL}}, Vector{Float64}}
    @test typeof(broadcast(5.0, (Cell, K))) == Field{Float64, 0, Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}, Tuple{}, Array{Float64, 0}}

    # Where -----------------------------------------

    TODO

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

    @test typeof(convert(Int64, cell_values)) == Field{Int64, 1, Tuple{Dimension{:Cell_, HORIZONTAL}}, Tuple{Dimension{:Cell_, HORIZONTAL}}, Vector{Int64}}

    # Concat -----------------------------
    a_data = [1.0 1.0 1.0;
              1.0 NaN 1.0;
              1.0 1.0 1.0]
    a = Field((Cell, K), a_data)
    b_data = [NaN NaN NaN;
              NaN 100.0 NaN;
              NaN NaN NaN]
    b = Field((Cell, K), b_data)

    result = [1.0 1.0 1.0;
              1.0 100.0 1.0;
              1.0 1.0 1.0]

    @test concat(a, b).data .== result
end