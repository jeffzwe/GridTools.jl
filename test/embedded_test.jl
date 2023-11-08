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

E2C_offset_provider = Connectivity(edge_to_cell_table, Cell, Edge, 2)
C2E_offset_provider = Connectivity(edge_to_cell_table, Edge, Cell, 2)

offset_provider = Dict{String, Connectivity}(
                   "E2C" => E2C_offset_provider,
                   "C2E" => C2E_offset_provider
                )

x = Field((Cell, K), reshape(collect(-3.0:8.0), (6, 2)))
y = Field((K, Edge), reshape(collect(1.0:6.0), (2, 3)))

X = Field((Vertex, K), OffsetArray(reshape(collect(1.:15.), 3, 5), -1:1, 0:4))
Y = Field((K, Edge), OffsetArray(reshape(ones(6), 3, 2), 1:3, 1:2))

mask = cat([true true false true true ; true false false false true ;true true true true true], [true false true false true ; true false false false true ;true true true true true], dims=3)
mask_offset = Field((Vertex, K, Edge), OffsetArray(mask, -1:1, 0:4, 1:2))

# Tests ------------------------------------------------------------------------------------------------------------------------------

@testset "Testset arithmetic broadcast" begin
	
	addition = a .+ b
	@test a_value+b_value == mean(addition.data)
	@test 0 == std(addition.data)

    subtraction = a .- b
    @test a_value-b_value == mean(subtraction.data)
	@test 0 == std(subtraction.data)

    mult = a .* b
    @test a_value*b_value == mean(mult.data)
	@test 0 == std(mult.data)

    div = a ./ b
    @test a_value/b_value == mean(div.data)
	@test 0 == std(div.data)

	@test isa(addition, Field)
    @test isa(subtraction, Field)
    @test isa(mult, Field)
    @test isa(div, Field)
	@test_throws AssertionError (a .+ e)
end

@testset "Testset Field-Offset call" begin
    GridTools.assign_op(offset_provider)   # TODO necessary since "Testset Field-Offset call" is not in a field_operator

    result = [
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

    GridTools.unassign_op()
end


@testset "Testset built_ins" begin
	
    
end