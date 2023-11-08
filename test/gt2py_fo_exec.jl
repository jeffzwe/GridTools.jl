
struct TestFailedException <: Exception
    message::String
end

macro to_py(expr::Expr)
    try
        eval(expr) 
        return true
    catch e
        throw(TestFailedException("The following test: $(namify(expr)) encountered following error: $e"))
    end
end

# Setup ------------------------------------------------------------------------------------------

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


# Tests ------------------------------------------------

a = Field(Cell, collect(1.:15.))
b = Field(Cell, collect(-1.:-1:-15.))
out = Field(Cell, zeros(Float64, 15))

@field_operator function fo_addition(a::Field{Float64, 1, Tuple{Cell_}}, b::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Cell_}}
    return a .+ b
end

@test @to_py fo_addition(a, b, backend = "py", out = out)

# ------------------------------------------------

a = Field(Cell, collect(1:15))
out = Field(Cell, zeros(Int64, 15))

@field_operator function fo_nested_if_else(f::Field{Int64, 1, Tuple{Cell_}})::Field{Int64, 1, Tuple{Cell_}}
    tmp = f
    if 1. .< 10.0
        tmp = f .+ 1
        if 30 > 5
            tmp = tmp .+ 20
            tmp = tmp .- 10
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
end

@test @to_py fo_nested_if_else(a, backend = "py", out = out)

# ------------------------------------------------
a = Field(Cell, collect(1.:15.))
out = Field(Edge, zeros(Float64, 12))

@field_operator function fo_remapping(a::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Edge_}}
    return a(E2C[1])
end

@test @to_py fo_remapping(a, offset_provider=offset_provider, backend = "py", out = out)

# ------------------------------------------------
a = Field(Cell, collect(1.:15.))
out = Field(Edge, zeros(Float64, 12))

@field_operator function fo_neighbor_sum(a::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Edge_}}
    return neighbor_sum(a(E2C), axis=E2CDim)
end

@test @to_py fo_neighbor_sum(a, offset_provider=offset_provider, backend = "py", out = out)

# ------------------------------------------------
a = Field(Cell, collect(1.:15.))
out = Field(Edge, zeros(Float64, 12))

@field_operator function fo_max_over(a::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Edge_}}
    return max_over(a(E2C), axis=E2CDim)
end

@test @to_py fo_max_over(a, offset_provider=offset_provider, backend = "py", out = out)

# ------------------------------------------------

a = Field(Cell, collect(1.:15.))
out = Field(Edge, zeros(Float64, 12))

@field_operator function fo_min_over(a::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Edge_}}
    return min_over(a(E2C), axis=E2CDim)
end

@test @to_py fo_min_over(a, offset_provider=offset_provider, backend = "py", out = out)

# ------------------------------------------------

a = Field(Cell, collect(1.:15.))
out = Field((Cell, K), zeros(15, 5))

@field_operator function fo_simple_broadcast(a::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Cell_, K_}}
        return broadcast(a, (Cell, K))
end
    
@test @to_py fo_simple_broadcast(a, backend = "py", out = out)

# ------------------------------------------------

j_out = Field((), fill(0.), (Cell, K))
py_out = Field((Cell, K), fill(0., (10, 10)))


@field_operator function fo_scalar_broadcast()::Field{Float64, 0, Tuple{Cell_, K_}}
    return broadcast(5., (Cell, K))
end

@test @to_py fo_scalar_broadcast(backend = "py", out = py_out)

# ------------------------------------------------

a = Field((Cell, K), reshape(collect(1.:12.), (6, 2)))
b = Field((Cell, K), fill(-1., (6, 2)))
mask = Field((Cell, K), rand(Bool, (6, 2)))
out = Field((Cell, K), zeros(6, 2))

@field_operator function fo_where(mask::Field{Bool, 2, Tuple{Cell_, K_}}, a::Field{Float64, 2, Tuple{Cell_, K_}}, b::Field{Float64, 2, Tuple{Cell_, K_}})::Field{Float64, 2, Tuple{Cell_, K_}}
        return where(mask, a, b)
end

@test @to_py fo_where(mask, a, b, backend = "py", out = out)

# -------------------------------------------------

a = Field((Cell, K), reshape(collect(1.:12.), (6, 2)))
out = Field((Cell, K), zeros(Int64, (6, 2)))

@field_operator function fo_astype(a::Field{Float64, 2, Tuple{Cell_, K_}})::Field{Int64, 2, Tuple{Cell_, K_}}
        return convert(Int64, a)
end

@test @to_py fo_astype(a, backend = "py", out = out)

# -------------------------------------------------

a = Field((Cell, K), reshape(collect(1.:12.), (6, 2)))
out = Field((Cell, K), zeros((6, 2)))

@field_operator function fo_sin(a::Field{Float64, 2, Tuple{Cell_, K_}})::Field{Float64, 2, Tuple{Cell_, K_}}
        return sin.(a)
end

@test @to_py fo_sin(a, backend = "py", out = out)

# -------------------------------------------------

a = Field((Cell, K), reshape(collect(1.:12.), (6, 2)))
out = Field((Cell, K), zeros((6, 2)))

@field_operator function fo_asinh(a::Field{Float64, 2, Tuple{Cell_, K_}})::Field{Float64, 2, Tuple{Cell_, K_}}
        return asinh.(a)
end

@test @to_py fo_asinh(a, backend = "py", out = out)

# -------------------------------------------------

# TODO OffsetArray is ignored for the moment

A = Field((Vertex, K), OffsetArray(reshape(collect(1.:15.), 3, 5), -1:1, 0:4))
a = Field((Vertex, K), reshape(collect(1.:15.), 3, 5))
arr_out = Field((Vertex, K), zeros((3, 5)))
off_out = Field((Vertex, K), zeros((3, 5)))

@field_operator function fo_offset_array(a::Field{Float64, 2, Tuple{Vertex_, K_}})::Field{Float64, 2, Tuple{Vertex_, K_}}
        return a .+ 10. ./ 2.
    end

fo_offset_array(a, backend = "py", out = arr_out)
fo_offset_array(A, backend = "py", out = off_out)

@test arr_out == off_out

# -------------------------------------------------

a = Field(Cell, collect(1.:15.))
b = Field(Cell, ones(15))
out = Field(Cell, zeros(15))

@field_operator function nested_fo(a::Field{Float64, 1, Tuple{Cell_}}, b::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Cell_}}
    res = fo_addition(a, b)
    return res .+ a
end