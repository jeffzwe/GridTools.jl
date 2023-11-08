
struct TestFailedException <: Exception
    message::String
end

function to_py(expr::Expr)
    try
        py_field_operator(expr, @__MODULE__)  
        return true
    catch e
        throw(TestFailedException("The following test: $(namify(expr)) encountered the following error:"))
    end
end


@testset "Testset Arithmetic Operators" begin
    
    expr = :(function test_addition(f::Field{Int64, 1, Tuple{Cell_}}, g::Field{Int64, 1, Tuple{Cell_}})
                return f + g
            end)
    @test to_py(expr)

    expr = :(function test_b_addition(f::Field{Int64, 1, Tuple{Cell_}}, g::Field{Int64, 1, Tuple{Cell_}})
              return f .+ g
            end)
    @test to_py(expr)
    
    expr = :(function test_bit_and(f::Field{Bool, 1, Tuple{Cell_}}, g::Field{Bool, 1, Tuple{Cell_}})
                return f .& g
            end)
    @test to_py(expr)

    expr = :(function test_bit_xor(f::Field{Bool, 1, Tuple{Cell_}}, g::Field{Bool, 1, Tuple{Cell_}})
                return f .⊻ g
            end)
    @test to_py(expr)

    expr = :(function test_not(f::Field{Bool, 1, Tuple{Cell_}}, g::Field{Bool, 1, Tuple{Cell_}})
                return .~f
            end)
    @test to_py(expr)
    
    expr = :(function test_bool_and(f::Field{Bool, 1, Tuple{Cell_}}, g::Field{Bool, 1, Tuple{Cell_}})
                return f && g
            end)
    @test to_py(expr)
    
    expr = :(function test_bool_or(f::Field{Bool, 1, Tuple{Cell_}}, g::Field{Bool, 1, Tuple{Cell_}})
                return f || g
            end)
    @test to_py(expr)
end

@testset "Testset function header annotation" begin

    expr = :(function test_no_annotation(inp)
                return inp
            end)
    @test_throws TestFailedException to_py(expr)

    # Type annotation error
    expr = :(function test_no_annotation_kwargs(f::Field{Float64, 1, Tuple{Cell_}}; g)
                return g
            end)
    @test_throws TestFailedException to_py(expr)

    expr = :(function test_simple_annotation(inp::Integer)
                return inp
            end)
    @test to_py(expr)

    expr = :(function test_kwargs(inp::Integer ; inp2::AbstractFloat)
                return inp
            end)
    @test to_py(expr)

    expr = :(function test_return_annotation(f::Field{Float64, 1, Tuple{Cell_}}, g::Field{Int64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Cell_}}
                return f
            end)
    @test to_py(expr)

    # Type deduction error
    expr = :(function test_incompatible_types(f::Field{Float64, 1, Tuple{Cell_}}, g::Field{Int64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Cell_}}
                return f .+ g
            end)
    @test_throws TestFailedException to_py(expr)
    
    # Return statement error
    expr = :(function test_no_return_stmt(f::Field{Float64, 1, Tuple{Cell_}}; g::Field{Int64, 1, Tuple{Cell_}})
                g .+ 1
            end)
    @test_throws TestFailedException to_py(expr)
end


@testset "Testset Preprocessing" begin
    expr = :(function test_multi_assign(f::Field{Float64, 1, Tuple{Cell_}})
                a, b = 5., 6.
                return f .+ a .- b
            end)
    @test to_py(expr)

    expr = :(function test_renaming_variables(f::Field{Float64, 1, Tuple{Cell_}})
                tmp = 1
                tmp = 2 + tmp
                tmp = 3 + tmp
                return tmp
            end)
    @test to_py(expr)

    expr = :(function test_ternary_expr(f::Field{Float64, 1, Tuple{Cell_}}, g::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Cell_}}
                return 1 < 2 ? f :  g
            end)
    @test to_py(expr)

    expr = :(function test_renaming_conditional(f::Field{Int64, 1, Tuple{Cell_}})
                if 1. .< 10.
                    tmp = f .+ 1
                elseif 1. < 10.
                    tmp = f .- 1
                else
                    tmp = f .+ 1
                end
                return tmp
            end)
    @test to_py(expr)

    expr = :(function test_renaming_nested_conditional(f::Field{Int64, 1, Tuple{Cell_}}, g::Field{Int64, 1, Tuple{Cell_}})::Field{Int64, 1, Tuple{Cell_}}
                tmp = f
                if 1. .< 10.
                    tmp = f .+ 1
                    if 1. .< 10.
                        tmp = tmp .- 1
                        tmp = tmp .* 1
                    elseif 1. .< 10.
                        tmp = 1. .< 10. ? tmp : tmp ./ 1
                    else 
                        tmp = tmp .* 1
                    end
                    tmp = tmp .+ 1
                elseif 1. .< 10.
                    tmp = f .- 1
                else
                    tmp = tmp .* 1
                    tmp = tmp .+ 1
                    tmp = tmp ./ 1
                end
                return tmp
            end)
    @test to_py(expr)

    expr = :(function test_compair_chain(f::Field{Int64, 1, Tuple{Cell_}}, g::Field{Int64, 1, Tuple{Cell_}})::Field{Bool, 1, Tuple{Cell_}}
                return 1 < 10 .< f .< g .< 50 > -1
            end)
    @test to_py(expr)

    # Only constants in condition, Should this actually fail?  TODO
    expr = :(function test_if_with_field(f::Field{Float64, 1, Tuple{Cell_}}, g::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Cell_}}
                return f .< 2 ? f :  g
            end)
    @test_throws TestFailedException to_py(expr)
end


@testset "Test built-ins and closure variables" begin

    expr = :(function test_where(mask::Field{Bool, 1, Tuple{Cell_}}, f::Field{Float64, 1, Tuple{Cell_}}, g::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Cell_}}
                return where(mask, f, g)
            end)
    @test to_py(expr)

    # Error due to non boolean mask argument
    expr = :(function test_where(mask::Field{Bool, 1, Tuple{Cell_}}, f::Field{Float64, 1, Tuple{Cell_}}, g::Field{Float64, 1, Tuple{Cell_}})::Field{Float64, 1, Tuple{Cell_}}
                return where(f, f, g)
            end)
    @test_throws TestFailedException to_py(expr)


    expr = :(function test_maxover(f::Field{Float64, 2, Tuple{Cell_, K_}})::Field{Float64, 1, Tuple{Edge_}}
                return max_over(f(E2C[1]), axis=K)
            end)
    @test to_py(expr)

    expr = :(function test_neighborsum(f::Field{Float64, 2, Tuple{Cell_, K_}})::Field{Float64, 1, Tuple{Edge_, K_}}
                return neighbor_sum(f(E2C), axis=E2CDim)
            end)
    @test to_py(expr)

    expr = :(function test_broadcast(f::Field{Float64, 1, Tuple{Cell_}})
                return broadcast(f, (Cell, K))
            end)
    @test to_py(expr)

    expr = :(function test_julia_builtin(f::Field{Float64, 2, Tuple{Cell_, K_}})::Field{Float64, 2, Tuple{Cell_, K_}}
                return sin.(f)
            end)
    @test to_py(expr)

end

