using Pkg
path_to_package = joinpath(@__DIR__, "..")  # Assuming the benchmarks.jl file is in the "benchmark" directory
push!(LOAD_PATH, path_to_package)
using BenchmarkTools
using GridTools

SUITE = BenchmarkGroup()

SUITE["arith_broadcast"] = BenchmarkGroup()

a = rand(1000, 1000); b = rand(1000,1000); c = rand(1000,1000)
af = Field((Cell, K), rand(1000, 1000)); bf = Field((Cell, K), rand(1000, 1000)); cf = Field((Cell, K), rand(1000, 1000))
SUITE["arith_broadcast"]["arrays"] = @benchmarkable a .+ b .- c
SUITE["arith_broadcast"]["fields"] = @benchmarkable af .+ bf .- cf

run(SUITE, verbose = true, seconds = 1)
