using BenchmarkTools
using Profile


# Benchmark and Profiling for simple array/field arithmetic ########################################################################################################

# add3(a, b, c) = a.+b.+c
# add3(rand(10,10), rand(10,10), rand(10,10))
# add3(Field((Cell, K), rand(10,10)),Field((Cell, K), rand(10,10)),Field((Cell, K), rand(10,10)))

# function prof()
#     for i in [1000, 10000, 100000, 1000000]
#         k =100
#         Profile.clear()
#         for _ in 0:1000
#             a=rand(i,k)
#             b=rand(i,k)
#             c=rand(i,k)
#             @profile add3(a, b, c)
#         end
#         print("Plain arrays: ")
#         Profile.print()

#         Profile.clear()
#         for j in 0:1000
#             a=Field((Cell, K), rand(i,k))
#             b=Field((Cell, K), rand(i,k))
#             c=Field((Cell, K), rand(i,k))
#             @profile add3(a, b, c)
#         end
#         print("Fields: ")
#         Profile.print()
        
#         break
#     end
# end

# function bench()
#     for i in [1000, 10000, 100000, 1000000]
#         k =100
#         println("Time for Fields for i = $i")
#         @btime (unos .+ dos .- tres) setup=(
#             unos=Field((Cell, K), rand($i,$k)); 
#             dos=Field((Cell, K), rand($i,$k)); 
#             tres=Field((Cell, K), rand($i,$k)))

#         println("Time for arrays for i = $i")
#         @btime (a .+ b .- c) setup=(
#             a=rand($i,$k); 
#             b=rand($i,$k); 
#             c=rand($i,$k))

#         # @btime (a + b - c) setup=(
#         # a=rand($i,$k); 
#         # b=rand($i,$k); 
#         # c=rand($i,$k))
#     end
# end

# Benchmark for Julia and Python implementations of advection ##############################################################################################################

include("../src/atlas/advection_test.jl")

println("Starting julia embedded benchmark")

bench_julia_embedded = @benchmark upwind_scheme(
        state.rho,
        δt,
        mesh.vol,
        metric.gac,
        state.vel[1],
        state.vel[2],
        state.vel[3],
        mesh.pole_edge_mask,
        mesh.dual_face_orientation,
        mesh.dual_face_normal_weighted_x,
        mesh.dual_face_normal_weighted_y,
        out = state_next.rho,
        offset_provider = mesh.offset_provider
    )

println("Finished Julia embedded benchmark")

include("../src/atlas/advection_test.jl")

println("Starting julia python benchmark")

bench_julia_python = @benchmark upwind_scheme(
        state.rho,
        δt,
        mesh.vol,
        metric.gac,
        state.vel[1],
        state.vel[2],
        state.vel[3],
        mesh.pole_edge_mask,
        mesh.dual_face_orientation,
        mesh.dual_face_normal_weighted_x,
        mesh.dual_face_normal_weighted_y,
        out = state_next.rho,
        offset_provider = mesh.offset_provider,
        backend = "py"
    )

println("Finished Julia python backend benchmark")

@pyinclude("/Users/jeffreyzweidler/Documents/CSCS/fvm_advection_sphere/tests/advection.py")   # TODO: Change to correct path on daint!

bench_python_embedded = @benchmark py"""

upwind_scheme(
    state.rho,
    δt,
    mesh.vol,
    metric.gac,
    state.vel[0],
    state.vel[1],
    state.vel[2],
    mesh.pole_edge_mask,
    mesh.dual_face_orientation,
    mesh.dual_face_normal_weighted_x,
    mesh.dual_face_normal_weighted_y,
    out=state_next.rho,
    offset_provider=mesh.offset_provider,
)
"""

println("Benchmark for julia embedded version: $bench_julia_embedded")
println("Benchmark for julia with python backend: $bench_julia_python")
println("Benchmark for python embedded version: $bench_python_embedded")