
include("GridTools.jl")
using .GridTools

using OffsetArrays

struct Cell_ <: Dimension end
struct K_ <: Dimension end
struct Edge_ <: Dimension end
struct Vertex_ <: Dimension end
struct V2VDim_ <: Dimension end
struct V2EDim_ <: Dimension end
struct E2VDim_ <: Dimension end
struct E2CDim_ <: Dimension end
struct C2EDim_ <: Dimension end
Cell = Cell_()
K = K_()
Edge = Edge_()
Vertex = Vertex_()
V2VDim = V2VDim_()
V2EDim = V2EDim_()
E2VDim = E2VDim_()
E2CDim = E2CDim_()
C2EDim = C2EDim_()

V2V = FieldOffset("V2V", source=(Vertex,), target=(Vertex, V2VDim))
E2V = FieldOffset("E2V", source=(Vertex,), target=(Edge, E2VDim))
V2E = FieldOffset("V2E", source=(Edge,), target=(Vertex, V2EDim))
E2C = FieldOffset("E2C", source=(Cell,), target=(Edge, E2CDim))
C2E = FieldOffset("C2E", source=(Edge,), target=(Cell, C2EDim))
Koff = FieldOffset("Koff", source=(K,), target=(K,))


a = Field((Vertex, K), reshape(collect(-3.0:2.0), (3, 2)))
b = Field((K, Edge), reshape(collect(1.0:6.0), (2, 3)))

A = Field((Vertex, K), OffsetArray(reshape(collect(1.:15.), 3, 5), -1:1, 0:4))
B = Field((K, Edge), OffsetArray(reshape(ones(6), 3, 2), 1:3, 1:2))

mask_b = cat([true true false true true ; true false false false true ;true true true true true], [true false true false true ; true false false false true ;true true true true true], dims=3)

mask = Field((Vertex, K, Edge), OffsetArray(mask_b, -1:1, 0:4, 1:2))



# include("AtlasMesh.jl")
# include("Advection.jl")
# include("StateContainer.jl")
# include("Metric.jl")
# include("AdvectionTest.jl")
