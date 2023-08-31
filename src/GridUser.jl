
include("GridTools.jl")
using .GridTools

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

include("AtlasMesh.jl")
include("Advection.jl")
include("StateContainer.jl")
include("Metric.jl")
include("AdvectionTest.jl")
