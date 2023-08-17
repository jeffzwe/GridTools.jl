
include("GridTools.jl")
using .GridTools

struct Cell_ <: Dimension end
struct K_ <: Dimension end
struct Edge_ <: Dimension end
struct E2C_ <: Dimension end
struct Vertex_ <: Dimension end
struct V2EDim_ <: Dimension end
struct E2VDim_ <: Dimension end
Cell = Cell_()
K = K_()
Edge = Edge_()
E2C = E2C_()
Vertex = Vertex_()
V2EDim = V2EDim_()
E2VDim = E2VDim_()

# V2V = 
# E2V = 
# V2E = 
# K = 
# Koff = 

# V2V = FieldOffset("V2V", source=Vertex, target=(Vertex, V2VDim))
# E2V = FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
# V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
# Koff = FieldOffset("Koff", source=K, target=(K,))

