
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

