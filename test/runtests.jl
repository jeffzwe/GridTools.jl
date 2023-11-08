# push!(LOAD_PATH, "/Users/jeffreyzweidler/Documents/CSCS/GridTools/src")

using GridTools
using Test
using OffsetArrays

include("../src/gt2py/gt2py.jl")

# Setup --------------------------------------

Cell_ = Dimension{:Cell_, HORIZONTAL}
K_ = Dimension{:K_, HORIZONTAL}
Edge_ = Dimension{:Edge_, HORIZONTAL}
Vertex_ = Dimension{:Vertex_, HORIZONTAL}
V2VDim_ = Dimension{:V2VDim_, LOCAL}
V2EDim_ = Dimension{:V2EDim_, LOCAL} 
E2VDim_ = Dimension{:E2VDim_, LOCAL} 
E2CDim_ = Dimension{:E2CDim_, LOCAL}
C2EDim_ = Dimension{:C2EDim_, LOCAL}
Cell = Cell_()
K = K_()
Edge = Edge_()
Vertex = Vertex_()
V2VDim = V2VDim_()
V2EDim = V2EDim_()
E2VDim = E2VDim_()
E2CDim = E2CDim_()
C2EDim = C2EDim_()

V2V = FieldOffset("V2V", source=Vertex, target=(Vertex, V2VDim))
E2V = FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2C = FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
C2E = FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))
Koff = FieldOffset("Koff", source=K, target=K)

@testset "GridTools tests" begin

    @testset "Embedded Testsuit" begin
        include("embedded_test.jl")
    end

    @testset "GT2Py Testsuit" begin
        include("gt2py_fo_parser.jl")
        include("gt2py_fo_exec.jl")
    end
end
