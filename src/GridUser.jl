
include("GridTools.jl")
using .GridTools



# @field_operator function hello(x::Integer,y::Integer)
#     a = V2V
#     b = E2V
#     return x + y + a
# end

# offset_provider = Dict{String, Integer}(
#                    "V2V" => 4,
#                    "V2E" => 3,
#                    "E2V" => 2,
#                    "Koff" => 1  # TODO(tehrengruber): using K here gives a terrible compilation error. Improve in GT4Py!
#                 )

# GridTools.hello(1, 2, offset_provider = offset_provider)




# Use this as soon as FieldOffset is implemented
# V2V = 
# E2V = 
# V2E = 
# K = 
# Koff = 

# V2V = FieldOffset("V2V", source=Vertex, target=(Vertex, V2VDim))
# E2V = FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
# V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
# Koff = FieldOffset("Koff", source=K, target=(K,))

