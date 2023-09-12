module GridTools

using Printf
using Statistics
using BenchmarkTools
using Profile
using Debugger
using Base: @propagate_inbounds
using MacroTools

import Base.Broadcast: Extruded, Style, BroadcastStyle, ArrayStyle ,Broadcasted

export Field, Dimension, Connectivity, FieldOffset, neighbor_sum, max_over, min_over, where, @field_operator, @test #, broadcast


# Lib ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
    abstract type Dimension

Create a new Dimension.

# Examples
```julia-repl
julia> struct Cell_ <: Dimension end
julia> Cell = Cell_()
```
"""
abstract type Dimension end


# Field struct --------------------------------------------------------------------

# TODO: check for #dimension at compile time and not runtime
# TODO: <: AbstractArray{T,N} is not needed... but then we have to define our own length and iterate function for Fields
# TODO: What type should dims and broadcast_dims have?
"""
    Field(dims::Tuple, data::Array, broadcast_dims::Tuple)

Fields store data as a multi-dimensional array, and are defined over a set of named dimensions. 

# Examples
```julia-repl
julia> new_field = Field((Cell, K), fill(1.0, (3,2)))
3x2  Field with dims (Main.GridTools.Cell_(), Main.GridTools.K_()) and broadcasted_dims (Main.GridTools.Cell_(), Main.GridTools.K_()):
 1.0  1.0
 1.0  1.0
 1.0  1.0
```

Fields also have a call operator. You can transform fields (or tuples of fields) over one domain to another domain by using the call operator of the source field with a field offset as argument.
As an example, you can use the field offset E2C below to transform a field over cells to a field over edges using edge-to-cell connectivities. The FieldOffset can take an index as argument which restricts the output dimension of the transform.

The call itself must happen in a field_operator since it needs the functionality of an offset_provider.


# Examples
```julia-repl
julia> E2C = gtx.FieldOffset("E2C", source=CellDim, target=(EdgeDim,E2CDim))
julia> field = Field((Cell,), ones(5))

...
julia> field(E2C())
julia> field(E2C(1))
...
```
"""
struct Field{T, N, T2 <: Tuple{Vararg{<:Dimension}}, T3 <: Tuple{Vararg{<:Dimension}}} <: AbstractArray{T,N}
    dims::T2
    data::Array{T,N}
    broadcast_dims::T3
    
    function Field(dims::T2, data::Array{T,N}, broadcast_dims::T3 = dims) where {T, N, T2 <: Tuple{Vararg{<:Dimension}}, T3 <: Tuple{Vararg{<:Dimension}}}
        if ndims(data) != 0 @assert length(dims) == ndims(data) end
        return new{T,N,T2,T3}(dims, data, broadcast_dims)
    end
end

Base.size(F::Field)::Tuple = size(F.data)
@propagate_inbounds Base.getindex(F::Field{T,N}, inds::Vararg{Int,N}) where {T,N} = F.data[inds...]
@propagate_inbounds Base.setindex!(F::Field{T,N}, val, inds::Vararg{Int,N}) where {T,N} = F.data[inds...] = val
Base.showarg(io::IO, F::Field, toplevel) = print(io, " Field with dims ", F.dims, " and broadcasted_dims ", F.broadcast_dims)

# TODO: Add to documentation of Field
function (field_call::Field)(conn_in::Tuple{Array{Integer}, Tuple{Vararg{<:Dimension}}, Tuple{Vararg{<:Dimension}}})::Field

    conn_data = conn_in[1]
    conn_source = conn_in[2]
    conn_target = conn_in[3]
    
    @assert maximum(conn_data) <= size(field_call)[1] && minimum(conn_data) >= 0

    if ndims(field_call) == 1
        res = map(x -> x == 0. ? 0. : getindex(field_call, Int.(x)), conn_data)
    else
        f(slice) = map(x -> x == 0. ? 0. : getindex(slice, Int.(x)), conn_data)
        res = cat(map(f, eachslice(field_call.data, dims=2))...,dims=ndims(conn_data)+1)
    end

    dims = deleteat!([field_call.dims...], findall(x->x in conn_source, [field_call.dims...]))
    dims = tuple(conn_target..., dims...)

    return Field(dims, res)
end


# Connectivity struct ------------------------------------------------------------

"""
    Connectivity(data::Array, source::Tuple, target::Tuple, dims::Int)

# Examples
```julia-repl
julia> new_connectivity = Connectivity(fill(1, (3,2)), Cell, (Edge, E2C), 2)
3x2  Field with dims (Main.GridTools.Cell_(), Main.GridTools.K_()) and broadcasted_dims (Main.GridTools.Cell_(), Main.GridTools.K_()):
 1  1
 1  1
 1  1
```
"""
struct Connectivity
    data::Array{Integer, 2}
    source::Tuple{Vararg{<:Dimension}}
    target::Tuple{Vararg{<:Dimension}}
    dims::Integer
end

# FieldOffset struct -------------------------------------------------------------

"""
    FieldOffset(name::String, source::Tuple, target::Tuple)

You can transform fields (or tuples of fields) over one domain to another domain by using the call operator of the source field with a field offset as argument. This transform uses the connectivity between the source and target domains to find the values of adjacent mesh elements.

# Examples
```julia-repl
julia> vertex2edge = FieldOffset("V2E", source=(Edge,), target=(Vertex, V2EDim))
FieldOffset("V2E", (Edge_(),), (Vertex_(), V2EDim_()))
```
"""
struct FieldOffset
    name::String
    source::Tuple{Vararg{<:Dimension}}
    target::Tuple{Vararg{<:Dimension}}

    function FieldOffset(name::String; source::Tuple{Vararg{<:Dimension}}, target::Tuple{Vararg{<:Dimension}})::FieldOffset
        new(name, source, target)
    end
end

function (f_off::FieldOffset)(ind::Integer)::Tuple{Array{Integer,1}, Tuple{Vararg{<:Dimension}}, Tuple{Vararg{<:Dimension}}}
    conn = OFFSET_PROVIDER[f_off.name]
    @assert all(x -> x in f_off.source, conn.source) && all(x -> x in f_off.target, conn.target)
    return (conn.data[:,ind], f_off.source, (f_off.target[1],))
end

function (f_off::FieldOffset)()::Tuple{Array{Integer,2}, Tuple{Vararg{<:Dimension}}, Tuple{Vararg{<:Dimension}}}
    conn = OFFSET_PROVIDER[f_off.name]
    @assert all(x -> x in f_off.source, conn.source) && all(x -> x in f_off.target, conn.target)
    return (conn.data, f_off.source, f_off.target)
end

# Constants  -----------------------------------------------------------------------------------
OFFSET_PROVIDER::Dict{String, Connectivity} = Dict{String, Connectivity}()


# Macros ----------------------------------------------------------------------
"""
    @field_operator

The field_operator macro takes a function definition and creates a run environment for the function call within the GridTools package. It enables the additional argument "offset_provider" etc.

# Examples
```julia-repl
julia> @field_operator hello(x) = x + x
hello (generic function with 1 method)
...
```
"""

macro field_operator(expr::Expr)
    
    wrap = :(function wrapper(args...; offset_provider::Union{Dict{String, Connectivity}, Nothing} = nothing, kwargs...)
        @assert isempty(OFFSET_PROVIDER)
        global OFFSET_PROVIDER = offset_provider
        f = $(esc(expr))
        try
            result = f(args...; kwargs...)
        finally
            global OFFSET_PROVIDER = Dict{String, Connectivity}()
        end
        return result
    end)

    return Expr(:(=), esc(namify(expr)), wrap)
end

# Built-ins ----------------------------------------------------------------------

"""
    broadcast(f::Field, b_dims::Tuple)

Sets the broadcast dimension of Field f to b_dims
"""
function broadcast(f::Field, b_dims::D)::Field where D <: Tuple{Vararg{<:Dimension}}
    @assert issubset(f.dims, b_dims)
    return Field(f.dims, f.data, b_dims)
end

function broadcast(n::Number, b_dims::D)::Field where D <: Tuple{Vararg{<:Dimension}}
    return Field((), fill(n), b_dims)
end


"""
    neighbor_sum(f::Field; axis::Dimension)

Sums along the axis dimension. Outputs a field with dimensions size(f.dims)-1.
"""
function neighbor_sum(field_in::Field; axis::Dimension)::Field
    dim = findall(x -> x == axis, field_in.dims)[1]
    return Field((field_in.dims[1:dim-1]..., field_in.dims[dim+1:end]...), dropdims(sum(field_in.data, dims=dim), dims=dim)) 
end
"""
    max_over(f::Field; axis::Dimension)

Gives the maximum along the axis dimension. Outputs a field with dimensions size(f.dims)-1.
"""
function max_over(field_in::Field; axis::Dimension)::Field
    dim = findall(x -> x == axis, field_in.dims)[1]
    return Field((field_in.dims[1:dim-1]..., field_in.dims[dim+1:end]...), dropdims(maximum(field_in.data, dims=dim), dims=dim)) 
end
"""
    min_over(f::Field; axis::Dimension)

Gives the minimum along the axis dimension. Outputs a field with dimensions size(f.dims)-1.
"""
function min_over(field_in::Field; axis::Dimension)::Field
    dim = findall(x -> x == axis, field_in.dims)[1]
    return Field((field_in.dims[1:dim-1]..., field_in.dims[dim+1:end]...), dropdims(minimum(field_in.data, dims=dim), dims=dim)) 
end


@inbounds where(mask::Field, a::Union{Field, Real}, b::Union{Field, Real})::Field = ifelse.(mask, a, b)

"""
    where(mask::Field, true, false)

The 'where' loops over each entry of the mask and returns values corresponding to the same indexes of either the true or the false branch.

# Arguments
- `mask::Field`: a field with eltype Boolean
- `true`: a tuple, a field, or a scalar
- `false`: a tuple, a field, or a scalar

# Examples
```julia-repl
julia> mask = Field((Cell, K), rand(Bool, (3,3)))
3x3  Field with dims (Cell_(), K_()) and broadcasted_dims (Cell_(), K_()):
 1  0  0
 0  1  0
 1  1  1
julia> a = Field((Cell, K), fill(1.0, (3,3)));
julia> b = Field((Cell, K), fill(2.0, (3,3)));
julia> where(mask, a, b)
3x3  Field with dims (Cell_(), K_()) and broadcasted_dims (Cell_(), K_()):
 1.0  2.0  2.0
 2.0  1.0  2.0
 1.0  1.0  1.0
```

The `where` function builtin also allows for nesting of tuples. In this scenario, it will first perform an unrolling:
`where(mask, ((a, b), (b, a)), ((c, d), (d, c)))` -->  `where(mask, (a, b), (c, d))` and `where(mask, (b, a), (d, c))` and then combine results to match the return type:
"""
where(mask::Field, t1::Tuple, t2::Tuple)::Field = map(x -> where(mask, x[1], x[2]), zip(t1, t2))



# Includes ------------------------------------------------------------------------------------

include("CustBroadcast.jl")

end

