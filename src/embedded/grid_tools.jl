module GridTools

using Printf
using Statistics
using BenchmarkTools
using Profile
using Base: @propagate_inbounds
using MacroTools
using OffsetArrays
using OffsetArrays: IdOffsetRange
using Debugger

import Base.Broadcast: Extruded, Style, BroadcastStyle, ArrayStyle ,Broadcasted

export Dimension, DimensionKind, HORIZONTAL, VERTICAL, LOCAL, Field, FieldShape, Connectivity, FieldOffset, shape, neighbor_sum, max_over, min_over, where, @field_operator, get_dim_name, get_dim_kind


# Lib ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dimension --------------------------------------------------------------------

abstract type DimensionKind end

struct HORIZONTAL <: DimensionKind end 
struct VERTICAL <: DimensionKind end 
struct LOCAL <: DimensionKind end

"""
    abstract type Dimension

Create a new Dimension.

# Examples
```julia-repl
julia> struct Cell_ <: Dimension end
julia> Cell = Cell_()
```
"""
struct Dimension{name, kind <: DimensionKind} end

Base.length(d::Dimension) = 1
Base.iterate(d::Dimension, state=1) = state==1 ? (d, state+1) : nothing
Base.getindex(d::Dimension, offset::Integer) = d, offset
get_dim_name(d::Dimension) = typeof(d).parameters[1]
get_dim_kind(d::Dimension) = typeof(d).parameters[2]

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

    function FieldOffset(name::String; source::Union{Dimension, Tuple{Vararg{<:Dimension}}}, target::Union{Dimension, Tuple{Vararg{<:Dimension}}})::FieldOffset
        if length(target) == 2 @assert typeof(target[2]).parameters[2] == LOCAL ("Second dimension in offset must be a local dimension.") end
        new(name, Tuple(source), Tuple(target))
    end
end

Base.getindex(f_off::FieldOffset, ind::Integer) = f_off, ind


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
struct Field{T <: Union{AbstractFloat, Integer, Bool}, N, BD <: Tuple{Vararg{<:Dimension}}, D <: Tuple{Vararg{<:Dimension}}} <: AbstractArray{T,N}
    dims::D
    data::AbstractArray{T,N}
    broadcast_dims::BD
    
    function Field(dims::D, data::AbstractArray{T,N}, broadcast_dims::BD = dims) where {T <: Union{AbstractFloat, Integer, Bool}, N, BD <: Tuple{Vararg{<:Dimension}}, D <: Tuple{Vararg{<:Dimension}}}
        if ndims(data) != 0 @assert length(dims) == ndims(data) end
        return new{T,N,BD,D}(dims, data, broadcast_dims)
    end

    function Field(dim::Dimension, data::AbstractArray{T,N}, broadcast_dims::Union{Dimension,BD} = dim) where {T <: Union{AbstractFloat, Integer, Bool}, N, BD <: Tuple{Vararg{<:Dimension}}}
        if ndims(data) != 0 @assert ndims(data) == 1 end
        return Field(Tuple(dim), data, Tuple(broadcast_dims))
    end
end

# Call functions to Field struct #TODO Add to documentation of Field
(field_call::Field)(t::Tuple{FieldOffset, <:Integer})::Field = field_call(t...)
function (field_call::Field)(f_off::FieldOffset, ind::Union{<:Integer, Nothing} = nothing)::Field

    conn = OFFSET_PROVIDER[f_off.name]

    conn_data = isnothing(ind) ? conn.data : conn.data[:,ind]
    f_target = isnothing(ind) ? f_off.target : f_off.target[1]

    @assert maximum(conn_data) <= size(field_call)[1] && minimum(conn_data) >= 0
    @assert all(x -> x in f_off.source, conn.source) && all(x -> x in f_off.target, conn.target)

    if ndims(field_call) == 1
        res = map(x -> x == 0 ? convert(eltype(field_call), 0) : getindex(field_call, Int.(x)), conn_data)
    else
        f(slice) = map(x -> x == 0 ? convert(eltype(field_call), 0) : getindex(slice, Int.(x)), conn_data)
        res = cat(map(f, eachslice(field_call.data, dims=2))...,dims=ndims(conn_data)+1)
    end

    dims = deleteat!([field_call.dims...], findall(x->x in f_off.source, [field_call.dims...]))
    dims = tuple(f_target..., dims...)

    return Field(dims, res)
end

function (field_call::Field)(t::Tuple{<:Dimension, <:Integer})::Field
    dim_ind = findall(x -> x == t[1], field_call.dims)[1]
    new_size = zeros(Integer, length(axes(field_call.data)))
    new_size[dim_ind] = t[2]
    return Field(field_call.dims, OffsetArray(field_call.data, new_size...), field_call.broadcast_dims)
end

# Field struct interfaces
Base.size(F::Field)::Tuple = size(F.data)
Base.axes(F::Field)::Tuple = axes(F.data)
@propagate_inbounds Base.getindex(F::Field{T,N}, inds::Vararg{Int,N}) where {T,N} = F.data[inds...]
@propagate_inbounds Base.setindex!(F::Field{T,N}, val, inds::Vararg{Int,N}) where {T,N} = F.data[inds...] = val
Base.showarg(io::IO, F::Field, toplevel) = print(io, " Field with buffer_dims ", typeof(F.dims), " and broadcasted_dims ", typeof(F.broadcast_dims))
function Base.promote(f1::Field, f2::Field)
    f1_new_data, f2_new_data = promote(f1.data, f2.data)
    return Field(f1.dims, f1_new_data, f1.broadcast_dims),Field(f2.dims, f2_new_data, f2.broadcast_dims)
end

# used for custom Broadcast #TODO Maybe move to CustBroadcast.jl
struct FieldShape
    dims::Tuple{Vararg{<:Dimension}}
    axes::Tuple{Vararg{<:AbstractUnitRange{Int64}}}
    broadcast_dims::Tuple{Vararg{<:Dimension}}
end

function shape(f::Field)
    return FieldShape(f.dims, axes(f), f.broadcast_dims)
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
    source::Dimension
    target::Dimension
    dims::Integer
end

# OFFSET_PROVIDER  -----------------------------------------------------------------------------------
OFFSET_PROVIDER::Dict{String, Connectivity} = Dict{String, Connectivity}()

# assign_op(dict::Nothing) = nothing
function assign_op(dict::Dict{String, Connectivity})
    global OFFSET_PROVIDER = dict
end

function unassign_op()
    global OFFSET_PROVIDER = Dict{String, Connectivity}()
end

# Field operator ----------------------------------------------------------------------

py_field_ops = Dict()

"""
    @field_operator

The field_operator macro takes a function definition and creates a run environment for the function call within the GridTools package. It enables the additional argument "offset_provider", "backend", etc.

# Examples
```julia-repl
julia> @field_operator addition(x) = x + x
addition (generic function with 1 method)
...
```
"""
macro field_operator(expr::Expr)
    
    wrap = :(function wrapper(args...; offset_provider::Dict{String, Connectivity} = Dict{String, Connectivity}(), backend::String = "embedded", out = nothing, kwargs...)
        @assert isempty(GridTools.OFFSET_PROVIDER)

        result = nothing

        @bp

        if backend == "embedded"
            GridTools.assign_op(offset_provider)
            try
                f = $(esc(expr))
                result = f(args...; kwargs...)
            finally
                GridTools.unassign_op()
            end
        elseif backend == "py"
            f = py_field_operator($(Expr(:quote, expr)), @__MODULE__)
            p_args = py_args(args)
            p_kwargs = py_args(kwargs)
            p_out = py_args(out)
            p_offset_provider = py_args(offset_provider)
            f(p_args..., out = p_out, offset_provider = p_offset_provider; p_kwargs...)
            result = p_out
        else 
            throw("The backend option you provided is not available")
        end
        return result
    end)

    # field_ops[namify(expr)] = FieldOperator object # TODO

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
@inbounds where(mask::Field, a::Union{Field, Real}, b::Union{Field, Real})::Field = ifelse.(mask, promote(a, b)...)


# Includes ------------------------------------------------------------------------------------

include("cust_broadcast.jl")
include("../gt2py/gt2py.jl")

end

