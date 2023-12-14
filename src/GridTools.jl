
# __precompile__(false)
module GridTools

using Printf
using Statistics
using BenchmarkTools
using Profile
using Base: @propagate_inbounds
using MacroTools
using OffsetArrays: IdOffsetRange
using Debugger

import Base.Broadcast: Extruded, Style, BroadcastStyle, ArrayStyle ,Broadcasted

export Dimension, DimensionKind, HORIZONTAL, VERTICAL, LOCAL, Field, Connectivity, FieldOffset, neighbor_sum, max_over, min_over, where, concat, @field_operator, slice, copyfield!, get_dim_name, get_dim_kind, get_dim_ind

const SKIP_NEIGHBOR_INDICATOR = -1 # TODO(tehrengruber): move from atlas submodule here


# Lib ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dimension --------------------------------------------------------------------

abstract type DimensionKind end

struct HORIZONTAL <: DimensionKind end 
struct VERTICAL <: DimensionKind end 
struct LOCAL <: DimensionKind end

"""
    struct Dimension


Create a new Dimension. Per default a dimension is horizontal.

# Examples
```julia-repl
julia> Cell_ = Dimension(:Cell_)
julia> Cell = Cell_()
```

In order to create a local dimension one needs to adhere to a particular naming scheme as shown below:

# Examples
```julia-repl
julia> V2EDim_ = Dimension(:V2E_, LOCAL)
julia> V2EDim = V2EDim_()
"""
struct Dimension{name, kind <: DimensionKind} end

function Dimension(sym::Symbol, kind = HORIZONTAL)
    return Dimension{sym, kind}
end

Base.length(d::Dimension)::Int64 = 1
Base.iterate(d::Dimension, state=1) = state==1 ? (d, state+1) : nothing
get_dim_name(d::Dimension) = string(typeof(d).parameters[1])[1:end-1]
get_dim_kind(d::Dimension) = typeof(d).parameters[2]
get_dim_ind(source::Tuple{Vararg{Dimension}}, ind::Dimension) = findfirst(x -> x == ind, source)

# FieldOffset struct -------------------------------------------------------------

"""
    FieldOffset(name::String, source::Tuple, target::Tuple)

You can transform fields (or tuples of fields) over one domain to another domain by using the call operator of the source field with a field offset as argument. This transform uses the connectivity between the source and target domains to find the values of adjacent mesh elements.

# Examples
```julia-repl
julia> V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
FieldOffset("V2E", Dimension{:Edge_, HORIZONTAL}(), (Dimension{:Vertex_, HORIZONTAL}(), Dimension{:V2E_, LOCAL}()))
```
"""
struct FieldOffset
    name::String
    source::Dimension
    target::Tuple{Vararg{Dimension}}

    function FieldOffset(name::String; source::Dimension, target::Union{Dimension, Tuple{Vararg{Dimension}}})::FieldOffset
        if length(target) > 1 @assert all(get_dim_kind.(Base.tail(target)) .== LOCAL) ("All but the first dimension in an offset must be local dimensions.") end
        return new(name, source, Tuple(target))
    end
end

Base.getindex(f_off::FieldOffset, ind::Integer) = f_off, ind


# Field struct --------------------------------------------------------------------

"""
    Field(dims::Tuple, data::Array, broadcast_dims::Tuple; origin::Dic{Dimension, Int64})

Fields store data as a multi-dimensional array, and are defined over a set of named dimensions. 

# Examples
```julia-repl
julia> new_field = Field((Cell, K), fill(1.0, (3,2)))
3×2 Float64 Field with dimensions ("Cell", "K") with indices 1:3×1:2:
 1.0  1.0
 1.0  1.0
 1.0  1.0
```

Each dimension can be offset by a certain index. This changes the index-range of the Field.

# Examples
```julia-repl
julia> new_field = Field((Cell, K), fill(1.0, (3,2)), origin = Dict(K => 1))
3×2 Float64 Field with dimensions ("Cell", "K") with indices 1:3×2:3:
 1.0  1.0
 1.0  1.0
 1.0  1.0
```

Fields also have a call operator. You can transform fields (or tuples of fields) over one domain to another domain by using the call operator of the source field with a field offset as argument.
As an example, you can use the field offset E2C below to transform a field over cells to a field over edges using edge-to-cell connectivities. The FieldOffset can take an index as argument which restricts the output dimension of the transform.

The call itself must happen in a field_operator since it needs the functionality of an offset_provider.


# Examples
```julia-repl
julia> E2C = FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
julia> field = Field(Cell, ones(5))

...
julia> field(E2C)
julia> field(E2C[1])
...
```
"""
# TODO: check for #dimension at compile time and not runtime
# TODO: <: AbstractArray{T,N} is not needed... but then we have to define our own length and iterate function for Fields
# TODO sequence of types: Do BD, T, N. 
# Error showing value of type Field{Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}, Float64, 2, Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}}:
# ERROR: CanonicalIndexError: getindex not defined for Field{Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}, Float64, 2, Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}}

struct Field{B_Dim <: Tuple{Vararg{Dimension}}, T <: Union{AbstractFloat, Integer}, N, Dim <: NTuple{N, Dimension}, D <: AbstractArray{T,N}} <: AbstractArray{T,N}
    dims::Dim
    data::D
    broadcast_dims::B_Dim
    origin::NTuple{N, Int64}
end

function Field(dims::Dim, data::D, broadcast_dims::B_Dim = dims; origin::Dict{<:Dimension, Int64} = Dict{Dimension, Int64}()) where {T <: Union{AbstractFloat, Integer}, N, B_Dim <: Tuple{Vararg{Dimension}}, Dim <: NTuple{N, Dimension}, D <: AbstractArray{T,N}}
    if ndims(data) != 0 @assert length(dims) == ndims(data) end
    offsets = Tuple([get(origin, dim, 0) for dim in dims])
    return Field(dims, data, broadcast_dims, offsets)
end

function Field(dim::Dimension, data::D, broadcast_dims::Union{Dimension,B_Dim} = dim; origin::Dict{<:Dimension, Int64} = Dict{Dimension, Int64}()) where {T <: Union{AbstractFloat, Integer}, N, B_Dim <: NTuple{N, Dimension}, D <: AbstractArray{T,N}}
    if ndims(data) != 0 @assert ndims(data) == 1 end
    return Field(Tuple(dim), data, Tuple(broadcast_dims), origin = origin)
end

struct FieldOffsetTS{Name, Source <: Dimension, Target <: Union{Tuple{<:Dimension, <:Dimension}, Tuple{<:Dimension}}}
end

to_type_stable_field_offset(offset::FieldOffset) = FieldOffsetTS{Symbol(offset.name), typeof(offset.source), Tuple{map(typeof,offset.target)...}}()

struct AllNeighbors
end

function remap_position(
        current_position::Tuple{Int64, Vararg{Int64}},
        dims::Tuple{Dimension, Vararg{Dimension}},
        offset::FieldOffsetTS{OffsetName, SourceDim, Tuple{TargetDim, TargetLocalDim}},
        nb_ind::AllNeighbors,
        conn) where {OffsetName, SourceDim <: Dimension, TargetDim <:Dimension, TargetLocalDim <: Dimension}  # TODO: restrict conn to type ::Connectivity
    if dims[1] == TargetDim()  # since we are mapping indices not field here the target source are flipped
        ind, actual_nb_ind, tail_position... = current_position
        _, local_dim, tail_dims... = dims
        #@assert local_dim isa Dimension && string(typeof(local_dim).parameters[1]) == (string(OffsetName)*"_") && typeof(local_dim).parameters[2] == LOCAL
        new_ind = conn.data[ind, actual_nb_ind]
    else
        new_ind, tail_position... = current_position
        dim, tail_dims... = dims
    end

    tail_position_exists, new_tail_position = remap_position(tail_position, tail_dims, offset, nb_ind, conn)
    position_exists = (new_ind != SKIP_NEIGHBOR_INDICATOR) && tail_position_exists
    return position_exists, (new_ind, new_tail_position...)
end
function remap_position(
        current_position::Tuple{Int64, Vararg{Int64}},
        dims::Tuple{Dimension, Vararg{Dimension}},
        offset::FieldOffsetTS{OffsetName, SourceDim, Tuple{TargetDim, TargetLocalDim}},
        nb_ind::Int64,
        conn) where {OffsetName, SourceDim <: Dimension, TargetDim <:Dimension, TargetLocalDim <: Dimension}  # TODO: restrict conn to type ::Connectivity
    if dims[1] == TargetDim()  # since we are mapping indices not field here the target source are flipped
        ind, tail_position... = current_position
        _, tail_dims... = dims
        new_ind = conn.data[ind, nb_ind]
    else
        new_ind, tail_position... = current_position
        dim, tail_dims... = dims
    end

    tail_position_exists, new_tail_position = remap_position(tail_position, tail_dims, offset, nb_ind, conn)
    position_exists = (new_ind != SKIP_NEIGHBOR_INDICATOR) && tail_position_exists
    return position_exists, (new_ind, new_tail_position...)
end
remap_position(current_position::Tuple{}, dims::Tuple{}, offset::FieldOffsetTS, nb_ind::Union{Int64, AllNeighbors}, conn) = (true, ())


function compute_remapped_field_info(
        field_size::Tuple{Int64, Vararg{Int64}},
        dims::Tuple{Dimension, Vararg{Dimension}},
        offset::FieldOffsetTS{OffsetName, SourceDim, Tuple{TargetDim, TargetLocalDim}},
        nb_ind::Union{Int64, AllNeighbors},
        conn) where {OffsetName, SourceDim <: Dimension, TargetDim <:Dimension, TargetLocalDim <: Dimension}
    length, tail_size... = field_size
    dim, tail_dims... = dims
    if dim == SourceDim()
        if nb_ind == AllNeighbors()
            new_dims_part = (TargetDim(), TargetLocalDim())
            # TODO: restrict indices to only what is needed
            new_size_part = size(conn.data)  # we just take the size of the connecitvity
        else
            new_dims_part = (TargetDim(),)
            # TODO: restrict indices to only what is needed
            new_size_part = size(conn.data)[1]  # we just take the size of the connecitvity
        end
    else
        new_dims_part = (dim,)
        new_size_part = (length,)
    end

    new_tail_dims, new_tail_size = compute_remapped_field_info(tail_size, tail_dims, offset, nb_ind, conn)
    new_dims = (new_dims_part..., new_tail_dims...)
    new_size = (new_size_part..., new_tail_size...)
    return new_dims, new_size
end
compute_remapped_field_info(
    field_size::Tuple{},
    dims::Tuple{},
    offset::FieldOffsetTS,
    nb_ind::Union{Int64, AllNeighbors},
    conn
)= ((), ())

function remap_ts(
        field::Field,
        offset::FieldOffsetTS{OffsetName, SourceDim, Tuple{TargetDim}},
        nb_ind::Int64)::Field where {OffsetName, SourceDim <: Dimension, TargetDim <:Dimension}
    conn = OFFSET_PROVIDER[string(OffsetName)]

    new_offsets = Dict(field.dims[i] => field.origin[i] for i in 1:length(field.dims))
    new_offsets[conn] = nb_ind
    return Field(field.dims, field.data, field.broadcast_dims, origin = new_offsets)
end


function remap_broadcast_dims(
    broadcast_dims::Tuple{T, Vararg{Dimension}},
    offset::FieldOffsetTS{OffsetName, SourceDim, Tuple{TargetDim, TargetLocalDim}},
    nb_ind::Union{Int64, AllNeighbors}
) where {T <: Dimension, OffsetName, SourceDim <: Dimension, TargetDim <:Dimension, TargetLocalDim <: Dimension}
    dim, dim_tail... = broadcast_dims
    if dim == SourceDim()
        if nb_ind == AllNeighbors()
            (TargetDim(), TargetLocalDim(), remap_broadcast_dims(dim_tail, offset, nb_ind)...)
        else
            (TargetDim(), remap_broadcast_dims(dim_tail, offset, nb_ind)...)
        end
    else
        (dim, remap_broadcast_dims(dim_tail, offset, nb_ind)...)
    end
end

remap_broadcast_dims(broadcast_dims::Tuple{}, offset::FieldOffsetTS, nb_ind::Union{Int64, AllNeighbors}) = ()


function remap_ts(
        field::Field,
        offset::FieldOffsetTS{OffsetName, SourceDim, Tuple{TargetDim, TargetLocalDim}},
        nb_ind::Union{Int64, AllNeighbors} = AllNeighbors())::Field  where {OffsetName, SourceDim <: Dimension, TargetDim <:Dimension, TargetLocalDim <: Dimension}
    conn = OFFSET_PROVIDER[string(OffsetName)]

    # compute new indices
    out_field_dims, out_field_size = compute_remapped_field_info(size(field.data), field.dims, offset, nb_ind, conn)
    #out_field = map(position -> begin
    #    neighbor_exists, new_position = remap_position(Tuple(position), out_field_dims, offset, nb_ind, conn)
    #    if neighbor_exists
    #        field.data[new_position...]
    #    else
    #        eltype(field.data)(0)
    #    end
    #end, CartesianIndices(map(len -> Base.OneTo(len), out_field_size)))
    out_field = Array{eltype(field.data)}(undef, out_field_size)
    for position in eachindex(IndexCartesian(), out_field)
        neighbor_exists, new_position = remap_position(Tuple(position), out_field_dims, offset, nb_ind, conn)
        if neighbor_exists
            out_field[position] = field.data[new_position...]
        end
    end
    # todo: origin
    return Field(out_field_dims, out_field, remap_broadcast_dims(field.broadcast_dims, offset, nb_ind))
end

(field::Field)(f_off::Tuple{FieldOffset, <:Integer})::Field = field(f_off...)
function (field::Field)(f_off::FieldOffset, nb_ind::Union{Int64, AllNeighbors} = AllNeighbors())::Field
    result = remap_ts(field, to_type_stable_field_offset(f_off), nb_ind)
    return result
end

# Field struct interfaces
Base.axes(F::Field)::Tuple = map((i,j) -> i .+ j, axes(F.data), F.origin)
Base.size(F::Field)::Tuple = size(F.data)
Base.convert(t::Type{T}, F::Field) where {T<:Number} = Field(F.dims, convert.(t, F.data), F.broadcast_dims)
@propagate_inbounds function Base.getindex(F::Field{BD,T,N}, inds::Vararg{Int,N}) where {BD, T,N}
    new_inds = inds .- F.origin
    return F.data[new_inds...]
end
@propagate_inbounds function Base.setindex!(F::Field{BD, T,N}, val, inds::Vararg{Int,N}) where {BD, T,N}
    new_inds = inds .- F.origin
    F.data[new_inds...] = val
end
Base.showarg(io::IO, @nospecialize(F::Field), toplevel) = print(io, eltype(F), " Field with dimensions ", get_dim_name.(F.broadcast_dims))
function slice(F::Field, inds...)::Field
    dim_ind = findall(x -> typeof(x) <: UnitRange{Int64}, inds)
    return Field(F.dims[dim_ind], view(F.data, inds...), F.broadcast_dims)
end

# Connectivity struct ------------------------------------------------------------

"""
    Connectivity(data::Array, source::Tuple, target::Tuple, dims::Int)

# Examples
```julia-repl
julia> new_connectivity = Connectivity(fill(1, (3,2)), Cell, Edge, 2)
Connectivity(Integer[1 1; 1 1; 1 1], Dimension{:Cell_, HORIZONTAL}(), Dimension{:Edge_, HORIZONTAL}(), 2
```
"""
struct Connectivity
    data::Array{Int64, 2}
    source::Dimension
    target::Dimension
    dims::Integer
end

# Field operator ----------------------------------------------------------------------

struct FieldOp
    name::Symbol
    f::Function
    expr::Expr
    closure_vars::Dict
end

# Includes ------------------------------------------------------------------------------------

include("embedded/builtins.jl")
include("embedded/cust_broadcast.jl")
include("gt2py/gt2py.jl")

# Helper functions for fields -------------------------------------------------------------------

"""
    copyfield!(target::Field, source::Field)

Takes a copies the data from a source field to a target fields (equivalent to target .= source). Also works with tuples of fields.

# Examples
```julia-repl
julia> copyfields!((a, b), (c, d))
"""
function copyfield!(target::Tuple{Vararg{Field}}, source::Tuple{Vararg{Field}})
    for i in 1:length(target)
        target[i] .= source[i]
    end
end
copyfield!(target, source) = target .= source

# Field operator functionalities ------------------------------------------------------------

OFFSET_PROVIDER::Union{Dict{String, Union{Connectivity, Dimension}}, Nothing} = nothing
FIELD_OPERATORS::Dict{Symbol, PyObject} = Dict{Symbol, PyObject}()

function (fo::FieldOp)(args...; offset_provider::Dict{String, Union{Connectivity, Dimension}} = Dict{String, Union{Connectivity, Dimension}}(), backend::String = "embedded", out = nothing, kwargs...)

    is_outermost_fo = isnothing(OFFSET_PROVIDER)
    if is_outermost_fo
        @assert !isnothing(out) "Must provide an out field."
        @assert typeof(out) <: Field || typeof(out) <: Tuple{Vararg{Field}} "Out argument is not a field."
        global OFFSET_PROVIDER = offset_provider
        out = backend_execution(Val{Symbol(backend)}(), fo, args, kwargs, out, is_outermost_fo)
        global OFFSET_PROVIDER = nothing
    else
        # TODO(tehrengruber): this breaks when fo execution fails and no cleanup is done. use try finally block
        #@assert isnothing(out)
        #@assert isempty(offset_provider)
        out = backend_execution(Val{Symbol(backend)}(), fo, args, kwargs, out, is_outermost_fo)
    end

    return out
end

function backend_execution(backend::Val{:embedded}, fo::FieldOp, args, kwargs, out, is_outermost_fo)
    if is_outermost_fo
        copyfield!(out, fo.f(args...; kwargs...))
        return
    else
        return fo.f(args...; kwargs...)
    end
end

function backend_execution(backend::Val{:py}, fo::FieldOp, args, kwargs, out, is_outermost_fo)
    
    if haskey(FIELD_OPERATORS, fo.name)
        f = FIELD_OPERATORS[fo.name]
    else
        f = py_field_operator(fo)
        FIELD_OPERATORS[fo.name] = f
    end
    p_args, p_kwargs, p_out, p_offset_provider = py_args.((args, kwargs, out, GridTools.OFFSET_PROVIDER))
    if is_outermost_fo
        f(p_args..., out = p_out, offset_provider = p_offset_provider; p_kwargs...)
        return
    else
        return f(p_args...; p_kwargs...)
    end
end

function get_closure_vars(expr::Expr, current_vars::Dict)::Dict

    expr_def = splitdef(expr)
    @assert all(typeof.(expr_def[:args]) .== Expr) && all(typeof.(expr_def[:kwargs]) .== Expr) ("Field operator parameters must be type annotated.")

    local_vars = Set()
    closure_names = Set()
    closure_vars = Dict()

    # catch all local variables
    postwalk(expr) do x
        if @capture(x, (name_ = value_) | (name_::type_))
            if typeof(name) == Symbol
                push!(local_vars, name)
            elseif typeof(name) == Expr
                if name.head == :tuple
                    push!(local_vars, name.args...)
                elseif name.head == :call
                    push!(local_vars, name.args[1])
                else
                    throw("For the following local variable: $name we dont provide support yet. Please report.") # TODO: verify
                end
            end
        end
        return x
    end

    # catch all dimensions
    postwalk(expr.args[1]) do x
        if typeof(x) == Symbol && x in keys(current_vars) && typeof(current_vars[x]) == DataType && current_vars[x] <: Dimension
            push!(closure_names, x)
        end
    end

    # catch all closure_variables
    prewalk(expr.args[2]) do x
        if @capture(x, name_(args__)) && !(name in local_vars) && !(name in math_ops)
            push!(closure_names, name)
            return Expr(:irgendoeppis, args...)  # small present for tehrengruber
        elseif typeof(x) == Symbol && !(x in local_vars) && !(x in math_ops)
            push!(closure_names, x)
            return x
        else 
            return x
        end
    end

    # update dictionary
    for name in closure_names
        closure_vars[name] = current_vars[name]
    end

    return closure_vars
end

macro module_vars()
    return esc(quote
            module_vars = Dict(name => Core.eval(@__MODULE__, name) for name in names(@__MODULE__)[6:end])
            local_vars = Base.@locals
            merge(module_vars, local_vars, GridTools.builtin_op)
        end)
end


"""
    @field_operator

The field_operator macro takes a function definition and creates a run environment for the function call within the GridTools package. It enables the additional argument "offset_provider", "backend" and "out.

# Examples
```julia-repl
julia> @field_operator addition(x::Int64) = x + x
...
```
"""
macro field_operator(expr::Expr)
    f_name = namify(expr)

    expr_dict = splitdef(expr)
    expr_dict[:name] = generate_unique_name(f_name)
    unique_expr = combinedef(expr_dict)
    
    return Expr(:(=), esc(f_name), :(FieldOp(namify($(Expr(:quote, expr))), $(esc(unique_expr)), $(Expr(:quote, expr)), get_closure_vars($(Expr(:quote, expr)), @module_vars))))
end

generate_unique_name(name::Symbol, value::Integer = 0) = Symbol("$(name)ᐞ$(value)")

end