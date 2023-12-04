# TODO
# When using a precompiled Module with PyCall there are issues that arise. See https://github.com/JuliaPy/PyCall.jl#using-pycall-from-julia-modules
# I've added my conversion attempt to GridTools/notes/Snippets.jl
# To test the precompilation remove the line below "__precompile__(false)" and attempt a precompilation (manually or via running the tests)

__precompile__(false)
module GridTools

using Printf
using Statistics
using BenchmarkTools
using Profile
using Base: @propagate_inbounds
using MacroTools
using OffsetArrays
using OffsetArrays: IdOffsetRange, no_offset_view
using Debugger

import Base.Broadcast: Extruded, Style, BroadcastStyle, ArrayStyle ,Broadcasted

export Dimension, DimensionKind, HORIZONTAL, VERTICAL, LOCAL, Field, Connectivity, FieldOffset, neighbor_sum, max_over, min_over, where, concat, @field_operator, @module_vars, get_dim_name, get_dim_kind, slice


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
Base.getindex(d::Dimension, offset::Integer) = d, offset            # TODO: Unnecessary?
get_dim_name(d::Dimension) = string(typeof(d).parameters[1])[1:end-1]
get_dim_kind(d::Dimension) = typeof(d).parameters[2]
get_dim_ind(source::Tuple{Vararg{Dimension}}, ind::Dimension) = findfirst(x -> x == ind, source)

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
    source::Dimension          # TODO: can source have multiple dimensions?
    target::Tuple{Vararg{Dimension}}

    function FieldOffset(name::String; source::Dimension, target::Union{Dimension, Tuple{Vararg{Dimension}}})::FieldOffset
        if length(target) > 1 @assert all(get_dim_kind.(Base.tail(target)) .== LOCAL) ("All but the first dimension in an offset must be local dimensions.") end
        new(name, source, Tuple(target))
    end
end

Base.getindex(f_off::FieldOffset, ind::Integer) = f_off, ind


# Field struct --------------------------------------------------------------------

# TODO: check for #dimension at compile time and not runtime
# TODO: <: AbstractArray{T,N} is not needed... but then we have to define our own length and iterate function for Fields
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
# TODO sequence of types: Do BD, T, N. 
# Error showing value of type Field{Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}, Float64, 2, Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}}:
# ERROR: CanonicalIndexError: getindex not defined for Field{Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}, Float64, 2, Tuple{Dimension{:Cell_, HORIZONTAL}, Dimension{:K_, HORIZONTAL}}}

struct Field{T <: Union{AbstractFloat, Integer}, N, BD <: Tuple{Vararg{Dimension}}, D <: Tuple{Vararg{Dimension}}} <: AbstractArray{T,N}
    dims::D
    data::AbstractArray{T,N}
    broadcast_dims::BD
    
    function Field(dims::D, data::AbstractArray{T,N}, broadcast_dims::BD = dims) where {T <: Union{AbstractFloat, Integer}, N, BD <: Tuple{Vararg{Dimension}}, D <: Tuple{Vararg{Dimension}}}
        if ndims(data) != 0 @assert length(dims) == ndims(data) end
        return new{T,N,BD,D}(dims, data, broadcast_dims)
    end

    function Field(dim::Dimension, data::AbstractArray{T,N}, broadcast_dims::Union{Dimension,BD} = dim) where {T <: Union{AbstractFloat, Integer}, N, BD <: Tuple{Vararg{Dimension}}}
        if ndims(data) != 0 @assert ndims(data) == 1 end
        return Field(Tuple(dim), data, Tuple(broadcast_dims))
    end
end


(field_call::Field)(f_off::Tuple{FieldOffset, <:Integer})::Field = field_call(f_off...)
function (field_call::Field)(f_off::FieldOffset, ind::Integer = 0)::Field

    conn = OFFSET_PROVIDER[f_off.name]

    if typeof(conn) <: Dimension
        new_size = zeros(Integer, length(axes(field_call.data)))
        new_size[get_dim_ind(field_call.dims, conn)] = ind
        return Field(field_call.dims, OffsetArray(field_call.data, new_size...), field_call.broadcast_dims)
    elseif typeof(conn) <: Connectivity

        conn_data = ind == 0 ? no_offset_view(conn.data) : no_offset_view(conn.data)[:,ind]
        f_target = ind == 0 ? f_off.target : f_off.target[1]

        conn_ind = get_dim_ind(field_call.dims, f_off.source)

        @assert maximum(conn_data) <= maximum(axes(field_call)[conn_ind]) "Indices of Connectivity $f_off are out of range for the called field"
        @assert !any(x -> x == 0, conn_data) && minimum(conn_data) > -2 "Illegal indices used in the Connectivity $f_off"
        @assert all(x -> x in f_off.source, conn.source) && all(x -> x in f_off.target, conn.target) "Source or target dimensions of Connectivity $f_off do not match the called field"     
        
        if ndims(field_call) == 1
            res = map(x -> x == -1 ? convert(eltype(field_call), 0) : getindex(field_call, Int.(x)), conn_data)
        else
            f(slice) = map(x -> x == -1 ? convert(eltype(field_call), 0) : getindex(slice, Int.(x)), conn_data)

            sliced_data = eachslice(field_call.data, dims=Tuple(deleteat!(collect(1:ndims(field_call)), conn_ind)))
            outsize = [size(field_call)...]
            Tuple(splice!(outsize, conn_ind, [size(conn_data)...]))
            res = reshape(hcat(map(f, sliced_data)...), Tuple(outsize))
        end

        new_dims = [field_call.dims[1:conn_ind-1]..., f_target..., field_call.dims[conn_ind+1:end]...]

        return Field(Tuple(new_dims), res)
    else
        throw("The datatype: $(typeof(conn)) is not supported for within an offset_provider")
    end
end

# Field struct interfaces
Base.size(F::Field)::Tuple = size(F.data)
Base.axes(F::Field)::Tuple = axes(F.data)
Base.convert(t::Type{T}, F::Field) where {T<:Number} = Field(F.dims, convert.(t, F.data), F.broadcast_dims)
@propagate_inbounds Base.getindex(F::Field{T,N}, inds::Vararg{Int,N}) where {T,N} = F.data[inds...]
@propagate_inbounds Base.setindex!(F::Field{T,N}, val, inds::Vararg{Int,N}) where {T,N} = F.data[inds...] = val
Base.showarg(io::IO, F::Field, toplevel) = print(io, eltype(F), " Field with dimensions ", get_dim_name.(F.broadcast_dims))
function Base.copyto!(target::Tuple{Vararg{Field}}, source::Tuple{Vararg{Field}})
    for i in 1:length(target)
        target[i] .= source[i]
    end
end
function slice(F::Field, inds...)::Field
    dim_ind = findall(x -> typeof(x) <: UnitRange{Int64}, inds)
    return Field(F.dims[dim_ind], view(F.data, inds...), F.broadcast_dims)
end



# TODO: where doesnt allow mixed types. Should we promote mixed types or should we throw an error?

# function Base.promote(f1::Field, f2::Field)
#     f1_new_data, f2_new_data = promote(f1.data, f2.data)
#     return Field(f1.dims, f1_new_data, f1.broadcast_dims), Field(f2.dims, f2_new_data, f2.broadcast_dims)
# end

# function Base.promote(scal::Real, f::Field)

# end

# function Base.promote(f::Field, scal::Real)

# end

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
    data::Array{Integer, 2}
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

OFFSET_PROVIDER::Union{Dict{String, Union{Connectivity, Dimension}}, Nothing} = nothing
FIELD_OPERATORS::Dict{Symbol, FieldOp} = Dict{Symbol, FieldOp}()

function (fo::FieldOp)(args...; offset_provider::Dict{String, Union{Connectivity, Dimension}} = Dict{String, Union{Connectivity, Dimension}}(), backend::String = "embedded", out = nothing, kwargs...)

    FIELD_OPERATORS[fo.name] = fo

    is_outermost_fo = isnothing(OFFSET_PROVIDER)
    if is_outermost_fo
        @assert !isnothing(out) "Must provide an out field."
        @assert typeof(out) <: Field || typeof(out) <: Tuple{Vararg{Field}} "Out argument is not a field."
        global OFFSET_PROVIDER = offset_provider
        out = backend_execution(Val{Symbol(backend)}(), fo, args, kwargs, out, is_outermost_fo)
        global OFFSET_PROVIDER = nothing
    else
        @assert isnothing(out)
        @assert isempty(offset_provider)
        out = backend_execution(Val{Symbol(backend)}(), fo, args, kwargs, out, is_outermost_fo)
    end

    return out
end

function backend_execution(backend::Val{:embedded}, fo::FieldOp, args, kwargs, out, is_outermost_fo)
    if is_outermost_fo
        copyto!(out, fo.f(args...; kwargs...))
        return
    else
        return fo.f(args...; kwargs...)
    end
end

function backend_execution(backend::Val{:py}, fo::FieldOp, args, kwargs, out, is_outermost_fo)
    f = py_field_operator(fo)
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

The field_operator macro takes a function definition and creates a run environment for the function call within the GridTools package. It enables the additional argument "offset_provider", "backend", etc.

# Examples
```julia-repl
julia> @field_operator addition(x) = x + x
addition (generic function with 1 method)
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

# Includes ------------------------------------------------------------------------------------

include("embedded/builtins.jl")
include("embedded/cust_broadcast.jl")
include("gt2py/gt2py.jl")

end

