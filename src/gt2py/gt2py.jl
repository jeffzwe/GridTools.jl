# Imports ---------------------------------------------------------------------------------
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3.10")
ENV["PYTHONBREAKPOINT"] = "pdb.set_trace"

using PyCall
using MacroTools
using MacroTools: prewalk, postwalk

gtx = pyimport("gt4py.next")
np = pyimport("numpy")

func_to_foast = gtx.ffront.func_to_foast
foast = gtx.ffront.field_operator_ast
type_info = gtx.type_system.type_info
ts = gtx.type_system.type_specifications
type_translation = gtx.type_system.type_translation
dialect_ast_enums = gtx.ffront.dialect_ast_enums
fbuiltins = gtx.ffront.fbuiltins
ClosureVarFolding = gtx.ffront.foast_passes.closure_var_folding.ClosureVarFolding
ClosureVarTypeDeduction = gtx.ffront.foast_passes.closure_var_type_deduction.ClosureVarTypeDeduction
DeadClosureVarElimination = gtx.ffront.foast_passes.dead_closure_var_elimination.DeadClosureVarElimination
UnpackedAssignPass = gtx.ffront.foast_passes.iterable_unpack.UnpackedAssignPass
FieldOperatorTypeDeduction = gtx.ffront.foast_passes.type_deduction.FieldOperatorTypeDeduction
FieldOperater = gtx.ffront.decorator.FieldOperator
roundtrip = gtx.program_processors.runners.roundtrip


concepts = pyimport("gt4py.eve.concepts")
SourceLocation = concepts.SourceLocation

py"""
import builtins
from typing import Any, Callable, Iterable, Mapping, Type, cast
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation
from gt4py.next.ffront import dialect_ast_enums, fbuiltins, field_operator_ast as foast
"""
include("preprocessing.jl")
include("single_static_assign.jl")
include("jast_to_foast.jl")

# Utils ------------------------------------------------------------------------------------

bin_op = Set([:(+), :(-), :(*), :(/), :(÷), :(^), :(%), :(&), :(|), :(⊻), :.+, :.-, :.*, :./, :.÷, :.^, :.%, :.&, :.|, :.⊻])
unary_op = Set([:(!), :(~), :.!, :.~])
comp_op = Set([:(==), :(!=), :(<), :(<=), :(>), :(>=), :.==, :.!=, :.<, :.<=, :.>, :.>=])

math_ops = union(bin_op, unary_op, comp_op)

scalar_types = Dict(
    :Bool => ts.ScalarKind."BOOL",
    :Int32 => ts.ScalarKind."INT32",
    :Int64 => ts.ScalarKind."INT64",
    :Int => ts.ScalarKind."INT64",
    :Integer => ts.ScalarKind."INT64",
    :(<:Integer) => ts.ScalarKind."INT64",
    :Float32 => ts.ScalarKind."FLOAT32",
    :Float64 => ts.ScalarKind."FLOAT64",
    :AbstractFloat => ts.ScalarKind."FLOAT64",
    :(<:AbstractFloat) => ts.ScalarKind."FLOAT64",
    :String => ts.ScalarKind."STRING"
)

py_scalar_types = Dict(
    Bool => py"bool",
    Int32 => np.int32,
    Int64 => np.int64,
    Int => np.int64,
    Integer => np.int64,
    Float32 => np.float32,
    Float64 => np.float64,
    AbstractFloat => np.float64,
)

py_dim_kind = Dict(
    HORIZONTAL => gtx.DimensionKind.HORIZONTAL,
    VERTICAL => gtx.DimensionKind.VERTICAL,
    LOCAL => gtx.DimensionKind.LOCAL
)

builtin_op = Dict(
    :max_over => gtx.max_over, 
    :min_over => gtx.min_over, 
    :broadcast => gtx.broadcast,
    :where => gtx.where,
    :neighbor_sum => gtx.neighbor_sum,
    :convert => gtx.astype,
    :as_offset => gtx.as_offset,
    :sin => gtx.sin,
    :cos => gtx.cos,
    :tan => gtx.tan,
    :asin => gtx.arcsin,
    :acos => gtx.arccos,
    :atan => gtx.arctan,
    :sinh => gtx.sinh,
    :cosh => gtx.cosh,
    :tanh => gtx.tanh,
    :asinh => gtx.arcsinh,
    :acosh => gtx.arccosh,
    :atanh => gtx.arctanh,
    :sqrt => gtx.sqrt,
    :exp => gtx.exp,
    :log => gtx.log,
    :gamma => gtx.gamma,
    :cbrt => gtx.cbrt,
    :floor => gtx.floor,
    :ceil => gtx.ceil,
    :trunc => gtx.trunc,
    :abs => gtx.abs,
    :isfinite => gtx.isfinite,
    :isinf => gtx.isinf,
    :isnan => gtx.isnan,
    :min => gtx.minimum,
    :max => gtx.maximum
)



disallowed_op = Set([])

CURRENT_MODULE = nothing


# Methods -----------------------------------------------------------------------------------

function py_field_operator(function_definition::Expr, module_::Module, backend = roundtrip.executor, grid_type = py"None"o, operator_attributes = Dict())

    global CURRENT_MODULE = module_

    foast_definition_node, closure_vars = jast_to_foast(function_definition)
    loc = foast_definition_node.location

    operator_attribute_nodes = Dict(
        key => foast.Constant(value=value, type=type_translation.from_value(value), location=loc)
        for (key, value) in operator_attributes
    )

    untyped_foast_node = foast.FieldOperator(
        id=(foast_definition_node.id),
        definition=(foast_definition_node),
        location=loc
        ;operator_attribute_nodes...
    )

    foast_node = FieldOperatorTypeDeduction.apply(untyped_foast_node)

    return FieldOperater(
            foast_node=foast_node,
            closure_vars=closure_vars,
            definition=py"None"o,
            backend=backend,
            grid_type=grid_type,
        )
end

function jast_to_foast(expr::Expr)
    expr, closure_vars, annotations = preprocess_definiton(expr)
    expr, closure_vars = remove_function_aliases(expr, closure_vars)                                                     # TODO Can be ommited once gt4py allows aliases
    foast_node = visit_jast(expr, closure_vars)
    foast_node = postprocess_definition(foast_node, closure_vars, annotations)
    return foast_node, closure_vars
end

function preprocess_definiton(expr::Expr)
    ssa = single_static_assign_pass(expr)
    sat = single_assign_target_pass(ssa)
    ucc = unchain_compairs_pass(sat)
    closure_vars = get_closure_vars(ucc)
    annotations = get_annotation(ucc)
    return (ucc, closure_vars, annotations)
end

function postprocess_definition(foast_node, closure_vars, annotations)
    foast_node = ClosureVarFolding.apply(foast_node, closure_vars)
    foast_node = DeadClosureVarElimination.apply(foast_node)
    foast_node = ClosureVarTypeDeduction.apply(foast_node, closure_vars)

    foast_node = FieldOperatorTypeDeduction.apply(foast_node)
    foast_node = UnpackedAssignPass.apply(foast_node)

    if haskey(annotations, "return")
        annotated_return_type = annotations["return"]
        @assert annotated_return_type == foast_node.type.returns  ("Annotated return type does not match deduced return type. Expected $(foast_node.type.returns), but got $annotated_return_type.")
    end

    return foast_node
end



function py_args(args::Tuple)
    out_args = []

    for i in args
        push!(out_args, convert_type(i))
    end

    return out_args
end

function py_args(args::Union{Base.Pairs, Dict})
    out_args = Dict()

    for i in args
        out_args[i.first] = convert_type(i.second)
    end

    return out_args
end
py_args(arg) = convert_type(arg)

function convert_type(a)
    if typeof(a) <: Field

        b_dims = []
        for dim in a.broadcast_dims
            kind = py_dim_kind[(get_dim_kind(dim))]
            push!(b_dims, gtx.Dimension(string(get_dim_name(dim))[1:end-1], kind = kind))      #TODO this requires strict naming rules for dimensions... not sure if we want that
        end

        if ndims(a.data) == length(b_dims)
            data = a.data
        else                                                                    # upscale array to new dimensions for gt4py
            data = upscale_data(a.dims, a.broadcast_dims, a.data)
        end

        return gtx.np_as_located_field(Tuple(b_dims)...)(np.asarray(data))     #TODO a.data gets passed as a list, not as a numpy array as stated in documentation of PyCall

    elseif typeof(a) <: Connectivity
    
        kind = py_dim_kind[(get_dim_kind(a.source))]
        source_dim  = gtx.Dimension(string(get_dim_name(a.source))[1:end-1], kind=kind)
        
        kind = py_dim_kind[(get_dim_kind(a.target))]
        target_dim = gtx.Dimension(string(get_dim_name(a.target))[1:end-1], kind=kind)

        return gtx.NeighborTableOffsetProvider(np.asarray(a.data .- 1), target_dim, source_dim, a.dims)      #TODO a.data gets passed as a list, not as a numpy array as stated in documentation of PyCall
    else 
        @assert Symbol(typeof(a)) in keys(scalar_types) ("The type of argument $(a) is not a valid argument type to a field operator")
        return a
    end
end


function upscale_data(dims::Tuple{Vararg{<:Dimension}}, b_dims::Tuple{Vararg{<:Dimension}}, data::Array)

    out_size = []

    for dim in b_dims
        if dim in dims
            ind = findfirst(x -> x == dim, dims) 
            push!(out_size, size(data)[ind])
        else
            push!(out_size, 1)
        end
    end

    return reshape(data, Tuple(out_size))
end

# -------------------------------------------------------------------------

# TODO Can be ommited once gt4py allows aliases

py_aliases = Dict(
    :convert => :astype,
    :asin => :arcsin,
    :acos => :arccos,
    :atan => :arctan,
    :asinh => :arcsinh,
    :acosh => :arccosh,
    :atanh => :arctanh,
    :min => :minimum,
    :max => :maximum
)

function remove_function_aliases(expr::Expr, closure_vars::Dict)

    expr = postwalk(expr) do x
        if typeof(x) == Symbol && x in keys(py_aliases)
            return py_aliases[x]
        else
            return x
        end
    end

    for (key, value) in closure_vars
        if Symbol(key) in keys(py_aliases)
            closure_vars[string(py_aliases[Symbol(key)])] = value
            delete!(closure_vars, key)
        end
    end

    return expr, closure_vars

end