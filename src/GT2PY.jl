# Imports ---------------------------------------------------------------------------------
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3.10")
ENV["PYTHONBREAKPOINT"] = "pdb.set_trace"

using PyCall
using MacroTools
using MacroTools: prewalk, postwalk
using Debugger

gtx = pyimport("gt4py.next")

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

concepts = pyimport("gt4py.eve.concepts")
SourceLocation = concepts.SourceLocation

py"""
import builtins
from typing import Any, Callable, Iterable, Mapping, Type, cast
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation
from gt4py.next.ffront import dialect_ast_enums, fbuiltins, field_operator_ast as foast
"""

include("Jast2Foast.jl")

# Globals ----------------------------------------------------------------------------------
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
    :astype => gtx.astype,
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

# Methods -----------------------------------------------------------------------------------

# Notes:
# Annotations should be an empty dictionary. Can change this later on.

function jast_to_foast(expr::Expr)
    annotations = get_annotation(expr)
    closure_vars = get_closure_vars(expr)
    foast_node = visit(expr, closure_vars)

    foast_node = ClosureVarFolding.apply(foast_node, closure_vars)
    foast_node = DeadClosureVarElimination.apply(foast_node)
    foast_node = ClosureVarTypeDeduction.apply(foast_node, closure_vars)
    foast_node = FieldOperatorTypeDeduction.apply(foast_node)
    foast_node = UnpackedAssignPass.apply(foast_node)

    if haskey(annotations, "return")
        # TODO(tehrengruber): use `type_info.return_type` when the type of the
        annotated_return_type = annotations["return"]
        #  arguments becomes available here
        @assert annotated_return_type == foast_node.type.returns  ("Annotated return type does not match deduced return type. Expected $(foast_node.type.returns), but got $annotated_return_type.")
    end

    return foast_node
end

function get_annotation(expr::Expr)
    out_ann = Dict()

    if expr.args[1].head == :(::)
        return_type = from_type_hint(expr.args[1].args[2])
        out_ann["return"] = return_type
    end
    return out_ann
end

function get_closure_vars(expr::Expr)
    j_closure_vars = get_j_cvars(expr)
    return translate_cvars(j_closure_vars)
end

function get_j_cvars(expr::Expr)
    local_vars = Set()
    closure_names = Set()
    closure_vars = Dict()

    # catch all local variables
    postwalk(expr) do x
        if @capture(x, (name_ = value_) | (name_::type_))
            if typeof(name) == Symbol
                push!(local_vars, name)
            elseif typeof(name) == Expr && name.head == :tuple
                push!(local_vars, name.args...)
            end
        end
        return x
    end

    # catch all closure_variables
    postwalk(expr.args[2]) do x
        if typeof(x) == Symbol && !(x in local_vars) && !(x in math_ops)
            push!(closure_names, x)
        end
        return x
    end

    # add name => type to dictionary
    for name in closure_names
        closure_vars[name] = eval(name)
    end

    return closure_vars 
end

function translate_cvars(j_closure_vars::Dict)
    py_cvars = Dict()

    for (key, value) in j_closure_vars
        new_value = nothing
        if typeof(value) == FieldOffset
            py_source = map(dim -> gtx.Dimension(get_dim_name(dim), kind=py_dim_kind[get_dim_kind(dim)]), value.source)
            py_target = map(dim -> gtx.Dimension(get_dim_name(dim), kind=py_dim_kind[get_dim_kind(dim)]), value.target)
            new_value = gtx.FieldOffset(
                value.name, 
                source= length(py_source) == 1 ? py_source[1] : py_source, 
                target= length(py_target) == 1 ? py_target[1] : py_target
            )
        elseif typeof(value) <: Dimension
            new_value = gtx.Dimension(get_dim_name(value), kind=py_dim_kind[get_dim_kind(value)])

        elseif typeof(value) <: Function
            if key in keys(builtin_op)
                new_value = builtin_op[key]
            elseif key in keys(GridTools.py_field_ops)
                new_value = GridTools.py_field_ops[key]
            end
        elseif isconst(@__MODULE__, Symbol(value))
            # TODO create FrozenNameSpace...
            new_value = "Constant"
        else 
            throw("Access to following type: $(typeof(value)) is not permitted within a field operator!")
        end
        py_cvars[string(key)] = new_value
    end
    return py_cvars
end



