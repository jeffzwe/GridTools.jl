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

function jast_to_foast(definition::Expr)
    definition = preprocess_definiton(definition)

    annotations = get_annotation(expr)
    closure_vars = get_closure_vars(expr)
    foast_node = visit(expr, closure_vars)

    foast_node = postprocess_definition(foast_node, closure_vars, annotations)

    return foast_node
end

function preprocess_definiton(definition::Expr)
    ssa = single_static_assign_pass(definition)
    sat = single_assign_target_pass(ssa)
    ucc = unchain_compairs_pass(sat)
    return ucc
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

function single_static_assign_pass(expr::Expr)
    var_trans = Dict()
    new_to_og = Dict()
    og_to_new = Dict()

    postwalk(expr) do x
        if @capture(x, name_ = value_)
            if name in keys(new_to_og)
                og_name = pop!(new_to_og, name)
                var_trans[og_name] = var_trans[og_name]+1
            else
                og_name = name
                var_trans[name] = 0
            end
            new_name = generate_unique_name(og_name, var_trans)
            new_to_og[new_name] = og_name
            og_to_new[og_name] = new_name
            
            return :($new_name = $value)
        elseif x in keys(og_to_new)
            return og_to_new[x]
        else
            return x
        end
    end
end

function generate_unique_name(name::Symbol, var_trans::Dict)
    return Symbol("$(name)ᐞ$(var_trans[name])")
end

function single_assign_target_pass(expr::Expr)
    return postwalk(expr) do x
        @capture(x, (t1_, t2_ = val1_, val2_) | ((t1_, t2_) = (val1_, val2_)) | (t1_, t2_ = (val1_, val2_)) | ((t1_, t2_) = val1_, val2_)) || return x
        return :($t1 = $val1; $t2 = $val2)
    end
end

function unchain_compairs_pass(expr::Expr)
    return postwalk(expr) do x
        if typeof(x) == Expr && x.head == :comparison
            return rec_unchain(x.args)
        else
            return x
        end
    end
end

function rec_unchain(args::Array)
    if length(args) == 3
        return Expr(:call, args[2], args[1], args[3])  # Alternative syntax: :($(args[2])($(args[1]), $(args[3])))
    else
        return Expr(:&&, Expr(:call, args[2], args[1], args[3]), rec_unchain(args[3:end]))
    end
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

    expr_def = splitdef(expr)
    @assert all(typeof.(expr_def[:args]) .== Expr) ("Field operator parameters must be type annotated.")
    @assert all(typeof.(expr_def[:kwargs]) .== Expr) ("Field operator parameters must be type annotated.")

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



