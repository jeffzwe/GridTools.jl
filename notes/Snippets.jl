

# where implementation with @generated #####################################################################

function dfs_index(exp::Expr)
    if typeof(exp.args[1]) == Symbol
        return [exp.args[2]]
    else
        return [dfs_index(exp.args[1])..., exp.args[2]]
    end 
end

## Recursive Version
# function fold_tuple(res::Expr, t::Expr, p::Expr, d::Vector, mask::Type{<:Field})::Expr
#     ind = d[1]
#     if length(d) > 1
#         @assert length(res.args) in (ind-1, ind)
#         if length(res.args) == ind-1
#             push!(res.args, Expr(:tuple))
#         end
#         res.args[ind] = fold_tuple(res.args[ind], t, p, d[2:length(d)], mask)
#     else
#         temp = :(ifelse.(mask, $t, $p))
#         push!(res.args, temp)
#     end
#     return res
# end

## Iterative version
function fold_tuple(t::Expr, p::Expr, mask::Type{<:Field})::Expr
   res = Expr(:tuple)
   for tuple_id in 1:length(t.args)             # for all tuples
        path = dfs_index(t.args[tuple_id])       # get the path of a tuple
        current_expr = res
        for path_step in 1:length(path)-1
            @assert length(current_expr.args) in (path[path_step]-1, path[path_step])
            if length(current_expr.args) == path[path_step]-1
                push!(current_expr.args, Expr(:tuple))
            end
            current_expr = current_expr.args[path[path_step]]
        end
        push!(current_expr.args, :(ifelse.(mask, $(t.args[tuple_id]), $(p.args[tuple_id]))))
   end
   return res
end

function unfold_tuple(tt::Type{<:Tuple}, tuple_name::Symbol, path::Vector = [])::Expr
    res = Expr(:tuple)
    for i in 1:length(tt.parameters)
        if tt.parameters[i] <: Tuple
            push!(res.args, unfold_tuple(tt.parameters[i], tuple_name, [path..., i]).args...)
        else
            current_expr = tuple_name
            for current_index in [path..., i]
                current_expr = Expr(:ref, current_expr, current_index)
            end
            push!(res.args, current_expr)
        end
    end
    return res
end

@inbounds @generated function where(mask::Field, tt::Tuple, pp::Tuple)   # TODO: Make @generated. Problem with naming String, vectors used like d::Vector and so on.
    @assert tt == pp
    t = unfold_tuple(tt, :tt)
    p = unfold_tuple(pp, :pp)

    ## Recursive Version
    # res = Expr(:tuple)

    # for i in 1:length(t.args)
    #     d = dfs_index(t.args[i])
    #     res = fold_tuple(res, t.args[i], p.args[i], d, mask)
    # end

    ## Iterative version
    res = fold_tuple(t, p, mask)

    return res
end


# First try precompile version gt2py.jl ###############################################################################

# Imports ---------------------------------------------------------------------------------
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3.10")
# ENV["PYTHONBREAKPOINT"] = "pdb.set_trace"

using PyCall
using MacroTools
using MacroTools: prewalk, postwalk

const gtx = PyNULL()
const np = PyNULL()

const func_to_foast = PyNULL()
const foast = PyNULL()
const type_info = PyNULL()
const ts = PyNULL()
const type_translation = PyNULL()
const dialect_ast_enums = PyNULL()
const fbuiltins = PyNULL()
const ClosureVarFolding = PyNULL()
const ClosureVarTypeDeduction = PyNULL()
const DeadClosureVarElimination = PyNULL()
const UnpackedAssignPass = PyNULL()
const FieldOperatorTypeDeduction = PyNULL()
const FieldOperater = PyNULL()
const roundtrip = PyNULL()
const SourceLocation = PyNULL()

const scalar_types = Dict()
const py_scalar_types = Dict()
const py_dim_kind = Dict()
const builtin_op = Dict()

function __init__()
    copy!(gtx, pyimport("gt4py.next"))
    copy!(np, pyimport("numpy"))

    copy!(func_to_foast, pyimport("gt4py.next.ffront.func_to_foast"))
    copy!(foast, pyimport("gt4py.next.ffront.field_operator_ast"))
    copy!(type_info, pyimport("gt4py.next.type_system.type_info"))
    copy!(ts, pyimport("gt4py.next.type_system.type_specifications"))
    copy!(type_translation, pyimport("gt4py.next.type_system.type_translation"))
    copy!(dialect_ast_enums, pyimport("gt4py.next.ffront.dialect_ast_enums"))
    copy!(fbuiltins, pyimport("gt4py.next.ffront.fbuiltins"))
    copy!(ClosureVarFolding, pyimport("gt4py.next.ffront.foast_passes.closure_var_folding").ClosureVarFolding)
    copy!(ClosureVarTypeDeduction, pyimport("gt4py.next.ffront.foast_passes.closure_var_type_deduction").ClosureVarTypeDeduction)
    copy!(DeadClosureVarElimination, pyimport("gt4py.next.ffront.foast_passes.dead_closure_var_elimination").DeadClosureVarElimination)
    copy!(UnpackedAssignPass, pyimport("gt4py.next.ffront.foast_passes.iterable_unpack").UnpackedAssignPass)
    copy!(FieldOperatorTypeDeduction, pyimport("gt4py.next.ffront.foast_passes.type_deduction").FieldOperatorTypeDeduction)
    copy!(FieldOperater, pyimport("gt4py.next.ffront.decorator").FieldOperator)
    copy!(roundtrip, pyimport("gt4py.next.program_processors.runners.roundtrip"))
    copy!(SourceLocation, pyimport("gt4py.eve.concepts").SourceLocation)


    # todo: place into functions
    scalar_types_init = Dict(
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

    py_scalar_types_init = Dict(
    Bool => py"bool",
    Int32 => np.int32,
    Int64 => np.int64,
    Int => np.int64,
    Integer => np.int64,
    Float32 => np.float32,
    Float64 => np.float64,
    AbstractFloat => np.float64,
    )

    py_dim_kind_init = Dict(
    HORIZONTAL => gtx.DimensionKind.HORIZONTAL,
    VERTICAL => gtx.DimensionKind.VERTICAL,
    LOCAL => gtx.DimensionKind.LOCAL
    )

    builtin_op_init = Dict(
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

    copy!(scalar_types, scalar_types_init)
    copy!(py_scalar_types, py_scalar_types_init)
    copy!(py_dim_kind, py_dim_kind_init)
    copy!(builtin_op, builtin_op_init)
end


