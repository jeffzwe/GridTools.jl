
# ---------------------------------------------------------------------------------------------------------------------------
# Sets and Dicts

bin_op = Set([:(+), :(-), :(*), :(/), :(÷), :(^), :(%), :(&), :(|), :(⊻), :.+, :.-, :.*, :./, :.÷, :.^, :.%, :.&, :.|, :.⊻])
unary_op = Set([:(!), :(~), :.!, :.~])
comp_op = Set([:(==), :(!=), :(<), :(<=), :(>), :(>=), :.==, :.!=, :.<, :.<=, :.>, :.>=])

math_ops = union(bin_op, unary_op, comp_op)

scalar_types = Dict(
    :Bool => ts.ScalarKind."BOOL",
    :Int32 => ts.ScalarKind."INT32",
    :Int64 => ts.ScalarKind."INT64",
    :(<:Integer) => ts.ScalarKind."INT64",
    :Float32 => ts.ScalarKind."FLOAT32",
    :Float64 => ts.ScalarKind."FLOAT64",
    :(<:AbstractFloat) => ts.ScalarKind."FLOAT64",
    :String => ts.ScalarKind."STRING"
)


# -----------------------------------------------------------------------------------------------------------------------------
# visit() instances

# TODO maybe? vcat, hcat, a.dims, 

function visit(sym::Symbol, loc)
    return foast.Name(id=string(sym), location=loc)
end

function visit(sym::Symbol)
    return visit_(Val{sym}())
end

function visit(expr::Expr, loc=nothing)
   return visit_(Val{expr.head}(), expr.args, loc)
end

function visit(expr::Expr, closure_vars::Dict)
    return visit_(Val{:function}(), expr.args, closure_vars)
end

function visit(constant::Any, loc=nothing)
    type_ = type_translation.from_value(constant)

    return foast.Constant(
        value=constant,
        location=loc,
        type=type_,
    )
end

function _builtin_type_constructor_symbols(captured_vars, loc)::Tuple
    py"""
    result: list[foast.Symbol] = []
    skipped_types = {"tuple"}
    python_type_builtins: dict[str, Callable[[Any], Any]] = {
        name: getattr(builtins, name)
        for name in set(fbuiltins.TYPE_BUILTIN_NAMES) - skipped_types
        if hasattr(builtins, name)
    }
    """
    captured_type_builtins = Dict(
            name => value
            for (name, value) in captured_vars
            if py"$name in fbuiltins.TYPE_BUILTIN_NAMES and $value is getattr(fbuiltins, name)"
    )
    py"""
    to_be_inserted = python_type_builtins | $captured_type_builtins
    for name, value in to_be_inserted.items():
        result.append(
            foast.Symbol(
                id=name,
                type=ts.FunctionType(
                    pos_only_args=[
                        ts.DeferredType(constraint=ts.ScalarType)
                    ],  # this is a constraint type that will not be inferred (as the function is polymorphic)
                    pos_or_kw_args={},
                    kw_only_args={},
                    returns=cast(ts.DataType, type_translation.from_type_hint(value)),
                ),
                namespace=dialect_ast_enums.Namespace.CLOSURE,
                location=$loc,
            )
        )
    """
    return py"result, to_be_inserted.keys()"
end

function visit_(sym::Val{:function}, args::Array, closure_vars::Dict)
    inner_loc = get_location(args[2].args[1])
    closure_var_symbols, skip_names = _builtin_type_constructor_symbols(closure_vars, inner_loc)

    for name in keys(closure_vars)
        if name in skip_names
            continue
        end

        push!(closure_var_symbols, 
            foast.Symbol(
                id=name,
                type=py"ts.DeferredType(constraint=None)",
                namespace=dialect_ast_enums.Namespace."CLOSURE",
                location=inner_loc,
            )
        )
    end

    function_header = args[1]
    function_body = visit(args[2], inner_loc)
    function_params = []

    if function_header.head == :(::)
        function_header = function_header.args[1]
    end

    for arg in Base.tail((Tuple(function_header.args)))
        function_params = vcat(function_params, visit(arg, inner_loc))
    end
    
    return foast.FunctionDefinition(
        id=string(function_header.args[1]),
        params= function_params,
        body=function_body,
        closure_vars=closure_var_symbols,
        location=inner_loc
    )
end

visit_(sym::Union{Val{:(+)}, Val{:.+}}) = dialect_ast_enums.BinaryOperator."ADD"
visit_(sym::Union{Val{:(-)}, Val{:.-}}) = dialect_ast_enums.BinaryOperator."SUB"
visit_(sym::Union{Val{:(*)}, Val{:.*}}) = dialect_ast_enums.BinaryOperator."MULT"
visit_(sym::Union{Val{:(/)}, Val{:./}}) = dialect_ast_enums.BinaryOperator."DIV"
visit_(sym::Union{Val{:(÷)}, Val{:.÷}}) = dialect_ast_enums.BinaryOperator."FLOOR_DIV"
visit_(sym::Union{Val{:(^)}, Val{:.^}}) = dialect_ast_enums.BinaryOperator."POW"
visit_(sym::Union{Val{:(%)}, Val{:.%}}) = dialect_ast_enums.BinaryOperator."MOD"
visit_(sym::Union{Val{:(&)}, Val{:.&}}) = dialect_ast_enums.BinaryOperator."BIT_AND"
visit_(sym::Union{Val{:(|)}, Val{:.|}}) = dialect_ast_enums.BinaryOperator."BIT_OR"
visit_(sym::Union{Val{:(⊻)}, Val{:.⊻}}) = dialect_ast_enums.BinaryOperator."BIT_XOR"
visit_(sym::Union{Val{:(!)}, Val{:.!}}) = dialect_ast_enums.UnaryOperator."NOT"
visit_(sym::Union{Val{:(~)}, Val{:.~}}) = dialect_ast_enums.UnaryOperator."INVERT"
visit_(sym::Union{Val{:(==)}, Val{:.==}}) = foast.CompareOperator."EQ"
visit_(sym::Union{Val{:(!=)}, Val{:.!=}}) = foast.CompareOperator."NOTEQ"
visit_(sym::Union{Val{:(<)}, Val{:.<}}) = foast.CompareOperator."LT"
visit_(sym::Union{Val{:(<=)}, Val{:.<=}}) = foast.CompareOperator."LTE"
visit_(sym::Union{Val{:(>)}, Val{:.>}}) = foast.CompareOperator."GT"
visit_(sym::Union{Val{:(>=)}, Val{:.>=}}) = foast.CompareOperator."GTE"

function visit_(sym::Val{:call}, args::Array, outer_loc)
    if args[1] in bin_op
        return foast.BinOp(
            op=visit(args[1]),
            left=visit(args[2], outer_loc),
            right=visit(args[3], outer_loc),
            location=outer_loc
        )
    elseif args[1] in unary_op
        return foast.UnaryOp(
            op=visit(args[1]),
            operand=visit(args[2], outer_loc),
            location=outer_loc
        )
    elseif args[1] in comp_op
        return foast.Compare(
            op=visit(args[1]),
            left=visit(args[2], outer_loc),
            right=visit(args[3], outer_loc),
            location=outer_loc
        )
    else
        return foast.Call(
            func=visit(args[1], outer_loc),
            args=[visit(x, outer_loc) for x in Base.tail(Tuple(args)) if (typeof(x) != Expr || x.head != :(kw))],
            kwargs=Dict(x.args[1] => visit(x.args[2], outer_loc) for x in Base.tail(Tuple(args)) if (typeof(x) == Expr && x.head == :kw)),
            location=outer_loc,
        )
    end
end


function visit_(sym::Val{:block}, args::Array, outer_loc)
    
    return foast.BlockStmt(
        stmts=[visit(args[i+1], get_location(args[i])) for i in 2:2:(length(args)-1)],
        location=outer_loc
    )
end

# TODO ALLES
function visit_(sym::Val{:(::)}, args::Array, outer_loc)
    if typeof(args[1]) != Symbol
        throw("Left side of a passed argument must be a variable name")  #TODO throw correct error
    end

    try
        eval(args)
    catch
        throw("Type Error encountered") #TODO throw correct error
    end

    new_type = from_type_hint(args[2])

    if !py"isinstance"(new_type, ts.DataType)
        throw("Invalid Parameter Annotation Error")
    end

    return foast.DataSymbol(id=string(args[1]), location=outer_loc, type=new_type)

end

function visit_(sym::Val{:parameters}, args::Array, outer_loc)
    return [visit(expr, outer_loc) for expr in args]
end

function visit_(sym::Val{:comparison}, args::Array, outer_loc)
    return visit(unchain_comp(args), outer_loc)
end

function visit_(sym::Val{:(.)}, args::Array, outer_loc)
    return foast.Attribute(
        value=visit(args[1], outer_loc), attr=string(args[2]), location=outer_loc
    )
end

function visit_(sym::Val{:if}, args::Array, outer_loc)
    return foast.IfStmt(
        condition= args[1].head != :block ? visit(args[1], outer_loc) : visit(args[1].args[2], get_location(args[1].args[1])),       # in elseif the condition is always a block. gt4py does not accept blocks in the condition
        true_branch=visit(args[2], outer_loc),
        false_branch = length(args) == 3 ? visit(args[3], outer_loc) : foast.BlockStmt(stmts=[], location=outer_loc),
        location = outer_loc
    )
end

function visit_(sym::Val{:elseif}, args::Array, outer_loc)
    return foast.BlockStmt(
        stmts=[visit_(Val{:if}(), args, outer_loc)],
        location=outer_loc
    )
end

function visit_(sym::Val{:(=)}, args::Array, outer_loc)
    # TODO closure_var_symbols
    if typeof(args[1]) == Expr
        if args[1].head == :(call)
            return visit_(Val{:function}(), args, outer_loc)
        elseif args[1].head == :(tuple)
            new_targets = []
            for elt in args[1].args
                push!(new_targets, foast.DataSymbol(
                    id=visit(elt, outer_loc).id,
                    location=outer_loc,
                    type=ts.DeferredType(constraint=ts.DataType),
                ))
            end
            return foast.TupleTargetAssign(
                targets=new_targets, value=visit(args[2], outer_loc), location=outer_loc
            )
        end
    else
        new_value = visit(args[2], outer_loc)
        constraint_type = ts.DataType
        if py"isinstance"(new_value, foast.TupleExpr)
            constraint_type = ts.TupleType
        elseif type_info.is_concrete(new_value.type) && py"type_info.type_class($(new_value.type)) is ts.ScalarType"
            constraint_type = ts.ScalarType
        end

        return foast.Assign(
            target=foast.DataSymbol(
                id=string(args[1]),
                location=outer_loc,
                type=ts.DeferredType(constraint=constraint_type),
            ),
            value=new_value,
            location=outer_loc,
        )
    end
end

function visit_(sym::Val{:(+=)}, args::Array, outer_loc)
    return visit(:($(args[1]) = $(args[1]) + $(args[2])), outer_loc)
end

function visit_(sym::Val{:(-=)}, args::Array, outer_loc)
    return visit(:($(args[1]) = $(args[1]) - $(args[2])), outer_loc)
end

function visit_(sym::Val{:(*=)}, args::Array, outer_loc)
    return visit(:($(args[1]) = $(args[1]) * $(args[2])), outer_loc)
end

function visit_(sym::Val{:(/=)}, args::Array, outer_loc)
    return visit(:($(args[1]) = $(args[1]) / $(args[2])), outer_loc)
end

function visit_(sym::Val{:tuple}, args::Array, outer_loc)
    return foast.TupleExpr(
        elts=[visit(expr, outer_loc) for expr in args],
        location=outer_loc
    )
end

function visit_(sym::Val{:&&}, args::Array, outer_loc)
    return foast.BinOp(
            op=visit(:(&)),
            left=visit(args[1], outer_loc),
            right=visit(args[2], outer_loc),
            location=outer_loc
        )
end

function visit_(sym::Val{:||}, args::Array, outer_loc)
    return foast.BinOp(
            op=visit(:(|)),
            left=visit(args[1], outer_loc),
            right=visit(args[2], outer_loc),
            location=outer_loc
        )
end

function visit_(sym::Val{:return}, args::Array, outer_loc)
    if isnothing(args[1])
        throw("Must return a value and not nothing at $outer_loc")
    end
    return foast.Return(
        value=visit(args[1], outer_loc),
        location=outer_loc
    )
end

function visit_(sym::Val{:ref}, args::Array, outer_loc)
    if typeof(args[2]) <: Integer
        return foast.Subscript(
            value=visit(args[1], outer_loc),
            index=args[2]-1, # Due to different indexing in python
            location=outer_loc,
        )
    else
        throw("Expected an integer index, got $(args[2])")
    end
end

function visit_(sym::Val{:for}, args::Array, outer_loc)
    throw("For-loops are not supported. For-loop encountered at $outer_loc")
end

# ----------------------------------------------------------------------------------------------------------------------------------
# Helper functions

function unchain_comp(args::Array)
    if length(args) == 3
        return Expr(:call, args[2], args[1], args[3])  # Alternative syntax: :($(args[2])($(args[1]), $(args[3])))
    else
        return Expr(:&&, Expr(:call, args[2], args[1], args[3]), unchain_comp(args[3:end]))
    end
end

function get_location(linenuno::LineNumberNode)
    return SourceLocation(string(linenuno.file), linenuno.line, 1, end_line=py"None", end_column=py"None")
end


function from_type_hint(sym::Symbol)
    if typeof(eval(sym)) == DataType
        try
            return ts.ScalarType(scalar_types[sym])
        catch
            throw("Non-trivial dtypes like $(sym) are not yet supported")
        end
    else
        throw("Feature not supported by gt4py. Needs parametric type specification")
    end
end

function from_type_hint(expr::Expr)
    @assert expr.head == :curly
    param_type = expr.args
    if param_type[1] == :Tuple
        return ts.TupleType(types=[recursive_make_symbol(arg) for arg in Base.tail(param_type)])  #TODO was macht das recursive_make_symbol?
    elseif param_type[1] == :Field
        @assert length(param_type) == 4 ("Field type requires three arguments, got $(length(param_type)-1) in $(param_type)")
        
        dim = []
        (dtype, ndims, dims) = param_type[2:end]

        for d in dims.args[2:end]
            if eval(d) <: Dimension{<:Any, HORIZONTAL}
                kind = gtx.common.DimensionKind."HORIZONTAL"
            elseif eval(d) <: Dimension{<:Any, VERTICAL}
                kind = gtx.common.DimensionKind."VERTICAL"
            else
                kind = gtx.common.DimensionKind."LOCAL"
            end
            push!(dim, gtx.common.Dimension(string(d), kind=kind))
        end
        @bp

        return ts.FieldType(dims=dim, dtype=scalar_types[dtype]) 
    end
end