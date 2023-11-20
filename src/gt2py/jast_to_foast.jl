
# TODO: place in submodule
# TODO add forbidden Symbols to a list where an error is thrown when used: :hcat, :vcat, etc

# First call, takes closure_vars and passes it to the function instance
function visit_jast(expr::Expr, closure_vars::Dict)
    return visit_function(expr.args, closure_vars)
end

function _builtin_type_constructor_symbols(captured_vars, loc)::Tuple
    result = []
    skipped_types = Set(["tuple"])
    python_type_builtins = Dict(
        name => py"getattr"(builtins, name)
        for name in setdiff(Set(fbuiltins.TYPE_BUILTIN_NAMES), skipped_types)
        if py"hasattr"(builtins, name)
    )

    captured_type_builtins = Dict(
            name => value
            for (name, value) in captured_vars
            if name in fbuiltins.TYPE_BUILTIN_NAMES && value == py"getattr"(fbuiltins, name)        # TODO originally was === . Doesnt work with pyobjects. Verify
    )

    to_be_inserted = union(python_type_builtins, captured_type_builtins)

    for (name, value) in to_be_inserted
        push!(result, 
            foast.Symbol(
                id=name, 
                type=ts.FunctionType(
                    pos_only_args=[ts.DeferredType(constraint=ts.ScalarType)],
                    pos_or_kw_args=Dict(),
                    kw_only_args=Dict(),
                    returns=cast(ts.DataType, type_translation.from_type_hint(value))
                ),
                namespace=dialect_ast_enums.Namespace."CLOSURE",
                location=loc
            )
        )
    end
    return result, keys(to_be_inserted)
end

function visit_function(args::Array, closure_vars::Dict)
    inner_loc = get_location(args[2].args[1])
    args[2].args = args[2].args[2:end]
    closure_var_symbols, skip_names = _builtin_type_constructor_symbols(closure_vars, inner_loc)

    for name in keys(closure_vars)
        if name in skip_names
            continue
        end

        push!(closure_var_symbols, 
            foast.Symbol(
                id=name,
                type=ts.DeferredType(constraint=py"None"o),
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

    for param in Base.tail((Tuple(function_header.args)))
        function_params = vcat(function_params, visit_types(param.args, closure_vars, inner_loc))
    end
    
    return foast.FunctionDefinition(
        id=string(function_header.args[1]),
        params= function_params,
        body=function_body,
        closure_vars=closure_var_symbols,
        location=inner_loc
    )
end

function visit_types(args::Array, closure_vars::Dict, outer_loc)
    if typeof(args[1]) != Symbol
        throw("Left side of a passed argument must be a variable name.")
    end

    new_type = from_type_hint(args[2], closure_vars)

    if !py"isinstance"(new_type, ts.DataType)
        throw("Invalid Parameter Annotation Error.")
    end

    return foast.DataSymbol(id=string(args[1]), location=outer_loc, type=new_type)

end



function visit(expr::Expr, loc=nothing)
   return visit_(Val{expr.head}(), expr.args, loc)
end

function visit(sym::Symbol, loc)
    if sym in keys(scalar_types)
        return foast.Name(id=lowercase(string(sym)), location=loc)
    else
        return foast.Name(id=string(sym), location=loc)
    end
end

# Called when Symbol is a Unary, Binary or Compare Operation
function visit(sym::Symbol)
    return visit_(Val{sym}())
end

# Catches Integers, Floats, Bool, Strings, etc. otherwise throws error
function visit(constant::Any, loc)
    try
        type_ = type_translation.from_value(constant)
        type_ = type_ == ts.ScalarType(kind=ts.ScalarKind."INT32") ? ts.ScalarType(kind=ts.ScalarKind."INT64") : type_

        return foast.Constant(
        value=constant,
        location=loc,
        type=type_,
        )
    catch e
        throw("Constants of type $(typeof(constant)) are not permitted.")
    end
end

function visit_(sym::Val{:parameters}, args::Array, outer_loc)
    return [visit(expr, outer_loc) for expr in args]
end

function visit_(sym::Val{:block}, args::Array, outer_loc)

    # TODO Due to bug in gt4py where a nested BlockStmt creates new local scope. Can skip for-loop if fixed
    stmts = [visit(args[i+1], get_location(args[i])) for i in 1:2:(length(args)-1)]
    new_stmts = []    

    for arg in stmts
        if py"isinstance"(arg, foast.BlockStmt)
            append!(new_stmts, arg.stmts)
        else
            push!(new_stmts, arg)
        end
    end
    
    return foast.BlockStmt(
        stmts=new_stmts,
        location=outer_loc
    )
end

function visit_(sym::Val{:if}, args::Array, outer_loc)
    
    if is_ternary_stmt(args)
        return foast.TernaryExpr(
            condition=visit(args[1], outer_loc),
            true_expr=visit(args[2], outer_loc),
            false_expr=visit(args[3], outer_loc),
            location=outer_loc,
            type=ts.DeferredType(constraint=ts.DataType),
        )
    else
        # condition is not allowed to be a BlockStmt. elseif condition is always a BlockStmt.
        if typeof(args[1]) == Expr && args[1].head == :block
            condition = visit(args[1].args[2], get_location(args[1].args[1]))
        else
            condition = visit(args[1], outer_loc)
        end

        return foast.IfStmt(
            condition = condition,
            true_branch=visit(args[2], outer_loc),
            false_branch = length(args) == 3 ? visit(args[3], outer_loc) : foast.BlockStmt(stmts=[], location=outer_loc),
            location = outer_loc
        )
    end
end

function visit_(sym::Val{:elseif}, args::Array, outer_loc)
    return foast.BlockStmt(
        stmts=[visit_(Val{:if}(), args, outer_loc)],
        location=outer_loc
    )
end

function visit_(sym::Val{:(=)}, args::Array, outer_loc)
    if typeof(args[1]) == Expr
        if args[1].head == :call
            return visit_(Val{:function}(), args, outer_loc)
        elseif args[1].head == :tuple
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
        elseif args[1].head == :(::)
            throw("Annotated assigns are not yet supported in gt4py.")
        end
    else
        new_value = visit(args[2], outer_loc)
        constraint_type = ts.DataType
        if py"isinstance"(new_value, foast.TupleExpr)
            constraint_type = ts.TupleType
        elseif type_info.is_concrete(new_value.type) && type_info.type_class(new_value.type) == ts.ScalarType   # TODO originally was === . Doesnt work with pyobjects. Verify
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

function visit_(sym::Val{:tuple}, args::Array, outer_loc)
    return foast.TupleExpr(
        elts=[visit(expr, outer_loc) for expr in args],
        location=outer_loc
    )
end

function visit_(sym::Val{:.}, args::Array, outer_loc)
    if typeof(args[2]) == QuoteNode     # TODO Frozen namespace  # return foast.Attribute(value=visit(args[1], outer_loc), attr=string(args[2]), location=outer_loc)
        throw("Access to object attributes is not permitted within a field_operator.")
    elseif typeof(args[2]) == Expr  # we have a function broadcast aka sin.(field)
        # arguments to (.)call are wrapped in a tuple expression
        func_args = args[2].args    
        pop!(args)
        append!(args, func_args)      
        return visit_(Val{:call}(), args, outer_loc)
    else
        throw("We shouldn't land here... Report.") # TODO verify
    end
end

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
    elseif args[1] in disallowed_op
        throw("The function $(args[1]) is currently not supported by gt4py.")
    elseif args[1] == :slice  # TODO: What to do here? Current: pass unsliced field
        return visit(args[2], outer_loc)
    else
        if args[1] == :astype args[2], args[3] = args[3], args[2] end
        return foast.Call(
            func=visit(args[1], outer_loc),
            args=[visit(x, outer_loc) for x in Base.tail(Tuple(args)) if (typeof(x) != Expr || x.head != :(kw))],
            kwargs=Dict(x.args[1] => visit(x.args[2], outer_loc) for x in Base.tail(Tuple(args)) if (typeof(x) == Expr && x.head == :kw)),
            location=outer_loc,
        )
    end
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

function visit_(sym::Val{:ref}, args::Array, outer_loc)
    if typeof(args[2]) <: Integer
        return foast.Subscript(
            value=visit(args[1], outer_loc),
            index=args[2]-1, # Due to different indexing in python
            location=outer_loc,
        )
    else
        throw("Expected an integer index, got $(args[2]).")
    end
end

function visit_(sym::Val{:return}, args::Array, outer_loc)
    if isnothing(args[1])
        throw("Must return a value and not nothing at $outer_loc.")
    end
    return foast.Return(
        value=visit(args[1], outer_loc),
        location=outer_loc
    )
end

function visit_(sym::Val{:comparison}, args::Array, outer_loc)
    throw("All compairs should have been eliminated in the preprocessing step. Report.")
end

function visit_(sym::Union{Val{:&&},Val{:.&&}}, args::Array, outer_loc)
    return foast.BinOp(
            op=visit(:(&)),
            left=visit(args[1], outer_loc),
            right=visit(args[2], outer_loc),
            location=outer_loc
        )
end

function visit_(sym::Union{Val{:||}, Val{:.||}}, args::Array, outer_loc)
    return foast.BinOp(
            op=visit(:(|)),
            left=visit(args[1], outer_loc),
            right=visit(args[2], outer_loc),
            location=outer_loc
        )
end

function visit_(sym::Val{:for}, args::Array, outer_loc)
    throw("For-loops are not supported. For-loop encountered at $outer_loc.")
end

# ----------------------------------------------------------------------------------------------------------------------------------
# Helper functions

function is_ternary_stmt(args::Array)
    if length(args) == 3
        if typeof(args[2]) == Expr && typeof(args[3]) == Expr
            if args[2].head == :block || args[3].head == :block
                return false
            else
                return true
            end
        else
            return true
        end
    else
        return false
    end
end

function get_location(linenuno::LineNumberNode)
    return SourceLocation(string(linenuno.file), linenuno.line, 1, end_line=py"None", end_column=py"None")
end

function from_type_hint(sym::Symbol, closure_vars::Dict)
    if sym == :Dimension || sym == :FieldOffset || sym == :Connectivity
        throw("$(string(sym)) is not a valid function argument")
    elseif sym == :Field
        throw("Feature not supported by gt4py. Needs parametric type specification. ")
    else
        if sym in keys(scalar_types)
            return ts.ScalarType(kind=scalar_types[sym])
        else
            throw("Non-trivial dtypes like $(sym) are not yet supported.")
        end
    end
end

function from_type_hint(expr::Expr, closure_vars::Dict)
    @assert expr.head == :curly
    param_type = expr.args
    if param_type[1] == :Tuple
        return ts.TupleType(types=[recursive_make_symbol(arg) for arg in Base.tail(param_type)])
    elseif param_type[1] == :Field
        @assert length(param_type) == 4 ("Field type requires three arguments, got $(length(param_type)-1) in $(param_type).")
        
        dim = []
        (dtype, ndims, dims) = param_type[2:end]

        for d in dims.args[2:end]
            @assert string(d) in keys(closure_vars)
            push!(dim, closure_vars[string(d)])
        end

        return ts.FieldType(dims=dim, dtype=ts.ScalarType(kind=scalar_types[dtype])) 
    else
        throw("The following kind of function argument is not yet supported: $param_type")
    end

end