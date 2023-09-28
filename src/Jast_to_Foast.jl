ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3.10")

using PyCall

foast = pyimport("gt4py.next.ffront.field_operator_ast")

type_system = pyimport("gt4py.next.type_system")
type_info = type_system.type_info
ts = type_system.type_specifications
type_translation = type_system.type_translation

concepts = pyimport("gt4py.eve.concepts")
SourceLocation = concepts.SourceLocation

common = pyimport("gt4py.next.common")


function visit(sym::Symbol, loc=nothing)
    return foast.Name(id=string(sym), location=loc)
end

function visit(expr::Expr, loc=nothing)
   return visit_(Val{$(expr.head)}(), expr.args, loc)
end

function visit(constant::Any, loc=nothing)
    try
        type_ = type_translation.from_value(constant)
    catch
        throw("Type not okay") # TODO what error should be thrown here
    end

    return foast.Constant(
        value=constant,
        location=loc,
        type=type_,
    )
end

function visit_(sym::Val{:function}, args::Array, outer_loc=nothing)
    inner_loc = get_location(args[2].args[1])
    closure_var_symbols, skip_names = #TODO

    function_header = args[1]
    function_body = visit(args[2], inner_loc)

    return foast.FunctionDefinition(
        id=string(function_header.args[1]),
        params= Iterators.flatten(map(expr -> visit(expr, inner_loc), Base.tail(function_header.args))),
        body=function_body,
        closure_vars=closure_var_symbols,
        location=inner_loc
    )
end

bin_op = Set([:(+), :(-), :(*), :(/), :(÷), :(^), :(%), :(&), :(|), :(⊻)])
unary_op = Set([:(!), :(~)])
comp_op = Set([:(==), :(!=), :(<), :(<=), :(>), :(>=)])

ops = Dict{Symbol, String}(
    :+ => "plus",
    :- => "minus",
    :* => "multiplies",
    :/ => "divides",
    :÷ => "floordiv",
    :^ => "power",
    :% => "mod",
    :& => "and_",
    :| => "or_",
    :⊻ => "xor_",
    :! => "not_",
    :~ => "invert",
    :(==) => "eq",
    :!= => "not_eq",
    :< => "less",
    :<= => "less_equal",
    :> => "greater",
    :>= => "greater_equal",

    :.+ => "plus",
    :.- => "minus",
    :.* => "multiplies",
    :./ => "divides",
    :.÷ => "floordiv",
    :.^ => "power",
    :.% => "mod",
    :.& => "and_",
    :.| => "or_",
    :.⊻ => "xor_",
    :.! => "not_",
    :.~ => "invert",
    :.== => "eq",
    :.!= => "not_eq",
    :.< => "less",
    :.<= => "less_equal",
    :.> => "greater",
    :.>= => "greater_equal"
)


function visit_(sym::Val{:call}, args::Array, outer_loc=nothing)
    if args[1] in bin_op
        return foast.BinOp(
            op=ops[args[1]],
            left=visit(args[2]),
            right=visit(args[3]),
            location=outer_loc
        )
    elseif args[1] in unary_op
        return foast.UnaryOp(
            op=ops[args[1]],
            operand=visit(args[2]),
            location=outer_loc
        )
    elseif args[1] in comp_op
        return foast.Compare(
            op=ops[args[1]],
            left=visit(args[2]),
            right=visit(args[3]),
            location=outer_loc
        )
    else
        arguments = 
        return foast.Call(
            func=args[1]
            args=[visit(x) for x in Base.tail(args) if (typeof(x) != Expr || x.head != :(kw))],
            kwargs=Dict(x.args[1] => visit(x.args[2]) for x in Base.tail(args) if (typeof(x) == Expr && x.head == :(kw))),
            location=self.get_location(node),
        )
    end
end

# visit_(sym::Val{:(+)}, outer_loc) = "plus"
# visit_(sym::Val{:(-)}, outer_loc) = "minus"
# visit_(sym::Val{:(*)}, outer_loc) = "multiplies"
# visit_(sym::Val{:(/)}, outer_loc) = "divides"
# visit_(sym::Val{:(÷)}, outer_loc) = "floordiv"
# visit_(sym::Val{:(^)}, outer_loc) = "power"
# visit_(sym::Val{:(%)}, outer_loc) = "mod"
# visit_(sym::Val{:(&)}, outer_loc) = "and_"
# visit_(sym::Val{:(|)}, outer_loc) = "or_"
# visit_(sym::Val{:(⊻)}, outer_loc) = "xor_"

# visit_(sym::Val{:(!)}, outer_loc) = "not_"
# visit_(sym::Val{:(~)}, outer_loc) = "invert"

# visit_(sym::Val{:(==)}, outer_loc) = "eq"
# visit_(sym::Val{:(!=)}, outer_loc) = "not_eq"
# visit_(sym::Val{:(<)}, outer_loc) = "less"
# visit_(sym::Val{:(<=)}, outer_loc) = "less_equal"
# visit_(sym::Val{:(>)}, outer_loc) = "greater"
# visit_(sym::Val{:(>=)}, outer_loc) = "greater_equal"

# for sym, gtsym in ...
#     @eval visit_(sym::Val{:($sym)}, outer_loc) = $gtsym
# end


function visit_(sym::Val{:block}, args::Array, outer_loc=nothing)
    return foast.BlockStmt(
        stmts=[visit(args[i+1], get_location(args[i])) for i in 2:2:length(args)],
        location=outer_loc
    )
end

scalar_types = Dict(
    :Bool => ts.ScalarKind.BOOL,
    :Int32 => ts.ScalarKind.INT32,
    :Int64 => ts.ScalarKind.INT64,
    :Integer => ts.ScalarKind.INT64,
    :Float32 => ts.ScalarKind.FLOAT32,
    :Float64 => ts.ScalarKind.FLOAT64,
    :AbstractFloat => ts.ScalarKind.FLOAT64,
    :String => ts.ScalarKind.STRING
)

# TODO ALLES
function visit_(sym::Val{:(::)}, args::Array, loc=nothing)
    if typeof(args[1]) != Symbol
        throw("Left side of a passed argument must be a variable name")  #TODO throw correct error
    end

    try
        eval(args)
    catch
        throw("Type Error encountered") #TODO throw correct error
    end

    if typeof(args[2]) == Expr # Parametric Type
        param_type = args[2]
        if param_type[1] == :Tuple
            return ts.TupleType(types=[recursive_make_symbol(arg) for arg in Base.tail(param_type)])  #TODO was macht das recursive_make_symbol?
        elseif param_type[1] == :Field
            @assert length(param_type) == 4 ("Field type requires three arguments, got $(length(param_type)-1) in $(param_type)")
            
            dim = [] # actually Vector(Union{Ellipsis, Dimension})() aber mit python objects
            (dtype, ndims, dim_args) = Base.tail(param_type)

            if typeof(dim_args) == Tuple
                for d in dim_args
                    push!(dim, common.Dimension(string(d), kind=eval(d).kind.value)) #TODO this works if we only need to pass a string to kind...
                    #TODO Otherwise we need this...

                    if eval(d).kind.value == "horizontal"
                        kind = common.DimensionKind.HORIZONTAL
                    elseif eval(d).kind.value == "vertical"
                        kind = common.DimensionKind.VERTICAL
                    else
                        kind = common.DimensionKind.LOCAL
                    end
                    push!(dim, common.Dimension(string(d), kind=kind))
                end
            else typeof(dim_args) == Ellipsis
                push!() #TODO create Python Ellipsis and push to dim
            end

            return ts.FieldType(dims=dim, dtype=dtype) 
        end
    else
        if typeof(eval(args[2])) == DataType
            try
                return ts.ScalarType(scalar_types[args[2]])
            catch
                throw("Non-trivial dtypes like $(args[2]) are not yet supported")
            end
        else
            throw("Feature not supported by gt4py. Needs parametric type specification")
        end
    end
end

function visit_(sym::Val{:parameters}, args::Array, outer_loc=nothing)
    return [visit(expr, outer_loc) for expr in args]
end

function visit_(sym::Val{:comparison}, args::Array, outer_loc=nothing)
    return visit(unchain_comp(args), outer_loc)
end

function visit_(sym::Val{:if}, args::Array, outer_loc=nothing)
    return foast.IfStmt(
        condition=visit(args[1], outer_loc),
        true_branch=visit(args[2], outer_loc),
        false_branch = length(args) == 3 ? visit(args[3], outer_loc) : foast.BlockStmt(stmts=[], location=outer_loc),
        location = outer_loc
    )
end

function visit_(sym::Val{:(=)}, args::Array, outer_loc=nothing)
    # TODO closure_var_symbols
    if args[1].head == :(call)
        function_header = args[1]
        function_body = visit_(:(return), args[2], outer_loc)

        return foast.FunctionDefinition(
        id=string(function_header.args[1]),
        params= Iterators.flatten(map(expr -> visit(expr, outer_loc), Base.tail(function_header.args))),
        body=function_body,
        closure_vars=closure_var_symbols,
        location=outer_loc
    )
    elseif args[1].head == :(tuple)
        new_targets = []
        for elt in args[1].args
            push!(new_targets, foast.DataSymbol(
                id=visit(elt).id,
                location=outer_loc,
                type=ts.DeferredType(constraint=ts.DataType),
            ))
        end
        return foast.TupleTargetAssign(
            targets=new_targets, value=visit(args[2]), location=outer_loc
        )
    else
        new_value = visit(args[2])
        constraint_type = ts.DataType
        if py"isinstance"(new_value, foast.TupleExpr)
            constraint_type = ts.TupleType
        elseif type_info.is_concrete(new_value.type) && py"(type_info.type_class(new_value.type) is ts.ScalarType"
            constraint_type = ts.ScalarType
        end

        return foast.Assign(
            target=foast.DataSymbol(
                id=visit(args[1]),
                location=outer_loc,
                type=ts.DeferredType(constraint=constraint_type),
            ),
            value=new_value,
            location=outer_loc,
        )
end

function visit_(sym::Val{:(+=)}, args::Array, outer_loc=nothing)
    return visit(:($(args[1]) = $(args[1]) + $(args[2])), outer_loc)
end

function visit_(sym::Val{:(-=)}, args::Array, outer_loc=nothing)
    return visit(:($(args[1]) = $(args[1]) - $(args[2])), outer_loc)
end

function visit_(sym::Val{:(*=)}, args::Array, outer_loc=nothing)
    return visit(:($(args[1]) = $(args[1]) * $(args[2])), outer_loc)
end

function visit_(sym::Val{:(/=)}, args::Array, outer_loc=nothing)
    return visit(:($(args[1]) = $(args[1]) / $(args[2])), outer_loc)
end

function visit_(sym::Val{:tuple}, args::Array, outer_loc=nothing)
    return foast.TupleExpr(
        elts=[visit(expr, outer_loc) for expr in args],
        location=outer_loc
    )
end

function visit_(sym::Val{:&&}, args::Array, outer_loc=nothing)
    return foast.BinOp(
            op=visit(:(&))
            left=visit(args[2])
            right=visit(args[3])
            location=outer_loc
        )
end

function visit_(sym::Val{:||}, args::Array, outer_loc=nothing)
    return foast.BinOp(
            op=visit(:(|))
            left=visit(args[2])
            right=visit(args[3])
            location=outer_loc
        )
end

function visit_(sym::Val{:return}, args::Array, outer_loc=nothing)
    if isnothing(args[1])
        throw("Must return a value and not nothing at $outer_loc")
    end
    return foast.Return(
        value=visit(args[1], outer_loc),
        location=outer_loc
    )
end

function visit_(sym::Val{:ref}, args::Array, outer_loc=nothing)
    if typeof(args[2]) <: Integer
        return foast.Subscript(
            value=visit(args[1]),
            index=args[2],
            location=outer_loc,
        )
    else
        throw("Expected an integer index, got $(args[2])")
    end
end

function visit_(sym::Val{:for}, args::Array, outer_loc=nothing)
    throw("For loops are not supported. For loop encountered at $outer_loc")
end



#TODO maybe? vect, hcat, vcat, x.dims. 



function unchain_comp(args::Array)
    if length(args) == 3
        return Expr(:call, args[2], args[1], args[3])  # Alternative syntax: :($(args[2])($(args[1]), $(args[3])))
    else
        return Expr(:&&, Expr(:call, args[2], args[1], args[3]), unchain_comp(args[3:end]))
    end
end

function translate_to_gt4py(args::Array)  #TODO how to get current Module? glaub ned mal noetig...   ; module_ = MyCurrentModule
    try
        eval(expr)
    catch
        throw("Type Error encountered")
    end

end

function get_location(linenuno::LineNumberNode)
    return SourceLocation(file=string(linenuno.file), line=linenuno.line, column=py"None", end_line=py"None", end_colum=py"None")
end

