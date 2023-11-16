# TODO: write docstring with small doctest example (if easily possible)

function single_assign_target_pass(expr::Expr)::Expr
    return postwalk(expr) do x
        @capture(x, (t1_, t2_ = val1_, val2_) | ((t1_, t2_) = (val1_, val2_)) | (t1_, t2_ = (val1_, val2_)) | ((t1_, t2_) = val1_, val2_)) || return x
        return :(begin $t1 = $val1; $t2 = $val2 end)
    end
end

function unchain_compairs_pass(expr::Expr)::Expr
    return postwalk(expr) do x
        if typeof(x) == Expr && x.head == :comparison
            return recursive_unchain(x.args)
        else
            return x
        end
    end
end

function recursive_unchain(args::Array)::Expr
    if length(args) == 3
        return Expr(:call, args[2], args[1], args[3])
    else
        return Expr(:&&, Expr(:call, args[2], args[1], args[3]), recursive_unchain(args[3:end]))
    end
end

function get_annotation(expr::Expr, closure_vars::Dict)::Dict
    out_ann = Dict()

    if expr.args[1].head == :(::)
        return_type = from_type_hint(expr.args[1].args[2], closure_vars)
        out_ann["return"] = return_type
    end
    return out_ann
end

function translate_closure_vars(j_closure_vars::Dict)::Dict
    py_closure_vars = Dict()

    for (key, value) in j_closure_vars
        new_value = nothing

        if typeof(value) <: FieldOffset
            py_source = map(dim -> gtx.Dimension(get_dim_name(dim), kind=py_dim_kind[get_dim_kind(dim)]), value.source)
            py_target = map(dim -> gtx.Dimension(get_dim_name(dim), kind=py_dim_kind[get_dim_kind(dim)]), value.target)
            new_value = gtx.FieldOffset(
                value.name, 
                source= length(py_source) == 1 ? py_source[1] : py_source, 
                target= length(py_target) == 1 ? py_target[1] : py_target
            )
        elseif typeof(value) <: Function
            new_value = builtin_py_op[key]

        elseif typeof(value) <: FieldOp 
            new_value = py_field_operator(value)
        elseif typeof(value) <: Dimension
            new_value = gtx.Dimension(get_dim_name(value), kind=py_dim_kind[get_dim_kind(value)])
        elseif typeof(value) <: DataType
            if value <: Dimension
                new_value = gtx.Dimension(string(value.parameters[1])[1:end-1], kind=py_dim_kind[value.parameters[2]])
            else
                key = lowercase(string(key))
                new_value = py_scalar_types[value]
            end
        elseif false #isconst(CURRENT_MODULE, Symbol(value))  
            # TODO handle constants with FrozenNameSpace...
        else
            throw("Access to following type: $(typeof(value)) is not permitted within a field operator.")
        end
        py_closure_vars[string(key)] = new_value
    end
    return py_closure_vars
end