

function single_assign_target_pass(expr::Expr)::Expr
    return postwalk(expr) do x
        @capture(x, (t1_, t2_ = val1_, val2_) | ((t1_, t2_) = (val1_, val2_)) | (t1_, t2_ = (val1_, val2_)) | ((t1_, t2_) = val1_, val2_)) || return x
        return :($t1 = $val1; $t2 = $val2)
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
        return Expr(:call, args[2], args[1], args[3])  # Alternative syntax: :($(args[2])($(args[1]), $(args[3])))
    else
        return Expr(:&&, Expr(:call, args[2], args[1], args[3]), recursive_unchain(args[3:end]))
    end
end

function get_annotation(expr::Expr)::Dict
    out_ann = Dict()

    if expr.args[1].head == :(::)
        return_type = from_type_hint(expr.args[1].args[2])
        out_ann["return"] = return_type
    end
    return out_ann
end

function get_closure_vars(expr::Expr)::Dict
    j_closure_vars = get_j_cvars(expr)
    return translate_cvars(j_closure_vars)
end

function get_j_cvars(expr::Expr)::Dict

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
    prewalk(expr.args[2]) do x
        if @capture(x, name_(args__)) && !(name in local_vars) && !(name in math_ops)
            push!(closure_names, name)
            return Expr(:irgendoeppis, args...)                                         # small present for tehrengruber
        elseif typeof(x) == Symbol && !(x in local_vars) && !(x in math_ops)
            push!(closure_names, x)
            return x
        else 
            return x
        end
    end

    # update dictionary
    for name in closure_names
        closure_vars[name] = Core.eval(CURRENT_MODULE, name)
    end

    return closure_vars 
end

function translate_cvars(j_closure_vars::Dict)::Dict
    py_cvars = Dict()

    for (key, value) in j_closure_vars
        new_value = nothing
        if typeof(value) == FieldOffset
            py_source = map(dim -> gtx.Dimension(string(get_dim_name(dim))[1:end-1], kind=py_dim_kind[get_dim_kind(dim)]), value.source)
            py_target = map(dim -> gtx.Dimension(string(get_dim_name(dim))[1:end-1], kind=py_dim_kind[get_dim_kind(dim)]), value.target)
            new_value = gtx.FieldOffset(
                value.name, 
                source= length(py_source) == 1 ? py_source[1] : py_source, 
                target= length(py_target) == 1 ? py_target[1] : py_target
            )
        elseif typeof(value) <: Dimension
            new_value = gtx.Dimension(string(get_dim_name(value))[1:end-1], kind=py_dim_kind[get_dim_kind(value)])

        elseif typeof(value) <: Function
            if key in keys(builtin_op)
                new_value = builtin_op[key]
            elseif key in keys(GridTools.py_field_ops)
                new_value = GridTools.py_field_ops[key]
            end
        elseif typeof(value) <: DataType      # Not necessary... Filter out DataTypes in the prewalk
            key = lowercase(string(key))
            new_value = py_scalar_types[value]
        elseif isconst(CURRENT_MODULE, Symbol(value))
            # TODO create FrozenNameSpace...
            new_value = "Constant"
        else 
            throw("Access to following type: $(typeof(value)) is not permitted within a field operator!")
        end
        py_cvars[string(key)] = new_value
    end
    return py_cvars
end