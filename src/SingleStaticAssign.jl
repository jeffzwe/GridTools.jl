
ASSIGNMENT_TRACKER = Dict{Symbol, Integer}()

function single_static_assign_pass(expr::Expr)
    try
        return s_visit(expr)
    finally
        global ASSIGNMENT_TRACKER = Dict{Symbol, Integer}()
    end
end

function s_visit(expr::Expr, assignment_tracker::Dict{Symbol, Integer} = ASSIGNMENT_TRACKER)
    return s_visit(Val{expr.head}(), expr.args, assignment_tracker)
end

function s_visit(sym::Symbol, assignment_tracker::Dict{Symbol, Integer} = ASSIGNMENT_TRACKER)
    return sym in keys(assignment_tracker) ? generate_unique_name(sym, assignment_tracker[sym]) : sym
end

function s_visit(constant::Any, assignment_tracker::Dict{Symbol, Integer})
    return constant
end

function s_visit(linuno::LineNumberNode, assignment_tracker::Dict{Symbol, Integer})
    return linuno
end

function s_visit(sym::Val{:function}, args::Array, assignment_tracker::Dict{Symbol, Integer})
    args[2] = s_visit(args[2], assignment_tracker)
    return Expr(typeof(sym).parameters[1], args...)
end

function s_visit(sym::Val{:(=)}, args::Array, assignment_tracker::Dict{Symbol, Integer})
    
    args[2] = s_visit(args[2], assignment_tracker)

    if args[1] in keys(assignment_tracker)
        assignment_tracker[args[1]] = assignment_tracker[args[1]]+1
    else
        assignment_tracker[args[1]] = 0
    end

    args[1] = generate_unique_name(args[1], assignment_tracker[args[1]])
    return Expr(typeof(sym).parameters[1], args...)
end

function s_visit(sym::Any, args::Array, assignment_tracker::Dict{Symbol, Integer})
    return Expr(typeof(sym).parameters[1], map(x -> s_visit(x, assignment_tracker), args)...)
end

function s_visit(sym::Union{Val{:if}, Val{:elseif}}, args::Array, assignment_tracker::Dict{Symbol, Integer})

    @bp
    
    args[1] = s_visit(args[1], assignment_tracker)
    assignment_tracker_true = copy(assignment_tracker)
    assignment_tracker_false = copy(assignment_tracker)

    length(args) == 2 ? push!(args, Expr(:block)) : nothing

    args[2] = s_visit(args[2], assignment_tracker_true)
    args[3] = s_visit(args[3], assignment_tracker_false)

    new_args = combine_variable_states(args[2], args[3], assignment_tracker_true, assignment_tracker_false)
    args[2] = new_args[1]
    args[3] = new_args[2]
    
    merge!(assignment_tracker, new_args[3])

    return Expr(typeof(sym).parameters[1], args...)
end

# TODO handle cases where if or else are not Expressions aka TernaryExpr
function combine_variable_states(true_expr::Union{Expr, Symbol}, false_expr::Union{Expr, Symbol}, assignment_tracker_true::Dict{Symbol, Integer}, assignment_tracker_false::Dict{Symbol, Integer})
    if assignment_tracker_true == assignment_tracker_false
        return true_expr, false_expr, assignment_tracker_true
    end

    assignment_tracker_comb = Dict{Symbol, Integer}()
    switch = false

    for key in union(keys(assignment_tracker_true), keys(assignment_tracker_false))
        if key in keys(assignment_tracker_true) && key in keys(assignment_tracker_false)
            if assignment_tracker_true[key] != assignment_tracker_false[key]
                if assignment_tracker_true[key] < assignment_tracker_false[key]
                    true_expr, false_expr, assignment_tracker_true, assignment_tracker_false = false_expr, true_expr, assignment_tracker_false, assignment_tracker_true
                    switch = !switch
                end
                # True branch is the up-to-date branch
                assignment_tracker_comb[key] = assignment_tracker_true[key]
                location, value = generate_unique_name(key, assignment_tracker_true[key]), generate_unique_name(key, assignment_tracker_false[key])

                false_expr = update_false_expr(false_expr, location, value)
            end
        end
    end

    return switch ? (false_expr, true_expr, assignment_tracker_comb) : (true_expr, false_expr, assignment_tracker_comb)
end


function update_false_expr(false_expr::Expr, location::Symbol, value::Symbol)
    if false_expr.head == :block
        return Expr(:block, [false_expr.args..., :($location = $value)]...)
    elseif false_expr.head == :elseif 
        return Expr(:elseif, false_expr.args[1], map(x -> update_false_expr(x, location, value), false_expr.args[2:end])...)
    else
        return Expr(:block, [false_expr, :($location = $value)]...)
    end
end

update_false_expr(false_expr::Symbol, location::Symbol, value::Symbol) = Expr(:block, [false_expr, :($location = $value)]...)


function generate_unique_name(name::Symbol, value::Integer)
    return Symbol("$(name)ᐞ$(value)")
end




