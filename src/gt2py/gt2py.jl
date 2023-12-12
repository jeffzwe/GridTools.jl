# TODO: place in submodule

# Imports ---------------------------------------------------------------------------------
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3.10")
# ENV["PYTHONBREAKPOINT"] = "pdb.set_trace"

using PyCall
using MacroTools
using MacroTools: prewalk, postwalk
using SpecialFunctions

const cast = PyNULL()
const builtins = PyNULL()

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
const gtfn_cpu = PyNULL()
const cmake = PyNULL()
const SourceLocation = PyNULL()

const scalar_types = Dict()
const py_scalar_types = Dict()
const py_dim_kind = Dict()
const builtin_py_op = Dict()
const py_backends = Dict()

function __init__()

    copy!(cast, pyimport("typing").cast)
    copy!(builtins, pyimport("builtins"))

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
    copy!(gtfn_cpu, pyimport("gt4py.next.program_processors.runners.gtfn"))
    copy!(cmake, pyimport("gt4py.next.otf.compilation.build_systems.cmake"))
    copy!(SourceLocation, pyimport("gt4py.eve.concepts").SourceLocation)

    init_py_utils()
end

include("preprocessing.jl")
include("single_static_assign.jl")
include("jast_to_foast.jl")

# Utils ------------------------------------------------------------------------------------

bin_op = Set([:(+), :(-), :(*), :(/), :(÷), :(^), :(%), :(&), :(|), :(⊻), :.+, :.-, :.*, :./, :.÷, :.^, :.%, :.&, :.|, :.⊻])
unary_op = Set([:(!), :(~), :.!, :.~, :(:)])
comp_op = Set([:(==), :(!=), :(<), :(<=), :(>), :(>=), :.==, :.!=, :.<, :.<=, :.>, :.>=])

math_ops = union(bin_op, unary_op, comp_op)

disallowed_op = Set([:hcat, :vcat, :concat]) #TODO: Link concat as soon as it's implemented in gt4py. Add further "forbidden" operations

builtin_op = Dict(
    :max_over => max_over, 
    :min_over => min_over, 
    :broadcast => GridTools.broadcast,
    :where => GridTools.where,
    :neighbor_sum => neighbor_sum,
    :convert => convert,
    :slice => nothing,   # TODO: how do we handle this?
    # :as_offset => gtx.as_offset,
    :pi => pi,
    :sin => sin,
    :cos => cos,
    :tan => tan,
    :asin => asin,
    :acos => acos,
    :atan => atan,
    :sinh => sinh,
    :cosh => cosh,
    :tanh => tanh,
    :asinh => asinh,
    :acosh => acosh,
    :atanh => atanh,
    :sqrt => sqrt,
    :exp => exp,
    :log => log,
    :gamma => gamma,
    :cbrt => cbrt,
    :floor => floor,
    :ceil => ceil,
    :trunc => trunc,
    :abs => abs,
    :isfinite => isfinite,
    :isinf => isinf,
    :isnan => isnan,
    :min => min,
    :max => max,
    Symbol("@bp") => nothing,
    Symbol("@run") => nothing
    )

CLOSURE_VARS::Dict = Dict()

# Methods -----------------------------------------------------------------------------------

function py_field_operator(fo, backend = py_backends["gpu"], grid_type = py"None"o, operator_attributes = Dict())

    foast_definition_node, closure_vars = jast_to_foast(fo.expr, fo.closure_vars)
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

function jast_to_foast(expr::Expr, closure_vars::Dict)
    expr, closure_vars, annotations = preprocess_definiton(expr, closure_vars)
    expr, closure_vars = remove_function_aliases(expr, closure_vars)             # TODO Can be ommited once gt4py allows aliases
    foast_node = visit_jast(expr, closure_vars)
    foast_node = postprocess_definition(foast_node, closure_vars, annotations)
    return foast_node, closure_vars
end

function preprocess_definiton(expr::Expr, closure_vars::Dict)
    sat = single_assign_target_pass(expr)
    ucc = unchain_compairs_pass(sat)
    ssa = single_static_assign_pass(ucc)
    closure_vars = translate_closure_vars(closure_vars)
    annotations = get_annotation(ssa, closure_vars)
    return (ssa, closure_vars, annotations)
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

py_args(args::Union{Base.Pairs, Dict}) = Dict(i.first => convert_type(i.second) for i in args)
py_args(args::Tuple) = [map(convert_type, args)...]
py_args(arg) = convert_type(arg)
py_args(n::Nothing) = nothing

function convert_type(a::Field)
    b_dims = []
    for dim in a.broadcast_dims
        kind = py_dim_kind[(get_dim_kind(dim))]
        push!(b_dims, gtx.Dimension(get_dim_name(dim), kind = kind)) # NOTE: this requires strict naming rules for dimensions
    end

    if typeof(a.data) <: Array
        new_data = a.data
    else
        new_data = np.asarray(a.data)
        @warn "Dtype of the Field: $a is not concrete. Data must be copied to Python which may affect performance. Try using dtypes <: Array."
    end

    offset = Dict(convert_type(dim) => a.origin[i] for (i, dim) in enumerate(a.dims))

    # py_field = gtx.as_field(b_dims, a.data, origin = offset)      #TODO: as_field copies the data. This effects performance and the data returned in out. 
    py_field = gtx.np_as_located_field(b_dims..., origin=offset)(new_data)

    return py_field
end

function convert_type(a::Connectivity)
    source_kind = py_dim_kind[(get_dim_kind(a.source))]
    source_dim  = gtx.Dimension(get_dim_name(a.source), kind=source_kind)
    
    target_kind = py_dim_kind[(get_dim_kind(a.target))]
    target_dim = gtx.Dimension(get_dim_name(a.target), kind=target_kind)

    @assert typeof(a.data) <: Array "Use concrete types for the data Array of the following Connectivity: $a."

    # account for different indexing in python
    return gtx.NeighborTableOffsetProvider(ifelse.(a.data .!= -1, a.data .- 1, a.data), target_dim, source_dim, a.dims)
end

function convert_type(a::Dimension)
    return gtx.Dimension(get_dim_name(a), kind=py_dim_kind[get_dim_kind(a)])
end

function convert_type(a::Any)
    @assert Symbol(typeof(a)) in keys(scalar_types) ("The type of argument $(a) is not a valid argument type to a field operator.")
    return a
end

# Remove aliases -------------------------------------------------------------------------

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

# Init helper ------------------------------------------------------------------------------

function init_py_utils()
    
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
        # TODO: Without quotes gt4py throws an error message. Report to gt4py that this needs to fail with a better error message.
        HORIZONTAL => gtx.DimensionKind."HORIZONTAL",
        VERTICAL => gtx.DimensionKind."VERTICAL",
        LOCAL => gtx.DimensionKind."LOCAL"
    )

    builtin_py_op_init = Dict(
    :max_over => gtx.max_over, 
    :min_over => gtx.min_over, 
    :broadcast => gtx.broadcast,
    :where => gtx.where,
    :neighbor_sum => gtx.neighbor_sum,
    :convert => gtx.astype,
    # :as_offset => gtx.as_offset,
    :pi => np.pi,
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

    py_backends_init = Dict(
        "cpu" => roundtrip.executor,
        "gpu" => gtfn_cpu.otf_compile_executor.CachedOTFCompileExecutor(
                    name="run_gtfn_cached_cmake",
                    otf_workflow=gtfn_cpu.workflow.CachedStep(step=gtfn_cpu.run_gtfn.executor.otf_workflow.replace(
                        compilation=gtfn_cpu.compiler.Compiler(
                            cache_strategy=gtfn_cpu.cache.Strategy.PERSISTENT,
                            builder_factory=cmake.CMakeFactory(cmake_build_type=cmake.BuildType.RELEASE)
                            )
                        ),
                    hash_function=gtfn_cpu.compilation_hash),
                )
    )

    copy!(scalar_types, scalar_types_init)
    copy!(py_scalar_types, py_scalar_types_init)
    copy!(py_dim_kind, py_dim_kind_init)
    copy!(builtin_py_op, builtin_py_op_init)
    copy!(py_backends, py_backends_init)
end