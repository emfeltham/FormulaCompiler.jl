# evaluator.jl - DerivativeEvaluator construction and setup

"""
    _is_numeric_vector(col) -> Bool

Check if a column is a numeric vector that can be converted to Float64.
"""
_is_numeric_vector(col::AbstractVector) = eltype(col) <: Number

"""
    build_derivative_evaluator(compiled, data, vars) -> DerivativeEvaluator
    build_derivative_evaluator(compiled, data, vars, chunk) -> DerivativeEvaluator
    build_derivative_evaluator(compiled, data) -> DerivativeEvaluator

Build a reusable automatic differentiation evaluator for computing Jacobians and marginal effects.

Constructs a specialized evaluator that computes derivatives of model matrix rows with
respect to continuous variables using ForwardDiff.jl. Supports both dual-number automatic
differentiation and finite differences with backend selection for optimal performance.

# Arguments
- `compiled::UnifiedCompiled`: Result of `compile_formula(model, data)`
- `data::NamedTuple`: Column-table data (from `Tables.columntable(df)`)
- `vars::Vector{Symbol}`: Continuous variables to differentiate with respect to
  - **Restriction**: Must be continuous predictors (Float64, Int64, Int32, Int)
  - **Categorical variables**: Not supported; use scenario system for categorical profiles
  - **Validation**: Function validates variable types and provides clear error messages
- `chunk`: ForwardDiff chunk size (`ForwardDiff.Chunk{N}()` or `:auto` for `Chunk{length(vars)}`)

# Returns
- `DerivativeEvaluator{...}`: Specialized evaluator with preallocated buffers
  - Supports both automatic differentiation and finite differences backends
  - Contains type-specialized closures and configurations for optimal performance
  - Reusable across multiple row evaluations and derivative computations

# Performance Characteristics
- **Construction**: One-time cost proportional to number of variables
- **AD backend**: Small allocations per call (ForwardDiff internals)
- **FD backend**: Zero bytes allocated (finite differences)
- **Specialization**: Type-stable evaluation with concrete closures and configurations
- **Validation**: Tested across diverse formula types and variable combinations

# Variable Type Handling
- **All numeric types**: Automatically converted to Float64 for differentiation (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64)
- **Type validation**: Rejects non-numeric variables with informative error messages
- **Memory optimization**: Conversion overhead incurred once during construction

# Example
```julia
using FormulaCompiler, GLM

# Setup model with diverse variable types
df = DataFrame(
    y = randn(1000),
    x = randn(1000),                    # Float64
    age = rand(Int16(18):Int16(80), 1000),  # Int16
    score = rand(UInt8(0):UInt8(100), 1000), # UInt8
    group = rand([\"A\", \"B\"], 1000)
)
model = lm(@formula(y ~ x * group + age + score), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build derivative evaluator for continuous variables (positional args)
vars = [:x, :age, :score]  # Mix of Float64, Int16, and UInt8
de = build_derivative_evaluator(compiled, data, vars)

# Jacobian computation
J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow!(J, de, 1)  # J[i,j] = ∂X[i]/∂vars[j]

# Marginal effects on linear predictor η = Xβ
β = coef(model)
g_eta = Vector{Float64}(undef, length(vars))
marginal_effects_eta_ad!(g_eta, de, β, 1)  # Small allocations
marginal_effects_eta_fd!(g_eta, de, β, 1)  # Zero allocations
```

# Method Overloads
```julia
# Primary version with explicit variables
de = build_derivative_evaluator(compiled, data, [:x, :z])

# With explicit chunk size
de = build_derivative_evaluator(compiled, data, [:x, :z], ForwardDiff.Chunk{4}())

# All continuous variables (convenience)
de = build_derivative_evaluator(compiled, data)  # Uses all continuous vars
```

# Use Cases
- **Marginal effects computation**: Economic and policy analysis
- **Sensitivity analysis**: Parameter robustness assessment
- **Gradient-based optimization**: Custom model fitting and inference
- **Bootstrap inference**: Repeated derivative computation across samples

# Error Handling
Provides clear validation with specific guidance:
```julia
# This will error with helpful message:
de = build_derivative_evaluator(compiled, data, [:x, :group])
# Error: Non-continuous/categorical vars: [:group]. Use scenario system for categorical profiles.
```

See also: [`derivative_modelrow!`](@ref), [`marginal_effects_eta_ad!`](@ref), [`continuous_variables`](@ref)
"""
function build_derivative_evaluator(
    # Primary version: explicit variables
    
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    vars::Vector{Symbol}
) where {T, Ops, S, O}
    return _build_derivative_evaluator_impl(compiled, data, vars, :auto)
end

# Version with explicit chunk size
function build_derivative_evaluator(
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    vars::Vector{Symbol},
    chunk
) where {T, Ops, S, O}
    return _build_derivative_evaluator_impl(compiled, data, vars, chunk)
end

# Convenience version: all continuous variables
function build_derivative_evaluator(
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple
) where {T, Ops, S, O}
    vars = continuous_variables(compiled, data)
    return _build_derivative_evaluator_impl(compiled, data, vars, :auto)
end

# Implementation function (extracted from original)
function _build_derivative_evaluator_impl(
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    vars::Vector{Symbol},
    chunk
) where {T, Ops, S, O}
    # Validate that all requested vars are continuous (no categorical derivatives)
    allowed = continuous_variables(compiled, data)
    bad = [v for v in vars if !(v in allowed)]
    if !isempty(bad)
        msg = "build_derivative_evaluator only supports continuous variables. " *
              "Non-continuous/categorical vars requested: $(bad). " *
              "For categorical profile workflows, use the scenario system " *
              "(e.g., create_scenario()) combined with derivative evaluators."
        throw(ArgumentError(msg))
    end

    nvars = length(vars)
    xbuf = Vector{Float64}(undef, nvars)
    # Prebuild fully concrete overrides + merged data (Float64 path)
    override_vecs = Vector{FDOverrideVector}(undef, nvars)
    pairs = Pair{Symbol,FDOverrideVector}[]
    for (i, s) in enumerate(vars)
        col = getproperty(data, s)
        # Convert integer columns to Float64 for derivative computation
        float_col = if _is_numeric_vector(col)
            convert(Vector{Float64}, col)
        else
            col::Vector{Float64}
        end
        ov = FDOverrideVector(float_col, 1, 0.0)
        override_vecs[i] = ov
        push!(pairs, s => ov)
    end
    # Merge with converted columns - use Float64 versions for derivative computation
    data_over = merge(data, NamedTuple(pairs))
    overrides = override_vecs
    rowvec_float = Vector{Float64}(undef, length(compiled))
    # Pre-cache column references to avoid getproperty allocations (as NTuple)
    # Use the converted Float64 columns for FD computation
    fd_columns = ntuple(i -> getproperty(data_over, vars[i]), nvars)

    # Create a mutable ref that will eventually point to the final evaluator
    beta_ref = Base.RefValue(Vector{Float64}())
    de_ref = Base.RefValue{DerivativeEvaluator}()

    # Build typed closures and configs using the ref (which is still uninitialized)
    g = DerivClosure(de_ref)
    ch = chunk === :auto ? ForwardDiff.Chunk{nvars}() : chunk
    cfg = ForwardDiff.JacobianConfig(g, xbuf, ch)

    gscalar = GradClosure(g, beta_ref)
    gradcfg = ForwardDiff.GradientConfig(gscalar, xbuf, ch)

    # Preallocate DiffResult containers for zero-allocation AD paths
    jac_result = ForwardDiff.DiffResult(
        Vector{Float64}(undef, length(compiled)),
        (Matrix{Float64}(undef, length(compiled), nvars),),
    )
    grad_result = ForwardDiff.DiffResult(0.0, Vector{Float64}(undef, nvars))

    # Build typed single-cache for manual dual path (Dual tag = Nothing)
    DualT = ForwardDiff.Dual{Nothing, Float64, nvars}
    compiled_dual_vec = UnifiedCompiled{DualT, Ops, S, O}(compiled.ops)
    rowvec_dual_vec = Vector{DualT}(undef, length(compiled))
    x_dual_vec = Vector{DualT}(undef, nvars)
    data_over_dual_vec, overrides_dual_vec = build_row_override_data_typed(data, vars, 1, DualT)
    # Precompute unit partials for each variable to avoid per-call construction
    partials_unit_vec = Vector{ForwardDiff.Partials{nvars, Float64}}(undef, nvars)
    for i in 1:nvars
        partials_unit_vec[i] = ForwardDiff.Partials{nvars, Float64}(ntuple(j -> (i == j ? 1.0 : 0.0), Val(nvars)))
    end

    # Final evaluator with concrete closure/config/result types
    de = DerivativeEvaluator{T, Ops, S, O, typeof(data), typeof(data_over), nvars, typeof(fd_columns), typeof(g), typeof(cfg), typeof(gscalar), typeof(gradcfg), typeof(jac_result), typeof(grad_result), typeof(data_over_dual_vec), typeof(overrides_dual_vec)}(
        compiled,
        data,
        vars,
        xbuf,
        overrides,
        data_over,
        DualCacheDict(),  # Initialize empty dual cache
        compiled_dual_vec,
        rowvec_dual_vec,
        overrides_dual_vec,
        data_over_dual_vec,
        x_dual_vec,
        partials_unit_vec,
        rowvec_float,
        g,
        cfg,
        gscalar,
        gradcfg,
        beta_ref,
        Vector{Float64}(undef, length(compiled)),
        jac_result,
        grad_result,
        1,
        Matrix{Float64}(undef, length(compiled), nvars),
        Vector{Float64}(undef, nvars),
        Vector{Float64}(undef, length(compiled)),
        Vector{Float64}(undef, length(compiled)),
        Vector{Float64}(undef, length(compiled)),
        Vector{Float64}(undef, nvars),
        fd_columns,
    )

    # Now initialize the ref to point to the final evaluator
    de_ref[] = de

    return de
end

