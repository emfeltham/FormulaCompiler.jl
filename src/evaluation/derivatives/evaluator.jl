# evaluator.jl - DerivativeEvaluator construction and setup

"""
    _is_numeric_vector(col) -> Bool

Check if a column is a numeric vector that can be converted to Float64.
"""
_is_numeric_vector(col::AbstractVector) = eltype(col) <: Number

"""
    derivativevaluator(backend, compiled, data, vars) -> FDEvaluator|ADEvaluator
    derivativevaluator(backend, compiled, data) -> FDEvaluator|ADEvaluator

Build a reusable automatic differentiation evaluator for computing Jacobians and marginal effects.

Constructs a specialized evaluator that computes derivatives of model matrix rows with
respect to continuous variables using ForwardDiff.jl. Supports both dual-number automatic
differentiation and finite differences with backend selection for optimal performance.

# Arguments
- `backend::Symbol`: Backend selection (`:ad` for automatic differentiation, `:fd` for finite differences)
- `compiled::UnifiedCompiled`: Result of `compile_formula(model, data)`
- `data::NamedTuple`: Column-table data (from `Tables.columntable(df)`)
- `vars::Vector{Symbol}`: Continuous variables to differentiate with respect to
  - **Restriction**: Must be continuous predictors (Float64, Int64, Int32, Int)
  - **Categorical variables**: Not supported; use scenario system for categorical profiles
  - **Validation**: Function validates variable types and provides clear error messages

# Returns
- `FDEvaluator{...}` or `ADEvaluator{...}`: Specialized evaluator with preallocated buffers
  - **FDEvaluator**: Finite differences backend with only FD infrastructure (6 type parameters)
  - **ADEvaluator**: Automatic differentiation backend with only AD infrastructure (9 type parameters)
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

# Build derivative evaluator for continuous variables
vars = [:x, :age, :score]  # Mix of Float64, Int16, and UInt8
de = derivativevaluator(:ad, compiled, data, vars)  # Automatic differentiation backend

# Jacobian computation
J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow!(J, de, 1)  # J[i,j] = ∂X[i]/∂vars[j]

# Marginal effects on linear predictor η = Xβ
β = coef(model)
g_eta = Vector{Float64}(undef, length(vars))
marginal_effects_eta!(g_eta, de, β, 1)  # Small allocations
marginal_effects_eta!(g_eta, de, β, 1)  # Zero allocations
```

# Method Overloads
```julia
# Primary version with explicit backend and variables
de = derivativevaluator(:ad, compiled, data, [:x, :z])  # Automatic differentiation
de = derivativevaluator(:fd, compiled, data, [:x, :z])  # Finite differences

# All continuous variables (convenience)
de = derivativevaluator(:ad, compiled, data)  # Uses all continuous vars with AD
de = derivativevaluator(:fd, compiled, data)  # Uses all continuous vars with FD
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
de = derivativevaluator(compiled, data, [:x, :group])
# Error: Non-continuous/categorical vars: [:group]. Use scenario system for categorical profiles.
```

See also: [`derivative_modelrow!`](@ref), [`marginal_effects_eta!`](@ref), [`continuous_variables`](@ref)
"""
# Main constructor with positional backend argument
function derivativevaluator(
    backend::Symbol, compiled, data, vars::Vector{Symbol}
)
    if backend === :fd
        return derivativeevaluator_fd(compiled, data, vars)
    elseif backend === :ad
        return derivativeevaluator_ad(compiled, data, vars)
    else
        throw(ArgumentError("Unknown backend: $backend. Use :fd or :ad"))
    end
end

# Convenience version: all continuous variables
function derivativevaluator(backend::Symbol, compiled, data)
    vars = continuous_variables(compiled, data)
    return derivativevaluator(backend, compiled, data, vars)
end


# Backend-specialized constructor functions

"""
    derivativeevaluator_fd(compiled, data, vars) -> FDEvaluator

Create a finite differences specialized FDEvaluator using Float64 counterfactual vectors.

Returns a concrete FDEvaluator with only FD infrastructure, no field pollution from AD.
Uses NumericCounterfactualVector{Float64} for type-stable counterfactual operations.
"""
function derivativeevaluator_fd(
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    vars::Vector{Symbol}
) where {T, Ops, S, O}
    # Validate continuous variables
    allowed = continuous_variables(compiled, data)
    bad = [v for v in vars if !(v in allowed)]
    if !isempty(bad)
        throw(ArgumentError("derivativeevaluator_fd only supports continuous variables. Non-continuous vars: $(bad)"))
    end

    # Build Float64 counterfactual system only
    data_counterfactual, counterfactuals = build_counterfactual_data(data, vars, 1, Float64)

    # Direct field construction - no complex type parameters needed
    return FDEvaluator(
        compiled,                                              # compiled_base
        data,                                                  # base_data
        vars,                                                  # vars
        counterfactuals,                                       # counterfactuals
        data_counterfactual,                                   # data_counterfactual
        Vector{Float64}(undef, length(compiled)),              # y_plus
        Vector{Float64}(undef, length(compiled)),              # yminus
        Vector{Float64}(undef, length(vars)),                  # xbase
        Matrix{Float64}(undef, length(compiled), length(vars)), # jacobian_buffer
        Vector{Float64}(undef, length(compiled)),              # xrow_buffer
        1                                                      # row
    )
end

"""
    derivativeevaluator_ad(compiled, data, vars) -> ADEvaluator

Create an automatic differentiation specialized ADEvaluator using Dual counterfactual vectors.

Returns a concrete ADEvaluator with only AD infrastructure, no field pollution from FD.
Uses NumericCounterfactualVector{Dual{...}} for type-stable dual number operations.
"""
function derivativeevaluator_ad(
    compiled::UnifiedCompiled{T, Ops, S, O},
    data::NamedTuple,
    vars::Vector{Symbol}
) where {T, Ops, S, O}
    # Validate continuous variables
    allowed = continuous_variables(compiled, data)
    bad = [v for v in vars if !(v in allowed)]
    if !isempty(bad)
        throw(ArgumentError("derivativeevaluator_ad only supports continuous variables. Non-continuous vars: $(bad)"))
    end

    nvars = length(vars)

    # Build Dual counterfactual system
    DualT = ForwardDiff.Dual{Nothing, Float64, nvars}
    data_counterfactual, counterfactuals = build_counterfactual_data(data, vars, 1, DualT)

    # Build dual-specialized compiled evaluator
    UB = typeof(compiled)
    OpsT = UB.parameters[2]
    ST = UB.parameters[3]
    OT = UB.parameters[4]
    compiled_dual = UnifiedCompiled{DualT, OpsT, ST, OT}(compiled.ops)

    # AD-specific buffers and infrastructure
    x_dual_vec = Vector{DualT}(undef, nvars)
    partials_unit_vec = Vector{ForwardDiff.Partials{nvars, Float64}}(undef, nvars)
    for i in 1:nvars
        partials_unit_vec[i] = ForwardDiff.Partials{nvars, Float64}(ntuple(j -> (i == j ? 1.0 : 0.0), Val(nvars)))
    end
    rowvec_dual_vec = Vector{DualT}(undef, length(compiled))

    # Build concrete ForwardDiff configuration
    de_ref = Base.RefValue{ADEvaluator}()
    g = DerivClosure(de_ref)
    ch = ForwardDiff.Chunk{nvars}()
    cfg = ForwardDiff.JacobianConfig(g, Vector{Float64}(undef, nvars), ch)

    # Direct field construction - concrete types
    beta_buf = Vector{Float64}(undef, length(compiled))
    beta_ref = Ref{Vector{Float64}}(beta_buf)

    de = ADEvaluator(
        compiled,                                              # compiled_base
        compiled_dual,                                         # compiled_dual
        data,                                                  # base_data
        vars,                                                  # vars
        counterfactuals,                                       # counterfactuals
        data_counterfactual,                                   # data_counterfactual
        x_dual_vec,                                           # x_dual_vec
        partials_unit_vec,                                    # partials_unit_vec
        rowvec_dual_vec,                                      # rowvec_dual_vec
        Matrix{Float64}(undef, length(compiled), length(vars)), # jacobian_buffer
        Vector{Float64}(undef, length(compiled)),              # xrow_buffer
        g,                                                     # g
        cfg,                                                   # cfg
        1,                                                     # row
        beta_ref,                                              # beta_ref
        beta_buf                                               # beta_buf
    )

    # Initialize the ref for the closure
    de_ref[] = de
    return de
end


