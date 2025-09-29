# UnifiedCompiler Main Compilation
# Entry point for unified formula compilation

include("types.jl")
include("execution.jl")
include("decomposition.jl")

# Import mixture validation functions
using ..FormulaCompiler: validate_mixture_consistency!

# Helper to extract fixed effects formula
function get_fixed_effects_formula(model)
    # For MixedModels, extract only fixed effects
    if isdefined(MixedModels, :LinearMixedModel) && isa(model, MixedModels.LinearMixedModel)
        return fixed_effects_form(model)
    elseif isdefined(MixedModels, :GeneralizedLinearMixedModel) && isa(model, MixedModels.GeneralizedLinearMixedModel)
        return fixed_effects_form(model)
    else
        # For GLM and other models, use the full formula
        return StatsModels.formula(model)
    end
end

# The fixed_effects_form function is imported at the module level

"""
    compile_formula(model, data) -> UnifiedCompiled

Compile a fitted statistical model into a zero-allocation, type-specialized evaluator.

Transforms statistical formulas into optimized computational engines using position mapping
that achieves ~50ns per row evaluation with zero allocations. The resulting evaluator
provides constant-time row access regardless of dataset size.

# Arguments
- `model`: Fitted statistical model (`GLM.LinearModel`, `GLM.GeneralizedLinearModel`, 
          `MixedModels.LinearMixedModel`, etc.)
- `data`: Data in Tables.jl format (preferably `Tables.columntable(df)` for optimal performance)

# Returns
- `UnifiedCompiled{T,Ops,S,O}`: Callable evaluator with embedded position mappings
  - Call as `compiled(output_vector, data, row_index)` for zero-allocation evaluation
  - `length(compiled)` returns number of model matrix columns

# Performance Characteristics
- **Compilation**: One-time cost for complex formulas
- **Evaluation**: Zero bytes allocated after warmup
- **Memory**: O(output_size) scratch space, reused across all evaluations
- **Scaling**: Evaluation time independent of dataset size

# Supported Models
- Linear models: `GLM.lm(@formula(y ~ x + group), df)`
- Generalized linear models: `GLM.glm(@formula(success ~ x), df, Binomial(), LogitLink())`
- Mixed models: `MixedModels.fit(MixedModel, @formula(y ~ x + (1|group)), df)` (fixed effects only)
- Custom contrasts: Models with `DummyCoding()`, `EffectsCoding()`, `HelmertCoding()`, etc.
- Standardized predictors: Models with `ZScore()` standardization

# Formula Features
- **Basic terms**: `x`, `log(z)`, `x^2`, `(x > 0)`, integer and float variables
- **Categorical variables**: Must use `CategoricalArrays.jl` format - raw strings not supported
- **Interactions**: `x * group`, `x * y * z`, `log(x) * group`
- **Functions**: `log`, `exp`, `sqrt`, `sin`, `cos`, `abs`, `^` (integer and fractional powers)
- **Boolean conditions**: `(x > 0)`, `(z >= mean(z))`, `(group == \"A\")`
- **Complex formulas**: `x * log(abs(z)) * group + sqrt(y) + (w > threshold)`

# Data Requirements
- **Categorical variables**: Must use `categorical(column)` before model fitting
- **Missing values**: Not supported - remove with `dropmissing()` or impute before compilation
- **Table format**: Use `Tables.columntable(df)` for optimal performance

# Example
```julia
using FormulaCompiler, GLM, DataFrames, Tables, CategoricalArrays

# Fit model
df = DataFrame(
    y = randn(1000), 
    x = randn(1000), 
    group = categorical(rand([\"A\", \"B\"], 1000))  # Required: use categorical()
)
model = lm(@formula(y ~ x * group + log(abs(x) + 1)), df)

# Compile once
data = Tables.columntable(df)  # Convert for optimal performance
compiled = compile_formula(model, data)

# Use many times (zero allocations)
output = Vector{Float64}(undef, length(compiled))
compiled(output, data, 1)     # Zero allocations
compiled(output, data, 500)   # Zero allocations

# Substantial speedup compared to modelmatrix(model)[row, :]
```

# Mixed Models Example
```julia
using MixedModels
mixed = fit(MixedModel, @formula(y ~ x + treatment + (1|subject)), df)
compiled = compile_formula(mixed, data)  # Compiles fixed effects: y ~ x + treatment
```

See also: [`modelrow!`](@ref), [`ModelRowEvaluator`](@ref)
"""
function compile_formula(model, data_example::NamedTuple)
    # Phase 2: Validate mixture columns are consistent
    validate_mixture_consistency!(data_example)
    
    # Extract schema-applied formula using standard API
    # For MixedModels, this extracts only fixed effects
    formula = get_fixed_effects_formula(model)
    
    # Decompose formula to operations (formula has schema info)
    ops_vec, scratch_size, output_size = decompose_formula(formula, data_example)
    
    # Convert to tuple for type stability
    ops_tuple = Tuple(ops_vec)
    
    # Create specialized compiled formula (Float64 by default)
    return UnifiedCompiled{Float64, typeof(ops_tuple), scratch_size, output_size}(ops_tuple)
end

# Export main functions
export UnifiedCompiled, compile_formula

"""
    compile_formula(formula::StatsModels.FormulaTerm, data) -> UnifiedCompiled

Compile a formula directly without a fitted model for zero-allocation evaluation.

This overload enables compilation from raw formulas, bypassing model fitting when only
the computational structure is needed. Useful for custom model implementations or
direct formula evaluation workflows.

# Arguments
- `formula::StatsModels.FormulaTerm`: Formula specification (e.g., from `@formula(y ~ x + group)`)
- `data`: Data in Tables.jl format (preferably `Tables.columntable(df)`)

# Returns
- `UnifiedCompiled{T,Ops,S,O}`: Zero-allocation evaluator, same interface as model-based compilation

# Performance
- **Compilation**: Fast for complex formulas
- **Evaluation**: Zero bytes allocated
- **Memory**: Identical performance to model-based compilation

# Example
```julia
using StatsModels, FormulaCompiler, Tables

# Direct formula compilation
formula = @formula(y ~ x * group + log(z))
data = Tables.columntable(df)
compiled = compile_formula(formula, data)

# Zero-allocation evaluation
output = Vector{Float64}(undef, length(compiled))
compiled(output, data, 1)  # Zero allocations
```

# Use Cases
- Custom model implementations requiring direct formula evaluation
- Performance-critical applications avoiding model fitting overhead
- Exploratory analysis with formula variations
- Integration with external statistical frameworks

See also: [`compile_formula(model, data)`](@ref) for model-based compilation
"""
function compile_formula(formula::StatsModels.FormulaTerm, data_example::NamedTuple)
    # Phase 2: Validate mixture columns are consistent
    validate_mixture_consistency!(data_example)
    
    ops_vec, scratch_size, output_size = decompose_formula(formula, data_example)
    ops_tuple = Tuple(ops_vec)
    return UnifiedCompiled{Float64, typeof(ops_tuple), scratch_size, output_size}(ops_tuple)
end
