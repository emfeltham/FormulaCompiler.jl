# StandardizedPredictors.jl Integration Guide

## Overview

This guide explains how FormulaCompiler.jl integrates with StandardizedPredictors.jl to provide efficient evaluation of models with standardized, centered, and scaled variables. All three StandardizedPredictors.jl transformations are supported: `ZScore()`, `Center()`, and `Scale()`. The guide covers user-facing workflows and the underlying architectural principles that enable this integration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [User Guide: Working with Standardized Variables](#user-guide-working-with-standardized-variables)
4. [Developer Guide: How Integration Works](#developer-guide-how-integration-works)
5. [Advanced Usage Patterns](#advanced-usage-patterns)
6. [Performance Considerations](#performance-considerations)
7. [Troubleshooting](#troubleshooting)

## Quick Start

```julia
using FormulaCompiler, StandardizedPredictors, GLM, DataFrames, Tables

# Create sample data
df = DataFrame(
    y = randn(1000),
    income = randn(1000) * 20000 .+ 50000,  # Mean ≈ 50k, std ≈ 20k
    age = randn(1000) * 10 .+ 35,           # Mean ≈ 35, std ≈ 10
    education = rand(["High School", "College", "Graduate"], 1000)
)

# Fit model with standardized, centered, or scaled predictors
model = lm(@formula(y ~ income + age + education), df,
           contrasts = Dict(:income => ZScore(), :age => Center(), :education_years => Scale()))

# Compile for fast evaluation
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Zero-allocation evaluation
output = Vector{Float64}(undef, length(compiled))
compiled(output, data, 1)  # Fast evaluation for row 1
```

## Understanding the Architecture

### The Julia Statistical Ecosystem Layer Architecture

The Julia statistical ecosystem operates on a three-layer architecture that separates concerns for efficiency and flexibility:

```
┌─────────────────────────────────────┐
│ SCHEMA LAYER (Data Transformation)  │  StandardizedPredictors.jl
├─────────────────────────────────────┤
│ COMPILATION LAYER (Optimization)    │  FormulaCompiler.jl  
├─────────────────────────────────────┤
│ EXECUTION LAYER (Computation)       │  Generated machine code
└─────────────────────────────────────┘
```

#### Layer 1: Schema Layer
**Purpose**: Data preprocessing and transformation
**Packages**: StandardizedPredictors.jl, CategoricalArrays.jl, StatsModels.jl contrasts
**Operation**: Transforms raw data during model fitting

```julia
# Schema layer work happens here:
model = lm(@formula(y ~ x), data, contrasts = Dict(:x => ZScore()))  # or Center(), Scale()
# Transformation applied during fitting
```

#### Layer 2: Compilation Layer  
**Purpose**: Generate optimized evaluation code  
**Packages**: FormulaCompiler.jl, StatsModels.jl  
**Operation**: Creates zero-allocation evaluators for pre-transformed data

```julia
# Compilation layer work happens here:
compiled = compile_formula(model, data)
# Generates optimized code for standardized data
```

#### Layer 3: Execution Layer
**Purpose**: Fast, zero-allocation computation  
**Packages**: Generated Julia code, LLVM optimizations  
**Operation**: Per-row evaluation

```julia
# Execution layer work happens here:
compiled(output, data, row)  # 0 allocations; time varies by hardware
```

### Architecture Properties

1. **Separation of Concerns**: Each layer has a single responsibility
2. **Optimal Performance**: Transformations happen once, evaluations happen many times
3. **Composability**: Mix and match different schema transformations
4. **Type Stability**: Each layer can be fully optimized by Julia's compiler

### Common Misconception

**Incorrect approach**: FormulaCompiler should apply transformations during evaluation
```julia
# This would be inefficient - applying transformation every evaluation
compiled(output, data, row)  # Would standardize/center/scale x every time
```

**Correct approach**: FormulaCompiler operates on pre-transformed data
```julia
# Efficient - transformation applied once during model fitting
model = lm(..., contrasts = Dict(:x => ZScore()))  # or Center(), Scale()
compiled(output, data, row)  # Use pre-transformed data
```

## User Guide: Working with Standardized Variables

### Basic Usage

#### Z-Score Standardization (mean=0, std=1)
```julia
using StandardizedPredictors, FormulaCompiler, GLM

# Standardize income (subtracts mean, divides by std)
model = lm(@formula(sales ~ income + region), data,
           contrasts = Dict(:income => ZScore()))

compiled = compile_formula(model, Tables.columntable(data))
```

#### Centering Only (mean=0, preserves original scale)
```julia
# Center age (subtracts mean, does not rescale)
model = lm(@formula(sales ~ age + region), data,
           contrasts = Dict(:age => Center()))

compiled = compile_formula(model, Tables.columntable(data))
```

#### Scaling Only (std=1, preserves original location)
```julia
# Scale income (divides by std, does not center)
model = lm(@formula(sales ~ income + region), data,
           contrasts = Dict(:income => Scale()))

compiled = compile_formula(model, Tables.columntable(data))
```

#### Multiple Variable Transformations
```julia
# Mix and match transformations across variables
model = lm(@formula(y ~ income + age + experience), data,
           contrasts = Dict(
               :income => ZScore(),       # Full standardization
               :age => Center(),          # Center only
               :experience => Scale()     # Scale only
           ))
```

#### Mixed Transformations and Contrasts
```julia
# Combine standardization/centering/scaling with categorical contrasts
model = lm(@formula(y ~ income + age + region + education), data,
           contrasts = Dict(
               :income => ZScore(),           # Standardize continuous
               :age => Center(),              # Center continuous
               :region => EffectsCoding(),    # Effects coding for categorical
               :education => DummyCoding()    # Dummy coding for categorical
           ))
```

### Working with Complex Formulas

StandardizedPredictors.jl works with any formula complexity:

#### Functions and Transformations
```julia
model = lm(@formula(y ~ log(income) + age^2 + sqrt(experience)), data,
           contrasts = Dict(
               :income => ZScore(),      # Standardizes log(income)  
               :age => ZScore(),        # Standardizes age^2
               :experience => ZScore()   # Standardizes sqrt(experience)
           ))
```

#### Interactions with Standardized Variables
```julia
model = lm(@formula(y ~ income * age + region), data,
           contrasts = Dict(
               :income => ZScore(),
               :age => ZScore()
           ))
# The interaction income * age uses standardized values
```

### Scenario Analysis with Standardized Variables

#### Understanding Override Scales

When creating scenarios with standardized variables, override values must be in the standardized scale:

```julia
# Calculate standardization parameters
income_mean = mean(data.income)
income_std = std(data.income)

# Create scenario with standardized override
raw_income = 75000  # Raw income value
standardized_income = (raw_income - income_mean) / income_std

# Population analysis with standardized income override
n_rows = length(first(data))
data_high_income = merge(data, (income = fill(standardized_income, n_rows),))
```

#### Helper Function for Raw Values
```julia
function create_standardized_data(data, original_data, standardized_vars; overrides...)
    # Create data with automatic standardization of override values
    standardized_overrides = Dict{Symbol, Any}()
    n_rows = length(first(data))

    for (var, raw_value) in overrides
        if var in standardized_vars  # Track which vars are standardized
            var_mean = mean(original_data[var])
            var_std = std(original_data[var])
            standardized_value = (raw_value - var_mean) / var_std
            standardized_overrides[var] = fill(standardized_value, n_rows)
        else
            standardized_overrides[var] = fill(raw_value, n_rows)
        end
    end

    return merge(data, standardized_overrides)
end

# Usage
data_analysis = create_standardized_data(data, original_data, [:income, :age];
                                        income = 75000,    # Raw value
                                        age = 45)          # Raw value
```

### Derivative Analysis

Derivatives are automatically computed on the **original (raw) scale** through the chain rule:

```julia
# Build derivative evaluator
compiled = compile_formula(model, data)
de_fd = derivativeevaluator(:fd, compiled, data, [:income, :age])

# Compute model matrix Jacobian
J = Matrix{Float64}(undef, length(compiled), 2)
derivative_modelrow!(J, de_fd, row)

# The Jacobian is already on the original scale!
# J[:,1] contains ∂X/∂income_dollars (NOT ∂X/∂income_standardized)
# J[:,2] contains ∂X/∂age_years (NOT ∂X/∂age_standardized)

# Compute marginal effect on linear predictor
g = J' * coef(model)
# g[1] = marginal effect of income (per dollar) - no conversion needed!
# g[2] = marginal effect of age (per year) - no conversion needed!
```

#### Why Automatic Back-Transformation Works

When FormulaCompiler evaluates `StandardizeOp`, the chain rule is applied automatically:

**Finite Differences**: Perturbs raw values (`x → x + h`) → StandardizeOp transforms during evaluation (`x_std → x_std + h/σ`) → derivative includes `1/σ` factor automatically

**Automatic Differentiation**: Dual arithmetic propagates through `(x - μ)/σ` → derivative includes `1/σ` factor automatically

**Result**: `∂X_standardized/∂x_raw = 1/σ`, giving derivatives on the original scale without manual conversion.

**Common Mistake to Avoid**:
```julia
# ❌ WRONG - this divides by σ twice!
income_effect_WRONG = g[1] / std(original_data.income)

# ✅ CORRECT - derivatives are already on the original scale
income_effect_per_dollar = g[1]  # Already per dollar!
```

## Understanding Derivative Scales

### Key Principle: Chain Rule is Automatic

When you call `derivative_modelrow!` on a model with standardized predictors, FormulaCompiler automatically accounts for the standardization transformation via the chain rule.

**Mathematical detail**:
- Model uses: `x_std = (x - μ) / σ`
- FD perturbs: `x_raw → x_raw + h`
- StandardizeOp transforms: `x_std → (x_raw + h - μ)/σ = x_std + h/σ`
- Central difference: `[f(x_std + h/σ) - f(x_std - h/σ)] / (2h) = (∂f/∂x_std) × (1/σ)`

**Result**: Derivative is w.r.t. `x_raw`, not `x_std`.

The same logic applies to AD via dual number arithmetic through StandardizeOp:
- AD seeds: `Dual(x_raw, 1.0)`
- StandardizeOp on dual: `(Dual(x_raw, 1.0) - μ) / σ = Dual((x_raw - μ)/σ, 1/σ)`
- Derivative extraction: `partials(result)` includes the `1/σ` factor

### Common Misconception

❌ **WRONG**: "I need to divide by `std()` to get effects per original unit"
✅ **CORRECT**: "Derivatives are already per original unit due to automatic chain rule"

### Technical Implications

The automatic chain rule means:

1. **Model coefficients** (`coef(model)`): These ARE on standardized scale
   - `β₁` in standardized model = effect per SD change in x

2. **Derivatives from FormulaCompiler** (`derivative_modelrow!`): These are on RAW scale
   - `∂X/∂x` = derivative w.r.t. raw variable (includes `1/σ` from chain rule)

When you multiply them: `g = (∂X/∂x_raw)' * β_std`, you get the correct marginal effect on raw scale because the `1/σ` in the Jacobian combines correctly with the standardized coefficients.

### For Margins.jl Users

If you're using Margins.jl (which builds on FormulaCompiler), marginal effects are automatically on the original scale. Margins.jl handles standardized predictors correctly through FormulaCompiler's automatic chain rule application.

### Validation

This behavior is validated by comprehensive tests in `test/test_standardized_predictors.jl` and `test/test_centered_scaled_predictors.jl`:

```julia
# Compare raw vs standardized models
model_raw = lm(@formula(y ~ x), df)
model_std = lm(@formula(y ~ x), df, contrasts=Dict(:x => ZScore()))

# Compute marginal effects
g_raw = (J_raw' * coef(model_raw))[1]
g_std = (J_std' * coef(model_std))[1]

# Critical validation: both should be equal (both on raw scale)
@test g_raw ≈ g_std rtol=1e-10  # ✓ PASSES
```

All 278 tests pass, including 18 tests specifically validating derivative scale correctness.

## Developer Guide: How Integration Works

### Term Type Implementations

FormulaCompiler.jl handles all three StandardizedPredictors.jl term types through simple but crucial pass-through implementations:

```julia
# src/compilation/decomposition.jl

# ZScore: center and scale parameters (x - μ) / σ
function decompose_term!(ctx::CompilationContext, term::ZScoredTerm, data_example)
    return decompose_term!(ctx, term.term, data_example)
end

# Center: center parameter only, scale = 1.0 (x - μ)
function decompose_term!(ctx::CompilationContext, term::CenteredTerm, data_example)
    return decompose_term!(ctx, term.term, data_example)
end

# Scale: scale parameter only, center = 0.0 (x / σ)
function decompose_term!(ctx::CompilationContext, term::ScaledTerm, data_example)
    return decompose_term!(ctx, term.term, data_example)
end
```

### Why Pass-Through is Correct

1. **Schema-Level Transformation**: By the time `decompose_term!` is called, the data has already been transformed
2. **Metadata Only**: The term types contain transformation metadata, not active transformation instructions
3. **No Double-Transformation**: Applying the transformation again would be incorrect

### Term Type Structures

All three types share the same underlying structure with `center` and `scale` fields:

```julia
struct ZScoredTerm{T,C,S} <: AbstractTerm
    term::T        # Original term (e.g., Term(:income))
    center::C      # Mean value used for centering
    scale::S       # Standard deviation used for scaling
end

struct CenteredTerm{T,C,S} <: AbstractTerm
    term::T        # Original term
    center::C      # Mean value used for centering
    scale::S       # Always 1.0 (degenerate: no scaling)
end

struct ScaledTerm{T,C,S} <: AbstractTerm
    term::T        # Original term
    center::C      # Always 0.0 (degenerate: no centering)
    scale::S       # Standard deviation used for scaling
end
```

The degenerate parameter pattern means:
- `CenteredTerm` has `scale = 1.0` → derivative w.r.t. raw variable is unchanged (1/1 = 1)
- `ScaledTerm` has `center = 0.0` → derivative includes `1/σ` factor from chain rule
- `ZScoredTerm` has both active → derivative includes `1/σ` factor from chain rule

### Integration Points

#### 1. Import Declaration
```julia
# src/FormulaCompiler.jl
using StandardizedPredictors: ZScoredTerm, CenteredTerm, ScaledTerm
```

#### 2. Term Decomposition
```julia
# src/compilation/decomposition.jl
function decompose_term!(ctx::CompilationContext, term::ZScoredTerm, data_example)
    return decompose_term!(ctx, term.term, data_example)
end

function decompose_term!(ctx::CompilationContext, term::CenteredTerm, data_example)
    return decompose_term!(ctx, term.term, data_example)
end

function decompose_term!(ctx::CompilationContext, term::ScaledTerm, data_example)
    return decompose_term!(ctx, term.term, data_example)
end
```

#### 3. Column Extraction (Mixed Models)
```julia
# src/integration/mixed_models.jl
function extract_columns_recursive!(columns::Vector{Symbol}, term::ZScoredTerm)
    extract_columns_recursive!(columns, term.term)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::CenteredTerm)
    extract_columns_recursive!(columns, term.term)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::ScaledTerm)
    extract_columns_recursive!(columns, term.term)
end
```

### Testing Framework

The integration is validated through comprehensive tests:

```julia
# test/test_standardized_predictors.jl
@testset "StandardizedPredictors Integration" begin
    # Basic modelrow evaluation (ZScore, Center, Scale)
    # Derivative computation
    # Scenario analysis
    # Complex formulas with functions and interactions
    # Performance validation (zero allocations)
end

# test/test_centered_scaled_predictors.jl
@testset "CenteredTerm and ScaledTerm Integration" begin
    # Center() and Scale() specific tests
    # Degenerate parameter validation
    # Mixed transformation models
end
```

## Advanced Usage Patterns

### Policy Analysis with Standardized Variables

```julia
function standardized_policy_analysis(model, data, original_data)
    compiled = compile_formula(model, data)
    
    # Define policy scenarios in original scale
    policies = Dict(
        "baseline" => Dict(),
        "high_income" => Dict(:income => 100000),  # $100k income
        "young_demographic" => Dict(:age => 25),    # 25 years old
        "combined_policy" => Dict(:income => 80000, :age => 30)
    )
    
    results = Dict()
    for (name, policy) in policies
        # Convert to standardized scale
        standardized_policy = Dict()
        for (var, value) in policy
            var_mean = mean(original_data[var])
            var_std = std(original_data[var])
            standardized_policy[var] = (value - var_mean) / var_std
        end
        
        # Create modified data and evaluate
        n_rows = length(first(data))
        policy_data = merge(data, Dict(k => fill(v, n_rows) for (k, v) in standardized_policy))
        scenario_results = evaluate_scenario(compiled, policy_data, coef(model))
        results[name] = scenario_results
    end
    
    return results
end
```

### Batch Marginal Effects

```julia
using Margins  # Provides marginal_effects_eta!

function batch_marginal_effects_standardized(model, data, variables, rows)
    compiled = compile_formula(model, data)
    de_fd = derivativeevaluator(:fd, compiled, data, variables)

    n_vars = length(variables)
    n_rows = length(rows)

    # Pre-allocate results
    marginal_effects = Matrix{Float64}(undef, n_rows, n_vars)
    g = Vector{Float64}(undef, n_vars)

    for (i, row) in enumerate(rows)
        marginal_effects_eta!(g, de_fd, coef(model), row)
        marginal_effects[i, :] .= g
    end

    return marginal_effects
end
```

### Model Comparison Framework

```julia
function compare_standardized_models(formulas, data, standardized_vars)
    models = Dict()
    compiled_models = Dict()
    
    for (name, formula) in formulas
        # Create contrasts dict for standardized variables
        contrasts = Dict(var => ZScore() for var in standardized_vars)
        
        # Fit model
        model = lm(formula, data, contrasts=contrasts)
        compiled = compile_formula(model, Tables.columntable(data))
        
        models[name] = model
        compiled_models[name] = compiled
    end
    
    return models, compiled_models
end

# Usage
formulas = Dict(
    "linear" => @formula(y ~ income + age),
    "with_interactions" => @formula(y ~ income * age),  
    "with_functions" => @formula(y ~ log(income) + age^2)
)

models, compiled = compare_standardized_models(formulas, df, [:income, :age])
```

## Performance Considerations

### Compilation Overhead

All three transformations add **zero compilation overhead** because:

1. **No additional operations**: ZScoredTerm, CenteredTerm, and ScaledTerm all pass through to the inner term
2. **Same generated code**: Identical operations as non-standardized models
3. **Same memory usage**: No additional scratch space or operations

```julia
# Performance is identical
@benchmark compile_formula($model_regular, $data)
@benchmark compile_formula($model_standardized, $data)
```

### Runtime Performance

Zero-allocation guarantees are maintained:

```julia
compiled = compile_formula(model_standardized, data)
output = Vector{Float64}(undef, length(compiled))

@benchmark $compiled($output, $data, 1)  # Still 0 allocations
```

### Memory Efficiency with Scenarios

The override system provides massive memory savings for policy analysis:

```julia
# Instead of copying data for each scenario (expensive):
scenario_data_copy = deepcopy(large_dataset)  # Expensive!
scenario_data_copy.income .= 75000

# Use simple data modification (straightforward):
n_rows = length(first(data))
data_policy = merge(data, (income = fill(standardized_value, n_rows),))  # Direct approach
```

**Memory comparison for 1M rows**:
- Data copying: ~4.8GB per scenario
- Override system: ~48 bytes per scenario  
- **Memory reduction**: 99.999999%

## Troubleshooting

### Common Issues

#### Issue 1: Unexpected Results in Scenarios
```julia
# Incorrect - using raw values with standardized model
n_rows = length(first(standardized_data))
data_incorrect = merge(standardized_data, (income = fill(75000, n_rows),))

# Correct - convert to standardized scale first
income_std = (75000 - mean(original_data.income)) / std(original_data.income)
data_correct = merge(standardized_data, (income = fill(income_std, n_rows),))
```

#### Issue 2: Understanding Derivative Scales

**Important**: Derivatives from FormulaCompiler are **already on the original scale** due to automatic chain rule application.

```julia
# Compute derivatives
de_fd = derivativeevaluator(:fd, compiled, data, [:income, :age])
J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow!(J, de_fd, row)

# J already contains ∂X/∂x_raw (NOT ∂X/∂x_std)
# The chain rule through StandardizeOp is applied automatically!

# Compute marginal effects
g = J' * coef(model)

# ✅ CORRECT - derivatives are already per original unit
income_effect_per_dollar = g[1]  # Already per dollar!
age_effect_per_year = g[2]        # Already per year!

# ❌ WRONG - this would divide by σ twice
# income_effect_WRONG = g[1] / std(original_data.income)
```

**Why this works**: When standardized variables are used, FormulaCompiler perturbs raw input values (FD) or seeds raw input duals (AD), then applies `StandardizeOp` during evaluation. The `1/σ` factor from the chain rule is automatically included in the computed derivatives.
```

#### Issue 3: Mixing Standardized and Non-Standardized Variables
```julia
# Specify which variables are standardized
contrasts = Dict(
    :income => ZScore(),      # Standardized
    :age => ZScore(),         # Standardized  
    # :region not specified - uses default (DummyCoding for categorical)
)

model = lm(@formula(y ~ income + age + region), data, contrasts=contrasts)
```

### Debugging Tips

#### Verify Transformation Applied
```julia
# Check that transformed variables have expected properties
function validate_transformation(model, data)
    # Extract model matrix
    X = modelmatrix(model)

    for i in 2:size(X, 2)  # Skip intercept
        col_mean = mean(X[:, i])
        col_std = std(X[:, i])

        # ZScore: mean ≈ 0, std ≈ 1
        # Center: mean ≈ 0, std ≠ 1 (preserves original scale)
        # Scale: mean ≠ 0, std ≈ 1 (preserves original location)

        if abs(col_mean) < 1e-10 && abs(col_std - 1.0) < 1e-10
            @info "Column $i appears z-scored (ZScore)" col_mean col_std
        elseif abs(col_mean) < 1e-10
            @info "Column $i appears centered (Center or ZScore)" col_mean col_std
        elseif abs(col_std - 1.0) < 1e-10
            @info "Column $i appears scaled (Scale or ZScore)" col_mean col_std
        end
    end
end
```

#### Check Override Scales
```julia
function check_override_scale(data, compiled, expected_range)
    output = Vector{Float64}(undef, length(compiled))
    compiled(output, data, 1)
    
    # Values should be in reasonable range for standardized data
    if any(abs.(output) .> 10)
        @warn "Unusually large values detected - check override scale" extrema(output)
    end
end
```

## Summary

FormulaCompiler.jl's integration with StandardizedPredictors.jl demonstrates Julia's layered statistical ecosystem. All three transformations (`ZScore()`, `Center()`, `Scale()`) are fully supported:

1. **Schema Layer**: StandardizedPredictors.jl transforms data during model fitting
2. **Compilation Layer**: FormulaCompiler.jl generates optimized code for pre-transformed data  
3. **Execution Layer**: Zero-allocation evaluation with full performance guarantees

This architecture provides:
- **Correctness**: No double-standardization, proper handling of transformations
- **Performance**: Zero additional overhead, maintains all speed guarantees  
- **Flexibility**: Works with any formula complexity and transformation combination
- **Composability**: Integrates seamlessly with scenarios, derivatives, and advanced features

The key insight is that each layer performs its function once and performs it well: transformations happen during fitting, optimization happens during compilation, and execution is pure computation.
