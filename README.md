# FormulaCompiler.jl

[![CI](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/ci.yml)
[![Documentation](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/docs.yml/badge.svg)](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://emfeltham.github.io/FormulaCompiler.jl/stable/)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://emfeltham.github.io/FormulaCompiler.jl/dev/)

Efficient per-row model matrix evaluation for Julia statistical models using position-mapped compilation and compile-time specialization.

## Key Features

- **Memory efficiency**: Optimized evaluation approach minimizes memory allocation during computation
- **Performance**: Faster single-row evaluation than building full model matrices
- **Compatibility**: Supports StatsModels.jl formulas, including interactions and transformations
- **Scenario analysis**: Memory-efficient variable override system for counterfactual analysis
- **Unified architecture**: Single compilation pipeline accommodates diverse formula structures
- **Ecosystem integration**: Compatible with GLM.jl, MixedModels.jl, and StandardizedPredictors.jl

## How It Works

Fit a model, convert data to a column table, compile once, then evaluate rows quickly without allocations.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/emfeltham/FormulaCompiler.jl")
```

## Quick Start

```julia
using FormulaCompiler, GLM, DataFrames, Tables

# Fit your model normally
df = DataFrame(
    y = randn(1000),
    x = randn(1000),
    z = abs.(randn(1000)) .+ 0.1,
    group = categorical(rand(["A", "B", "C"], 1000))
)

model = lm(@formula(y ~ x * group + log(z)), df)

# Compile once for efficient repeated evaluation  
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# Memory-efficient evaluation suitable for repeated calls
compiled(row_vec, data, 1)  # Zero allocations after warmup
```

## Core Interfaces

### Optimized Interface (Recommended for Performance-Critical Applications)

```julia
# Pre-compile for optimal performance
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# Evaluate individual rows
compiled(row_vec, data, 1)    # Row 1
compiled(row_vec, data, 100)  # Row 100

# Evaluate multiple rows
matrix = Matrix{Float64}(undef, 10, length(compiled))
modelrow!(matrix, compiled, data, 1:10)
```

### Convenient Interface

```julia
# Single row (allocating)
row_values = modelrow(model, data, 1)

# Multiple rows (allocating)
matrix = modelrow(model, data, [1, 5, 10, 50])

# In-place with automatic caching
row_vec = Vector{Float64}(undef, size(modelmatrix(model), 2))
modelrow!(row_vec, model, data, 1)  # Uses cache
```

### Object-Based Interface

```julia
# Create evaluator object
evaluator = ModelRowEvaluator(model, df)

# Efficient evaluation
result = evaluator(1)           # Row 1
evaluator(row_vec, 1)          # In-place evaluation
```

## Advanced Analysis: Population and Counterfactual Patterns

FormulaCompiler uses a unified row-wise architecture where population analysis is achieved through simple loops over individual observations:

```julia
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Population analysis pattern: individual + averaging
function compute_population_effects(compiled, data, var_overrides)
    n_rows = length(first(data))
    effects = Vector{Float64}(undef, n_rows)
    row_vec = Vector{Float64}(undef, length(compiled))

    # Build counterfactual data for the variable we want to override
    data_cf, cf_vecs = build_counterfactual_data(data, [:treatment], 1)
    treatment_cf = cf_vecs[1]

    for row in 1:n_rows
        # Set up counterfactual for this row
        update_counterfactual_row!(treatment_cf, row)
        update_counterfactual_replacement!(treatment_cf, true)  # Apply treatment

        # Evaluate with override
        compiled(row_vec, data_cf, row)
        effects[row] = sum(row_vec)  # or whatever summary you need
    end

    return mean(effects)  # Population average
end

# Policy analysis using counterfactual vectors
population_effect = compute_population_effects(compiled, data, [:treatment])
```

### Multi-Variable Sensitivity Analysis

Analyze multiple variables systematically using counterfactual vectors:

```julia
# Define analysis grid manually
treatment_values = [false, true]
dose_values = [50.0, 100.0, 150.0]
region_values = ["North", "South"]

# Create counterfactual data structure
data_cf, cf_vecs = build_counterfactual_data(data, [:treatment, :dose, :region], 1)
treatment_cf, dose_cf, region_cf = cf_vecs

# Evaluate all combinations (2×3×2 = 12 scenarios)
results = Matrix{Float64}(undef, 12, length(compiled))
row_vec = Vector{Float64}(undef, length(compiled))
scenario_idx = 1

for treatment in treatment_values
    for dose in dose_values
        for region in region_values
            # Set up counterfactual values
            update_counterfactual_replacement!(treatment_cf, treatment)
            update_counterfactual_replacement!(dose_cf, dose)
            update_counterfactual_replacement!(region_cf, region)

            # Evaluate for representative row (e.g., row 1)
            compiled(row_vec, data_cf, 1)
            results[scenario_idx, :] .= row_vec
            scenario_idx += 1
        end
    end
end
```

### Dynamic Counterfactual Modification

```julia
# Build counterfactual vectors for dynamic analysis
data_cf, cf_vecs = build_counterfactual_data(data, [:x, :y, :z], 1)
x_cf, y_cf, z_cf = cf_vecs

# Modify counterfactuals iteratively
row_vec = Vector{Float64}(undef, length(compiled))

# Initial state
update_counterfactual_replacement!(x_cf, 1.0)
compiled(row_vec, data_cf, 1)  # Baseline with x=1.0

# Add y override
update_counterfactual_replacement!(y_cf, 100.0)
compiled(row_vec, data_cf, 1)  # With x=1.0, y=100.0

# Bulk update multiple variables
update_counterfactual_replacement!(x_cf, 2.0)
update_counterfactual_replacement!(z_cf, 0.5)
compiled(row_vec, data_cf, 1)  # With x=2.0, y=100.0, z=0.5

# Reset individual variables by using original data values
original_y = getproperty(data, :y)[1]
update_counterfactual_replacement!(y_cf, original_y)  # Remove y override
```

## Ecosystem Integration

### GLM.jl Models

```julia
using GLM

# Linear models
linear_model = lm(@formula(mpg ~ hp * cyl + log(wt)), mtcars)
data_mtcars = Tables.columntable(mtcars)
compiled = compile_formula(linear_model, data_mtcars)

# Generalized linear models
logit_model = glm(@formula(vs ~ hp + wt), mtcars, Binomial(), LogitLink())
compiled_logit = compile_formula(logit_model, data_mtcars)

# Both work identically
row_vec = Vector{Float64}(undef, length(compiled))
compiled(row_vec, data_mtcars, 1)
```

### MixedModels.jl Integration

Automatically extracts fixed effects from mixed models:

```julia
using MixedModels

# Mixed model with random effects
mixed_model = fit(MixedModel, @formula(y ~ x + z + (1|group) + (1+x|cluster)), df)

# Compiles only the fixed effects: y ~ x + z
compiled = compile_formula(mixed_model, Tables.columntable(df))

# Random effects are automatically stripped
fixed_form = fixed_effects_form(mixed_model)  # Returns: y ~ x + z
```

### StandardizedPredictors.jl Integration

Standardized predictors are integrated (currently only `ZScore()`):

```julia
using StandardizedPredictors

contrasts = Dict(:x => ZScore(), :z => ZScore())
model = lm(@formula(y ~ x + z + group), df, contrasts=contrasts)
compiled = compile_formula(model, Tables.columntable(df))  # Standardization built-in
```

### Derivatives and Marginal Effects

FormulaCompiler provides memory-efficient computation of derivatives and marginal effects with standard errors:

```julia
# Build derivative evaluators with concrete types
vars = [:x, :z]
de_ad = derivativeevaluator_ad(compiled, data, vars)  # Returns ADEvaluator, zero allocations, higher accuracy (preferred)
de_fd = derivativeevaluator_fd(compiled, data, vars)  # Returns FDEvaluator, zero allocations, alternative
β = coef(model)

# Single-row marginal effect gradient (η case)
gβ = Vector{Float64}(undef, length(compiled))
me_eta_grad_beta!(gβ, de_fd, β, 1, :x)  # Zero allocations

# Standard error via delta method
Σ = vcov(model)
se = delta_method_se(gβ, Σ)  # Efficient computation

# Average marginal effects with FD backend
rows = 1:100
gβ_ame = Vector{Float64}(undef, length(compiled))
accumulate_ame_gradient!(gβ_ame, de_fd, β, rows, :x)  # Zero allocations per row
se_ame = delta_method_se(gβ_ame, Σ)

println("AME standard error for x: ", se_ame)
```

**Key capabilities:**
- **Dual backends**: Both achieve zero allocations. `derivativeevaluator_ad(...)` (ADEvaluator) preferred for higher accuracy and performance. `derivativeevaluator_fd(...)` (FDEvaluator) alternative with explicit step control.
- **Type dispatch**: Method selection based on concrete evaluator types, no keywords needed
- **η and μ cases**: Linear predictors and link function transformations
- **Delta method**: Standard error computation for marginal effects
- **Validated implementation**: Cross-validated against reference implementations

## Advanced Features

### Memory Efficiency

Counterfactual analysis uses type-stable CounterfactualVector for single-row perturbations:

```julia
# Traditional approach: copy entire columns for changes
traditional = copy(data.x)  # Full column copy
traditional[500_000] = 42.0  # Change one value

# FormulaCompiler approach: CounterfactualVector with O(1) memory overhead
cf_vec = counterfactualvector(data.x, 500_000)  # ~32 bytes
update_counterfactual_replacement!(cf_vec, 42.0)

# Identical interface, but minimal memory usage
traditional[500_000] == cf_vec[500_000]  # true, but cf_vec uses ~99.999% less memory
```

## Supported Formula Features

- **Basic terms**: `x`, `log(z)`, `x^2`, `(x > 0)`  
- **Boolean variables**: `Vector{Bool}` treated as continuous (false → 0.0, true → 1.0)
- **Categorical variables**: All contrast types, ordered/unordered
- **Interactions**: `x * group`, `x * y * z`, `log(z) * group`  
- **Functions**: `log`, `exp`, `sqrt`, `sin`, `cos`, `abs`, `^`, boolean operators
- **Complex formulas**: `x * log(z) * group + sqrt(abs(y)) + (x > mean(x))`
- **Standardized predictors**: ZScore, custom transformations
- **Mixed models**: Automatic fixed-effects extraction

## Performance Tips

1. Pre-compile formulas for repeated evaluation:
   ```julia
   data = Tables.columntable(df)
   compiled = compile_formula(model, data)  # Do once
   # Then call compiled() millions of times
   ```

2. Use column-table format for best performance:
   ```julia
   data = Tables.columntable(df)  # Convert once, reuse many times
   ```

3. Pre-allocate output vectors:
   ```julia
   row_vec = Vector{Float64}(undef, length(compiled))  # Reuse across calls
   ```

4. Batch operations when possible:
   ```julia
   # Better: batch evaluation
   matrix = Matrix{Float64}(undef, 1000, length(compiled))
   modelrow!(matrix, compiled, data, 1:1000)
   
   # Avoid: many single evaluations with allocation
   results = [modelrow(model, data, i) for i in 1:1000]
   ```

## Performance Characteristics

Comparative performance evaluation across formula types:

```julia
using BenchmarkTools

# Traditional approach (full model matrix construction)
@benchmark modelmatrix(model)[1, :]
# Note: Constructs entire model matrix, computationally intensive for large datasets

# FormulaCompiler optimized approach
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

@benchmark compiled(row_vec, data, 1)
# Performance improvement with reduced memory allocation
```

**Derivative computation**:
- Both ForwardDiff (`:ad`) and finite differences (`:fd`) backends achieve zero allocations
- `:ad` backend preferred: Higher accuracy (machine precision) and faster performance
- `:fd` backend alternative: Explicit step size control
- All marginal effects computations utilize preallocated buffers

## Architecture

FormulaCompiler achieves efficiency through a unified compilation pipeline that transforms statistical formulas into specialized, type-stable execution paths:

- **Position mapping**: Operations utilize compile-time position specialization
- **Adaptive dispatch**: Threshold-based approach (≤25 operations: recursive dispatch, >25 operations: generated functions) chosen based on Julia's compilation limits
- **Unified design**: Single compilation system accommodates diverse formula structures without special-case handling

## Use Cases

- Monte Carlo simulations: Millions of model evaluations
- Bootstrap resampling: Repeated matrix construction
- Marginal effects: Numerical derivatives require many evaluations
- Policy analysis: Evaluate many counterfactual scenarios
- Real-time applications: Low-latency prediction serving
- Large-scale inference: Memory-efficient batch processing

## Contributing

Contributions are welcome!

## Related Packages

- [Margins.jl](https://github.com/juliangehring/Margins.jl): Marginal effects (built on this package)
- [GLM.jl](https://github.com/JuliaStats/GLM.jl): Generalized linear models
- [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl): Mixed-effects models
- [StandardizedPredictors.jl](https://github.com/beacon-biosignals/StandardizedPredictors.jl): Standardized predictors
- [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl): Formula interface

## Documentation

- **[DIAGRAMS.md](DIAGRAMS.md)**: Complete visual guide with system architecture, technical implementation, and usage workflows
- **[categorical_handling.md](categorical_handling.md)**: Detailed explanation of categorical variable and interaction handling
- **[docs/diagrams/](docs/diagrams/)**: Individual diagram files for embedding in documentation

## License

MIT License. See [LICENSE](LICENSE) for details.
