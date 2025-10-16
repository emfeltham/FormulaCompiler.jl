# FormulaCompiler.jl

[![CI](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/ci.yml)
[![Documentation](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/docs.yml/badge.svg)](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://emfeltham.github.io/FormulaCompiler.jl/stable/)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://emfeltham.github.io/FormulaCompiler.jl/dev/)

FormulaCompiler.jl provides efficient per-row model matrix evaluation for Julia statistical models through position-mapped compilation and compile-time specialization. The package transforms StatsModels.jl formulas into specialized evaluators that achieve zero-allocation performance for repeated row-wise operations.

## Overview

The package resolves StatsModels.jl formula complexity at compile time, thereby enabling type-stable execution. This approach supports efficient counterfactual analysis, marginal effects computation, and scenarios requiring many model matrix evaluations.

Key characteristics include:

- Zero-allocation evaluation after initial compilation
- O(1) memory overhead for counterfactual analysis via variable override system
- Support for categorical mixtures in marginal effects calculations
- Dual automatic differentiation (via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)) and finite difference backends for derivatives
- Built on the StatsModels.jl ecosystem (GLM.jl, MixedModels.jl, StandardizedPredictors.jl) allowing users to work with the existing and flexible domain-specific language.

## Installation

The package is registered, and the current version is available via

```julia
using Pkg
Pkg.add(url="https://github.com/emfeltham/FormulaCompiler.jl")
```

## Basic Usage

```julia
using FormulaCompiler, GLM, DataFrames, Tables

# Fit model
df = DataFrame(
    y = randn(1000),
    x = randn(1000),
    z = abs.(randn(1000)) .+ 0.1,
    group = categorical(rand(["A", "B", "C"], 1000))
)

model = lm(@formula(y ~ x * group + log(z)), df)

# Compile formula once
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# Evaluate rows without allocation
compiled(row_vec, data, 1)  # Zero allocations after warmup
```

## Core Interfaces

### Direct Compilation Interface

The most efficient interface for performance-critical applications:

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

### Convenience Interface

Alternative interface with automatic caching:

```julia
# Single row (allocating)
row_values = modelrow(model, data, 1)

# Multiple rows (allocating)
matrix = modelrow(model, data, [1, 5, 10, 50])

# In-place with automatic caching
row_vec = Vector{Float64}(undef, size(modelmatrix(model), 2))
modelrow!(row_vec, model, data, 1)
```

### Object-Based Interface

Stateful evaluator for repeated use:

```julia
evaluator = ModelRowEvaluator(model, df)
result = evaluator(1)           # Row 1
evaluator(row_vec, 1)          # In-place evaluation
```

## Counterfactual Analysis

FormulaCompiler implements a type-stable counterfactual vector system that enables variable substitution with O(1) memory overhead, avoiding data duplication. This is particularly useful for policy analysis and treatment effect evaluation.

### Basic Counterfactual Evaluation

```julia
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build counterfactual data structure
data_cf, cf_vecs = build_counterfactual_data(data, [:treatment], 1)
treatment_cf = cf_vecs[1]

# Evaluate with counterfactual values
row_vec = Vector{Float64}(undef, length(compiled))
update_counterfactual_replacement!(treatment_cf, true)
compiled(row_vec, data_cf, 1)
```

### Multi-Variable Sensitivity Analysis

```julia
# Define analysis grid
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
            update_counterfactual_replacement!(treatment_cf, treatment)
            update_counterfactual_replacement!(dose_cf, dose)
            update_counterfactual_replacement!(region_cf, region)

            compiled(row_vec, data_cf, 1)
            results[scenario_idx, :] .= row_vec
            scenario_idx += 1
        end
    end
end
```

### Memory Efficiency

The counterfactual vector system uses O(1) memory regardless of dataset size:

```julia
# Traditional approach: full column copy
traditional = copy(data.x)
traditional[500_000] = 42.0

# CounterfactualVector approach: ~32 bytes overhead
cf_vec = counterfactualvector(data.x, 500_000)
update_counterfactual_replacement!(cf_vec, 42.0)

# Identical interface, minimal memory usage
@assert traditional[500_000] == cf_vec[500_000]
```

## Ecosystem Integration

### GLM.jl Models

```julia
using GLM

# Linear models
linear_model = lm(@formula(mpg ~ hp * cyl + log(wt)), mtcars)
compiled = compile_formula(linear_model, Tables.columntable(mtcars))

# Generalized linear models
logit_model = glm(@formula(vs ~ hp + wt), mtcars, Binomial(), LogitLink())
compiled_logit = compile_formula(logit_model, Tables.columntable(mtcars))
```

### MixedModels.jl Integration

The package automatically extracts fixed effects from mixed models:

```julia
using MixedModels

mixed_model = fit(MixedModel, @formula(y ~ x + z + (1|group) + (1+x|cluster)), df)
compiled = compile_formula(mixed_model, Tables.columntable(df))  # Fixed effects only
```

### StandardizedPredictors.jl Integration

Standardized predictors (via [StandardizedPredictors.jl](https://github.com/beacon-biosignals/StandardizedPredictors.jl)) are integrated during compilation (currently limited to currently `ZScore`):

```julia
using StandardizedPredictors

contrasts = Dict(:x => ZScore(), :z => ZScore())
model = lm(@formula(y ~ x + z + group), df, contrasts=contrasts)
compiled = compile_formula(model, Tables.columntable(df))
```

## Categorical Mixtures

FormulaCompiler supports categorical mixtures—weighted combinations of categorical levels—for efficient profile-based marginal effects computation:

```julia
using FormulaCompiler, Margins

# Create mixture specification
reference_grid = DataFrame(
    x = [1.0, 2.0, 3.0],
    group = mix("Treatment" => 0.6, "Control" => 0.4)
)

# Compile and evaluate
compiled = compile_formula(model, Tables.columntable(reference_grid))
output = Vector{Float64}(undef, length(compiled))
compiled(output, reference_grid, 1)

# Multiple mixture variables
grid = DataFrame(
    age = [30, 40, 50],
    treatment = mix("Control" => 0.3, "Drug_A" => 0.4, "Drug_B" => 0.3),
    dose = mix("Low" => 0.25, "Medium" => 0.5, "High" => 0.25)
)
compiled = compile_formula(model, Tables.columntable(grid))
```

Mixture evaluation maintains zero-allocation performance through compile-time specialization. Each mixture specification generates a specialized evaluation method.

## Derivatives

FormulaCompiler provides computational primitives for computing derivatives of model matrix rows with respect to continuous variables. For marginal effects, standard errors, and complete statistical workflows, see [Margins.jl](https://github.com/emfeltham/Margins.jl).

### Computational Primitives

The package provides zero-allocation Jacobian computation using both automatic differentiation (ForwardDiff) and finite differences:

```julia
using FormulaCompiler, GLM

# Build derivative evaluator
vars = [:x, :z]
de = derivativeevaluator(:ad, compiled, data, vars)

# Compute Jacobian: J[i,j] = ∂(model_matrix[i])/∂vars[j]
J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow!(J, de, 1)  # Zero allocations
```

### Marginal Effects

For marginal effects computation, use [Margins.jl](https://github.com/emfeltham/Margins.jl), which provides:

```julia
using Margins

# Marginal effects on linear predictor η = Xβ
g_eta = Vector{Float64}(undef, length(vars))
marginal_effects_eta!(g_eta, de, coef(model), 1)

# Marginal effects on mean response μ (with link function)
g_mu = Vector{Float64}(undef, length(vars))
marginal_effects_mu!(g_mu, de, coef(model), LogitLink(), 1)

# Standard errors via delta method
se = delta_method_se(g_eta, vcov(model))
```

### Backend Selection

While automatic differentiation is the strongly preferred default option, two backends are available:

- `:ad` (automatic differentiation via ForwardDiff): Recommended for standard formulas. Provides machine-precision accuracy and approximately 20% faster performance than finite differences.
- `:fd` (finite differences): Recommended for formulas containing boolean predicates. Guarantees zero allocations for all formula types.

Backend selection is specified when constructing the evaluator:

```julia
# Automatic differentiation (recommended)
de_ad = derivativeevaluator(:ad, compiled, data, [:x, :z])

# Finite differences (for boolean predicates)
de_fd = derivativeevaluator(:fd, compiled, data, [:x, :z])
```

Both backends achieve zero-allocation performance through pre-allocated buffers and in-place operations.

### Link Function Derivatives

FormulaCompiler provides computational primitives for the following GLM link functions (used by Margins.jl for computing marginal effects on the mean response μ):

- Identity
- Log
- Logit
- Probit
- Cloglog
- Cauchit
- Inverse
- Sqrt
- InverseSquare

## Supported Formula Features

The package supports the complete StatsModels.jl formula language:

- Basic terms: `x`, `log(z)`, `x^2`, `(x > 0)`
- Boolean variables: treated as continuous (false → 0.0, true → 1.0)
- Categorical variables: all contrast types, ordered and unordered
- Categorical mixtures: weighted combinations (e.g., `mix("A" => 0.6, "B" => 0.4)`)
- Interactions: two-way and multi-way (`x * group`, `x * y * z`)
- Mathematical functions: `log`, `exp`, `sqrt`, `sin`, `cos`, `abs`, `^`
- Boolean predicates: `(x > 0)`, `(z >= mean(z))`
- Complex formulas: `x * log(z) * group + sqrt(abs(y))`
- Standardized predictors: ZScore and custom transformations
- Mixed models: automatic fixed-effects extraction

## Performance Considerations

### Compilation Pattern

The package is designed for scenarios where compilation cost is amortized over many evaluations:

```julia
# Compile once
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# Evaluate many times (zero allocations)
for i in 1:n_rows
    compiled(row_vec, data, i)
end
```

### Memory Layout

Column-table format provides optimal performance:

```julia
data = Tables.columntable(df)  # Convert once, reuse
```

### Pre-allocation

Pre-allocate output vectors for zero-allocation performance:

```julia
row_vec = Vector{Float64}(undef, length(compiled))  # Reuse across calls
```

### Batch Operations

Batch evaluation is more efficient than individual allocating calls:

```julia
# Efficient: batch evaluation
matrix = Matrix{Float64}(undef, 1000, length(compiled))
modelrow!(matrix, compiled, data, 1:1000)

# Inefficient: repeated allocation
results = [modelrow(model, data, i) for i in 1:1000]
```

## Performance Characteristics

### Evaluation Performance

Typical performance for formula evaluation after compilation:

```julia
using BenchmarkTools

# Traditional approach
@benchmark modelmatrix(model)[1, :]  # Constructs entire matrix

# FormulaCompiler approach
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))
@benchmark compiled(row_vec, data, 1)  # Zero allocations
```

FormulaCompiler typically achieves a significant speedup compared to full model matrix construction for single-row evaluation, with zero allocations after compilation.

### Derivative Performance

Both derivative backends achieve zero-allocation performance:

- Automatic differentiation (`:ad`): approximately 50-60ns per row for standard formulas
- Finite differences (`:fd`): approximately 65-85ns per row for standard formulas

All operations use pre-allocated buffers, validated through comprehensive allocation tests.

## Architecture

The package uses a compilation pipeline based on position mapping:

- Position mapping: Formula terms are mapped to fixed scratch and output positions during compilation, with all position information encoded in type parameters.
- Type specialization: Each unique formula generates a specialized evaluation method with concrete types throughout.
- Adaptive dispatch: Small formulas (≤10 operations) use recursive dispatch; larger formulas use generated functions to respect Julia's compilation limits.
- Zero-allocation execution: All memory layouts are determined at compile time, enabling allocation-free runtime evaluation.

## Use Cases

The package is designed for applications requiring many model matrix evaluations:

- Marginal effects computation requiring numerical derivatives
- Monte Carlo simulations requiring millions of model evaluations
- Bootstrap resampling with repeated matrix construction
- Large-scale inference with memory constraints

## Related Packages

- [Margins.jl](https://github.com/juliangehring/Margins.jl): Marginal effects computation (uses FormulaCompiler.jl)
- [GLM.jl](https://github.com/JuliaStats/GLM.jl): Generalized linear models
- [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl): Mixed-effects models
- [StandardizedPredictors.jl](https://github.com/beacon-biosignals/StandardizedPredictors.jl): Standardized predictors
- [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl): Statistical formula interface
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl): Automatic differentiation

## Documentation

Additional documentation is available in the repository:

- [DIAGRAMS.md](DIAGRAMS.md): System architecture and usage workflows with visual guides
- [categorical_handling.md](categorical_handling.md): Categorical variable and interaction handling
- [docs/diagrams/](docs/diagrams/): Individual diagram files

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.

## License

MIT License. See [LICENSE](LICENSE) for details.
