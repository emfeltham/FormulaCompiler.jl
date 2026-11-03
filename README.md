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

The package is registered in the General registry:

```julia
using Pkg
Pkg.add("FormulaCompiler")
```

## Basic Usage

```julia
using FormulaCompiler, GLM, DataFrames, Tables, CategoricalArrays

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
for i in 1:10
    compiled(view(matrix, i, :), data, i)
end
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

The examples in this section assume a model whose data has a Bool column `treatment`, a Float64 column `dose` and a String column `region`. The counterfactual functions are not exported.

### Basic Counterfactual Evaluation

```julia
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build counterfactual data structure
data_cf, cf_vecs = FormulaCompiler.build_counterfactual_data(data, [:treatment], 1)
treatment_cf = cf_vecs[1]

# Evaluate with counterfactual values
row_vec = Vector{Float64}(undef, length(compiled))
FormulaCompiler.update_counterfactual_replacement!(treatment_cf, true)
compiled(row_vec, data_cf, 1)
```

### Multi-Variable Sensitivity Analysis

```julia
# Define analysis grid
treatment_values = [false, true]
dose_values = [50.0, 100.0, 150.0]
region_values = ["North", "South"]

# Create counterfactual data structure
data_cf, cf_vecs = FormulaCompiler.build_counterfactual_data(data, [:treatment, :dose, :region], 1)
treatment_cf, dose_cf, region_cf = cf_vecs

# Evaluate all combinations (2×3×2 = 12 scenarios)
results = Matrix{Float64}(undef, 12, length(compiled))
row_vec = Vector{Float64}(undef, length(compiled))
scenario_idx = 1

for treatment in treatment_values
    for dose in dose_values
        for region in region_values
            FormulaCompiler.update_counterfactual_replacement!(treatment_cf, treatment)
            FormulaCompiler.update_counterfactual_replacement!(dose_cf, dose)
            FormulaCompiler.update_counterfactual_replacement!(region_cf, region)

            compiled(row_vec, data_cf, 1)
            results[scenario_idx, :] .= row_vec
            scenario_idx += 1
        end
    end
end
```

### Memory Efficiency

A counterfactual vector wraps a column instead of copying it, so its memory does not depend on the number of rows:

```julia
x = randn(1_000_000)

# Traditional approach: full column copy (8.0 MB for 1,000,000 Float64 values)
traditional = copy(x)
traditional[500_000] = 42.0

# CounterfactualVector approach: the wrapper allocates 32 bytes
cf_vec = FormulaCompiler.counterfactualvector(x, 500_000)
FormulaCompiler.update_counterfactual_replacement!(cf_vec, 42.0)

# Identical interface, minimal memory usage
@assert traditional[500_000] == cf_vec[500_000]
```

(Measured with `@allocated` on an Apple M4 Pro, Julia 1.13.3.)

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

All three [StandardizedPredictors.jl](https://github.com/beacon-biosignals/StandardizedPredictors.jl) transformations are supported: `ZScore()`, `Center()`, and `Scale()`:

```julia
using StandardizedPredictors

contrasts = Dict(:x => ZScore(), :z => Center(), :w => Scale())
model = lm(@formula(y ~ x + z + w + group), df, contrasts=contrasts)
compiled = compile_formula(model, Tables.columntable(df))
```

## Categorical Mixtures

FormulaCompiler supports categorical mixtures—weighted combinations of categorical levels—for efficient profile-based marginal effects computation:

```julia
# Mixture of the levels of `group` in the model from Basic Usage
reference_grid = DataFrame(
    x = [1.0, 2.0, 3.0],
    z = [1.0, 1.0, 1.0],
    group = fill(mix("A" => 0.6, "B" => 0.4), 3)
)

# Compile and evaluate
grid_data = Tables.columntable(reference_grid)
compiled_grid = compile_formula(model, grid_data)
output = Vector{Float64}(undef, length(compiled_grid))
compiled_grid(output, grid_data, 1)  # group:B = 0.4, group:C = 0.0

# Several mixture columns work the same way, e.g. for a model with
# categorical `treatment` and `dose`:
#   treatment = fill(mix("Control" => 0.3, "Drug_A" => 0.4, "Drug_B" => 0.3), n)
#   dose = fill(mix("Low" => 0.25, "Medium" => 0.5, "High" => 0.25), n)
```

Mixture evaluation is allocation-free. The mixture's levels and weights are read from the data at evaluation time, so a compiled formula can be reused with different mixtures (a new mixture does not trigger recompilation).

## Derivatives

FormulaCompiler provides computational primitives for zero-allocation derivative computation. For marginal effects, cf. [Margins.jl](https://github.com/emfeltham/Margins.jl).

```julia
using FormulaCompiler, GLM

# Build derivative evaluator
vars = [:x, :z]
de = derivativeevaluator(:ad, compiled, data, vars)

# Compute Jacobian: J[i,j] = ∂modelmatrix[i]/∂vars[j]
J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow!(J, de, 1)  # Row 1
```

### Backend Selection

Two backends are available; automatic differentiation is the recommended default:

- `:ad` (automatic differentiation via ForwardDiff.jl): exact derivatives up to floating-point rounding.
- `:fd` (finite differences): central differences with an automatically chosen step; an alternative backend, for example to cross-check AD results.

For the Basic Usage model with `vars = [:x, :z]`, `derivative_modelrow!` took a median of about 35 ns with `:ad` and about 70 ns with `:fd` (BenchmarkTools, Apple M4 Pro, Julia 1.13.3). Both backends allocated 0 bytes after warmup, here and for formulas with boolean predicates (`(x > 0)`, `!(x > 0)`, `(z >= w)`). Both use pre-allocated buffers and in-place operations.

### Supported Link Functions

The following GLM link functions are supported for marginal effects on the mean response μ:

- Identity
- Log
- Logit
- Probit
- Cloglog
- Cauchit
- Inverse
- Sqrt
- InverseSquare
- Power
- NegativeBinomial

Any other `GLM.Link` falls back to forward-mode automatic differentiation of `GLM.linkinv`.

## Supported Formula Features

The package supports the StatsModels.jl formula language, with the functions listed below:

- Basic terms: `x`, `log(z)`, `x^2`, `(x > 0)`
- Boolean variables: treated as continuous (false → 0.0, true → 1.0)
- Categorical variables: all contrast types, ordered and unordered
- Categorical mixtures: weighted combinations (e.g., `mix("A" => 0.6, "B" => 0.4)`)
- Interactions: two-way and multi-way (`x * group`, `x * y * z`)
- Mathematical functions: `exp`, `log`, `log1p`, `sqrt`, `abs`, `sin`, `cos`, `inv`, and
  `+`, `-`, `*`, `/`, `^`. Any other function in a formula (including a user-defined one) is
  rejected by `compile_formula` with an error listing these; precompute it as a data column
- Boolean predicates: `(x > 0)`, `(z >= w)`, `!(x > 0)`
- Complex formulas: `x * log(z) * group + sqrt(abs(y))`
- Standardized predictors: ZScore, Center, Scale, and custom transformations
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
for i in 1:nrow(df)
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
# Efficient: batch evaluation with pre-allocation
matrix = Matrix{Float64}(undef, 1000, length(compiled))
for i in 1:1000
    compiled(view(matrix, i, :), data, i)
end

# Inefficient: repeated allocation
results = [modelrow(model, data, i) for i in 1:1000]
```

## Performance Characteristics

### Evaluation Performance

Single-row evaluation after compilation, compared with building the model matrix:

```julia
using BenchmarkTools, StatsModels

# Traditional approach: build the full model matrix from the data, take one row
@benchmark modelmatrix(formula($model), $df)[1, :]

# FormulaCompiler approach
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))
@benchmark $compiled($row_vec, $data, 1)
```

For the Basic Usage model (1,000 rows, 7 columns), on an Apple M4 Pro with Julia 1.13.3, the compiled row took a median of about 10 ns and allocated 0 bytes (also checked with `@allocated` after a warmup call). Building the model matrix took about 31 μs and allocated about 270 KB. (`modelmatrix(model)` itself returns the matrix stored in the fitted model, so it does not measure construction, and it cannot evaluate new or counterfactual data.)

### Derivative Performance

Both derivative backends allocate nothing after warmup. For the Basic Usage model with two variables, `derivative_modelrow!` took a median of about 35 ns (`:ad`) and 70 ns (`:fd`) per row in the setting above. Allocation-free evaluation is checked in the test suite.

### Marginal Effects

In the benchmark of the accompanying paper, which compares [Margins.jl](https://github.com/emfeltham/Margins.jl) (built on FormulaCompiler) with R's marginaleffects on matched estimands, the geometric means of the ratio of R to Julia run time were 42x at 5,000 rows, 19x at 50,000 rows and 17x at 500,000 rows (see the Citation section).

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
- Applications requiring low-latency prediction
- Large-scale inference with memory constraints

## Related Packages

- [Margins.jl](https://github.com/emfeltham/Margins.jl): Marginal effects computation (uses FormulaCompiler.jl)
- [GLM.jl](https://github.com/JuliaStats/GLM.jl): Generalized linear models
- [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl): Mixed-effects models
- [StandardizedPredictors.jl](https://github.com/beacon-biosignals/StandardizedPredictors.jl): Standardized predictors
- [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl): Statistical formula interface
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl): Automatic differentiation

## Documentation

Additional documentation is available in the [docs/](docs/) directory and the [online documentation](https://emfeltham.github.io/FormulaCompiler.jl/dev/).

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{feltham_formulacompilerjl_2026,
  title = {{{FormulaCompiler}}.Jl and {{Margins}}.Jl: {{Efficient Marginal Effects}} in {{Julia}}},
  shorttitle = {{{FormulaCompiler}}.Jl and {{Margins}}.Jl},
  author = {Feltham, Eric},
  year = {2026},
  month = jan,
  number = {arXiv:2601.07065},
  eprint = {2601.07065},
  primaryclass = {stat},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2601.07065},
  archiveprefix = {arXiv},
  keywords = {Statistics - Computation},
}
```
