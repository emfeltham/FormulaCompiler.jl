# FormulaCompiler.jl

[![CI](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/ci.yml)
[![Documentation](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/docs.yml/badge.svg)](https://github.com/emfeltham/FormulaCompiler.jl/actions/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://emfeltham.github.io/FormulaCompiler.jl/stable/)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://emfeltham.github.io/FormulaCompiler.jl/dev/)

Efficient model matrix evaluation for Julia statistical models. Implements position-mapping compilation to achieve performance improvements over traditional model matrix construction through compile-time specialization.

## Key Features

- **Memory efficiency**: Optimized evaluation approach minimizes memory allocation during computation
- **Performance improvement**: Computational advantages over traditional model matrix construction methods
- **Comprehensive compatibility**: Supports all valid StatsModels.jl formulas, including complex interactions and functions
- **Scenario analysis**: Memory-efficient variable override system for counterfactual analysis
- **Unified architecture**: Single compilation pipeline accommodates diverse formula structures
- **Ecosystem integration**: Compatible with GLM.jl, MixedModels.jl, and StandardizedPredictors.jl

## How It Works

The workflow involves fitting a statistical model, preparing data in column-table format, compiling the formula for optimized evaluation, and then evaluating individual rows efficiently.

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
compiled(row_vec, data, 1)  # Efficient evaluation
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

## Advanced Scenario Analysis

Create data scenarios with variable overrides for policy analysis and counterfactuals:

```julia
data = Tables.columntable(df)

# Create policy scenarios
baseline = create_scenario("baseline", data)
treatment = create_scenario("treatment", data; 
    treatment = true,
    dose = 100.0
)
policy_change = create_scenario("policy", data;
    x = mean(df.x),           # Set to population mean
    group = "A",              # Override categorical
    regulatory = true         # Add policy variable
)

# Evaluate scenarios
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

compiled(row_vec, baseline.data, 1)      # Original data
compiled(row_vec, treatment.data, 1)     # With treatment
compiled(row_vec, policy_change.data, 1) # Policy scenario
```

### Scenario Grids

Generate comprehensive scenario combinations:

```julia
# Create all combinations
policy_grid = create_scenario_grid("policy_analysis", data, Dict(
    :treatment => [false, true],
    :dose => [50.0, 100.0, 150.0],
    :region => ["North", "South"]
))

# Evaluates 2×3×2 = 12 scenarios
results = Matrix{Float64}(undef, length(policy_grid), length(compiled))
for (i, scenario) in enumerate(policy_grid)
    compiled(view(results, i, :), scenario.data, 1)
end
```

### Dynamic Scenario Modification

```julia
scenario = create_scenario("dynamic", data; x = 1.0)

# Modify scenarios iteratively
set_override!(scenario, :y, 100.0)           # Add override
update_scenario!(scenario; x = 2.0, z = 0.5) # Bulk update  
remove_override!(scenario, :y)               # Remove override
```

## Ecosystem Integration

### GLM.jl Models

```julia
using GLM

# Linear models
linear_model = lm(@formula(mpg ~ hp * cyl + log(wt)), mtcars)
compiled = compile_formula(linear_model)

# Generalized linear models
logit_model = glm(@formula(vs ~ hp + wt), mtcars, Binomial(), LogitLink())
compiled_logit = compile_formula(logit_model)

# Both work identically
row_vec = Vector{Float64}(undef, length(compiled))
compiled(row_vec, Tables.columntable(mtcars), 1)
```

### MixedModels.jl Integration

Automatically extracts fixed effects from mixed models:

```julia
using MixedModels

# Mixed model with random effects
mixed_model = fit(MixedModel, @formula(y ~ x + z + (1|group) + (1+x|cluster)), df)

# Compiles only the fixed effects: y ~ x + z
compiled = compile_formula(mixed_model)

# Random effects are automatically stripped
fixed_form = fixed_effects_form(mixed_model)  # Returns: y ~ x + z
```

### StandardizedPredictors.jl integration

Standardized predictors are integrated (currently only `ZScore()`):

```julia
using StandardizedPredictors

contrasts = Dict(:x => ZScore(), :z => ZScore())
model = lm(@formula(y ~ x + z + group), df, contrasts=contrasts)
compiled = compile_formula(model)  # Standardization built-in
```

### Derivatives and Marginal Effects

FormulaCompiler provides memory-efficient computation of derivatives and marginal effects with standard errors:

```julia
# Build derivative evaluator
vars = [:x, :z]
de = build_derivative_evaluator(compiled, data; vars=vars)
β = coef(model)

# Single-row marginal effect gradient (η case)
gβ = Vector{Float64}(undef, length(compiled))
me_eta_grad_beta!(gβ, de, β, 1, :x)  # Minimal memory allocation

# Standard error via delta method
Σ = vcov(model)
se = delta_method_se(gβ, Σ)  # Efficient computation

# Average marginal effects with backend selection
rows = 1:100
gβ_ame = Vector{Float64}(undef, length(compiled))
accumulate_ame_gradient!(gβ_ame, de, β, rows, :x; backend=:fd)  # Memory-efficient computation
se_ame = delta_method_se(gβ_ame, Σ)

println("AME standard error for x: ", se_ame)
```

**Key capabilities:**
- **Dual backends**: `:fd` (memory-efficient) and `:ad` (higher numerical accuracy)  
- **η and μ cases**: Linear predictors and link function transformations
- **Delta method**: Standard error computation for marginal effects
- **Validated implementation**: Cross-validated against reference implementations

## Advanced Features

### Memory Efficiency

The scenario system employs `OverrideVector` for efficient data representation:

```julia
# Traditional approach: memory allocation for large datasets
traditional = fill(42.0, 1_000_000)  # ~8MB allocation

# FormulaCompiler approach: reduced memory overhead
efficient = OverrideVector(42.0, 1_000_000)  # ~32 bytes allocation

# Identical computational interface
traditional[500_000] == efficient[500_000]  # true
```

## Supported Formula Features

- Basic terms: `x`, `log(z)`, `x^2`, `(x > 0)`
- Categorical variables: All contrast types, ordered/unordered
- Interactions: `x * group`, `x * y * z`, `log(z) * group`
- Functions: `log`, `exp`, `sqrt`, `sin`, `cos`, `abs`, `^`, boolean operators
- Complex formulas: `x * log(z) * group + sqrt(abs(y)) + (x > mean(x))`
- Standardized predictors: ZScore, custom transformations
- Mixed models: Automatic fixed-effects extraction

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
- ForwardDiff-based operations involve some per-call allocations due to automatic differentiation requirements
- Finite difference backend provides alternative with validation against automatic differentiation results
- Marginal effects computations utilize preallocated buffers

The automatic differentiation backend for batch gradient operations (`accumulate_ame_gradient!`) involves allocations. Users with strict memory constraints should utilize the `:fd` backend, which provides mathematically equivalent results with reduced memory overhead.

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
