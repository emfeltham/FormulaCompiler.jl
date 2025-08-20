# FormulaCompiler.jl

High-performance, zero-allocation model matrix evaluation/update for Julia statistical models. Useful for efficient marginal effect calculations, simulations, among other purposes.

## Key Features

- **Zero-allocation evaluation**: ~50-100ns per row, 0 bytes allocated
- **10x+ speedup** over `modelmatrix()` for single-row evaluations  
- **Advanced scenarios**: Override variables for policy analysis and counterfactuals
- **Analytical derivatives**: Built-in derivative compilation for marginal effects
- **Full ecosystem compatibility**: Works with GLM.jl, MixedModels.jl, StandardizedPredictors.jl
- **Extensible architecture**: Evaluator trees enable custom backends and analysis
- **Generality**: The system should work for any possible formula that works with StatsModels.jl.

## Installation

```julia
using Pkg
Pkg.add("FormulaCompiler")
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

# Compile once for fast evaluation
compiled = compile_formula(model)
data = Tables.columntable(df)
row_vec = Vector{Float64}(undef, length(compiled))

# Zero-allocation evaluation (call millions of times)
compiled(row_vec, data, 1)  # ~50ns, 0 allocations
```

## Core Interfaces

### Zero-Allocation Interface (Fastest)

```julia
# Pre-compile for maximum performance
compiled = compile_formula(model)
data = Tables.columntable(df)
row_vec = Vector{Float64}(undef, length(compiled))

# Evaluate single rows
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

# Zero-allocation evaluation
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
compiled = compile_formula(model)
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

## Advanced Features

### Evaluator Tree Access

Access the internal computation tree for advanced analysis:

```julia
compiled = compile_formula(model)

# Extract evaluator tree
evaluator = extract_root_evaluator(compiled)

# Analyze formula structure
variables = get_variable_dependencies(compiled)
complexity = count_evaluator_nodes(compiled)
summary = get_evaluator_summary(compiled)

# Pretty print tree structure
print_evaluator_tree(compiled)
```

### Analytical Derivatives

Built-in analytical derivative compilation (for integration with Margins.jl):

```julia
# Compute derivatives analytically
root_evaluator = extract_root_evaluator(compiled)
x_derivative = compute_derivative_evaluator(root_evaluator, :x)

# Evaluate derivatives with zero allocation
deriv_vec = Vector{Float64}(undef, 1)
evaluate!(x_derivative, deriv_vec, data, 1)
```

### Memory Efficiency

The scenario system uses `OverrideVector` for memory efficiency:

```julia
# Traditional approach: allocates 8MB for 1M rows
traditional = fill(42.0, 1_000_000)

# FormulaCompiler: allocates ~32 bytes
efficient = OverrideVector(42.0, 1_000_000)

# Both provide identical interface
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
   compiled = compile_formula(model)  # Do once
   # Then call compiled() millions of times
   ```

2. Use column-table format for best performance:
   ```julia
   data = Tables.columntable(df)  # Convert once
   ```

3. Pre-allocate output vectors:
   ```julia
   row_vec = Vector{Float64}(undef, length(compiled))  # Reuse
   ```

4. Batch operations when possible:
   ```julia
   # Better: batch evaluation
   matrix = Matrix{Float64}(undef, 1000, length(compiled))
   modelrow!(matrix, compiled, data, 1:1000)
   
   # Avoid: many single evaluations with allocation
   results = [modelrow(model, data, i) for i in 1:1000]
   ```

5. Clear cache periodically in long-running applications:
   ```julia
   clear_model_cache!()
   ```

## Benchmarks

Performance on a complex formula `y ~ x * group + log(z) + x^2 + sqrt(abs(z))`:

```julia
using BenchmarkTools

# Traditional approach
@benchmark modelmatrix(model)[1, :]
# ~10.2 μs (1 allocation: 896 bytes)

# FormulaCompiler
@benchmark compiled(row_vec, data, 1)
# ~85.3 ns (0 allocations: 0 bytes)

# Speedup: ~120x faster, zero allocations
```

## Use Cases

- Monte Carlo simulations: Millions of model evaluations
- Bootstrap resampling: Repeated matrix construction
- Marginal effects: Numerical derivatives require many evaluations
- Policy analysis: Evaluate many counterfactual scenarios
- Real-time applications: Low-latency prediction serving
- Large-scale inference: Memory-efficient batch processing

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Related Packages

- [Margins.jl](https://github.com/juliangehring/Margins.jl): Marginal effects (uses FormulaCompiler for speed)
- [GLM.jl](https://github.com/JuliaStats/GLM.jl): Generalized linear models
- [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl): Mixed-effects models
- [StandardizedPredictors.jl](https://github.com/beacon-biosignals/StandardizedPredictors.jl): Standardized predictors
- [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl): Formula interface

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built for the Julia statistical ecosystem. Optimized for performance, designed for usability.*