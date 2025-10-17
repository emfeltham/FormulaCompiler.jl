# Population Analysis Using Row-Wise Functions

This guide shows how to perform population-level analysis using row-wise marginal effects functions. The key insight is that **population effects are simply averages over individual effects** - simple loops over rows are all you need.

> **Note**: Marginal effects functions (`marginal_effects_eta!`, `marginal_effects_mu!`, etc.) are provided by [Margins.jl](https://github.com/emfeltham/Margins.jl). FormulaCompiler provides the computational primitives (`derivative_modelrow!`) that Margins.jl builds upon. Install Margins.jl to use the examples in this guide: `using Pkg; Pkg.add(url="https://github.com/emfeltham/Margins.jl")`

## Core Principle

```julia
# Population Average Marginal Effect (AME)
AME = (1/n) × Σᵢ ∂f(xᵢ)/∂x

# Implementation: Individual effects + averaging
population_effects = Vector{Float64}(undef, n_rows)
for (i, row) in enumerate(rows)
    marginal_effects_eta!(temp_buffer, de, β, row)  # Existing row-wise function
    population_effects[i] = temp_buffer[var_index]
end
population_ame = mean(population_effects)  # Simple arithmetic
```

## Computing Population Marginal Effects

### Basic Pattern

Use `marginal_effects_eta!()` in a loop to compute Average Marginal Effects (AME):

```julia
using FormulaCompiler, Margins, GLM, DataFrames, Tables

# Setup: Fit model and compile
df = DataFrame(
    y = randn(1000),
    x = randn(1000),
    age = rand(18:80, 1000),
    income = exp.(randn(1000))
)
model = lm(@formula(y ~ x + age + log(income)), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build derivative evaluator for variables of interest
de = derivativeevaluator(:fd, compiled, data, [:x, :age])
β = coef(model)

# Compute population marginal effects
n_rows = length(df.y)
temp_buffer = Vector{Float64}(undef, length(de.vars))
ame_results = zeros(length(de.vars))

for row in 1:n_rows
    # Compute marginal effects for this individual
    marginal_effects_eta!(temp_buffer, de, β, row)

    # Accumulate for population average
    ame_results .+= temp_buffer
end

# Population AME = average over individuals
ame_results ./= n_rows

println("Population Average Marginal Effects:")
for (i, var) in enumerate(de.vars)
    println("  $var: $(ame_results[i])")
end
```

### Weighted Population Effects

For survey data or other weighted analyses:

```julia
# Assume we have survey weights
weights = rand(0.5:0.1:2.0, n_rows)  # Example weights
total_weight = sum(weights)

# Weighted population marginal effects
weighted_ame = zeros(length(de.vars))

for row in 1:n_rows
    marginal_effects_eta!(temp_buffer, de, β, row)

    # Weight each individual's contribution
    weighted_ame .+= weights[row] .* temp_buffer
end

# Weighted average
weighted_ame ./= total_weight

println("Weighted Population AME:")
for (i, var) in enumerate(de.vars)
    println("  $var: $(weighted_ame[i])")
end
```

## Scenario Analysis Using Data Modification

### Single Variable Scenario

Evaluate "what if all individuals had a specific value for a variable":

```julia
# Scenario: What if everyone had income = 50,000?
scenario_var = :income
scenario_value = log(50000)  # Model uses log(income)

# Create counterfactual data with modified variable
cf_data = merge(data, (income = fill(scenario_value, n_rows),))

# Evaluate scenario for each individual
scenario_predictions = Vector{Float64}(undef, n_rows)
output_buffer = Vector{Float64}(undef, length(compiled))

for row in 1:n_rows
    # Evaluate model with counterfactual data
    compiled(output_buffer, cf_data, row)

    # Extract prediction (assuming single outcome)
    scenario_predictions[row] = dot(β, output_buffer)
end

# Population-level scenario results
population_scenario_mean = mean(scenario_predictions)
println("Population mean under scenario: $population_scenario_mean")

# Compare with baseline
baseline_predictions = Vector{Float64}(undef, n_rows)
for row in 1:n_rows
    compiled(output_buffer, data, row)
    baseline_predictions[row] = dot(β, output_buffer)
end

baseline_mean = mean(baseline_predictions)
scenario_effect = population_scenario_mean - baseline_mean
println("Population scenario effect: $scenario_effect")
```

### Multiple Variable Scenarios

Policy analysis with multiple variables changed simultaneously:

```julia
# Policy scenario: Universal basic income + education investment
policy_vars = [:income, :education_years]
policy_values = [log(30000), 16.0]  # $30k income, 16 years education

# Create counterfactual data with multiple modified variables
cf_data = merge(data, (
    income = fill(policy_values[1], n_rows),
    education_years = fill(policy_values[2], n_rows)
))

# Evaluate policy for each individual
policy_predictions = Vector{Float64}(undef, n_rows)

for row in 1:n_rows
    # Evaluate under policy
    compiled(output_buffer, cf_data, row)
    policy_predictions[row] = dot(β, output_buffer)
end

# Policy impact analysis
policy_effect = mean(policy_predictions) - baseline_mean
println("Policy effect: $policy_effect")

# Distribution analysis
println("Policy effect distribution:")
println("  Min: $(minimum(policy_predictions - baseline_predictions))")
println("  Max: $(maximum(policy_predictions - baseline_predictions))")
println("  Std: $(std(policy_predictions - baseline_predictions))")
```

## Comparing Multiple Scenarios

Systematic policy comparison:

```julia
# Define scenarios to compare
scenarios = [
    (name="Baseline", vars=Symbol[], values=Float64[]),
    (name="Income +20%", vars=[:income], values=[log(1.2)]),  # Relative increase
    (name="Education +2yr", vars=[:education_years], values=[2.0]),  # Absolute increase
    (name="Combined", vars=[:income, :education_years], values=[log(1.2), 2.0])
]

scenario_results = Dict{String, Vector{Float64}}()

for scenario in scenarios
    if isempty(scenario.vars)
        # Baseline: use original data
        predictions = Vector{Float64}(undef, n_rows)
        for row in 1:n_rows
            compiled(output_buffer, data, row)
            predictions[row] = dot(β, output_buffer)
        end
        scenario_results[scenario.name] = predictions
    else
        # Counterfactual scenario - build modified data
        # Handle relative vs absolute changes
        modified_cols = Dict{Symbol, Vector{Float64}}()

        for (i, (var, value)) in enumerate(zip(scenario.vars, scenario.values))
            if var == :income  # Relative change
                # Add log(1.2) to each individual's log(income)
                modified_cols[var] = getproperty(data, var) .+ value
            else  # Absolute change
                # Add fixed value to each individual's value
                modified_cols[var] = getproperty(data, var) .+ value
            end
        end

        cf_data = merge(data, NamedTuple(modified_cols))

        predictions = Vector{Float64}(undef, n_rows)
        for row in 1:n_rows
            compiled(output_buffer, cf_data, row)
            predictions[row] = dot(β, output_buffer)
        end
        scenario_results[scenario.name] = predictions
    end
end

# Compare scenarios
println("Scenario Comparison:")
baseline = scenario_results["Baseline"]
for (name, predictions) in scenario_results
    if name != "Baseline"
        effect = mean(predictions) - mean(baseline)
        println("  $name: $(round(effect, digits=4))")
    end
end
```

## Performance Tips for Large Datasets

### Buffer Reuse

For large datasets, reuse buffers to minimize allocations:

```julia
# Pre-allocate all buffers
n_rows = length(df.y)
n_vars = length(de.vars)
temp_buffer = Vector{Float64}(undef, n_vars)
output_buffer = Vector{Float64}(undef, length(compiled))
ame_accumulator = zeros(n_vars)

# Efficient loop with buffer reuse
for row in 1:n_rows
    # Reuse temp_buffer for each row
    marginal_effects_eta!(temp_buffer, de, β, row)
    ame_accumulator .+= temp_buffer
end

ame_results = ame_accumulator ./ n_rows
```

### Batch Processing

For very large datasets, process in batches:

```julia
function compute_population_ame_batched(de, β, batch_size=1000)
    n_rows = length(getproperty(de.base_data, first(de.vars)))
    n_vars = length(de.vars)

    ame_accumulator = zeros(n_vars)
    temp_buffer = Vector{Float64}(undef, n_vars)

    for batch_start in 1:batch_size:n_rows
        batch_end = min(batch_start + batch_size - 1, n_rows)

        for row in batch_start:batch_end
            marginal_effects_eta!(temp_buffer, de, β, row)
            ame_accumulator .+= temp_buffer
        end

        # Optional: progress reporting
        if batch_end % (10 * batch_size) == 0
            println("Processed $(batch_end)/$(n_rows) observations")
        end
    end

    return ame_accumulator ./ n_rows
end

# Usage
ame_results = compute_population_ame_batched(de, β, 5000)
```

### Parallel Processing

For massive datasets, use threading:

```julia
using Base.Threads

function compute_population_ame_parallel(de, β)
    n_rows = length(getproperty(de.base_data, first(de.vars)))
    n_vars = length(de.vars)

    # Thread-local accumulators
    thread_accumulators = [zeros(n_vars) for _ in 1:nthreads()]

    @threads for row in 1:n_rows
        tid = threadid()
        temp_buffer = Vector{Float64}(undef, n_vars)

        marginal_effects_eta!(temp_buffer, de, β, row)
        thread_accumulators[tid] .+= temp_buffer
    end

    # Combine thread results
    total_accumulator = zeros(n_vars)
    for acc in thread_accumulators
        total_accumulator .+= acc
    end

    return total_accumulator ./ n_rows
end

# Usage (requires Julia started with multiple threads)
ame_results = compute_population_ame_parallel(de, β)
```

## Implementation Properties

### Memory Efficiency

- **O(1) memory usage**: Independent of dataset size
- **No data copying**: CounterfactualVector provides transparent value substitution
- **Buffer reuse**: Same temporary arrays used across all rows

### Performance Characteristics

- **Zero allocations**: After warmup, row-wise functions allocate 0 bytes
- **Cache efficiency**: Sequential row processing optimizes memory access
- **Compiler optimization**: Simple loops enable aggressive optimization

### Simplicity

- **No special API**: Uses existing, well-tested row-wise functions
- **Clear semantics**: Population = individual + averaging (mathematically obvious)
- **Easy debugging**: Can inspect individual-level results before averaging

### Extensibility

- **Custom aggregation**: Not limited to simple averages (can use quantiles, weighted averages, etc.)
- **Conditional analysis**: Easy to subset rows or apply complex filters
- **Scenario combinations**: Natural composition of multiple counterfactual changes

## Migration from Population Functions

If you were previously using population-specific functions:

```julia
# OLD: Population-specific API (removed in v1.1.0)
scenario = create_scenario("policy", data; income = 50000)  # REMOVED
compiled_scenario = compile_formula(model, scenario.data)
population_effect = compute_population_effect(compiled_scenario, ...)

# CURRENT: Simple data modification + loops
cf_data = merge(data, (income = fill(log(50000), n_rows),))

effects = Vector{Float64}(undef, n_rows)
for row in 1:n_rows
    compiled(output_buffer, cf_data, row)
    effects[row] = dot(β, output_buffer)
end
population_effect = mean(effects)
```

The row-wise approach is:
- **Faster**: Simple loops with minimal overhead
- Extensible aggregation: Supports custom aggregation functions beyond simple averaging
- **More transparent**: Can examine individual-level results
- **Memory efficient**: O(1) usage vs O(n) for data copying approaches

## Summary

Population analysis in FormulaCompiler is achieved through:

1. **Individual-level computation**: Use existing row-wise functions (`marginal_effects_eta!`, etc.)
2. **Counterfactual scenarios**: Use `CounterfactualVector` system for single-observation perturbations
3. **Simple aggregation**: Apply standard arithmetic (mean, weighted average, etc.)

This simple loop-based approach provides superior performance, flexibility, and clarity.