# Counterfactual Analysis

FormulaCompiler.jl provides efficient counterfactual analysis through simple, direct data manipulation and loop patterns.

## Overview

FormulaCompiler enables counterfactual analysis through three straightforward approaches:

1. **Direct data modification** - Use `merge()` for simple scenarios (1-10 comparisons)
2. **Batch contrast evaluation** - Use `contrastevaluator()` for categorical contrasts (100+ comparisons)
3. **Population analysis** - Use simple loops over rows for aggregated effects

All approaches maintain zero-allocation performance and scale efficiently with dataset size.

## Approach 1: Direct Data Modification

### Basic Treatment Effect Analysis

The simplest approach for counterfactual analysis is to create modified versions of your data and compare outcomes:

```julia
using FormulaCompiler, GLM, DataFrames, Tables, Statistics

# Setup data and model
df = DataFrame(
    y = randn(1000),
    x = randn(1000),
    treatment = rand(Bool, 1000),
    age = rand(18:80, 1000)
)

model = lm(@formula(y ~ x * treatment + age), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)
β = coef(model)

# Create counterfactual scenarios
n_rows = length(data.treatment)
data_treated = merge(data, (treatment = fill(true, n_rows),))
data_control = merge(data, (treatment = fill(false, n_rows),))

# Compare individual outcomes under different treatments
row_vec = Vector{Float64}(undef, length(compiled))

# Individual 1: treated vs control
compiled(row_vec, data_treated, 1)
effect_treated = dot(β, row_vec)

compiled(row_vec, data_control, 1)
effect_control = dot(β, row_vec)

individual_effect = effect_treated - effect_control
println("Individual 1 treatment effect: $(round(individual_effect, digits=3))")
```

### Population Average Treatment Effects

Calculate average treatment effects across the population:

```julia
# Population analysis: loop over all individuals
n_individuals = nrow(df)
treatment_effects = Vector{Float64}(undef, n_individuals)

for i in 1:n_individuals
    # Effect if treated
    compiled(row_vec, data_treated, i)
    outcome_treated = dot(β, row_vec)

    # Effect if control
    compiled(row_vec, data_control, i)
    outcome_control = dot(β, row_vec)

    treatment_effects[i] = outcome_treated - outcome_control
end

# Summary statistics
avg_effect = mean(treatment_effects)
std_effect = std(treatment_effects)
println("Average treatment effect: $(round(avg_effect, digits=3)) ± $(round(std_effect, digits=3))")
```

### Multi-Variable Counterfactuals

Modify multiple variables simultaneously:

```julia
# Policy scenario: everyone gets treatment + standardized age
standard_age = 40
data_policy = merge(data, (
    treatment = fill(true, n_rows),
    age = fill(standard_age, n_rows)
))

# Compare baseline vs policy for each individual
policy_effects = Vector{Float64}(undef, n_individuals)

for i in 1:n_individuals
    # Baseline outcome
    compiled(row_vec, data, i)
    baseline = dot(β, row_vec)

    # Policy outcome
    compiled(row_vec, data_policy, i)
    policy = dot(β, row_vec)

    policy_effects[i] = policy - baseline
end

avg_policy_effect = mean(policy_effects)
println("Average policy effect: $(round(avg_policy_effect, digits=3))")
```

### Multiple Scenario Comparison

Compare several policy scenarios:

```julia
# Define scenarios to compare
scenarios = [
    ("baseline", data),
    ("universal_treatment", merge(data, (treatment = fill(true, n_rows),))),
    ("universal_control", merge(data, (treatment = fill(false, n_rows),))),
    ("young_treated", merge(data, (treatment = fill(true, n_rows), age = fill(30, n_rows)))),
    ("old_treated", merge(data, (treatment = fill(true, n_rows), age = fill(60, n_rows))))
]

# Evaluate all scenarios
results = Dict{String, Vector{Float64}}()

for (name, scenario_data) in scenarios
    outcomes = Vector{Float64}(undef, n_individuals)

    for i in 1:n_individuals
        compiled(row_vec, scenario_data, i)
        outcomes[i] = dot(β, row_vec)
    end

    results[name] = outcomes
end

# Compare scenario means
println("\nScenario comparison:")
for (name, outcomes) in results
    println("  $(name): mean = $(round(mean(outcomes), digits=3))")
end
```

## Approach 2: Categorical Contrasts with ContrastEvaluator

For repeated categorical variable comparisons, use the zero-allocation contrast evaluator:

### Basic Contrast Evaluation

```julia
using CategoricalArrays

# Data with categorical variable
df_cat = DataFrame(
    y = randn(1000),
    x = randn(1000),
    region = categorical(rand(["North", "South", "East", "West"], 1000))
)

model_cat = lm(@formula(y ~ x * region), df_cat)
data_cat = Tables.columntable(df_cat)
compiled_cat = compile_formula(model_cat, data_cat)

# Create contrast evaluator for zero-allocation batch processing
evaluator = contrastevaluator(compiled_cat, data_cat, [:region])
contrast_buf = Vector{Float64}(undef, length(compiled_cat))

# Single contrast: North vs South for individual 1
contrast_modelrow!(contrast_buf, evaluator, 1, :region, "North", "South")
regional_difference = dot(coef(model_cat), contrast_buf)
println("North vs South effect: $(round(regional_difference, digits=3))")
```

### Batch Contrast Processing

Process many contrasts with zero allocations:

```julia
# Compare all individuals: North vs South
n_rows = nrow(df_cat)
regional_effects = Vector{Float64}(undef, n_rows)

for i in 1:n_rows
    contrast_modelrow!(contrast_buf, evaluator, i, :region, "North", "South")
    regional_effects[i] = dot(coef(model_cat), contrast_buf)
end

println("Average North vs South effect: $(round(mean(regional_effects), digits=3))")
```

### Gradient Computation for Uncertainty

Compute parameter gradients for standard errors:

```julia
# Parameter gradient for delta method (FormulaCompiler computational primitive)
∇β = Vector{Float64}(undef, length(compiled_cat))
contrast_gradient!(∇β, evaluator, 1, :region, "North", "South", coef(model_cat))

# Standard error using delta method (requires Margins.jl)
using Margins
vcov_matrix = vcov(model_cat)
se = delta_method_se(∇β, vcov_matrix)
println("Standard error: $(round(se, digits=3))")
```

## Approach 3: Grid Analysis Patterns

### Systematic Parameter Exploration

Explore multiple parameter combinations:

```julia
# Define parameter grid
treatment_values = [false, true]
age_values = [30, 40, 50, 60]
x_values = [-1.0, 0.0, 1.0]

# Create all combinations
n_scenarios = length(treatment_values) * length(age_values) * length(x_values)
scenario_results = Matrix{Float64}(undef, n_scenarios, n_individuals)

scenario_idx = 1
for treat in treatment_values
    for age_val in age_values
        for x_val in x_values
            # Create scenario data
            scenario_data = merge(data, (
                treatment = fill(treat, n_rows),
                age = fill(age_val, n_rows),
                x = fill(x_val, n_rows)
            ))

            # Evaluate for all individuals
            for i in 1:n_individuals
                compiled(row_vec, scenario_data, i)
                scenario_results[scenario_idx, i] = dot(β, row_vec)
            end

            scenario_idx += 1
        end
    end
end

# Analyze results
scenario_means = [mean(scenario_results[i, :]) for i in 1:n_scenarios]
best_scenario = argmax(scenario_means)
println("Best scenario index: $best_scenario with mean outcome: $(round(scenario_means[best_scenario], digits=3))")
```

### Efficient Batched Evaluation

For very large grids, batch the evaluation:

```julia
function evaluate_scenario_grid(compiled, base_data, param_values, β)
    """Efficiently evaluate parameter grid"""
    n_rows = length(first(base_data))
    row_vec = Vector{Float64}(undef, length(compiled))

    results = Dict()

    for (name, values) in param_values
        # Create scenario
        scenario_data = merge(base_data, Dict(name => fill(values, n_rows)))

        # Evaluate population
        outcomes = Vector{Float64}(undef, n_rows)
        for i in 1:n_rows
            compiled(row_vec, scenario_data, i)
            outcomes[i] = dot(β, row_vec)
        end

        results[name => values] = mean(outcomes)
    end

    return results
end

# Usage
param_grid = Dict(
    :treatment => [true, false],
    :age => [30, 40, 50, 60]
)

grid_results = evaluate_scenario_grid(compiled, data, param_grid, β)
```

## Advanced Patterns

### Sensitivity Analysis

Test model sensitivity to parameter changes:

```julia
# Vary age systematically
age_range = 20:10:70
sensitivity_results = Vector{Float64}(undef, length(age_range))

for (idx, age_val) in enumerate(age_range)
    scenario_data = merge(data, (age = fill(age_val, n_rows),))

    outcomes = Vector{Float64}(undef, n_individuals)
    for i in 1:n_individuals
        compiled(row_vec, scenario_data, i)
        outcomes[i] = dot(β, row_vec)
    end

    sensitivity_results[idx] = mean(outcomes)
end

# Plot or analyze sensitivity
println("Age sensitivity:")
for (age_val, result) in zip(age_range, sensitivity_results)
    println("  Age $age_val: $(round(result, digits=3))")
end
```

### Bootstrap Confidence Intervals

Compute uncertainty via bootstrap:

```julia
using Random

function bootstrap_treatment_effect(df, model_formula, n_boot=1000)
    Random.seed!(123)
    n_obs = nrow(df)

    boot_effects = Vector{Float64}(undef, n_boot)

    for b in 1:n_boot
        # Bootstrap sample
        boot_indices = rand(1:n_obs, n_obs)
        boot_df = df[boot_indices, :]

        # Fit model
        boot_model = lm(model_formula, boot_df)
        boot_data = Tables.columntable(boot_df)
        boot_compiled = compile_formula(boot_model, boot_data)
        boot_β = coef(boot_model)

        # Create treatment scenarios
        n_boot_rows = nrow(boot_df)
        treated_data = merge(boot_data, (treatment = fill(true, n_boot_rows),))
        control_data = merge(boot_data, (treatment = fill(false, n_boot_rows),))

        # Compute average effect
        row_vec = Vector{Float64}(undef, length(boot_compiled))
        effects = Vector{Float64}(undef, n_boot_rows)

        for i in 1:n_boot_rows
            boot_compiled(row_vec, treated_data, i)
            treated = dot(boot_β, row_vec)

            boot_compiled(row_vec, control_data, i)
            control = dot(boot_β, row_vec)

            effects[i] = treated - control
        end

        boot_effects[b] = mean(effects)
    end

    return boot_effects
end

# Compute bootstrap CI
boot_results = bootstrap_treatment_effect(df, @formula(y ~ x * treatment + age), 500)
ci_lower = quantile(boot_results, 0.025)
ci_upper = quantile(boot_results, 0.975)
println("95% CI: [$(round(ci_lower, digits=3)), $(round(ci_upper, digits=3))]")
```

## Best Practices

### When to Use Each Approach

**Direct data modification** (`merge()`):
- Simple scenarios (1-10 comparisons)
- Exploratory analysis
- Quick prototyping
- Small to medium datasets

**Contrast evaluator** (`contrastevaluator()`):
- Categorical variable comparisons
- Batch processing (100+ contrasts)
- Need for uncertainty quantification
- Production pipelines

**Simple loops**:
- Population-level analysis
- Any scenario type
- Maximum flexibility
- Large-scale analysis

### Performance Tips

1. **Pre-allocate buffers**: Reuse `row_vec` and result vectors
2. **Compile once**: Cache compiled formulas across scenarios
3. **Batch operations**: Group related evaluations
4. **Use views**: Avoid unnecessary copies with `view()`

### Memory Efficiency

```julia
# Good: Pre-allocate and reuse
row_vec = Vector{Float64}(undef, length(compiled))
results = Vector{Float64}(undef, n_individuals)

for i in 1:n_individuals
    compiled(row_vec, scenario_data, i)
    results[i] = dot(β, row_vec)
end

# Avoid: Allocating each iteration
for i in 1:n_individuals
    row_vec = modelrow(compiled, scenario_data, i)  # Allocates!
    results[i] = dot(β, row_vec)
end
```

## Statistical Considerations

### Causal Interpretation

Remember that counterfactual estimates depend on modeling assumptions:
- **Unconfoundedness**: No unmeasured confounders
- **Positivity**: All individuals have positive probability of each treatment
- **Consistency**: Treatment definition is well-specified
- **Model specification**: Correct functional form

### Uncertainty Quantification

Account for parameter uncertainty:
- Use bootstrap for confidence intervals
- Apply delta method for analytic standard errors
- Consider robust/clustered standard errors when appropriate

### Sensitivity Analysis

Test robustness:
- Vary model specifications
- Check sensitivity to parameter ranges
- Examine heterogeneous effects across subgroups

## Integration with Statistical Workflows

### Model Comparison

```julia
# Compare models under fixed counterfactual
models = [
    lm(@formula(y ~ x + treatment), df),
    lm(@formula(y ~ x * treatment), df),
    lm(@formula(y ~ x * treatment + age), df)
]

scenario_data = merge(data, (treatment = fill(true, n_rows),))

for (i, model) in enumerate(models)
    compiled = compile_formula(model, data)
    β = coef(model)
    row_vec = Vector{Float64}(undef, length(compiled))

    predictions = Vector{Float64}(undef, n_individuals)
    for j in 1:n_individuals
        compiled(row_vec, scenario_data, j)
        predictions[j] = dot(β, row_vec)
    end

    println("Model $i mean prediction: $(round(mean(predictions), digits=3))")
end
```

## Further Reading

- [Advanced Features](advanced_features.md) - Additional computational patterns
- [Categorical Mixtures](categorical_mixtures.md) - Profile-based marginal effects
- [Examples](@ref) - Real-world applications
- [API Reference](@ref) - Complete function documentation
