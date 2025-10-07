# Counterfactual Analysis with FormulaCompiler.jl

## Overview

Counterfactual analysis examines hypothetical scenarios by systematically varying specific variables while holding others constant. FormulaCompiler.jl enables efficient counterfactual analysis through simple, direct approaches that maintain zero-allocation performance.

## Key Concepts

### What is Counterfactual Analysis?

Counterfactual analysis answers "what if" questions:
- **Treatment effects**: What would happen if individual i received treatment vs. control?
- **Policy evaluation**: How would outcomes change under a new policy?
- **Standardization**: What if all individuals had the same covariate profile?
- **Sensitivity**: How robust are results to parameter changes?

### FormulaCompiler's Approach

FormulaCompiler uses **simple data manipulation** rather than complex infrastructure:

1. **Direct modification**: Use `merge()` to create scenario data
2. **Loop-based evaluation**: Process individuals with simple loops
3. **Zero allocations**: Maintain performance throughout
4. **Memory efficiency**: No data duplication overhead

This "individual analysis + averaging" pattern is simpler, faster, and more flexible than specialized population infrastructure.

## Basic Usage

### Simple Treatment Effect

```julia
using FormulaCompiler, GLM, DataFrames, Tables, Statistics

# Prepare data
df = DataFrame(
    wage = rand(20000:80000, 1000),
    experience = rand(0:30, 1000),
    education = rand(["HS", "College", "Graduate"], 1000),
    treated = rand(Bool, 1000)
)

model = lm(@formula(wage ~ experience + education + treated), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)
β = coef(model)

# Create counterfactual scenarios
n_rows = length(data.treated)
data_treated = merge(data, (treated = fill(true, n_rows),))
data_control = merge(data, (treated = fill(false, n_rows),))

# Individual effect: person 100
row_vec = Vector{Float64}(undef, length(compiled))

compiled(row_vec, data_treated, 100)
outcome_treated = dot(β, row_vec)

compiled(row_vec, data_control, 100)
outcome_control = dot(β, row_vec)

individual_effect = outcome_treated - outcome_control
println("Individual 100 treatment effect: \$$(round(individual_effect, digits=2))")
```

### Population Average Effect

```julia
# Average treatment effect across population
n_individuals = nrow(df)
treatment_effects = Vector{Float64}(undef, n_individuals)

for i in 1:n_individuals
    compiled(row_vec, data_treated, i)
    treated_outcome = dot(β, row_vec)

    compiled(row_vec, data_control, i)
    control_outcome = dot(β, row_vec)

    treatment_effects[i] = treated_outcome - control_outcome
end

ate = mean(treatment_effects)
se_ate = std(treatment_effects) / sqrt(n_individuals)
println("Average Treatment Effect: \$$(round(ate, digits=2)) ± \$$(round(1.96*se_ate, digits=2))")
```

## Research Applications

### Labor Economics: Minimum Wage Analysis

```julia
# Wage determination model
df_labor = DataFrame(
    wage = 10.0 .+ rand(1000) * 20,
    experience = rand(1:30, 1000),
    education = rand(["HS", "College"], 1000),
    region = rand(["Urban", "Rural"], 1000)
)

model = lm(@formula(wage ~ experience + education + region), df_labor)
data = Tables.columntable(df_labor)
compiled = compile_formula(model, data)
β = coef(model)

# Policy scenarios
wage_policies = [12.0, 15.0, 18.0, 20.0]
n_rows = nrow(df_labor)

for min_wage in wage_policies
    # Scenario: all wages at or above minimum
    scenario_wages = max.(data.wage, min_wage)
    data_policy = merge(data, (wage = scenario_wages,))

    # Evaluate population effect
    outcomes = Vector{Float64}(undef, n_rows)
    row_vec = Vector{Float64}(undef, length(compiled))

    for i in 1:n_rows
        compiled(row_vec, data_policy, i)
        outcomes[i] = dot(β, row_vec)
    end

    avg_outcome = mean(outcomes)
    println("Minimum wage \$$(min_wage): mean outcome = \$$(round(avg_outcome, digits=2))")
end
```

### Health Economics: Treatment Effect Heterogeneity

```julia
# Treatment effect by age group
df_health = DataFrame(
    outcome = randn(1000),
    age = rand(20:80, 1000),
    treated = rand(Bool, 1000),
    comorbidities = rand(0:5, 1000)
)

model = lm(@formula(outcome ~ age * treated + comorbidities), df_health)
data = Tables.columntable(df_health)
compiled = compile_formula(model, data)
β = coef(model)

# Analyze by age groups
age_groups = [(20, 40, "Young"), (40, 60, "Middle"), (60, 80, "Old")]

for (min_age, max_age, label) in age_groups
    # Select age group
    age_mask = min_age .<= df_health.age .< max_age
    group_indices = findall(age_mask)

    if isempty(group_indices)
        continue
    end

    # Treatment effects for this group
    n_rows = length(data.treated)
    data_treated = merge(data, (treated = fill(true, n_rows),))
    data_control = merge(data, (treated = fill(false, n_rows),))

    effects = Vector{Float64}(undef, length(group_indices))
    row_vec = Vector{Float64}(undef, length(compiled))

    for (idx, i) in enumerate(group_indices)
        compiled(row_vec, data_treated, i)
        treated = dot(β, row_vec)

        compiled(row_vec, data_control, i)
        control = dot(β, row_vec)

        effects[idx] = treated - control
    end

    println("$(label) age group ($min_age-$max_age): ATE = $(round(mean(effects), digits=3))")
end
```

### Public Finance: Tax Policy Analysis

```julia
# Tax policy simulation
df_tax = DataFrame(
    income = rand(20000:200000, 1000),
    dependents = rand(0:5, 1000),
    region = rand(["Urban", "Rural"], 1000)
)

model = lm(@formula(income ~ dependents + region), df_tax)
data = Tables.columntable(df_tax)
compiled = compile_formula(model, data)
β = coef(model)

# Progressive tax scenarios
tax_scenarios = [
    ("flat_10", income -> 0.10 * income),
    ("flat_20", income -> 0.20 * income),
    ("progressive", income -> income < 50000 ? 0.10 * income : 0.10 * 50000 + 0.25 * (income - 50000))
]

n_rows = nrow(df_tax)
row_vec = Vector{Float64}(undef, length(compiled))

for (name, tax_fn) in tax_scenarios
    # After-tax income scenario
    after_tax = tax_fn.(data.income)
    data_scenario = merge(data, (income = data.income .- after_tax,))

    # Population outcomes
    outcomes = Vector{Float64}(undef, n_rows)
    for i in 1:n_rows
        compiled(row_vec, data_scenario, i)
        outcomes[i] = dot(β, row_vec)
    end

    println("$(name): mean after-tax outcome = \$$(round(mean(outcomes), digits=2))")
end
```

## Advanced Techniques

### Grid Analysis

Systematic exploration of parameter space:

```julia
# Multi-dimensional policy grid
experience_levels = [5, 10, 15, 20]
education_levels = ["HS", "College", "Graduate"]

n_scenarios = length(experience_levels) * length(education_levels)
scenario_results = Matrix{Float64}(undef, n_scenarios, n_rows)

scenario_idx = 1
for exp in experience_levels
    for edu in education_levels
        # Create scenario
        data_scenario = merge(data, (
            experience = fill(exp, n_rows),
            education = fill(edu, n_rows)
        ))

        # Evaluate
        for i in 1:n_rows
            compiled(row_vec, data_scenario, i)
            scenario_results[scenario_idx, i] = dot(β, row_vec)
        end

        scenario_idx += 1
    end
end

# Find optimal scenario
scenario_means = [mean(scenario_results[i, :]) for i in 1:n_scenarios]
best_idx = argmax(scenario_means)
println("Best scenario: $(best_idx) with mean = $(round(scenario_means[best_idx], digits=2))")
```

### Bootstrap Uncertainty Quantification

```julia
using Random

function bootstrap_ate(df, formula, n_boot=1000)
    Random.seed!(123)
    n_obs = nrow(df)
    boot_effects = Vector{Float64}(undef, n_boot)

    for b in 1:n_boot
        # Bootstrap sample
        boot_indices = rand(1:n_obs, n_obs)
        boot_df = df[boot_indices, :]

        # Fit model
        boot_model = lm(formula, boot_df)
        boot_data = Tables.columntable(boot_df)
        boot_compiled = compile_formula(boot_model, boot_data)
        boot_β = coef(boot_model)

        # Treatment scenarios
        n_boot_rows = nrow(boot_df)
        treated_data = merge(boot_data, (treated = fill(true, n_boot_rows),))
        control_data = merge(boot_data, (treated = fill(false, n_boot_rows),))

        # Compute ATE
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

# Compute confidence interval
boot_results = bootstrap_ate(df, @formula(wage ~ experience + education + treated), 500)
ci_lower = quantile(boot_results, 0.025)
ci_upper = quantile(boot_results, 0.975)
println("95% CI for ATE: [\$$(round(ci_lower, digits=2)), \$$(round(ci_upper, digits=2))]")
```

### Sensitivity Analysis

```julia
# Test robustness to parameter variations
function sensitivity_analysis(compiled, data, β, param, param_range)
    n_rows = length(first(data))
    row_vec = Vector{Float64}(undef, length(compiled))

    results = Vector{Float64}(undef, length(param_range))

    for (idx, value) in enumerate(param_range)
        # Create scenario
        data_scenario = merge(data, (param => fill(value, n_rows),))

        # Evaluate population
        outcomes = Vector{Float64}(undef, n_rows)
        for i in 1:n_rows
            compiled(row_vec, data_scenario, i)
            outcomes[i] = dot(β, row_vec)
        end

        results[idx] = mean(outcomes)
    end

    return results
end

# Analyze sensitivity to experience
exp_range = 0:5:30
sensitivity_results = sensitivity_analysis(compiled, data, β, :experience, exp_range)

println("Experience sensitivity:")
for (exp, result) in zip(exp_range, sensitivity_results)
    println("  Experience $(exp): outcome = $(round(result, digits=2))")
end
```

## Categorical Contrasts

For categorical variables, use the `contrastevaluator()` for efficient batch processing:

```julia
using CategoricalArrays

# Categorical data
df_cat = DataFrame(
    outcome = randn(1000),
    x = randn(1000),
    region = categorical(rand(["North", "South", "East", "West"], 1000))
)

model = lm(@formula(outcome ~ x * region), df_cat)
data = Tables.columntable(df_cat)
compiled = compile_formula(model, data)

# Contrast evaluator for zero-allocation batch processing
evaluator = contrastevaluator(compiled, data, [:region])
contrast_buf = Vector{Float64}(undef, length(compiled))

# Batch contrasts
n_rows = nrow(df_cat)
regional_effects = Vector{Float64}(undef, n_rows)

for i in 1:n_rows
    contrast_modelrow!(contrast_buf, evaluator, i, :region, "North", "South")
    regional_effects[i] = dot(coef(model), contrast_buf)
end

avg_contrast = mean(regional_effects)
println("Average North vs South effect: $(round(avg_contrast, digits=3))")
```

## Best Practices

### Categorical Variable Handling

**Critical**: When creating counterfactual data with categorical variables, preserve the categorical array structure:

```julia
using CategoricalArrays

# Correct: Preserves categorical structure
data_modified = merge(data, (treatment = fill("Drug", n_rows),))  # For string-valued categoricals

# For CategoricalArray, extract existing value:
drug_val = df.treatment[findfirst(==("Drug"), df.treatment)]
data_modified = merge(data, (treatment = fill(drug_val, n_rows),))
```

**Why this matters**: Categorical variables require knowledge of all possible levels for correct contrast coding. Using `fill()` or array comprehensions with plain strings loses this information, causing incorrect model matrix generation, especially in interaction terms.

### Performance Optimization

1. **Pre-allocate buffers**: Reuse output vectors
   ```julia
   row_vec = Vector{Float64}(undef, length(compiled))  # Once
   for i in 1:n  # Reuse
       compiled(row_vec, data, i)
   end
   ```

2. **Compile once**: Cache compiled formulas
   ```julia
   compiled = compile_formula(model, data)  # Once
   # Use many times with different scenarios
   ```

3. **Batch scenarios**: Group related evaluations
   ```julia
   scenarios = [scenario1, scenario2, scenario3]
   for scenario_data in scenarios
       # Process all individuals for this scenario
   end
   ```

### Statistical Rigor

**Causal Assumptions**:
- Unconfoundedness (no unmeasured confounders)
- Positivity (all treatments observable)
- Consistency (well-defined treatments)
- Correct model specification

**Uncertainty Quantification**:
- Bootstrap for confidence intervals
- Delta method for analytic standard errors
- Consider robust/clustered errors for panel data

**Sensitivity Testing**:
- Vary model specifications
- Test parameter ranges
- Examine subgroup heterogeneity

## Performance Characteristics

### Memory Efficiency

FormulaCompiler's approach using `merge()` is memory-efficient because:

1. **Named tuples share data**: `merge()` creates new structure but shares underlying column data
2. **Only modified columns copied**: Unchanged columns remain shared
3. **No allocation in hot loops**: The `compiled()` calls maintain zero allocations

Example memory usage:
```julia
# Original data: ~8MB for 1M rows × 1 Float64 column
original = (x = rand(1_000_000),)

# Modified data: Only adds ~8MB for new column, shares x
modified = merge(original, (y = fill(1.0, 1_000_000),))

# vs. full copy: Would be ~16MB
full_copy = (x = copy(original.x), y = fill(1.0, 1_000_000))
```

### Computational Performance

- **Core evaluation**: ~50ns per row (typical), 0 bytes allocated
- **Loop overhead**: Minimal, dominated by evaluation time
- **Scalability**: Linear in number of individuals and scenarios
- **Contrast evaluator**: 0 bytes for batch categorical contrasts

## Comparison with Alternative Approaches

### vs. Data Copying

**Data copying approach** (not recommended):
```julia
# Creates full duplicate for each scenario
data_treated_df = copy(df)
data_treated_df.treated .= true  # Modifies copy
```

**FormulaCompiler approach**:
```julia
# Shares underlying data, only new treatment column
data_treated = merge(data, (treated = fill(true, n_rows),))
```

### vs. Specialized Infrastructure

FormulaCompiler uses simple patterns instead of specialized population infrastructure:

- **Simpler**: Direct data manipulation > complex abstraction layers
- **Faster**: Loop patterns are highly optimized in Julia
- **More flexible**: Easy to customize analysis
- **Transparent**: Clear what's happening at each step

## Further Reading

- [Scenario Analysis Guide](docs/src/guide/scenarios.md) - Detailed patterns and examples
- [Advanced Features](docs/src/guide/advanced_features.md) - High-performance techniques
- [Examples](docs/src/examples.md) - Real-world applications across domains
- [API Reference](docs/src/api.md) - Complete function documentation

## References

### Econometric Theory
- Angrist & Pischke (2009): *Mostly Harmless Econometrics*
- Imbens & Rubin (2015): *Causal Inference for Statistics, Social, and Biomedical Sciences*

### Computational Methods
- Efron & Tibshirani (1994): *An Introduction to the Bootstrap*
- Cameron & Trivedi (2005): *Microeconometrics: Methods and Applications*
