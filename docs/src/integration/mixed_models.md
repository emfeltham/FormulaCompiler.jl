# MixedModels.jl Integration

FormulaCompiler.jl integrates with MixedModels.jl by automatically extracting fixed effects from mixed-effects models, enabling zero-allocation evaluation of the fixed-effects portion.

MixedModels is a weak dependency: the support lives in a package extension that Julia loads
automatically once both packages are loaded (`using MixedModels, FormulaCompiler`, in either
order). FormulaCompiler itself does not load MixedModels.

## Overview

Mixed models contain both fixed and random effects. FormulaCompiler.jl focuses on the fixed effects portion, which is often needed for:
- Marginal effects calculation
- Prediction with population-level effects
- Bootstrap inference on fixed effects
- Policy analysis scenarios

## Basic Integration

```julia
using MixedModels, FormulaCompiler, DataFrames, Tables

# Example dataset
df = DataFrame(
    y = randn(1000),
    x = randn(1000),
    treatment = rand(Bool, 1000),
    age = rand(18:80, 1000),
    group = rand(1:10, 1000),
    cluster = rand(1:50, 1000)
)
data = Tables.columntable(df)

# Fit mixed model
mixed_model = fit(MixedModel, @formula(y ~ x + treatment + (1|group) + (1+x|cluster)), df)

# FormulaCompiler automatically extracts fixed effects: y ~ x + treatment
compiled = compile_formula(mixed_model, data)
length(compiled)  # 3: intercept, x, treatment
```

## Fixed Effects Extraction

### Automatic Extraction

```julia
# Mixed model with various random effects structures
mixed_model = fit(MixedModel, @formula(y ~ x * treatment + age + (1|group) + (x|cluster)), df)

# Fixed effects formula (a FormulaTerm with the schema applied)
fixed_formula = fixed_effects_form(mixed_model)
println("Fixed effects: ", fixed_formula)  # y ~ 1 + x + treatment + age + x & treatment

# Compile fixed effects only
compiled = compile_formula(mixed_model, data)
```

### Manual Fixed Effects

```julia
using GLM

# The fixed-effects part of a MixedModels formula is the first element of its rhs tuple
fixed_form = mixed_model.formula.rhs[1]  # MatrixTerm of the fixed-effects terms
manual_model = lm(FormulaTerm(mixed_model.formula.lhs, fixed_form), df)
compiled_manual = compile_formula(manual_model, data)

# Equivalent, via the exported hook
manual_model = lm(fixed_effects_form(mixed_model), df)
```

## Use Cases

### Population-Level Predictions

```julia
using LinearAlgebra

function population_predictions(mixed_model, scenarios)
    results = Dict{String, Vector{Float64}}()

    for (scenario_name, scenario_data) in scenarios
        # Compile fixed effects (compile_formula caches by model and column types)
        compiled = compile_formula(mixed_model, scenario_data)
        row_vec = Vector{Float64}(undef, length(compiled))
        n_obs = Tables.rowcount(scenario_data)
        predictions = Vector{Float64}(undef, n_obs)

        for i in 1:n_obs
            compiled(row_vec, scenario_data, i)
            # This gives the fixed effects linear predictor
            predictions[i] = dot(fixef(mixed_model), row_vec)
        end

        results[scenario_name] = predictions
    end

    return results
end

# Example usage with data modification
n_rows = length(data.treatment)
data_control = merge(data, (treatment = fill(false, n_rows),))  # No treatment
data_treated = merge(data, (treatment = fill(true, n_rows),))   # Treatment

predictions = population_predictions(mixed_model,
    ["baseline" => data_control, "treatment" => data_treated])
```

### Marginal Effects for Fixed Effects

```julia
function marginal_effects_mixed(mixed_model, data, variable)
    compiled = compile_formula(mixed_model, data)
    fixed_coefs = fixef(mixed_model)
    
    # Create perturbed data
    delta = 0.01
    original_values = data[variable]
    perturbed_values = original_values .+ delta
    perturbed_data = merge(data, NamedTuple{(variable,)}((perturbed_values,)))
    
    row_vec_orig = Vector{Float64}(undef, length(compiled))
    row_vec_pert = Vector{Float64}(undef, length(compiled))
    
    n_obs = Tables.rowcount(data)
    marginal_effects = Vector{Float64}(undef, n_obs)
    
    for i in 1:n_obs
        compiled(row_vec_orig, data, i)
        compiled(row_vec_pert, perturbed_data, i)
        
        pred_orig = dot(fixed_coefs, row_vec_orig)
        pred_pert = dot(fixed_coefs, row_vec_pert)
        
        marginal_effects[i] = (pred_pert - pred_orig) / delta
    end
    
    return marginal_effects
end

# With y ~ x * treatment + age: β_x + β_{x & treatment} × treatment in each row
me_x = marginal_effects_mixed(mixed_model, data, :x)
```

## Performance Comparison

### Comparison with modelmatrix

```julia
using BenchmarkTools

# Setup mixed model
mixed_model = fit(MixedModel, @formula(y ~ x + treatment + (1|group)), df)

# Row of the fixed-effects matrix stored in the fitted model
function traditional_mixed_row(model, row_idx)
    X = modelmatrix(model)  # Fixed-effects design matrix (stored, not rebuilt)
    return X[row_idx, :]
end

# FormulaCompiler approach
compiled = compile_formula(mixed_model, data)
row_vec = Vector{Float64}(undef, length(compiled))

function fc_mixed_row(compiled, data, row_vec, row_idx)
    compiled(row_vec, data, row_idx)
    return row_vec
end

# Benchmark
println("Traditional mixed model row extraction:")
@benchmark traditional_mixed_row($mixed_model, 1)

println("FormulaCompiler mixed model row extraction:")
@benchmark fc_mixed_row($compiled, $data, $row_vec, 1)
```

`modelmatrix` returns the matrix stored in the fitted model, so the first benchmark measures a row copy (it allocates the returned row). The compiled formula evaluates the row from the data without allocating, and also works for data the model was not fitted on, such as counterfactual scenarios.

## Advanced Examples

### Bootstrap Fixed Effects

```julia
function bootstrap_fixed_effects(mixed_model, data, n_bootstrap=1000)
    compiled = compile_formula(mixed_model, data)
    n_obs = Tables.rowcount(data)
    n_coefs = length(compiled)
    
    # Get response variable
    y_var = Symbol(mixed_model.formula.lhs)
    y = data[y_var]
    
    bootstrap_coefs = Matrix{Float64}(undef, n_bootstrap, n_coefs)
    row_vec = Vector{Float64}(undef, n_coefs)
    
    for boot in 1:n_bootstrap
        # Bootstrap sample
        sample_idx = rand(1:n_obs, n_obs)
        
        # Build design matrix for bootstrap sample
        X_boot = Matrix{Float64}(undef, n_obs, n_coefs)
        y_boot = Vector{Float64}(undef, n_obs)
        
        for (i, idx) in enumerate(sample_idx)
            compiled(row_vec, data, idx)
            X_boot[i, :] .= row_vec
            y_boot[i] = y[idx]
        end
        
        # Estimate fixed effects (OLS approximation)
        bootstrap_coefs[boot, :] = X_boot \ y_boot
    end
    
    return bootstrap_coefs
end

# Usage
boot_coefs = bootstrap_fixed_effects(mixed_model, data, 200)

# Confidence intervals
using Statistics
conf_intervals = [
    (quantile(boot_coefs[:, j], 0.025), quantile(boot_coefs[:, j], 0.975))
    for j in 1:size(boot_coefs, 2)
]
```

### Policy Scenario Analysis

```julia
using Statistics

function analyze_policy_scenarios(mixed_model, base_data)
    compiled = compile_formula(mixed_model, base_data)
    fixed_coefs = fixef(mixed_model)
    
    # Define policy scenarios using data modification
    n_obs = Tables.rowcount(base_data)

    policy_configs = [
        ("baseline", false, nothing, nothing),          # No treatment, original x and support
        ("universal_treatment", true, nothing, nothing), # Treatment, original x and support
        ("targeted_treatment", true, quantile(base_data.x, 0.75), nothing), # Treatment + top 25% x
        ("enhanced_policy", true, mean(base_data.x), 1.0)  # Treatment + mean x + support
        # (additional_support is not in this model's formula, so it does not change predictions)
    ]

    results = Dict{String, NamedTuple}()
    row_vec = Vector{Float64}(undef, length(compiled))

    for (name, treatment_val, x_val, support_val) in policy_configs
        # Create modified data for this policy
        modified_data = base_data  # Start with baseline

        if name != "baseline"
            # Build modification dictionary
            mods = Dict{Symbol, Any}()
            mods[:treatment] = fill(treatment_val, n_obs)

            if x_val !== nothing
                mods[:x] = fill(x_val, n_obs)
            end

            if support_val !== nothing
                mods[:additional_support] = fill(support_val, n_obs)
            end

            modified_data = merge(base_data, NamedTuple(mods))
        end

        predictions = Vector{Float64}(undef, n_obs)

        for i in 1:n_obs
            compiled(row_vec, modified_data, i)
            predictions[i] = dot(fixed_coefs, row_vec)
        end
        
        results[name] = (
            mean_outcome = mean(predictions),
            std_outcome = std(predictions),
            quantiles = [quantile(predictions, q) for q in [0.25, 0.5, 0.75]]
        )
    end
    
    return results
end

# Run analysis
policy_results = analyze_policy_scenarios(mixed_model, data)

# Display results
for (policy, stats) in policy_results
    println("Policy: $policy")
    println("  Mean outcome: $(round(stats.mean_outcome, digits=3))")
    println("  Std outcome: $(round(stats.std_outcome, digits=3))")
    println("  Quartiles: $(round.(stats.quantiles, digits=3))")
    println()
end
```

## Integration Notes

### What's Included
- Fixed effects terms only
- Interaction terms involving fixed effects
- Functions applied to fixed effects predictors

### What's Excluded
- Random effects terms `(1|group)`, `(x|group)`
- Random intercepts and slopes
- Cross-level interactions involving random effects

### Validation
```julia
function validate_mixed_model_integration(mixed_model, data)
    compiled = compile_formula(mixed_model, data)
    
    # Extract fixed effects design matrix from MixedModels.jl
    mm_fixed = modelmatrix(mixed_model)
    
    # Compare with FormulaCompiler
    row_vec = Vector{Float64}(undef, length(compiled))
    
    for i in 1:min(10, size(mm_fixed, 1))
        compiled(row_vec, data, i)
        original_row = mm_fixed[i, :]
        
        if !isapprox(row_vec, original_row, rtol=1e-12)
            @warn "Mismatch in row $i"
            return false
        end
    end
    
    println("✓ FormulaCompiler matches MixedModels.jl fixed effects matrix")
    return true
end

validate_mixed_model_integration(mixed_model, data)
```

## Best Practices

### When to Use FormulaCompiler with Mixed Models

**Supported patterns:**
- Population-level predictions
- Fixed effects marginal effects
- Policy scenario analysis
- Bootstrap inference on fixed effects

**Not suitable for:**
- Predictions requiring random effects (BLUPs)
- Individual-level predictions in clustered data
- Random effects inference
- Cross-level interaction effects

### Performance Considerations

```julia
# For repeated evaluations, compile once
mixed_model = fit(MixedModel, @formula(y ~ x + (1|group)), df)
compiled = compile_formula(mixed_model, data)  # Do this once

# Then evaluate many times; scenarios must have the same column types as `data`
scenarios = [merge(data, (x = fill(v, length(data.x)),)) for v in (-1.0, 0.0, 1.0)]
row_vec = Vector{Float64}(undef, length(compiled))
for scenario_data in scenarios
    for individual in 1:100
        compiled(row_vec, scenario_data, individual)
        # Process fixed effects prediction...
    end
end
```

### Memory Efficiency

```julia
# Mixed models can be large - use data modification for memory efficiency
base_data = Tables.columntable(df)

# Instead of creating many copies of df,
# use merge() to replace just the variable of interest (one new column)
n_rows = length(base_data.x)
modified_data = merge(base_data, (x = fill(0.5, n_rows),))

# Evaluate
compiled = compile_formula(mixed_model, base_data)
row_vec = Vector{Float64}(undef, length(compiled))
compiled(row_vec, modified_data, 1)
```