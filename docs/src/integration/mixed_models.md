# MixedModels.jl Integration

FormulaCompiler.jl integrates with MixedModels.jl by automatically extracting fixed effects from mixed-effects models, enabling zero-allocation evaluation of the fixed-effects portion.

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
    group = rand(1:10, 1000),
    cluster = rand(1:50, 1000)
)

# Fit mixed model
mixed_model = fit(MixedModel, @formula(y ~ x + treatment + (1|group) + (1+x|cluster)), df)

# FormulaCompiler automatically extracts fixed effects: y ~ x + treatment
compiled = compile_formula(mixed_model, Tables.columntable(df))
```

## Fixed Effects Extraction

### Automatic Extraction

```julia
# Mixed model with various random effects structures
mixed_model = fit(MixedModel, @formula(y ~ x * treatment + age + (1|group) + (x|cluster)), df)

# Fixed effects formula: y ~ x * treatment + age
fixed_formula = fixed_effects_form(mixed_model)
println("Fixed effects: ", fixed_formula)

# Compile fixed effects only
compiled = compile_formula(mixed_model, data)
```

### Manual Fixed Effects

```julia
# If you need more control, extract manually
fixed_form = mixed_model.formula.rhs.terms[1]  # Gets fixed effects terms
manual_model = lm(FormulaTerm(mixed_model.formula.lhs, fixed_form), df)
compiled_manual = compile_formula(manual_model, data)
```

## Use Cases

### Population-Level Predictions

```julia
function population_predictions(mixed_model, data, scenarios)
    # Compile fixed effects
    compiled = compile_formula(mixed_model, data)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    results = Dict{String, Vector{Float64}}()
    
    for (scenario_name, scenario) in scenarios
        n_obs = Tables.rowcount(scenario.data)
        predictions = Vector{Float64}(undef, n_obs)
        
        for i in 1:n_obs
            compiled(row_vec, scenario.data, i)
            # This gives the fixed effects linear predictor
            predictions[i] = dot(fixef(mixed_model), row_vec)
        end
        
        results[scenario_name] = predictions
    end
    
    return results
end

# Example usage with CounterfactualVector
data_cf, cf_vecs = build_counterfactual_data(data, [:treatment], 1)
treatment_cf = cf_vecs[1]

# Baseline scenario
update_counterfactual_replacement!(treatment_cf, false)  # No treatment
baseline_predictions = population_predictions_single(mixed_model, data, "baseline")

# Treatment scenario
update_counterfactual_replacement!(treatment_cf, true)   # Treatment
treatment_predictions = population_predictions_single(mixed_model, data_cf, "treatment")
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
    perturbed_data = (; data..., variable => perturbed_values)
    
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
```

## Performance Benefits

### Comparison with modelmatrix

```julia
using BenchmarkTools

# Setup mixed model
mixed_model = fit(MixedModel, @formula(y ~ x + treatment + (1|group)), df)
data = Tables.columntable(df)

# Traditional approach: extract full model matrix
function traditional_mixed_row(model, row_idx)
    # Get fixed effects design matrix
    X = modelmatrix(model.optsum.lm)  # Linear model component
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
boot_coefs = bootstrap_fixed_effects(mixed_model, data, 1000)

# Confidence intervals
using Statistics
conf_intervals = [
    (quantile(boot_coefs[:, j], 0.025), quantile(boot_coefs[:, j], 0.975))
    for j in 1:size(boot_coefs, 2)
]
```

### Policy Scenario Analysis

```julia
function analyze_policy_scenarios(mixed_model, base_data)
    compiled = compile_formula(mixed_model, base_data)
    fixed_coefs = fixef(mixed_model)
    
    # Create counterfactual data for policy analysis
    data_cf, cf_vecs = build_counterfactual_data(base_data, [:treatment, :x, :additional_support], 1)
    treatment_cf, x_cf, support_cf = cf_vecs

    # Define policy scenarios with CounterfactualVector
    policy_configs = [
        ("baseline", false, nothing, nothing),          # No treatment, original x and support
        ("universal_treatment", true, nothing, nothing), # Treatment, original x and support
        ("targeted_treatment", true, quantile(base_data.x, 0.75), nothing), # Treatment + top 25% x
        ("enhanced_policy", true, mean(base_data.x), 1.0)  # Treatment + mean x + support
    ]

    results = Dict{String, NamedTuple}()
    row_vec = Vector{Float64}(undef, length(compiled))

    for (name, treatment_val, x_val, support_val) in policy_configs
        # Set counterfactual values
        update_counterfactual_replacement!(treatment_cf, treatment_val)

        if x_val !== nothing
            update_counterfactual_replacement!(x_cf, x_val)
        else
            update_counterfactual_replacement!(x_cf, getproperty(base_data, :x)[1])  # Use original
        end

        if support_val !== nothing
            update_counterfactual_replacement!(support_cf, support_val)
        else
            update_counterfactual_replacement!(support_cf, getproperty(base_data, :additional_support)[1])  # Use original
        end

        # Choose appropriate data source
        current_data = (name == "baseline") ? base_data : data_cf
        n_obs = Tables.rowcount(current_data)
        predictions = Vector{Float64}(undef, n_obs)
        
        for i in 1:n_obs
            if name != "baseline"
                update_counterfactual_row!(treatment_cf, i)
                update_counterfactual_row!(x_cf, i)
                update_counterfactual_row!(support_cf, i)
            end
            compiled(row_vec, current_data, i)
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
    mm_fixed = modelmatrix(mixed_model.optsum.lm)
    
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
```

## Best Practices

### When to Use FormulaCompiler with Mixed Models

**Good use cases:**
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

# Then evaluate many times
row_vec = Vector{Float64}(undef, length(compiled))
for scenario in many_scenarios
    for individual in many_individuals
        compiled(row_vec, scenario.data, individual)
        # Process fixed effects prediction...
    end
end
```

### Memory Efficiency

```julia
# Mixed models can be large - use scenarios for memory efficiency
large_mixed_model = fit(MixedModel, complex_formula, large_df)
base_data = Tables.columntable(large_df)

# Instead of creating many copies of large_df
# Use CounterfactualVector to override just the variables of interest
data_cf, cf_vecs = build_counterfactual_data(base_data, [:key_variable], 1)
key_cf = cf_vecs[1]
update_counterfactual_replacement!(key_cf, new_value)

# Evaluate with minimal memory overhead
compiled = compile_formula(large_mixed_model, base_data)
# ... use compiled with data_cf
```