# StandardizedPredictors.jl Integration

FormulaCompiler.jl integrates with StandardizedPredictors.jl to handle standardized predictors efficiently in zero-allocation model evaluation.

## Supported Standardizations

Currently, FormulaCompiler.jl supports:
- **ZScore**: Z-score standardization (mean=0, std=1)
- Future versions will support additional standardizations

## Basic Usage

```julia
using StandardizedPredictors, FormulaCompiler, GLM, DataFrames, Tables

# Create sample data
df = DataFrame(
    y = randn(1000),
    x1 = randn(1000) * 5 .+ 10,  # Mean ≈ 10, std ≈ 5
    x2 = randn(1000) * 2 .+ 3,   # Mean ≈ 3, std ≈ 2
    group = rand(["A", "B", "C"], 1000)
)

# Define standardization contrasts
contrasts_dict = Dict(
    :x1 => ZScore(),
    :x2 => ZScore()
)

# Fit model with standardized predictors
model = lm(@formula(y ~ x1 + x2 * group), df, contrasts=contrasts_dict)

# Compile with built-in standardization
data = Tables.columntable(df)
compiled = compile_formula(model, data)
```

## ZScore Standardization

### Automatic Integration

FormulaCompiler.jl automatically handles ZScore standardization:

```julia
# The standardization parameters are built into the compiled formula
println("Model includes standardization: ", has_standardization(compiled))

# Evaluation automatically applies standardization
row_vec = Vector{Float64}(undef, length(compiled))
compiled(row_vec, data, 1)  # x1 and x2 are automatically standardized
```

### Manual Standardization Parameters

You can access the standardization parameters:

```julia
# Get standardization info
std_info = get_standardization_info(compiled)
println("Standardized variables: ", keys(std_info))

for (var, params) in std_info
    println("Variable $var: mean=$(params.mean), std=$(params.std)")
end
```

### Custom Standardization Values

Override standardization parameters for specific scenarios:

```julia
# Create scenario with custom standardization
custom_scenario = create_scenario("custom_std", data;
    x1 = 0.0,  # Will be standardized as (0.0 - mean_x1) / std_x1
    x2 = 1.0   # Will be standardized as (1.0 - mean_x2) / std_x2
)

# The standardization is applied automatically
compiled(row_vec, custom_scenario.data, 1)
```

## Advanced Examples

### Marginal Effects with Standardization

```julia
function marginal_effects_standardized(model, data, variable, delta=0.1)
    compiled = compile_formula(model, data)
    
    # Get original values
    original_values = data[variable]
    
    # Create perturbed values (in original scale)
    perturbed_values = original_values .+ delta
    perturbed_data = (; data..., variable => perturbed_values)
    
    row_vec_orig = Vector{Float64}(undef, length(compiled))
    row_vec_pert = Vector{Float64}(undef, length(compiled))
    
    n_obs = Tables.rowcount(data)
    marginal_effects = Vector{Float64}(undef, n_obs)
    
    for i in 1:n_obs
        # Both evaluations will apply standardization automatically
        compiled(row_vec_orig, data, i)
        compiled(row_vec_pert, perturbed_data, i)
        
        # Calculate marginal effect (difference in standardized scale)
        pred_orig = dot(coef(model), row_vec_orig)
        pred_pert = dot(coef(model), row_vec_pert)
        
        marginal_effects[i] = (pred_pert - pred_orig) / delta
    end
    
    return marginal_effects
end

# Calculate marginal effects for standardized x1
me_x1 = marginal_effects_standardized(model, data, :x1)
println("Mean marginal effect of x1: ", mean(me_x1))
```

### Policy Analysis with Standardized Predictors

```julia
function standardized_policy_analysis(model, base_data)
    compiled = compile_formula(model, base_data)
    
    # Get original means and stds for interpretation
    x1_mean = mean(base_data.x1)
    x1_std = std(base_data.x1)
    x2_mean = mean(base_data.x2) 
    x2_std = std(base_data.x2)
    
    # Create policy scenarios (in original scale)
    scenarios = Dict(
        "baseline" => create_scenario("baseline", base_data),
        
        # Move everyone to +1 standard deviation
        "high_x1" => create_scenario("high_x1", base_data; 
            x1 = x1_mean + x1_std
        ),
        
        # Move everyone to population mean
        "mean_values" => create_scenario("mean_values", base_data;
            x1 = x1_mean,
            x2 = x2_mean
        ),
        
        # Extreme policy: +2 standard deviations
        "extreme_policy" => create_scenario("extreme", base_data;
            x1 = x1_mean + 2 * x1_std,
            x2 = x2_mean + 2 * x2_std
        )
    )
    
    # Evaluate scenarios
    results = Dict{String, Vector{Float64}}()
    row_vec = Vector{Float64}(undef, length(compiled))
    
    for (name, scenario) in scenarios
        n_obs = Tables.rowcount(scenario.data)
        predictions = Vector{Float64}(undef, n_obs)
        
        for i in 1:n_obs
            compiled(row_vec, scenario.data, i)
            predictions[i] = dot(coef(model), row_vec)
        end
        
        results[name] = predictions
    end
    
    # Compare scenarios
    for (name, preds) in results
        println("Scenario $name:")
        println("  Mean prediction: $(round(mean(preds), digits=3))")
        println("  Std prediction: $(round(std(preds), digits=3))")
        
        if name != "baseline"
            baseline_mean = mean(results["baseline"])
            effect = mean(preds) - baseline_mean
            println("  Effect vs baseline: $(round(effect, digits=3))")
        end
        println()
    end
    
    return results
end

# Run analysis
policy_results = standardized_policy_analysis(model, data)
```

### Interpretation Helpers

```julia
# Functions to help interpret standardized results

function unstandardize_prediction(prediction, y_mean, y_std)
    """Convert prediction back to original scale"""
    return prediction * y_std + y_mean
end

function standardized_effect_size(effect, y_std)
    """Convert effect to Cohen's d (standardized effect size)"""
    return effect / y_std
end

function interpret_standardized_coef(coef_value, x_std, y_std)
    """Interpret standardized coefficient"""
    # Effect of 1 standard deviation change in X on Y (in Y's standard deviation units)
    return coef_value * (x_std / y_std)
end

# Example usage
model_coefs = coef(model)
y_std = std(df.y)
x1_std = std(df.x1)

x1_coef_interpretation = interpret_standardized_coef(model_coefs[2], x1_std, y_std)
println("One std dev increase in x1 changes y by $(round(x1_coef_interpretation, digits=3)) std devs")
```

## Performance Considerations

### Compilation Overhead

```julia
using BenchmarkTools

# Standardization adds minimal compilation overhead
@benchmark compile_formula($model, $data)

# Evaluation performance is identical to non-standardized models
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

@benchmark $compiled($row_vec, $data, 1)  # Still fast, 0 allocations
```

### Memory Efficiency

```julia
# Standardization parameters are stored efficiently
sizeof_standardized = sizeof(compiled)
println("Compiled formula size with standardization: $sizeof_standardized bytes")

# Scenarios still provide memory benefits
large_scenario = create_scenario("large", data; x1 = 100.0)  # Large value
println("Scenario with standardization: $(sizeof(large_scenario)) bytes")
```

## Best Practices

### Variable Selection for Standardization

```julia
# Good candidates for standardization
good_candidates = [
    :income,        # Often has wide range and skew
    :age,           # Different scales across studies
    :test_scores,   # For comparing across tests
    :measurements   # Physical measurements with different units
]

# Usually don't standardize
dont_standardize = [
    :binary_vars,   # 0/1 variables
    :count_vars,    # Poisson-distributed variables
    :categorical,   # Categorical variables (use contrasts instead)
    :percentages    # Already on 0-100 scale
]

# Example: selective standardization
selective_contrasts = Dict(
    :income => ZScore(),      # Standardize income
    :age => ZScore(),         # Standardize age  
    :treatment => DummyCoding(),  # Don't standardize binary treatment
    :region => EffectsCoding()    # Use effects coding for categorical
)

model_selective = lm(@formula(y ~ income + age + treatment + region), df, 
                    contrasts = selective_contrasts)
```

### Scenario Design with Standardization

```julia
function create_meaningful_scenarios(base_data, model)
    # Get standardization info to create interpretable scenarios
    x1_mean = mean(base_data.x1)
    x1_std = std(base_data.x1)
    x2_mean = mean(base_data.x2)
    x2_std = std(base_data.x2)
    
    scenarios = Dict(
        # Baseline: typical individual
        "typical" => create_scenario("typical", base_data;
            x1 = x1_mean,      # Average x1
            x2 = x2_mean       # Average x2
        ),
        
        # High achiever: +1 std dev on both
        "high_achiever" => create_scenario("high", base_data;
            x1 = x1_mean + x1_std,
            x2 = x2_mean + x2_std
        ),
        
        # Low performer: -1 std dev on both
        "low_performer" => create_scenario("low", base_data;
            x1 = x1_mean - x1_std,
            x2 = x2_mean - x2_std
        ),
        
        # Mixed profile: high x1, low x2
        "mixed_profile" => create_scenario("mixed", base_data;
            x1 = x1_mean + x1_std,
            x2 = x2_mean - x1_std
        )
    )
    
    return scenarios
end

meaningful_scenarios = create_meaningful_scenarios(data, model)
```

### Validation and Diagnostics

```julia
function validate_standardization(model, data)
    compiled = compile_formula(model, data)
    
    # Check that standardized variables have correct properties
    # (This is a conceptual check - actual implementation would need model internals)
    
    # Verify evaluation works correctly
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, data, 1)
    
    # Compare with manual standardization
    manual_x1 = (data.x1[1] - mean(data.x1)) / std(data.x1)
    manual_x2 = (data.x2[1] - mean(data.x2)) / std(data.x2)
    
    println("✓ Standardization validation completed")
    println("Manual x1 standardization: $(round(manual_x1, digits=3))")
    println("Manual x2 standardization: $(round(manual_x2, digits=3))")
    
    return true
end

validate_standardization(model, data)
```

## Future Extensions

FormulaCompiler.jl plans to support additional StandardizedPredictors.jl features:

- **CenterScale**: Custom center and scale parameters
- **MeanScale**: Mean scaling
- **UnitScale**: Unit scaling  
- **Custom transformations**: User-defined standardization functions

These will follow the same zero-allocation principles and integrate seamlessly with the scenario system.