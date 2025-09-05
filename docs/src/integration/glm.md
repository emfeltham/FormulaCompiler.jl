# GLM.jl Integration

FormulaCompiler.jl seamlessly integrates with GLM.jl to provide zero-allocation model matrix evaluation for both linear and generalized linear models.

## Supported Models

FormulaCompiler.jl works with all GLM.jl model types:
- Linear models (`lm`)
- Generalized linear models (`glm`)
- All distribution families and link functions
- Custom contrasts and standardized predictors

## Basic Usage

### Linear Models

```julia
using GLM, FormulaCompiler, DataFrames, Tables

# Create sample data
df = DataFrame(
    y = randn(1000),
    x1 = randn(1000),
    x2 = randn(1000),
    group = rand(["A", "B", "C"], 1000)
)

# Fit linear model
model = lm(@formula(y ~ x1 + x2 * group), df)

# Compile for zero-allocation evaluation
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# Zero-allocation evaluation
compiled(row_vec, data, 1)  # Zero allocations; time varies by hardware
```

### Generalized Linear Models

```julia
using CategoricalArrays

# Binary outcome data
df_binary = DataFrame(
    success = rand(Bool, 1000),
    x = randn(1000),
    treatment = categorical(rand(["control", "treatment"], 1000)),
    age = rand(20:80, 1000)
)

# Logistic regression
logit_model = glm(
    @formula(success ~ x + treatment + age), 
    df_binary, 
    Binomial(), 
    LogitLink()
)

# Compile and use
compiled_logit = compile_formula(logit_model, Tables.columntable(df_binary))
row_vec = Vector{Float64}(undef, length(compiled_logit))
compiled_logit(row_vec, Tables.columntable(df_binary), 1)
```

## Distribution Families

FormulaCompiler.jl works with all GLM.jl distribution families:

### Gaussian (Normal) - Identity Link

```julia
normal_model = glm(@formula(y ~ x1 + x2), df, Normal(), IdentityLink())
compiled_normal = compile_formula(normal_model, data)
```

### Binomial - Logit Link

```julia
# Logistic regression
logit_model = glm(@formula(success ~ x + age), df_binary, Binomial(), LogitLink())

# Probit regression  
probit_model = glm(@formula(success ~ x + age), df_binary, Binomial(), ProbitLink())

# Complementary log-log
cloglog_model = glm(@formula(success ~ x + age), df_binary, Binomial(), CloglogLink())
```

### Poisson - Log Link

```julia
# Count data
df_count = DataFrame(
    count = rand(Poisson(2), 1000),
    x = randn(1000),
    exposure = rand(0.5:0.1:2.0, 1000)
)

poisson_model = glm(@formula(count ~ x + log(exposure)), df_count, Poisson(), LogLink())
compiled_poisson = compile_formula(poisson_model, Tables.columntable(df_count))
```

### Gamma - Inverse Link

```julia
# Positive continuous data
df_gamma = DataFrame(
    response = rand(Gamma(2, 3), 1000),
    x = randn(1000),
    factor = rand(["low", "high"], 1000)
)

gamma_model = glm(@formula(response ~ x + factor), df_gamma, Gamma(), InverseLink())
compiled_gamma = compile_formula(gamma_model, Tables.columntable(df_gamma))
```

## Advanced GLM Features

### Custom Contrasts

```julia
using StatsModels

# Define custom contrasts
contrasts_dict = Dict(
    :treatment => DummyCoding(base="control"),
    :group => EffectsCoding(),
    :region => HelmertCoding()
)

# Fit model with custom contrasts
df_contrasts = DataFrame(
    y = randn(1000),
    x = randn(1000),
    treatment = categorical(rand(["control", "low", "high"], 1000)),
    group = categorical(rand(["A", "B", "C", "D"], 1000)),
    region = categorical(rand(["North", "South", "East", "West"], 1000))
)

model_contrasts = lm(
    @formula(y ~ x + treatment + group + region), 
    df_contrasts, 
    contrasts = contrasts_dict
)

compiled_contrasts = compile_formula(model_contrasts, Tables.columntable(df_contrasts))
```

### Weights and Offsets

```julia
# Weighted regression
weights = rand(0.5:0.1:2.0, 1000)
weighted_model = lm(@formula(y ~ x1 + x2), df, wts=weights)

# GLM with offset
df_offset = DataFrame(
    y = rand(Poisson(2), 1000),
    x = randn(1000),
    offset_var = log.(rand(0.5:0.1:2.0, 1000))
)

offset_model = glm(
    @formula(y ~ x + offset(offset_var)), 
    df_offset, 
    Poisson(), 
    LogLink()
)

# FormulaCompiler handles both
compiled_weighted = compile_formula(weighted_model, Tables.columntable(df))
compiled_offset = compile_formula(offset_model, Tables.columntable(df_offset))
```

## Performance Comparisons

### Benchmark Against modelmatrix()

```julia
using BenchmarkTools

# Setup
df = DataFrame(
    y = randn(1000),
    x1 = randn(1000), 
    x2 = randn(1000),
    group = categorical(rand(["A", "B", "C"], 1000))
)

model = lm(@formula(y ~ x1 * x2 + group + log(abs(x1) + 1)), df)
data = Tables.columntable(df)

# Traditional approach
function traditional_single_row(model, row_idx)
    mm = modelmatrix(model)
    return mm[row_idx, :]
end

# FormulaCompiler approach
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

function fc_single_row(compiled, data, row_vec, row_idx)
    compiled(row_vec, data, row_idx)
    return row_vec
end

# Benchmark comparison
# Note: Absolute times vary by hardware and Julia version; see the Benchmark Protocol.
println("Traditional approach:")
@benchmark traditional_single_row($model, 1)

println("\nFormulaCompiler approach:")
@benchmark fc_single_row($compiled, $data, $row_vec, 1)

# Expected results (indicative):
# Traditional: ~10μs, 1 allocation
# FormulaCompiler: tens of ns, 0 allocations (order-of-magnitude faster)
```

### Large Model Performance

```julia
# Create a large, complex model
function create_large_model(n_obs=10000)
    df = DataFrame(
        y = randn(n_obs),
        x1 = randn(n_obs),
        x2 = randn(n_obs),
        x3 = randn(n_obs),
        x4 = randn(n_obs),
        group1 = categorical(rand(1:10, n_obs)),
        group2 = categorical(rand(1:5, n_obs)),
        group3 = categorical(rand(1:3, n_obs))
    )
    
    # Complex formula with interactions
    formula = @formula(y ~ (x1 + x2 + x3 + x4) * (group1 + group2) + 
                          log(abs(x1) + 1) * group3 + 
                          sqrt(abs(x2) + 1) + 
                          x3^2 + 
                          (x4 > 0))
    
    model = lm(formula, df)
    return model, df
end

# Test performance on large model
large_model, large_df = create_large_model(10000)
large_data = Tables.columntable(large_df)
large_compiled = compile_formula(large_model, large_data)

println("Large model performance:")
println("Model matrix size: ", size(modelmatrix(large_model)))
println("Compilation time:")
@time compile_formula(large_model, large_data)

println("Single row evaluation:")
row_vec = Vector{Float64}(undef, length(large_compiled))
@benchmark $large_compiled($row_vec, $large_data, 1)
```

## Real-world Applications

### Marginal Effects Calculation

```julia
function calculate_marginal_effects(model, data, variable_col, delta=0.01)
    compiled = compile_formula(model, data)
    n_rows = Tables.rowcount(data)
    
    # Get original variable values
    original_values = data[variable_col]
    
    # Create scenarios with perturbed values
    perturbed_values = original_values .+ delta
    perturbed_data = (; data..., variable_col => perturbed_values)
    
    row_vec_original = Vector{Float64}(undef, length(compiled))
    row_vec_perturbed = Vector{Float64}(undef, length(compiled))
    
    marginal_effects = Matrix{Float64}(undef, n_rows, length(compiled))
    
    for i in 1:n_rows
        # Original prediction
        compiled(row_vec_original, data, i)
        
        # Perturbed prediction  
        compiled(row_vec_perturbed, perturbed_data, i)
        
        # Marginal effect
        marginal_effects[i, :] .= (row_vec_perturbed .- row_vec_original) ./ delta
    end
    
    return marginal_effects
end

# Example usage
marginal_fx = calculate_marginal_effects(model, data, :x1)
```

### Bootstrap Confidence Intervals

```julia
function bootstrap_glm_coefficients(model, data, n_bootstrap=1000)
    compiled = compile_formula(model, data)
    n_obs = Tables.rowcount(data)
    n_coefs = length(compiled)
    
    # Get original response variable
    y = data[Symbol(model.mf.f.lhs)]
    
    bootstrap_coefs = Matrix{Float64}(undef, n_bootstrap, n_coefs)
    row_vec = Vector{Float64}(undef, n_coefs)
    
    for boot in 1:n_bootstrap
        # Bootstrap sample
        sample_idx = rand(1:n_obs, n_obs)
        
        # Create design matrix for bootstrap sample
        X_boot = Matrix{Float64}(undef, n_obs, n_coefs)
        y_boot = Vector{Float64}(undef, n_obs)
        
        for (i, idx) in enumerate(sample_idx)
            compiled(row_vec, data, idx)
            X_boot[i, :] .= row_vec
            y_boot[i] = y[idx]
        end
        
        # Fit bootstrap model (simplified OLS)
        bootstrap_coefs[boot, :] = X_boot \ y_boot
    end
    
    return bootstrap_coefs
end
```

### Prediction Intervals

```julia
function prediction_intervals(model, data, confidence_level=0.95)
    compiled = compile_formula(model, data)
    n_obs = Tables.rowcount(data)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Get model coefficients and residual variance
    coefs = coef(model)
    σ² = deviance(model) / dof_residual(model)
    
    # Critical value
    α = 1 - confidence_level
    t_crit = quantile(TDist(dof_residual(model)), 1 - α/2)
    
    predictions = Vector{Float64}(undef, n_obs)
    lower_bounds = Vector{Float64}(undef, n_obs)
    upper_bounds = Vector{Float64}(undef, n_obs)
    
    for i in 1:n_obs
        compiled(row_vec, data, i)
        
        # Point prediction
        pred = dot(coefs, row_vec)
        predictions[i] = pred
        
        # Prediction standard error
        # SE = sqrt(σ² * (1 + x'(X'X)⁻¹x))
        # Simplified for demonstration
        se = sqrt(σ² * (1 + sum(row_vec.^2) / n_obs))
        
        # Confidence bounds
        margin = t_crit * se
        lower_bounds[i] = pred - margin
        upper_bounds[i] = pred + margin
    end
    
    return (predictions = predictions, lower = lower_bounds, upper = upper_bounds)
end
```

## Integration Best Practices

### Model Validation

```julia
function validate_glm_integration(model, data)
    compiled = compile_formula(model, data)
    
    # Compare first few rows with modelmatrix
    mm = modelmatrix(model)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    for i in 1:min(10, size(mm, 1))
        compiled(row_vec, data, i)
        original_row = mm[i, :]
        
        if !isapprox(row_vec, original_row, rtol=1e-12)
            @warn "Mismatch in row $i"
            println("Original: ", original_row)
            println("Compiled: ", row_vec)
            return false
        end
    end
    
    println("✓ FormulaCompiler matches GLM.jl modelmatrix for all tested rows")
    return true
end

# Validate integration
validate_glm_integration(model, data)
```

### Memory Usage Comparison

```julia
function compare_memory_usage(model, data)
    # Traditional approach
    traditional_memory = @allocated modelmatrix(model)
    
    # FormulaCompiler approach
    compilation_memory = @allocated compile_formula(model, data)
    
    compiled = compile_formula(model, data)
    row_vec = Vector{Float64}(undef, length(compiled))
    evaluation_memory = @allocated compiled(row_vec, data, 1)
    
    println("Memory Usage Comparison:")
    println("Traditional modelmatrix(): ", traditional_memory, " bytes")
    println("FormulaCompiler compilation: ", compilation_memory, " bytes")
    println("FormulaCompiler evaluation: ", evaluation_memory, " bytes")
    
    if evaluation_memory == 0
        println("✓ Zero-allocation evaluation achieved")
    else
        @warn "Non-zero allocation detected in evaluation"
    end
end

compare_memory_usage(model, data)
```

## Troubleshooting

### Common Issues

1. **Type instability**: Ensure all variables have consistent types
2. **Missing values**: Handle `missing` values before compilation
3. **Categorical levels**: Ensure categorical variables have the same levels in test data
4. **Formula complexity**: Very complex formulas may have longer compilation times

### Debugging Tools

```julia
# Check compilation success
function debug_compilation(model, data)
    try
        compiled = compile_formula(model, data)
        println("✓ Compilation successful")
        println("Formula length: ", length(compiled))
        return compiled
    catch e
        @error "Compilation failed" exception = e
        return nothing
    end
end

# Performance diagnostics
function diagnose_performance(model, data)
    compiled = compile_formula(model, data)
    row_vec = Vector{Float64}(undef, length(compiled))
    
    # Check allocation
    alloc = @allocated compiled(row_vec, data, 1)
    if alloc > 0
        @warn "Non-zero allocation detected: $alloc bytes"
    end
    
    # Check timing
    time_ns = @elapsed compiled(row_vec, data, 1) * 1e9
    if time_ns > 1000  # > 1μs
        @warn "Evaluation slower than expected: $(round(time_ns))ns"
    end
    
    println("Performance: $(round(time_ns))ns, $(alloc) bytes")
end
```
