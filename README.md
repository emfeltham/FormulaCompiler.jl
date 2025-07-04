# EfficientModelMatrices.jl Usage Guide

## Overview

EfficientModelMatrices.jl provides a general-purpose solution for efficient model matrix updates using the modern StatsModels.jl API. It avoids the deprecated `ModelFrame` and instead builds on `schema`, `apply_schema`, and `modelcols`.

## Key Features

- **Selective Updates**: Only recompute matrix columns when their source variables change
- **Modern API**: Uses current StatsModels.jl best practices
- **General Purpose**: Works with any StatsModels formula, not tied to specific modeling packages
- **Memory Efficient**: Reuses matrices instead of constant reallocation
- **Scalable**: Performance scales with changed variables, not total variables

## Basic Usage

### 1. Creating from Existing Model Matrix (RECOMMENDED)

The most efficient approach is to reuse the model matrix that's already computed and stored in your fitted model:

```julia
using EfficientModelMatrices, DataFrames, StatsModels, GLM

# Create and fit a model
df = DataFrame(
    x1 = randn(1000),
    x2 = randn(1000),
    cat = repeat(["A", "B", "C"], 334)[1:1000],
    y = randn(1000)
)

model = glm(@formula(y ~ x1 + x2 + cat + x1&x2), df, Normal())

# Create cached matrix from existing model matrix (very fast!)
cached_mm = cached_modelmatrix(model)
```

This approach:
- Reuses the already-computed model matrix from `modelmatrix(model)`
- Avoids rebuilding the matrix from scratch
- Preserves all the original contrasts and coding exactly
- Is typically 10-50x faster than rebuilding from formula

### 2. Creating from Formula (Alternative)

If you don't have a fitted model yet:

```julia
# Define formula
formula = @formula(y ~ x1 + x2 + cat + x1&x2 + x1&cat)

# Create cached model matrix from formula
cached_mm = cached_modelmatrix(formula, df)
```

### 3. Efficient Updates

```julia
# Modify only some variables
df_new = copy(df)
df_new.x1 = randn(1000)  # Only x1 changed

# Efficient update - only recomputes columns involving x1
update!(cached_mm, df_new; changed_vars=[:x1])

# Or let it detect changes automatically (less efficient but safer)
update!(cached_mm, df_new)
```

## Advanced Usage

### Selective Variable Updates

```julia
# Update specific variables without rebuilding data
selective_update!(cached_mm, Dict(
    :x1 => randn(1000),
    :x2 => zeros(1000)
))
```

### Batch Updates

```julia
# Update multiple cached matrices efficiently
cached_matrices = [cached_mm1, cached_mm2, cached_mm3]
new_data_list = [df_new1, df_new2, df_new3]

batch_update!(cached_matrices, new_data_list)
```

### Dependency Analysis

```julia
# See which matrix columns depend on a variable
deps = get_dependency_info(cached_mm, :x1)
println("x1 affects columns: $deps")

# See which variables affect a matrix column
vars = get_affected_variables(cached_mm, 5)
println("Column 5 depends on: $vars")
```

## Integration Examples

### 1. With Marginal Effects Computation (Primary Use Case)

```julia
# Your existing margins workflow
function compute_marginal_effects_efficient(model, df, variables)
    # Create cached matrix from existing model matrix (key optimization!)
    cached_mm = cached_modelmatrix(model)
    
    # Store original matrix for baseline
    X_base = copy(cached_mm.matrix)
    
    # Compute derivatives efficiently
    effects = Dict()
    h = 0.001
    
    for var in variables
        # Perturb just this variable
        df_pert = copy(df)
        df_pert[!, var] .+= h
        
        # Efficient update - only recomputes columns involving this variable
        update!(cached_mm, df_pert; changed_vars=[var])
        
        # Compute marginal effect using the updated matrix
        effects[var] = compute_derivative(X_base, cached_mm.matrix, coef(model), h)
    end
    
    return effects
end

# This mirrors your existing approach in build_continuous_design.jl
# but with a cleaner, more general API
function build_design_matrices_with_cache(model, df, variables)
    # Reuse existing model matrix
    cached_mm = cached_modelmatrix(model)
    X_base = cached_mm.matrix
    
    # Build derivative matrices efficiently
    Xdx_list = []
    h = sqrt(eps(Float64))
    
    for var in variables
        df_pert = copy(df)
        df_pert[!, var] .+= h
        
        # This replaces your expensive modelmatrix! calls
        update!(cached_mm, df_pert; changed_vars=[var])
        
        # Compute finite difference
        Xdx = (cached_mm.matrix - X_base) ./ h
        push!(Xdx_list, copy(Xdx))
        
        # Reset for next variable
        cached_mm.matrix .= X_base
    end
    
    return X_base, Xdx_list
end
```

### 2. With Bootstrap/Simulation

```julia
# Efficient bootstrap
function bootstrap_estimates(formula, df, n_bootstrap=1000)
    cached_mm = cached_modelmatrix(formula, df)
    estimates = []
    
    for i in 1:n_bootstrap
        # Resample data
        boot_indices = sample(1:nrow(df), nrow(df), replace=true)
        df_boot = df[boot_indices, :]
        
        # Efficient update
        update!(cached_mm, df_boot)
        
        # Fit model and store estimate
        estimate = compute_estimate(cached_mm.matrix, df_boot.y)
        push!(estimates, estimate)
    end
    
    return estimates
end
```

### 3. With Cross-Validation

```julia
function cross_validate_efficient(formula, df, k_folds=5)
    cached_mm = cached_modelmatrix(formula, df)
    fold_results = []
    
    for fold in 1:k_folds
        # Get training data for this fold
        train_indices = get_train_indices(fold, k_folds, nrow(df))
        df_train = df[train_indices, :]
        
        # Efficient update
        update!(cached_mm, df_train)
        
        # Fit and evaluate
        result = fit_and_evaluate(cached_mm.matrix, df_train.y, ...)
        push!(fold_results, result)
    end
    
    return fold_results
end
```

## Performance Considerations

### When to Use

**Good use cases:**
- Marginal effects computation (changing one variable at a time)
- Bootstrap/simulation (similar structure, different data)
- Cross-validation (subset of same data)
- Sensitivity analysis (perturbing specific variables)
- Interactive modeling (user changes parameters)

**Less beneficial:**
- One-off matrix computations
- Completely different formulas each time
- Very small datasets (overhead not worth it)

### Optimization Tips

1. **Specify `changed_vars`** when you know which variables changed
2. **Reuse `CachedModelMatrix`** objects across computations
3. **Use `batch_update!`** for multiple similar updates
4. **Consider `selective_update!`** for programmatic variable changes

## Memory Management

```julia
# The cached matrix holds references to the original data structure
# For large datasets, be mindful of memory usage

# Create cached matrix
cached_mm = cached_modelmatrix(formula, df)

# Clear reference to original data if needed
df = nothing
GC.gc()

# The cached matrix still works with new data
update!(cached_mm, df_new)
```

## Error Handling

```julia
# The package provides informative errors
try
    cached_mm = cached_modelmatrix(formula, df)
    update!(cached_mm, df_wrong_structure)
catch e
    println("Error: $e")
    # Handle appropriately
end
```

## Integration with StatsModels Ecosystem

### With GLM.jl

```julia
using GLM

model = glm(@formula(y ~ x1 + x2), df, Normal())
cached_mm = cached_modelmatrix(formula(model), df; model=model)
```

### With MixedModels.jl

```julia
using MixedModels

model = fit(MixedModel, @formula(y ~ x1 + x2 + (1|group)), df)
# Extract fixed effects part
fe_formula = extract_fixed_effects(model)  # Your helper function
cached_mm = cached_modelmatrix(fe_formula, df; model=model)
```

### With Custom Contrasts

```julia
# Define custom contrasts
contrasts = Dict(:cat => DummyCoding())

cached_mm = cached_modelmatrix(formula, df; contrasts=contrasts)
```

## Testing and Validation

```julia
# Validate that cached updates match full recomputation
function validate_update(formula, df, df_new, changed_vars)
    # Full recomputation
    X_full = modelmatrix(formula, df_new)
    
    # Cached update
    cached_mm = cached_modelmatrix(formula, df)
    update!(cached_mm, df_new; changed_vars=changed_vars)
    
    # Should be identical
    @assert cached_mm.matrix ≈ X_full
end
```

## Best Practices

1. **Build cache once, reuse many times**
2. **Always specify `changed_vars` when known**
3. **Use `selective_update!` for programmatic changes**
4. **Consider memory usage for very large datasets**
5. **Validate critical computations against full recomputation**
6. **Use batch operations when updating multiple matrices**

This package provides a solid foundation for efficient model matrix operations while staying compatible with the modern StatsModels.jl ecosystem.
