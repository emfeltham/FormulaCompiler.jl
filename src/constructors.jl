# EfficientModelMatrices.jl
# A general-purpose package for efficient model matrix updates using modern StatsModels.jl API

# =============================================================================
# CONSTRUCTORS: Multiple entry points for different use cases
# =============================================================================

# """
#     cached_modelmatrix(formula, data; model=nothing, contrasts=Dict()) -> CachedModelMatrix

# Create a cached model matrix from a formula and data.

# # Arguments
# - `formula`: A StatsModels formula (e.g., `@formula(y ~ x1 + x2 + x1&x2)`)
# - `data`: Tabular data (DataFrame, NamedTuple, etc.)
# - `model=nothing`: Optional fitted model to reuse contrasts/schema from
# - `contrasts=Dict()`: Contrast specifications for categorical variables

# # Returns
# A `CachedModelMatrix` that can be efficiently updated when data changes.

# # Example
# ```julia
# df = DataFrame(x1=randn(100), x2=randn(100), cat=repeat(["A","B"], 50))
# formula = @formula(y ~ x1 + x2 + cat + x1&x2)
# cached_mm = cached_modelmatrix(formula, df)

# # Later, efficiently update when data changes
# df_new = DataFrame(x1=randn(100), x2=df.x2, cat=df.cat)  # Only x1 changed
# update!(cached_mm, df_new; changed_vars=[:x1])
# ```
# """
# function cached_modelmatrix(formula, data; model=nothing, contrasts=Dict())
#     # Apply schema using modern StatsModels API
#     if isnothing(model)
#         # Create new schema
#         sch = schema(formula, data; contrasts=contrasts)
#         applied_formula = apply_schema(formula, sch)
#     else
#         # Reuse existing model's schema
#         applied_formula = formula(model)
#     end
    
#     # Build the initial matrix using standard StatsModels
#     X = modelmatrix(applied_formula, data)
    
#     # Build dependency cache
#     cache = build_dependency_cache(applied_formula.rhs, data)
    
#     return CachedModelMatrix(X, cache, applied_formula.rhs)
# end

"""
    cached_modelmatrix(model, data) -> CachedModelMatrix

Create a cached model matrix by reusing the existing matrix from a fitted model.
This is often much faster than rebuilding from scratch.

# Arguments
- `model`: A fitted model (GLM, MixedModel, etc.) with existing modelmatrix
- `data`: Data for validation

# Returns
A `CachedModelMatrix` wrapping the existing model matrix for efficient updates.

# Example
```julia
# Fit a model
model = glm(@formula(y ~ x1 + x2 + cat + x1&x2), df, Normal())

# Create cached matrix from existing model matrix (very fast!)
cached_mm = cached_modelmatrix(model, data)

# Later, efficiently update when data changes
update!(cached_mm, df_new; changed_vars=[:x1])
```
"""
function cached_modelmatrix(model, data)
    # Extract existing model matrix (the key optimization!)
    X = modelmatrix(model)
        
    # Get the schema-applied formula
    applied_formula = formula(model)
    
    # Build dependency cache
    cache = build_dependency_cache(applied_formula.rhs, data)
    
    # Validate dimensions
    if nrow(data) != size(X, 1)
        throw(DimensionMismatch(
            "Model matrix has $(size(X, 1)) rows but data has $(nrow(data)) rows"
        ))
    end
    
    return CachedModelMatrix(X, cache, applied_formula.rhs)
end
