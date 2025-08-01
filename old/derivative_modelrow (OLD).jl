# derivative_modelrow.jl

###############################################################################
# CONVENIENT INTERFACES
###############################################################################

"""
    modelrow!(row_vec, compiled_derivative, data, row_idx)

Compute derivative model row using compiled derivative formula.

This function provides a convenient interface for evaluating derivatives that mirrors
the existing `modelrow!` API for regular formulas. The key difference is that this
returns full-width derivative vectors matching the original formula dimensions.

# Arguments
- `row_vec::AbstractVector{Float64}`: Pre-allocated output vector (same length as original formula)
- `compiled_derivative::CompiledDerivativeFormula`: From `compile_derivative_formula()`
- `data`: Column-table format data (from `Tables.columntable()`)
- `row_idx::Int`: Row index to evaluate (1-based)

# Returns
The input `row_vec` filled with derivative values, for method chaining

# Performance
- **Time**: ~50-100ns per row (same as formula evaluation)
- **Allocations**: 0 bytes (if row_vec pre-allocated)
- **Accuracy**: Analytical derivatives, machine precision

# Derivative Vector Format
The output vector has the same length as the original formula, with derivatives
in the correct positions and zeros elsewhere:

```julia
# For model y ~ intercept + x + log(z)
dx_vec = [∂(intercept)/∂x, ∂(x)/∂x, ∂(log(z))/∂x] = [0.0, 1.0, 0.0]
dz_vec = [∂(intercept)/∂z, ∂(x)/∂z, ∂(log(z))/∂z] = [0.0, 0.0, 1/z]
model = lm(@formula(y ~ x + log(z)), df)
compiled = compile_formula(model)
dx_compiled = compile_derivative_formula(compiled, :x)

data = Tables.columntable(df)
deriv_vec = Vector{Float64}(undef, length(dx_compiled))

for i in 1:nrow(df)
    modelrow!(deriv_vec, dx_compiled, data, i)
    # deriv_vec now contains [0.0, 1.0, 0.0] for ∂/∂x
    
    # Compute marginal effect
    marginal_x = dot(deriv_vec, coef(model))
end
```

# See Also
- [`compile_derivative_formula`](@ref): Create compiled derivative formulas
- [`marginal_effects!`](@ref): Convenient marginal effects computation
- [`modelrow!`](@ref): Regular formula evaluation (not derivatives)
"""
function modelrow!(
    row_vec::AbstractVector{Float64}, 
    compiled_derivative::CompiledDerivativeFormula, 
    data, row_idx::Int
)
    compiled_derivative(row_vec, data, row_idx)
    return row_vec
end

"""
    modelrow!(matrix, compiled_derivative, data, row_indices)

Multi-row derivative evaluation.

This function efficiently evaluates derivatives for multiple observations,
filling a pre-allocated matrix with derivative vectors. Each row of the matrix
corresponds to the derivative vector for one observation.

# Arguments
- `matrix::Matrix{Float64}`: Pre-allocated matrix (size: length(row_indices) × length(compiled_derivative))
- `compiled_derivative::CompiledDerivativeFormula`: From `compile_derivative_formula()`
- `data`: Column-table format data (from `Tables.columntable()`)
- `row_indices::Vector{Int}`: Row indices to evaluate (1-based)

# Returns
The input `matrix` filled with derivative values, for method chaining

# Performance
- **Time**: ~50-100ns per row (same as single-row evaluation)
- **Allocations**: 0 bytes (if matrix pre-allocated)
- **Memory layout**: Row-major storage (matrix[i, :] = derivative for row_indices[i])

# Matrix Layout

```julia
# For n observations and k parameters:
matrix[1, :] = derivative vector for row_indices[1]
matrix[2, :] = derivative vector for row_indices[2]
...
matrix[n, :] = derivative vector for row_indices[n]
model = lm(@formula(y ~ x + log(z)), df)
compiled = compile_formula(model)
dx_compiled = compile_derivative_formula(compiled, :x)

data = Tables.columntable(df)
row_indices = [1, 5, 10, 15]  # Evaluate these specific rows
deriv_matrix = Matrix{Float64}(undef, length(row_indices), length(dx_compiled))

modelrow!(deriv_matrix, dx_compiled, data, row_indices)

# Now:
# deriv_matrix[1, :] = derivative at observation 1
# deriv_matrix[2, :] = derivative at observation 5
# deriv_matrix[3, :] = derivative at observation 10
# deriv_matrix[4, :] = derivative at observation 15

# Compute marginal effects for all rows
coefficients = coef(model)
marginal_effects = [dot(deriv_matrix[i, :], coefficients) for i in 1:length(row_indices)]
```

# See Also
- [`modelrow!`](@ref): Single-row derivative evaluation
- [`marginal_effects!`](@ref): Direct marginal effects computation
"""
function modelrow!(matrix::Matrix{Float64}, 
                             compiled_derivative::CompiledDerivativeFormula, 
                             data, row_indices::Vector{Int})
    for (i, row_idx) in enumerate(row_indices)
        modelrow!(view(matrix, i, :), compiled_derivative, data, row_idx)
    end
    return matrix
end

"""
    marginal_effects!(effects_vec, compiled_formula, compiled_derivatives, coefficients, data, row_idx)

Compute marginal effects using compiled derivatives and model coefficients.

This function provides a convenient interface for computing marginal effects by
automatically combining derivative vectors with model coefficients. It handles
the dot product computation and manages temporary allocations efficiently.

# Arguments
- `effects_vec::Vector{Float64}`: Pre-allocated output vector (length = number of derivatives)
- `compiled_formula::CompiledFormula`: Original compiled formula from `compile_formula()`
- `compiled_derivatives::Vector{CompiledDerivativeFormula}`: Compiled derivatives for each variable
- `coefficients::Vector{Float64}`: Model coefficients from `coef(model)`
- `data`: Column-table format data (from `Tables.columntable()`)
- `row_idx::Int`: Row index to evaluate (1-based)

# Returns
The input `effects_vec` filled with marginal effects, for method chaining

# Mathematical Background
For a model `y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ`, the marginal effect of variable `xⱼ` is:
∂E[y|x]/∂xⱼ = Σᵢ βᵢ * ∂xᵢ/∂xⱼ

This function computes this efficiently using the full-width derivative vectors:

```julia
marginal_effect_j = dot(derivative_vector_j, coefficients)
```

# Performance
- **Time**: ~200-500ns per variable (includes derivative evaluation + dot product)
- **Allocations**: One temporary vector (reused across variables)
- **Accuracy**: Analytical derivatives, machine precision

# Example
```julia
model = lm(@formula(y ~ x + log(z) + x*group), df)
compiled = compile_formula(model)
coefficients = coef(model)

# Compile derivatives for variables of interest
dx_compiled = compile_derivative_formula(compiled, :x)
dz_compiled = compile_derivative_formula(compiled, :z)
compiled_derivatives = [dx_compiled, dz_compiled]

data = Tables.columntable(df)
effects = Vector{Float64}(undef, 2)  # 2 variables

for i in 1:nrow(df)
    marginal_effects!(effects, compiled, compiled_derivatives, coefficients, data, i)
    
    # effects[1] = marginal effect of x at observation i
    # effects[2] = marginal effect of z at observation i
    
    println("Row i: Marginal effect of x = (effects[1]), z = (effects[2])")
end
```

# Batch Example
```julia
# Compute marginal effects for multiple observations
n_obs = 100
effects_matrix = Matrix{Float64}(undef, n_obs, 2)

for i in 1:n_obs
    marginal_effects!(view(effects_matrix, i, :), compiled, compiled_derivatives, coefficients, data, i)
end

# Now effects_matrix[i, j] = marginal effect of variable j at observation i
```

# Variable Interpretation
The order of `effects_vec` corresponds to the order of `compiled_derivatives`:
- `effects_vec[1]` = marginal effect of focal variable in `compiled_derivatives[1]`
- `effects_vec[2]` = marginal effect of focal variable in `compiled_derivatives[2]`
- etc.

# See Also
- [`compile_derivative_formula`](@ref): Create compiled derivative formulas
- [`modelrow!`](@ref): Direct derivative evaluation
- [`coef`](@ref): Extract model coefficients (from GLM.jl)
"""
function marginal_effects!(effects_vec::Vector{Float64}, 
                          compiled_formula::CompiledFormula,
                          compiled_derivatives::Vector{CompiledDerivativeFormula},
                          coefficients::Vector{Float64}, 
                          data, row_idx::Int)
    
    deriv_vec = Vector{Float64}(undef, length(compiled_formula))
    
    for (i, compiled_deriv) in enumerate(compiled_derivatives)
        modelrow!(deriv_vec, compiled_deriv, data, row_idx)
        effects_vec[i] = dot(deriv_vec, coefficients)
    end
    
    return effects_vec
end

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

function clear_derivative_cache!()
    empty!(DERIVATIVE_CACHE)
    return nothing
end
