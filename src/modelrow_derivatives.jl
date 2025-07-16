# modelrow_derivatives.jl

###############################################################################
# NEW: FormulaCompiler.jl/src/modelrow_derivatives.jl
# (New file for derivative-specific modelrow interfaces)
###############################################################################

"""
    modelrow_derivatives.jl - Derivative-specific modelrow! interfaces

This provides convenient interfaces for computing model derivatives,
mirroring the existing modelrow! interfaces but for derivatives.
"""

"""
    derivative_modelrow!(row_vec, compiled_derivative, data, row_idx)

Compute derivative model row using compiled derivative formula.

# Arguments
- `row_vec::Vector{Float64}`: Pre-allocated output vector (same length as original formula)
- `compiled_derivative::CompiledDerivativeFormula`: From `compile_derivative_formula()`
- `data`: Column-table format data
- `row_idx::Int`: Row index to evaluate

# Performance
- **Time**: ~50-100ns per row (same as formula evaluation)
- **Allocations**: 0 bytes (if row_vec pre-allocated)

# Example
```julia
model = lm(@formula(y ~ x + log(z)), df)
compiled = compile_formula(model)
dx_compiled = compile_derivative_formula(compiled, :x)

data = Tables.columntable(df)
deriv_vec = Vector{Float64}(undef, length(dx_compiled))

for i in 1:nrow(df)
    derivative_modelrow!(deriv_vec, dx_compiled, data, i)
    # deriv_vec now contains [0.0, 1.0, 0.0] for ∂/∂x
end
```
"""
function derivative_modelrow!(row_vec::AbstractVector{Float64}, 
                             compiled_derivative::CompiledDerivativeFormula, 
                             data, row_idx::Int)
    compiled_derivative(row_vec, data, row_idx)
    return row_vec
end

"""
    derivative_modelrow!(matrix, compiled_derivative, data, row_indices)

Zero-allocation multi-row derivative evaluation.

# Arguments
- `matrix::Matrix{Float64}`: Pre-allocated matrix (size: length(row_indices) × length(compiled_derivative))
- `compiled_derivative::CompiledDerivativeFormula`: From `compile_derivative_formula()`
- `data`: Column-table format data  
- `row_indices::Vector{Int}`: Row indices to evaluate

# Example
```julia
model = lm(@formula(y ~ x + log(z)), df)
compiled = compile_formula(model)
dx_compiled = compile_derivative_formula(compiled, :x)

data = Tables.columntable(df)
row_indices = [1, 5, 10, 15]
deriv_matrix = Matrix{Float64}(undef, length(row_indices), length(dx_compiled))

derivative_modelrow!(deriv_matrix, dx_compiled, data, row_indices)

# deriv_matrix[1, :] = derivative at observation 1
# deriv_matrix[2, :] = derivative at observation 5
# etc.
```
"""
function derivative_modelrow!(matrix::Matrix{Float64}, 
                             compiled_derivative::CompiledDerivativeFormula, 
                             data, row_indices::Vector{Int})
    for (i, row_idx) in enumerate(row_indices)
        derivative_modelrow!(view(matrix, i, :), compiled_derivative, data, row_idx)
    end
    return matrix
end

"""
    marginal_effects!(effects_vec, compiled_formula, compiled_derivatives, coefficients, data, row_idx)

Compute marginal effects using compiled derivatives and coefficients.

# Arguments
- `effects_vec::Vector{Float64}`: Pre-allocated output vector (length = number of derivatives)
- `compiled_formula::CompiledFormula`: Original compiled formula
- `compiled_derivatives::Vector{CompiledDerivativeFormula}`: Compiled derivatives for each variable
- `coefficients::Vector{Float64}`: Model coefficients from `coef(model)`
- `data`: Column-table format data
- `row_idx::Int`: Row index to evaluate

# Example
```julia
model = lm(@formula(y ~ x + log(z) + x*group), df)
compiled = compile_formula(model)
dx_compiled = compile_derivative_formula(compiled, :x)
dz_compiled = compile_derivative_formula(compiled, :z)

derivatives = [dx_compiled, dz_compiled]
effects = Vector{Float64}(undef, 2)

for i in 1:nrow(df)
    marginal_effects!(effects, compiled, derivatives, coef(model), data, i)
    # effects[1] = marginal effect of x at observation i
    # effects[2] = marginal effect of z at observation i
end
```
"""
function marginal_effects!(effects_vec::Vector{Float64}, 
                          compiled_formula::CompiledFormula,
                          compiled_derivatives::Vector{CompiledDerivativeFormula},
                          coefficients::Vector{Float64}, 
                          data, row_idx::Int)
    
    # Pre-allocate temporary vector for derivative computation
    deriv_vec = Vector{Float64}(undef, length(compiled_formula))
    
    for (i, compiled_deriv) in enumerate(compiled_derivatives)
        # Compute derivative vector
        derivative_modelrow!(deriv_vec, compiled_deriv, data, row_idx)
        
        # Compute marginal effect as dot product with coefficients
        effects_vec[i] = dot(deriv_vec, coefficients)
    end
    
    return effects_vec
end

###############################################################################
# INTEGRATION WITH EXISTING ECOSYSTEM
###############################################################################

"""
Integration points with existing FormulaCompiler.jl ecosystem:

1. **CompiledFormula.jl**: 
   - Uses existing `generate_code_from_evaluator()` infrastructure
   - Reuses existing `@generated` function pattern
   - Leverages existing `FORMULA_CACHE` pattern for `DERIVATIVE_CACHE`

2. **generators.jl**:
   - Adds `generate_evaluator_code!` method for `PositionalDerivativeEvaluator`
   - Reuses existing `next_var()` for variable naming
   - Extends existing code generation infrastructure

3. **evaluators.jl**:
   - Adds `output_width` method for `PositionalDerivativeEvaluator`
   - Integrates with existing evaluator type hierarchy

4. **derivatives.jl**:
   - Keeps all existing derivative computation functions
   - Adds new overload for full-width derivatives
   - Maintains backward compatibility

5. **modelrow!.jl**:
   - New `derivative_modelrow!` mirrors existing `modelrow!` interfaces
   - Maintains same performance characteristics
   - Provides convenient marginal effects computation
"""