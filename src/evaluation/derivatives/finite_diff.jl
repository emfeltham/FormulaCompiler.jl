# finite_diff.jl - Clean & performant finite differences implementation

# Mathematical constant for optimal central difference step sizing
const FD_AUTO_EPS_SCALE = cbrt(eps(Float64))  # ≈ 6.055e-6

"""
    derivative_modelrow!(J, de::FDEvaluator, row) -> J

Primary finite differences API - zero allocations, concrete type dispatch.

Computes full Jacobian matrix ∂X[i]/∂vars[j] using central differences with
adaptive step sizing. Matches automatic_diff.jl signature for seamless backend switching.

# Performance Characteristics
- **Memory**: 0 bytes allocated (uses pre-allocated FDEvaluator buffers)
- **Speed**: ~65ns per variable with mathematical optimizations
- **Accuracy**: Adaptive step sizing balances truncation/roundoff error

# Mathematical Method
Central differences: ∂f/∂x ≈ [f(x+h) - f(x-h)] / (2h)
Step sizing: h = ε^(1/3) * max(1, |x|) for numerical stability

# Arguments
- `J::AbstractMatrix{Float64}`: Pre-allocated Jacobian buffer of size `(n_terms, n_vars)`
- `de::FDEvaluator`: Pre-built evaluator from `derivativeevaluator_fd(compiled, data, vars)`
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `J`: The same matrix passed in, containing `J[i,j] = ∂X[i]/∂vars[j]`

# Example
```julia
using FormulaCompiler, GLM

# Setup model and data
model = lm(@formula(y ~ x * group + log(abs(z) + 1)), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build FD evaluator
de_fd = derivativeevaluator_fd(compiled, data, [:x, :z])

# Zero-allocation finite differences
J = Matrix{Float64}(undef, length(compiled), length(de_fd.vars))
derivative_modelrow!(J, de_fd, 1)  # 0 bytes allocated
```

See also: [`derivativeevaluator_fd`](@ref)
"""
function derivative_modelrow!(J::AbstractMatrix{Float64}, de::FDEvaluator, row::Int)
    @assert size(J, 1) == length(de) "Expected $(length(de)) output terms, got $(size(J, 1))"
    @assert size(J, 2) == length(de.vars) "Expected $(length(de.vars)) variables, got $(size(J, 2))"

    # Extract core performance patterns from existing optimized implementation
    yplus = de.y_plus
    yminus = de.yminus
    xbase = de.xbase
    nterms = length(de)
    nvars = length(de.vars)

    # Efficient CounterfactualVector setup - batch row updates
    @inbounds for i in 1:nvars
        update_counterfactual_row!(de.counterfactuals[i], row)
    end

    # Cache base values to avoid stale replacement reads
    @inbounds for j in 1:nvars
        xbase[j] = de.counterfactuals[j].base[row]
    end

    # Main finite differences loop - preserves mathematical optimizations
    @inbounds for j in 1:nvars
        x = xbase[j]

        # Reset all counterfactuals to base values for this variable
        for k in 1:nvars
            update_counterfactual_replacement!(de.counterfactuals[k], xbase[k])
        end

        # Optimal adaptive step sizing
        h = FD_AUTO_EPS_SCALE * max(1.0, abs(x))

        # f(x + h) evaluation
        update_counterfactual_replacement!(de.counterfactuals[j], x + h)
        de.compiled_base(yplus, de.data_counterfactual, row)

        # f(x - h) evaluation
        update_counterfactual_replacement!(de.counterfactuals[j], x - h)
        de.compiled_base(yminus, de.data_counterfactual, row)

        # Central difference computation - single division, fast inner loop
        inv_2h = 1.0 / (2.0 * h)
        @fastmath for i in 1:nterms
            J[i, j] = (yplus[i] - yminus[i]) * inv_2h
        end
    end

    return J
end

"""
    fd_jacobian_column!(Jk, de::FDEvaluator, row, var_idx) -> Jk

Single-column finite differences - zero allocations, optimized for partial Jacobian computation.

More efficient than full Jacobian when only one variable's derivatives are needed.
Uses same mathematical core as derivative_modelrow! but avoids unnecessary computations.

# Arguments
- `Jk::Vector{Float64}`: Pre-allocated buffer of length `n_terms`
- `de::FDEvaluator`: Pre-built evaluator from `derivativeevaluator_fd(compiled, data, vars)`
- `row::Int`: Row index to evaluate (1-based indexing)
- `var_idx::Int`: Variable index (1-based) corresponding to `de.vars[var_idx]`

# Returns
- `Jk`: The same vector passed in, containing `Jk[i] = ∂X[i]/∂de.vars[var_idx]`

# Performance Characteristics
- **Memory**: 0 bytes allocated (uses pre-allocated FDEvaluator buffers)
- **Speed**: More efficient than full Jacobian for single variables
- **Method**: Central differences with adaptive step sizing

# Example
```julia
# Single variable derivatives (more efficient than full Jacobian)
de_fd = derivativeevaluator_fd(compiled, data, [:x, :z])
Jk = Vector{Float64}(undef, length(compiled))
fd_jacobian_column!(Jk, de_fd, 1, 1)  # Derivatives w.r.t. first variable (:x)
```

See also: [`derivative_modelrow!`](@ref), [`derivativeevaluator_fd`](@ref)
"""
function fd_jacobian_column!(Jk::Vector{Float64}, de::FDEvaluator, row::Int, var_idx::Int)
    @assert 1 ≤ var_idx ≤ length(de.vars) "var_idx $var_idx out of bounds [1, $(length(de.vars))]"
    @assert length(Jk) == length(de) "Expected output length $(length(de)), got $(length(Jk))"

    # Optimal buffer reuse patterns from full Jacobian
    yplus = de.y_plus
    yminus = de.yminus
    xbase = de.xbase  # Use buffer for efficient base value caching
    nterms = length(de)
    nvars = length(de.vars)

    # Batch counterfactual setup
    @inbounds for i in 1:nvars
        update_counterfactual_row!(de.counterfactuals[i], row)
    end

    # Cache all base values to buffer (avoid repeated data access)
    @inbounds for j in 1:nvars
        xbase[j] = de.counterfactuals[j].base[row]
    end

    # Get target variable from cached buffer
    x = xbase[var_idx]

    # Reset all counterfactuals using cached base values
    @inbounds for k in 1:nvars
        update_counterfactual_replacement!(de.counterfactuals[k], xbase[k])
    end

    # Single variable finite difference computation
    h = FD_AUTO_EPS_SCALE * max(1.0, abs(x))

    # f(x + h)
    @inbounds update_counterfactual_replacement!(de.counterfactuals[var_idx], x + h)
    de.compiled_base(yplus, de.data_counterfactual, row)

    # f(x - h)
    @inbounds update_counterfactual_replacement!(de.counterfactuals[var_idx], x - h)
    de.compiled_base(yminus, de.data_counterfactual, row)

    # Central difference for single column
    inv_2h = 1.0 / (2.0 * h)
    @inbounds @fastmath for i in 1:nterms
        Jk[i] = (yplus[i] - yminus[i]) * inv_2h
    end

    return Jk
end
