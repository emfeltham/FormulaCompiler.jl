# automatic_diff.jl
# ForwardDiff automatic differentiation implementations

"""
    derivative_modelrow!(J, de::ADEvaluator, row) -> J

Primary automatic differentiation API - zero allocations, concrete type dispatch.

Computes full Jacobian matrix ∂X[i]/∂vars[j] using ForwardDiff dual numbers with
machine precision accuracy. Matches finite_diff.jl signature for seamless backend switching.

# Arguments
- `J::AbstractMatrix{Float64}`: Preallocated Jacobian buffer of size `(n_terms, n_vars)`
- `de::ADEvaluator`: AD evaluator built by `derivativevaluator(:ad, compiled, data, vars)`
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `J`: The same matrix passed in, now containing `J[i,j] = ∂X[i]/∂vars[j]` for the specified row

# Performance Characteristics
- **Memory**: 0 bytes allocated (uses pre-allocated ADEvaluator buffers)
- **Speed**: ~49ns per variable with dual number optimizations
- **Accuracy**: Machine precision derivatives via ForwardDiff dual arithmetic

# Mathematical Method
Automatic differentiation: f(x + εe) = f(x) + ε∇f(x)ᵀe
Computes exact derivatives accounting for all formula transformations.

# Dependencies
Requires complete OVERRIDE.md Phase 1-3 implementation:
- Pre-converted dual data in data_counterfactual
- CounterfactualVector system with update_counterfactual_for_var! API

# Example
```julia
using FormulaCompiler, GLM

# Setup model with interactions and functions
model = lm(@formula(y ~ x * group + log(abs(z) + 1)), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build AD evaluator for continuous variables
vars = [:x, :z]
de = derivativevaluator(:ad, compiled, data, vars)  # Note: :ad backend

# Zero-allocation Jacobian computation
J = Matrix{Float64}(undef, length(compiled), length(vars))
derivative_modelrow!(J, de, 1)  # 0 bytes allocated

# Zero-allocation marginal effects
β = coef(model)
g = Vector{Float64}(undef, length(vars))
marginal_effects_eta!(g, de, β, 1)  # 0 bytes allocated
```

"""
function derivative_modelrow!(J::AbstractMatrix{Float64}, de::ADEvaluator, row::Int)
    # AD evaluator method (dispatch on concrete ADEvaluator type)
    @assert size(J, 1) == length(de) "Jacobian row mismatch: expected $(length(de)) terms"
    @assert size(J, 2) == length(de.vars) "Jacobian column mismatch: expected $(length(de.vars)) variables"

    # Direct field access - no buffer indirection needed
    x_dual_vec = de.x_dual_vec
    partials_unit_vec = de.partials_unit_vec
    rowvec_dual_vec = de.rowvec_dual_vec

    # Set up dual numbers with proper base values and unit partials
    N = length(de.vars)
    @inbounds for i in 1:N
        # Get base value and create dual number with unit partials
        base_val = getproperty(de.base_data, de.vars[i])[row]
        x_dual_vec[i] = typeof(x_dual_vec[i])(Float64(base_val), partials_unit_vec[i])
        # Update counterfactual vectors for this row
        update_counterfactual_for_var!(de.counterfactuals, de.vars, de.vars[i], row, x_dual_vec[i])
    end

    # Evaluate using specialized dual system
    de.compiled_dual(rowvec_dual_vec, de.data_counterfactual, row)

    # Extract gradients from dual results
    @inbounds for i in 1:size(J,1)
        parts = ForwardDiff.partials(rowvec_dual_vec[i])
        for j in 1:N
            J[i,j] = parts[j]
        end
    end
    return J
end


# Type barrier function for zero-allocation matrix multiply
@noinline function _matrix_multiply_eta!(
    g::AbstractVector{Float64},
    jacobian_buffer::Matrix{Float64},
    βref::Vector{Float64}
)
    @inbounds @fastmath for j in eachindex(g)
        acc = 0.0
        for i in 1:size(jacobian_buffer, 1)
            acc += jacobian_buffer[i, j] * βref[i]
        end
        g[j] = acc
    end
    return g
end

"""
    marginal_effects_eta!(g, de::ADEvaluator, β, row) -> g

Compute marginal effects on η = Xβ using automatic differentiation - zero allocations.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction),
not all variables in the dataset.

More efficient than computing full Jacobian when only marginal effects are needed.
Uses same mathematical core as derivative_modelrow! but optimized for gradient computation.

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(de.vars)`
- `de::ADEvaluator`: AD evaluator built by `derivativevaluator(:ad, compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients of length `length(de)` (may be any numeric type)
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `g`: The same vector passed in, now containing marginal effects `∂η/∂vars[j]` for the specified row

# Performance Characteristics
- **Memory**: 0 bytes allocated after warmup (uses pre-allocated ADEvaluator buffers)
- **Speed**: Optimized for gradient computation using type barrier and @fastmath
- **Type handling**: Zero-allocation conversion from any coefficient type to Float64

# Mathematical Method
Computes marginal effects: ∂η/∂x = (∂X/∂x)ᵀβ where X is model matrix row and η = Xβ.
Equivalent to `g = J'β` where J is the Jacobian from `derivative_modelrow!`.
"""
function marginal_effects_eta!(
    g::AbstractVector{Float64},
    de::ADEvaluator,
    β::AbstractVector{<:Real},
    row::Int
)
    # Simple bounds checks without string interpolation to avoid allocations
    length(g) == length(de.vars) || throw(DimensionMismatch("gradient length mismatch"))
    length(β) == length(de) || throw(DimensionMismatch("beta length mismatch"))

    # Zero-allocation beta type handling
    if β isa Vector{Float64}
        de.beta_ref[] = β              # No conversion needed
    else
        copyto!(de.beta_buf, β)        # Convert to Float64 once
        de.beta_ref[] = de.beta_buf    # Point to converted buffer
    end

    # Use derivative_modelrow! for Jacobian computation (leverages OVERRIDE.md counterfactual system)
    derivative_modelrow!(de.jacobian_buffer, de, row)

    # Type barrier for zero-allocation matrix multiply: g = J'β
    _matrix_multiply_eta!(g, de.jacobian_buffer, de.beta_ref[])

    return g
end

"""
    marginal_effects_eta!(g, Gβ, de::ADEvaluator, β, row) -> (g, Gβ)

Compute both marginal effects and parameter gradients for η = Xβ using automatic differentiation - zero allocations.

Extended version of marginal_effects_eta! that simultaneously computes marginal effects and
parameter gradients for all variables. The parameter gradient matrix is essentially free since
it's just the transpose of the already-computed Jacobian matrix.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction).

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(de.vars)`
- `Gβ::Matrix{Float64}`: Preallocated parameter gradient matrix of size `(length(de), length(de.vars))`
- `de::ADEvaluator`: AD evaluator built by `derivativevaluator(:ad, compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients of length `length(de)` (may be any numeric type)
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `(g, Gβ)`: Tuple containing marginal effects and parameter gradients
  - `g[j] = ∂η/∂vars[j]` for all variables in de.vars
  - `Gβ[i,j] = ∂(∂η/∂vars[j])/∂β[i]` for all model parameters and variables

# Performance Characteristics
- **Memory**: 0 bytes allocated (parameter gradients are transpose of computed Jacobian)
- **Speed**: Negligible overhead compared to regular marginal_effects_eta!
- **Efficiency**: Parameter gradient computation is essentially free

# Mathematical Method
- Marginal effects: Same as regular marginal_effects_eta!
- Parameter gradients: `Gβ = J'` where J is the Jacobian matrix

# Example
```julia
# Simultaneous marginal effects + parameter gradients for all variables
vars = [:x, :z]
de = derivativevaluator(:ad, compiled, data, vars)
g = Vector{Float64}(undef, length(vars))
Gβ = Matrix{Float64}(undef, length(de), length(vars))

# Get both marginal effects and parameter gradients
marginal_effects_eta!(g, Gβ, de, β, 1)

# g now contains marginal effects ∂η/∂vars
# Gβ now contains parameter gradients ∂(∂η/∂vars[j])/∂β[i]
```
"""
function marginal_effects_eta!(
    g::AbstractVector{Float64},
    Gβ::AbstractMatrix{Float64},
    de::ADEvaluator,
    β::AbstractVector{<:Real},
    row::Int
)
    # Simple bounds checks without string interpolation to avoid allocations
    length(g) == length(de.vars) || throw(DimensionMismatch("gradient length mismatch"))
    size(Gβ, 1) == length(de) || throw(DimensionMismatch("Gβ first dimension must match length(de)"))
    size(Gβ, 2) == length(de.vars) || throw(DimensionMismatch("Gβ second dimension must match length(de.vars)"))
    length(β) == length(de) || throw(DimensionMismatch("beta length mismatch"))

    # Zero-allocation beta type handling
    if β isa Vector{Float64}
        de.beta_ref[] = β              # No conversion needed
    else
        copyto!(de.beta_buf, β)        # Convert to Float64 once
        de.beta_ref[] = de.beta_buf    # Point to converted buffer
    end

    # Use derivative_modelrow! for Jacobian computation (leverages OVERRIDE.md counterfactual system)
    derivative_modelrow!(de.jacobian_buffer, de, row)

    # Extract parameter gradients (essentially free - transpose of Jacobian)
    # Gβ[i,j] = ∂(∂η/∂vars[j])/∂β[i] = J[i,j]
    Gβ .= de.jacobian_buffer

    # Type barrier for zero-allocation matrix multiply: g = J'β
    _matrix_multiply_eta!(g, de.jacobian_buffer, de.beta_ref[])

    return g, Gβ
end

# Batch operations for multiple rows - maximum performance

"""
    derivative_modelrow!(J, de::ADEvaluator, rows) -> J

Batch automatic differentiation for multiple rows - zero allocations after warmup.

Computes Jacobian matrices ∂X[i]/∂vars[j] for multiple rows efficiently by reusing
buffers and minimizing setup overhead. Optimal for bootstrap inference, AME computation,
and systematic sensitivity analysis.

# Arguments
- `J::Array{Float64,3}`: Preallocated Jacobian tensor of size `(length(rows), n_terms, n_vars)`
  - `J[k, i, j] = ∂X[i]/∂vars[j]` for row `rows[k]`
- `de::ADEvaluator`: AD evaluator built by `derivativevaluator(:ad, compiled, data, vars)`
- `rows::AbstractVector{Int}`: Row indices to evaluate (1-based indexing)

# Returns
- `J`: The same tensor passed in, now containing Jacobians for all specified rows

# Performance Characteristics
- **Memory**: 0 bytes allocated (reuses single-row buffers efficiently)
- **Speed**: ~49ns per variable per row with optimized batch processing
- **Scaling**: Linear in number of rows with minimal overhead

# Mathematical Method
Applies single-row automatic differentiation across multiple rows:
`J[k, i, j] = ∂(model_matrix_row[i])/∂(vars[j])` for row `rows[k]`

# Example
```julia
# Batch Jacobian computation for rows 1, 5, 10
rows = [1, 5, 10]
J_batch = Array{Float64,3}(undef, length(rows), length(compiled), length(vars))
derivative_modelrow!(J_batch, de, rows)  # 0 bytes allocated

# Access: J_batch[1, :, :] = Jacobian for row 1
#         J_batch[2, :, :] = Jacobian for row 5
#         J_batch[3, :, :] = Jacobian for row 10
```
"""
function derivative_modelrow!(
    J::AbstractArray{Float64,3},
    de::ADEvaluator,
    rows::AbstractVector{Int}
)
    # Validate dimensions
    size(J, 1) == length(rows) || throw(DimensionMismatch("J first dimension must match length(rows)"))
    size(J, 2) == length(de) || throw(DimensionMismatch("J second dimension must match length(de)"))
    size(J, 3) == length(de.vars) || throw(DimensionMismatch("J third dimension must match length(de.vars)"))

    # Batch processing: reuse single-row logic efficiently
    for (k, row) in enumerate(rows)
        J_k = view(J, k, :, :)  # View into k-th slice
        derivative_modelrow!(J_k, de, row)  # Zero-allocation single-row operation
    end

    return J
end

"""
    marginal_effects_eta!(G, de::ADEvaluator, β, rows) -> G

Batch marginal effects computation for multiple rows - zero allocations after warmup.

Computes marginal effects ∂η/∂x for multiple rows efficiently with single beta setup
and optimized buffer reuse. More efficient than repeated single-row calls for AME
computation, bootstrap inference, and systematic analysis.

# Arguments
- `G::Matrix{Float64}`: Preallocated gradient matrix of size `(length(rows), length(de.vars))`
  - `G[k, j] = ∂η/∂vars[j]` for row `rows[k]`
- `de::ADEvaluator`: AD evaluator built by `derivativevaluator(:ad, compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients (same β used for all rows)
- `rows::AbstractVector{Int}`: Row indices to evaluate (1-based indexing)

# Returns
- `G`: The same matrix passed in, now containing marginal effects for all specified rows

# Performance Characteristics
- **Memory**: 0 bytes allocated (single beta setup + reused buffers)
- **Speed**: Faster than N single-row calls due to reduced setup overhead
- **Scaling**: Linear in number of rows with beta conversion done once

# Mathematical Method
Computes: `G[k, j] = ∂η/∂vars[j]` for row `rows[k]` where η = Xβ
Uses optimized single beta setup followed by efficient row iteration.

# Example
```julia
# Batch marginal effects for bootstrap sample
rows = [1, 3, 7, 12, 15]
G_batch = Matrix{Float64}(undef, length(rows), length(vars))
marginal_effects_eta_batch!(G_batch, de, β, rows)  # 0 bytes allocated

# Access: G_batch[1, :] = marginal effects for row 1
#         G_batch[2, :] = marginal effects for row 3
```
"""
function marginal_effects_eta!(
    G::AbstractMatrix{Float64},
    de::ADEvaluator,
    β::AbstractVector{<:Real},
    rows::AbstractVector{Int}
)
    # Validate dimensions
    size(G, 1) == length(rows) || throw(DimensionMismatch("G first dimension must match length(rows)"))
    size(G, 2) == length(de.vars) || throw(DimensionMismatch("G second dimension must match length(de.vars)"))
    length(β) == length(de) || throw(DimensionMismatch("beta length mismatch"))

    # Single beta setup for entire batch (key optimization)
    if β isa Vector{Float64}
        de.beta_ref[] = β              # No conversion needed
    else
        copyto!(de.beta_buf, β)        # Convert to Float64 once
        de.beta_ref[] = de.beta_buf    # Point to converted buffer
    end

    # Batch processing: reuse setup, iterate rows efficiently
    for (k, row) in enumerate(rows)
        # Compute Jacobian for this row
        derivative_modelrow!(de.jacobian_buffer, de, row)

        # Compute marginal effects using pre-setup beta
        G_k = view(G, k, :)  # View into k-th row
        _matrix_multiply_eta!(G_k, de.jacobian_buffer, de.beta_ref[])
    end

    return G
end
