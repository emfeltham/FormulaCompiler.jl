# marginal_effects.jl
# Marginal effects implementations with concrete type dispatch


"""
    marginal_effects_mu!(g, de::AbstractDerivativeEvaluator, β, link, row) -> g

Compute marginal effects on μ = g⁻¹(η) using automatic differentiation or finite differences - zero allocations.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction),
not all variables in the dataset.

Computes response-scale marginal effects by applying the chain rule:
∂μ/∂x = (∂μ/∂η) × (∂η/∂x) = g'(η) × (∂η/∂x)

More efficient than separate computations by leveraging the existing zero-allocation
η-scale computation and applying link function derivatives.

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(de.vars)`
- `de::AbstractDerivativeEvaluator`: Evaluator built by `derivativevaluator(:ad/:fd, compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients of length `length(de)` (may be any numeric type)
- `link`: GLM link function (e.g., `GLM.LogitLink()`, `GLM.LogLink()`, `GLM.IdentityLink()`)
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `g`: The same vector passed in, now containing μ-scale marginal effects `∂μ/∂vars[j]` for the specified row

# Performance Characteristics
- **Memory**: 0 bytes allocated after warmup (reuses η-scale computation + link scaling)
- **Speed**: Marginally slower than η-scale due to link derivative computation
- **Type handling**: Zero-allocation conversion from any coefficient type to Float64

# Mathematical Method
Two-step computation using chain rule:
1. Compute η-scale marginal effects: ∂η/∂x via `marginal_effects_eta!`
2. Apply link derivative scaling: ∂μ/∂x = g'(η) × ∂η/∂x

Supported link functions: Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt

# Example
```julia
# Response-scale marginal effects for logistic regression
link = GLM.LogitLink()
g_mu = Vector{Float64}(undef, length(vars))
marginal_effects_mu!(g_mu, de, β, link, 1)  # 0 bytes allocated

# For logit: ∂μ/∂x = σ(η)(1-σ(η)) × ∂η/∂x where σ is sigmoid
```
"""
function marginal_effects_mu!(
    g::AbstractVector{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{<:Real},
    link,
    row::Int
)
    # Step 1: Compute η-scale marginal effects (reuses our zero-allocation implementation)
    marginal_effects_eta!(g, de, β, row)

    # Step 2: Compute η = Xβ for link derivative evaluation
    de.compiled_base(de.xrow_buffer, de.base_data, row)
    η = dot(β, de.xrow_buffer)

    # Step 3: Apply chain rule with link function derivative
    link_derivative = _dmu_deta(link, η)

    # Step 4: Scale all marginal effects: ∂μ/∂x = g'(η) × ∂η/∂x
    @inbounds @fastmath for i in eachindex(g)
        g[i] *= link_derivative
    end

    return g
end

"""
    marginal_effects_eta!(g, Gβ, de::AbstractDerivativeEvaluator, β, row) -> (g, Gβ)

Compute both marginal effects and parameter gradients for η = Xβ - zero allocations.

Wrapper function that dispatches to the appropriate backend (AD or FD) implementation.
Simultaneously computes marginal effects and parameter gradients for all variables,
with parameter gradients being essentially free since they're the computed Jacobian.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction).

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(de.vars)`
- `Gβ::Matrix{Float64}`: Preallocated parameter gradient matrix of size `(length(de), length(de.vars))`
- `de::AbstractDerivativeEvaluator`: Evaluator built by `derivativevaluator(:ad/:fd, compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients of length `length(de)` (may be any numeric type)
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `(g, Gβ)`: Tuple containing marginal effects and parameter gradients
  - `g[j] = ∂η/∂vars[j]` for all variables in de.vars
  - `Gβ[i,j] = ∂(∂η/∂vars[j])/∂β[i]` for all model parameters and variables

# Performance Characteristics
- **Memory**: 0 bytes allocated (parameter gradients are copy/transpose of computed Jacobian)
- **Speed**: Negligible overhead compared to regular marginal_effects_eta!
- **Backend**: Automatically uses AD or FD based on evaluator type

# Example
```julia
# Works with both AD and FD backends
vars = [:x, :z]
de_ad = derivativevaluator(:ad, compiled, data, vars)
de_fd = derivativevaluator(:fd, compiled, data, vars)

g = Vector{Float64}(undef, length(vars))
Gβ = Matrix{Float64}(undef, length(de_ad), length(vars))

# Both calls have identical interface
marginal_effects_eta!(g, Gβ, de_ad, β, 1)  # Uses AD
marginal_effects_eta!(g, Gβ, de_fd, β, 1)  # Uses FD
```
"""
function marginal_effects_eta!(
    g::AbstractVector{Float64},
    Gβ::AbstractMatrix{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{<:Real},
    row::Int
)
    # Dispatch to appropriate backend implementation
    return marginal_effects_eta!(g, Gβ, de, β, row)
end


"""
    marginal_effects_mu!(G, de::AbstractDerivativeEvaluator, β, link, rows) -> G

Batch μ-scale marginal effects computation for multiple rows - zero allocations after warmup.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction),
not all variables in the dataset.

Computes response-scale marginal effects ∂μ/∂x for multiple rows efficiently by leveraging
the batch η-scale computation and applying link function derivatives per row. More efficient
than repeated single-row calls for AME computation on response scale.

# Arguments
- `G::Matrix{Float64}`: Preallocated gradient matrix of size `(length(rows), length(de.vars))`
  - `G[k, j] = ∂μ/∂vars[j]` for row `rows[k]`
- `de::AbstractDerivativeEvaluator`: Evaluator built by `derivativevaluator(:ad/:fd, compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients (same β used for all rows)
- `link`: GLM link function (e.g., `GLM.LogitLink()`, `GLM.LogLink()`)
- `rows::AbstractVector{Int}`: Row indices to evaluate (1-based indexing)

# Returns
- `G`: The same matrix passed in, now containing μ-scale marginal effects for all specified rows

# Performance Characteristics
- **Memory**: 0 bytes allocated (single beta setup + efficient η computation + link scaling)
- **Speed**: Efficient batch processing with minimal per-row overhead
- **Scaling**: Linear in number of rows with optimized η computation

# Mathematical Method
Two-step batch computation:
1. Compute η-scale marginal effects using `marginal_effects_eta_batch!`
2. Apply link derivatives per row: ∂μ/∂x = g'(η_k) × ∂η/∂x for each row k

# Example
```julia
# Batch μ-scale marginal effects for logistic model
link = GLM.LogitLink()
rows = [1, 3, 7, 12, 15]
G_mu_batch = Matrix{Float64}(undef, length(rows), length(vars))
marginal_effects_mu_batch!(G_mu_batch, de, β, link, rows)  # 0 bytes allocated

# Access: G_mu_batch[1, :] = μ-scale marginal effects for row 1
#         G_mu_batch[2, :] = μ-scale marginal effects for row 3
```
"""
function marginal_effects_mu!(
    G::AbstractMatrix{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{<:Real},
    link,
    rows::AbstractVector{Int}
)
    # Validate dimensions
    size(G, 1) == length(rows) || throw(DimensionMismatch("G first dimension must match length(rows)"))
    size(G, 2) == length(de.vars) || throw(DimensionMismatch("G second dimension must match length(de.vars)"))
    length(β) == length(de) || throw(DimensionMismatch("beta length mismatch"))

    # Step 1: Compute η-scale marginal effects for all rows (reuse our efficient batch function)
    marginal_effects_eta!(G, de, β, rows)

    # Step 2: Apply link function derivatives per row
    for (k, row) in enumerate(rows)
        # Compute η = Xβ for this row
        de.compiled_base(de.xrow_buffer, de.base_data, row)
        η = dot(β, de.xrow_buffer)

        # Apply link derivative to this row's marginal effects
        link_derivative = _dmu_deta(link, η)
        G_k = view(G, k, :)  # View into k-th row

        @inbounds @fastmath for j in eachindex(G_k)
            G_k[j] *= link_derivative
        end
    end

    return G
end

"""
    marginal_effects_mu!(g, Gβ, de::AbstractDerivativeEvaluator, β, link, row) -> (g, Gβ)

Compute both marginal effects and parameter gradients for μ = g⁻¹(η) - zero allocations.

Extended version of marginal_effects_mu! that simultaneously computes marginal effects and
parameter gradients for all variables using the chain rule. More computationally intensive
than η-scale due to link function second derivatives, but still efficient.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction).

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(de.vars)`
- `Gβ::Matrix{Float64}`: Preallocated parameter gradient matrix of size `(length(de), length(de.vars))`
- `de::AbstractDerivativeEvaluator`: Evaluator built by `derivativevaluator(:ad/:fd, compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients of length `length(de)` (may be any numeric type)
- `link`: GLM link function (e.g., `GLM.LogitLink()`, `GLM.LogLink()`, `GLM.IdentityLink()`)
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `(g, Gβ)`: Tuple containing marginal effects and parameter gradients
  - `g[j] = ∂μ/∂vars[j]` for all variables in de.vars
  - `Gβ[i,j] = ∂(∂μ/∂vars[j])/∂β[i]` for all model parameters and variables

# Performance Characteristics
- **Memory**: 0 bytes allocated (reuses η-scale computation + chain rule application)
- **Speed**: Slower than η-scale due to link function second derivatives
- **Backend**: Automatically dispatches to AD or FD implementation

# Mathematical Method
Two-step computation using chain rule:
1. Get η-scale marginal effects + parameter gradients: `marginal_effects_eta!(g, Gβ, de, β, row)`
2. Apply full chain rule: `∂(∂μ/∂x_j)/∂β = g'(η) × J_j + (J_j'β) × g''(η) × X_row`

Supported link functions: Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt

# Example
```julia
# Response-scale marginal effects + parameter gradients for logistic regression
link = GLM.LogitLink()
vars = [:x, :z]
de = derivativevaluator(:ad, compiled, data, vars)

g = Vector{Float64}(undef, length(vars))
Gβ = Matrix{Float64}(undef, length(de), length(vars))

marginal_effects_mu!(g, Gβ, de, β, link, 1)
# g contains μ-scale marginal effects
# Gβ contains parameter gradients for uncertainty quantification
```
"""
function marginal_effects_mu!(
    g::AbstractVector{Float64},
    Gβ::AbstractMatrix{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{<:Real},
    link,
    row::Int
)
    # Step 1: Get η-scale marginal effects + all parameter gradients
    marginal_effects_eta!(g, Gβ, de, β, row)  # Gβ = J

    # Step 2: Compute η = Xβ and link derivatives
    de.compiled_base(de.xrow_buffer, de.base_data, row)
    η = dot(β, de.xrow_buffer)
    g_prime = _dmu_deta(link, η)      # dμ/dη
    g_double_prime = _d2mu_deta2(link, η)  # d²μ/dη²

    # Step 3: Apply chain rule to marginal effects
    g .*= g_prime  # g = g'(η) × ∂η/∂x

    # Step 4: Apply FULL chain rule to all parameter gradients
    # ∂(∂μ/∂x_j)/∂β = g'(η) × J_j + (J_j'β) × g''(η) × X_row
    @inbounds for j in 1:size(Gβ, 2)  # For each variable
        # Extract j-th column (parameter gradient for variable j)
        Jj = view(Gβ, :, j)
        Jj_dot_beta = dot(Jj, β)  # J_j'β (scalar)

        # Apply chain rule to each parameter
        for i in 1:size(Gβ, 1)  # For each parameter
            Gβ[i, j] = g_prime * Gβ[i, j] + Jj_dot_beta * g_double_prime * de.xrow_buffer[i]
        end
    end

    return g, Gβ
end

########################### OLD ###########################

# Parameter gradient function (uses variable index for zero-allocation performance)
# FD implementation
function me_mu_grad_beta!(
    gβ::Vector{Float64},
    de::FDEvaluator,
    β::Vector{Float64},
    row::Int,
    var_idx::Int,
    link
)
    @assert length(gβ) == length(de)
    @assert 1 ≤ var_idx ≤ length(de.vars) "var_idx $var_idx out of bounds [1, $(length(de.vars))]"

    # Step 1: Compute X_row and η, store X_row in xrow_buffer
    de.compiled_base(de.xrow_buffer, de.base_data, row)
    η = dot(β, de.xrow_buffer)

    # Step 2: Get link function derivatives
    g_prime = _dmu_deta(link, η)      # dμ/dη
    g_double_prime = _d2mu_deta2(link, η)  # d²μ/dη²

    # Step 3: Get single Jacobian column J_k using indexed version (NO LINEAR SEARCH!)
    fd_jacobian_column_pos!(de.y_plus, de, row, var_idx)  # Now y_plus contains J_k

    # Step 4: Compute J_k' * β (scalar)
    Jk_dot_beta = dot(de.y_plus, β)

    # Step 5: Apply chain rule formula: gβ = g'(η) * J_k + (J_k' * β) * g''(η) * X_row
    # xrow_buffer contains X_row, y_plus contains J_k
    @inbounds @fastmath for i in eachindex(gβ)
        gβ[i] = g_prime * de.y_plus[i] + Jk_dot_beta * g_double_prime * de.xrow_buffer[i]
    end

    return gβ
end

# Parameter gradient function - AD implementation
function me_mu_grad_beta!(
    gβ::Vector{Float64},
    de::ADEvaluator,
    β::Vector{Float64},
    row::Int,
    var_idx::Int,
    link
)
    @assert length(gβ) == length(de)
    @assert 1 ≤ var_idx ≤ length(de.vars) "var_idx $var_idx out of bounds [1, $(length(de.vars))]"

    # Step 1: Compute X_row and η, store X_row in xrow_buffer
    de.compiled_base(de.xrow_buffer, de.base_data, row)
    η = dot(β, de.xrow_buffer)

    # Step 2: Get link function derivatives
    g_prime = _dmu_deta(link, η)      # dμ/dη
    g_double_prime = _d2mu_deta2(link, η)  # d²μ/dη²

    # Step 3: Get single Jacobian column J_k using AD (column from full Jacobian)
    derivative_modelrow!(de.jacobian_buffer, de, row)
    J_k = view(de.jacobian_buffer, :, var_idx)  # Column var_idx

    # Step 4: Compute J_k' * β (scalar)
    Jk_dot_beta = dot(J_k, β)

    # Step 5: Apply chain rule formula: gβ = g'(η) * J_k + (J_k' * β) * g''(η) * X_row
    # xrow_buffer contains X_row, J_k is the Jacobian column
    @inbounds @fastmath for i in eachindex(gβ)
        gβ[i] = g_prime * J_k[i] + Jk_dot_beta * g_double_prime * de.xrow_buffer[i]
    end

    return gβ
end
