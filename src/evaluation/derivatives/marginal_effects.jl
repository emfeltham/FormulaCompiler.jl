# marginal_effects.jl - Marginal effects implementations with backend selection

"""
    marginal_effects_eta!(g, de, beta, row; backend=:ad)

Fill `g` with marginal effects of η = Xβ w.r.t. `de.vars` at `row`.
Implements: `g = J' * β`, where `J = ∂X/∂vars`.

Arguments:
- `backend::Symbol`: `:ad` (ForwardDiff) or `:fd` (finite differences)

Backends and allocations:
- `:ad`: Uses ForwardDiff automatic differentiation. Small allocations (≤288 bytes) 
  due to AD internals, but faster and more accurate.
- `:fd`: Uses zero-allocation finite differences. Strict 0 bytes after warmup,
  but slightly slower due to multiple function evaluations.
- Allocating convenience (`marginal_effects_eta`) allocates the result vector by design.

Recommendations:
- Use `:fd` backend for strict zero-allocation requirements
- Use `:ad` backend for speed and numerical accuracy (default)
"""
function marginal_effects_eta!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    backend::Symbol = :ad,
)
    @assert length(g) == length(de.vars)
    @assert length(beta) == length(de)
    
    # Select backend for Jacobian computation
    if backend === :fd
        # Zero-allocation finite difference path using evaluator
        derivative_modelrow_fd!(de.jacobian_buffer, de, row)
    elseif backend === :ad
        # ForwardDiff automatic differentiation path
        derivative_modelrow!(de.jacobian_buffer, de, row)
    else
        throw(ArgumentError("Invalid backend: $backend. Use :ad or :fd"))
    end
    
    # Matrix multiplication: g = J' * β (always zero-allocation)
    mul!(g, transpose(de.jacobian_buffer), beta)
    return g
end

function marginal_effects_eta(
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    backend::Symbol = :ad,
)
    g = Vector{Float64}(undef, length(de.vars))
    marginal_effects_eta!(g, de, beta, row; backend=backend)
    return g
end

"""
    marginal_effects_mu!(g, de, beta, row; link, backend=:ad)

Compute marginal effects of μ = g⁻¹(η) at `row` via chain rule: `dμ/dx = (dμ/dη) * (dη/dx)`.

Arguments:
- `link`: Link function (e.g., `IdentityLink()`, `LogLink()`, `LogitLink()`)
- `backend::Symbol`: `:ad` (ForwardDiff) or `:fd` (finite differences)

Backends and allocations:
- `:ad`: Uses ForwardDiff via η path. Small allocations (≤256 bytes) due to AD internals,
  but faster and more accurate.
- `:fd`: Uses zero-allocation finite differences. Strict 0 bytes after warmup,
  but slightly slower due to multiple function evaluations.
- Allocating convenience (`marginal_effects_mu`) allocates the result vector by design.

Recommendations:
- Use `:fd` backend for strict zero-allocation requirements
- Use `:ad` backend for speed and numerical accuracy (default)
"""
function marginal_effects_mu!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    link=GLM.IdentityLink(),
    backend::Symbol = :ad,
)
    # Compute dη/dx using selected backend and preallocated buffer
    marginal_effects_eta!(de.eta_gradient_buffer, de, beta, row; backend=backend)
    # Compute η at row using preallocated buffer
    de.compiled_base(de.xrow_buffer, de.base_data, row)
    η = dot(beta, de.xrow_buffer)
    scale = _dmu_deta(link, η)
    @inbounds @fastmath for j in eachindex(de.eta_gradient_buffer)
        g[j] = scale * de.eta_gradient_buffer[j]
    end
    return g
end

function marginal_effects_mu(
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    link=GLM.IdentityLink(),
    backend::Symbol = :ad,
)
    g = Vector{Float64}(undef, length(de.vars))
    marginal_effects_mu!(g, de, beta, row; link=link, backend=backend)
    return g
end

"""
    me_eta_grad_beta!(gβ, de, β, row, var)

Compute ∂m/∂β for m = marginal effect on η w.r.t. `var`: gβ .= J_k.

Arguments:
- `gβ::Vector{Float64}`: Preallocated buffer of length `n_terms`
- `de::DerivativeEvaluator`: Built by `build_derivative_evaluator`
- `β::Vector{Float64}`: Model coefficients
- `row::Int`: Row index (1-based)
- `var::Symbol`: Variable for marginal effect (must be in `de.vars`)

Returns:
- The same `gβ` buffer, with gradient of η marginal effect w.r.t. parameters

Notes:
- Zero allocations per call after warmup
- For η marginal effects: ∂m/∂β = ∂X/∂var (single Jacobian column)
- Uses zero-allocation single-column FD implementation
"""
function me_eta_grad_beta!(
    gβ::Vector{Float64},
    de::DerivativeEvaluator,
    β::Vector{Float64},
    row::Int,
    var::Symbol,
)
    @assert length(gβ) == length(de)
    
    # For η marginal effects, ∂m/∂β = J_k (single Jacobian column)
    # Use zero-allocation single-column FD
    fd_jacobian_column!(gβ, de, row, var)
    return gβ
end

"""
    me_mu_grad_beta!(gβ, de, β, row, var; link=GLM.IdentityLink())

Compute ∂m/∂β for m = marginal effect on μ w.r.t. `var` using chain rule.

Arguments:
- `gβ::Vector{Float64}`: Preallocated buffer of length `n_terms`
- `de::DerivativeEvaluator`: Built by `build_derivative_evaluator`
- `β::Vector{Float64}`: Model coefficients
- `row::Int`: Row index (1-based)
- `var::Symbol`: Variable for marginal effect (must be in `de.vars`)
- `link`: GLM link function

Returns:
- The same `gβ` buffer, with gradient of μ marginal effect w.r.t. parameters

Formula:
- gβ = g'(η) * J_k + (J_k' * β) * g''(η) * X_row
- where g'(η) = dμ/dη, g''(η) = d²μ/dη², J_k = ∂X/∂var

Notes:
- Zero allocations per call after warmup
- Uses preallocated evaluator buffers
- Implements full chain rule for μ marginal effects
"""
function me_mu_grad_beta!(
    gβ::Vector{Float64},
    de::DerivativeEvaluator,
    β::Vector{Float64},
    row::Int,
    var::Symbol;
    link=GLM.IdentityLink(),
)
    @assert length(gβ) == length(de)
    
    # Step 1: Compute X_row and η, store X_row in xrow_buffer
    de.compiled_base(de.xrow_buffer, de.base_data, row)
    η = dot(β, de.xrow_buffer)
    
    # Step 2: Get link function derivatives
    g_prime = _dmu_deta(link, η)      # dμ/dη
    g_double_prime = _d2mu_deta2(link, η)  # d²μ/dη²
    
    # Step 3: Get single Jacobian column J_k and store in fd_yplus buffer (reuse as temporary)
    fd_jacobian_column!(de.fd_yplus, de, row, var)  # Now fd_yplus contains J_k
    
    # Step 4: Compute J_k' * β (scalar)
    Jk_dot_beta = dot(de.fd_yplus, β)
    
    # Step 5: Apply chain rule formula: gβ = g'(η) * J_k + (J_k' * β) * g''(η) * X_row
    # xrow_buffer contains X_row, fd_yplus contains J_k
    @inbounds @fastmath for i in eachindex(gβ)
        gβ[i] = g_prime * de.fd_yplus[i] + Jk_dot_beta * g_double_prime * de.xrow_buffer[i]
    end
    
    return gβ
end