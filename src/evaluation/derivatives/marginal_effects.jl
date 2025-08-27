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
        # Zero-allocation finite difference path
        derivative_modelrow_fd_pos!(de.jacobian_buffer, de, row)
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