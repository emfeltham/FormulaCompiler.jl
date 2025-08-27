# automatic_diff.jl
# ForwardDiff automatic differentiation implementations

"""
    derivative_modelrow!(J, deval, row) -> AbstractMatrix{Float64}

Fill `J` with the Jacobian of one model row with respect to `deval.vars`.

Arguments:
- `J::AbstractMatrix{Float64}`: Preallocated buffer of size `(n_terms, n_vars)`.
- `deval::DerivativeEvaluator`: Built by `build_derivative_evaluator`.
- `row::Int`: Row index (1-based).

Returns:
- The same `J` buffer, with `J[i, j] = ∂X[i]/∂vars[j]` for the given row.

Notes:
- Orientation is `(n_terms, n_vars)`; `n_terms == length(compiled)`.
- Small allocations (~368 bytes) due to ForwardDiff internals. For strict zero-allocation
  requirements, use `derivative_modelrow_fd!` instead.
"""
function derivative_modelrow!(J::AbstractMatrix{Float64}, de::DerivativeEvaluator, row::Int)
    @assert size(J, 1) == length(de) "Jacobian row mismatch: expected $(length(de)) terms"
    @assert size(J, 2) == length(de.vars) "Jacobian column mismatch: expected $(length(de.vars)) variables"
    # Set row and seed x with base values
    de.row = row
    for (i, s) in enumerate(de.vars)
        de.xbuf[i] = getproperty(de.base_data, s)[row]
    end
    ForwardDiff.jacobian!(J, de.g, de.xbuf, de.cfg)
    return J
end

"""
    derivative_modelrow(deval, row) -> Matrix{Float64}

Allocating convenience wrapper that returns the Jacobian for one row.
"""
function derivative_modelrow(de::DerivativeEvaluator, row::Int)
    J = Matrix{Float64}(undef, length(de), length(de.vars))
    derivative_modelrow!(J, de, row)
    return J
end

"""
    marginal_effects_eta_grad!(g, de, beta, row)

Compute marginal effects on η = Xβ via ForwardDiff.gradient! of the scalar
function h(x) = dot(β, Xrow(x)). Uses the existing AD closure to evaluate rows.
"""
function marginal_effects_eta_grad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
)
    @assert length(g) == length(de.vars)
    @assert length(beta) == length(de)
    # Seed x with base values
    de.row = row
    for (i, s) in enumerate(de.vars)
        de.xbuf[i] = getproperty(de.base_data, s)[row]
    end
    # Use stored scalar closure/config, update beta reference
    de.beta_ref[] = (beta isa Vector{Float64} ? beta : Vector{Float64}(beta))
    ForwardDiff.gradient!(g, de.gscalar, de.xbuf, de.gradcfg)
    return g
end

function marginal_effects_eta_grad(
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
)
    g = Vector{Float64}(undef, length(de.vars))
    marginal_effects_eta_grad!(g, de, beta, row)
    return g
end