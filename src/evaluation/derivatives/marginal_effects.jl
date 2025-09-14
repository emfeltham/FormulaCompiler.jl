# marginal_effects.jl - Marginal effects implementations with backend selection

"""
    marginal_effects_eta!(g, evaluator, beta, row; backend=:ad) -> g

Compute marginal effects on the linear predictor η = Xβ with respect to continuous variables.

Fills a preallocated gradient vector with marginal effects ∂η/∂x where η is the linear
predictor and x represents the continuous variables. Uses the mathematical relationship
g = J'β where J is the Jacobian matrix ∂X/∂x and β are the model coefficients.

# Arguments
- `g::AbstractVector{Float64}`: Preallocated gradient buffer of length `length(evaluator.vars)`
  - Will be overwritten with marginal effects ∂η/∂x
- `evaluator::DerivativeEvaluator`: Built by `build_derivative_evaluator(compiled, data; vars=...)`
- `beta::AbstractVector{<:Real}`: Model coefficients (typically from `coef(model)`)
- `row::Int`: Row index to evaluate (1-based indexing)
- `backend::Symbol`: Computational backend (`:ad` or `:fd`, default `:ad`)

# Returns
- `g`: The same vector passed in, now containing marginal effects for each variable

# Backend Selection
- **`:ad` (default)**: ForwardDiff automatic differentiation
  - Small allocations per call (ForwardDiff internals)
  - High numerical accuracy via dual numbers
  - Faster computation for complex formulas
- **`:fd`**: Finite differences
  - Zero bytes allocated after warmup
  - Good numerical accuracy with adaptive step sizes
  - Optimal for allocation-sensitive applications

# Mathematical Foundation
Computes the gradient of the linear predictor:
```
∂η/∂x = ∂(Xβ)/∂x = (∂X/∂x)'β = J'β
```
where J[i,j] = ∂X[i]/∂vars[j] is the model matrix Jacobian.

# Example
```julia
using FormulaCompiler, GLM

# Setup model with interactions
model = lm(@formula(y ~ x * group + log(abs(z) + 1)), df)
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Build evaluator and compute marginal effects
vars = [:x, :z]
de = build_derivative_evaluator(compiled, data; vars=vars)
β = coef(model)
g = Vector{Float64}(undef, length(vars))

# Automatic differentiation (accurate, small allocations)
marginal_effects_eta!(g, de, β, 1; backend=:ad)
println(\"∂η/∂x = \$(g[1]), ∂η/∂z = \$(g[2])\")

# Finite differences (zero allocations)
marginal_effects_eta!(g, de, β, 1; backend=:fd)
```

# Use Cases
- **Economic analysis**: Policy impact assessment via marginal effects
- **Sensitivity analysis**: Parameter robustness evaluation
- **Bootstrap inference**: Repeated marginal effects computation across samples
- **Delta method**: Standard error computation for marginal effects

See also: [`marginal_effects_mu!`](@ref) for effects on response scale, [`derivative_modelrow!`](@ref), [`build_derivative_evaluator`](@ref)
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

"""
    marginal_effects_eta_ad_pos!(g, de, beta, row) -> g

Positional, zero-allocation AD path for η-scale marginal effects. Avoids keyword
overhead; equivalent to `marginal_effects_eta!(g, de, beta, row; backend=:ad)`.
"""
@inline function marginal_effects_eta_ad_pos!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int
)
    @assert length(g) == length(de.vars)
    @assert length(beta) == length(de)
    derivative_modelrow!(de.jacobian_buffer, de, row)
    mul!(g, transpose(de.jacobian_buffer), beta)
    return g
end

"""
    marginal_effects_eta_fd_pos!(g, de, beta, row) -> g

Positional, zero-allocation FD path for η-scale marginal effects. Avoids keyword
overhead; equivalent to `marginal_effects_eta!(g, de, beta, row; backend=:fd)`.
"""
@inline function marginal_effects_eta_fd_pos!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int
)
    @assert length(g) == length(de.vars)
    @assert length(beta) == length(de)
    derivative_modelrow_fd_pos!(de.jacobian_buffer, de, row)
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
    marginal_effects_mu!(g, evaluator, beta, row; link=IdentityLink(), backend=:ad) -> g

Compute marginal effects on the response scale μ with respect to continuous variables.

Computes marginal effects ∂μ/∂x where μ is the expected response after applying the
inverse link function. Uses the chain rule: ∂μ/∂x = (∂μ/∂η)(∂η/∂x) where η = Xβ is
the linear predictor and μ = g⁻¹(η) with g being the link function.

# Arguments
- `g::AbstractVector{Float64}`: Preallocated gradient buffer of length `length(evaluator.vars)`
  - Will be overwritten with marginal effects ∂μ/∂x
- `evaluator::DerivativeEvaluator`: Built by `build_derivative_evaluator(compiled, data; vars=...)`
- `beta::AbstractVector{<:Real}`: Model coefficients (typically from `coef(model)`)
- `row::Int`: Row index to evaluate (1-based indexing)
- `link`: GLM link function (default `IdentityLink()`)
  - Supported: `IdentityLink`, `LogLink`, `LogitLink`, `ProbitLink`, `CloglogLink`, etc.
- `backend::Symbol`: Computational backend (`:ad` or `:fd`, default `:ad`)

# Returns
- `g`: The same vector passed in, now containing marginal effects ∂μ/∂x for each variable

# Backend Selection
- **`:ad` (default)**: ForwardDiff automatic differentiation
  - Small allocations per call (ForwardDiff internals)
  - High numerical accuracy via dual numbers
  - Faster computation for complex formulas
- **`:fd`**: Finite differences  
  - Zero bytes allocated after warmup
  - Good numerical accuracy with chain rule implementation
  - Optimal for allocation-sensitive applications

# Mathematical Foundation
Uses the chain rule for link function derivatives:
```
∂μ/∂x = (∂μ/∂η) × (∂η/∂x)
```
where:
- `∂η/∂x` is computed via Jacobian methods (AD or FD)
- `∂μ/∂η` is the derivative of the inverse link function at η = Xβ

# Link Function Support
```julia
# Identity: μ = η (linear models)
marginal_effects_mu!(g, de, β, row; link=IdentityLink())

# Logistic: μ = 1/(1+exp(-η)) (logistic regression)  
marginal_effects_mu!(g, de, β, row; link=LogitLink())

# Log: μ = exp(η) (Poisson regression)
marginal_effects_mu!(g, de, β, row; link=LogLink())

# Probit: μ = Φ(η) (probit regression)
marginal_effects_mu!(g, de, β, row; link=ProbitLink())
```

# Example
```julia
using FormulaCompiler, GLM

# Logistic regression model
df_binary = DataFrame(success = rand(Bool, 1000), x = randn(1000), z = randn(1000))
model = glm(@formula(success ~ x + log(abs(z) + 1)), df_binary, Binomial(), LogitLink())
data = Tables.columntable(df_binary)
compiled = compile_formula(model, data)

# Build evaluator
vars = [:x, :z]
de = build_derivative_evaluator(compiled, data; vars=vars)
β = coef(model)
g = Vector{Float64}(undef, length(vars))

# Marginal effects on probability scale
marginal_effects_mu!(g, de, β, 1; link=LogitLink(), backend=:ad)
println(\"∂P(success)/∂x = \$(g[1])\")
println(\"∂P(success)/∂z = \$(g[2])\")

# Compare with effects on logit scale  
marginal_effects_eta!(g, de, β, 1; backend=:ad)
println(\"∂logit(P)/∂x = \$(g[1])\")
```

# Use Cases
- **Economic interpretation**: Effects on meaningful outcome scales (probabilities, rates, etc.)
- **Policy analysis**: Impact assessment on interpretable response measures
- **Medical research**: Treatment effects on probability or survival scales
- **Comparative analysis**: Standardized effect sizes across different link functions

See also: [`marginal_effects_eta!`](@ref) for effects on linear predictor scale, [`derivative_modelrow!`](@ref)
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

"""
    marginal_effects_mu_ad_pos!(g, de, beta, row, link) -> g

Positional, zero-allocation AD path for μ-scale marginal effects. Avoids keyword
overhead; equivalent to `marginal_effects_mu!(g, de, beta, row; link=link, backend=:ad)`.
"""
@inline function marginal_effects_mu_ad_pos!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    link
)
    # Compute dη/dx using AD positional path
    marginal_effects_eta_ad_pos!(de.eta_gradient_buffer, de, beta, row)
    # Compute η at row using preallocated buffer
    de.compiled_base(de.xrow_buffer, de.base_data, row)
    η = dot(beta, de.xrow_buffer)
    scale = _dmu_deta(link, η)
    @inbounds @fastmath for j in eachindex(de.eta_gradient_buffer)
        g[j] = scale * de.eta_gradient_buffer[j]
    end
    return g
end

"""
    marginal_effects_mu_fd_pos!(g, de, beta, row, link) -> g

Positional, zero-allocation FD path for μ-scale marginal effects. Avoids keyword
overhead; equivalent to `marginal_effects_mu!(g, de, beta, row; link=link, backend=:fd)`.
"""
@inline function marginal_effects_mu_fd_pos!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    link
)
    # Compute dη/dx using FD positional path
    marginal_effects_eta_fd_pos!(de.eta_gradient_buffer, de, beta, row)
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
