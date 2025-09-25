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
de = build_derivative_evaluator(compiled, data, vars)
β = coef(model)
g = Vector{Float64}(undef, length(vars))

# Automatic differentiation (accurate, small allocations)
marginal_effects_eta_ad!(g, de, β, 1)
println(\"∂η/∂x = \$(g[1]), ∂η/∂z = \$(g[2])\")

# Finite differences (zero allocations)
marginal_effects_eta_fd!(g, de, β, 1)
```

# Use Cases
- **Sensitivity analysis**: Parameter robustness evaluation
- **Bootstrap inference**: Repeated marginal effects computation across samples
- **Delta method**: Standard error computation for marginal effects

See also: [`marginal_effects_mu!`](@ref) for effects on response scale, [`derivative_modelrow!`](@ref), [`build_derivative_evaluator`](@ref)
"""

"""
    marginal_effects_eta_ad!(g, de, beta, row, var_indices) -> g

High-performance AD path for η-scale marginal effects with indexed variables.
Computes marginal effects ∂η/∂x for variables specified by var_indices.
"""
@inline function marginal_effects_eta_ad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    var_indices::AbstractVector{Int}
)
    @assert length(g) == length(var_indices)
    @assert length(beta) == length(de)
    derivative_modelrow!(de.jacobian_buffer, de, row)
    # Compute ∂η/∂x = (∂X/∂x)' × β for requested variables only
    @inbounds for (i, var_idx) in enumerate(var_indices)
        deta_dx = 0.0
        for p in 1:length(beta)
            deta_dx += de.jacobian_buffer[p, var_idx] * beta[p]
        end
        g[i] = deta_dx
    end
    return g
end

"""
    marginal_effects_eta_fd!(g, de, beta, row, var_indices) -> g

High-performance FD path for η-scale marginal effects with indexed variables.
Computes marginal effects ∂η/∂x for variables specified by var_indices.
"""
@inline function marginal_effects_eta_fd!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    var_indices::AbstractVector{Int}
)
    @assert length(g) == length(var_indices)
    @assert length(beta) == length(de)
    derivative_modelrow_fd_pos!(de.jacobian_buffer, de, row)
    # Extract only the requested variables using indices
    @inbounds for (i, var_idx) in enumerate(var_indices)
        g[i] = dot(view(de.jacobian_buffer, :, var_idx), beta)
    end
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
de = build_derivative_evaluator(compiled, data, vars)
β = coef(model)
g = Vector{Float64}(undef, length(vars))

# Marginal effects on probability scale
marginal_effects_mu!(g, de, β, 1; link=LogitLink(), backend=:ad)
println(\"∂P(success)/∂x = \$(g[1])\")
println(\"∂P(success)/∂z = \$(g[2])\")

# Compare with effects on logit scale  
marginal_effects_eta_ad!(g, de, β, 1)
println(\"∂logit(P)/∂x = \$(g[1])\")
```

# Use Cases
- **Interpretation**: Effects on meaningful outcome scales (probabilities, rates, etc.)
- **Policy analysis**: Impact assessment on interpretable response measures
- **Medical research**: Treatment effects on probability or survival scales
- **Comparative analysis**: Standardized effect sizes across different link functions

See also: [`marginal_effects_eta!`](@ref) for effects on linear predictor scale, [`derivative_modelrow!`](@ref)
"""

"""
    marginal_effects_mu_ad!(g, de, beta, row, var_indices, link) -> g

High-performance AD path for μ-scale marginal effects with indexed variables.
Computes marginal effects ∂μ/∂x for variables specified by var_indices.
"""
@inline function marginal_effects_mu_ad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    var_indices::AbstractVector{Int},
    link
)
    @assert length(g) == length(var_indices)
    # Compute Jacobian once (reads data and evaluates formula once)
    derivative_modelrow!(de.jacobian_buffer, de, row)

    # Extract η from dual evaluation results (no redundant computation)
    # The rowvec_dual_vec contains X_row as the value part of duals
    η = 0.0
    @inbounds for i in eachindex(de.rowvec_dual_vec)
        η += beta[i] * ForwardDiff.value(de.rowvec_dual_vec[i])
    end

    # Compute link function derivative once
    scale = _dmu_deta(link, η)

    # Apply chain rule: ∂μ/∂x = scale × (J'β) for requested variables only
    @inbounds for (i, var_idx) in enumerate(var_indices)
        deta_dx = 0.0
        for p in 1:length(beta)
            deta_dx += de.jacobian_buffer[p, var_idx] * beta[p]
        end
        g[i] = scale * deta_dx
    end
    return g
end

"""
    marginal_effects_mu_fd!(g, de, beta, row, var_indices, link) -> g

High-performance FD path for μ-scale marginal effects with indexed variables.
Computes marginal effects ∂μ/∂x for variables specified by var_indices.
"""
@inline function marginal_effects_mu_fd!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    var_indices::AbstractVector{Int},
    link
)
    @assert length(g) == length(var_indices)
    # Compute dη/dx using FD path for the specific variables
    marginal_effects_eta_fd!(de.eta_gradient_buffer, de, beta, row, var_indices)
    # Compute η at row using preallocated buffer
    de.compiled_base(de.xrow_buffer, de.base_data, row)
    η = dot(beta, de.xrow_buffer)
    scale = _dmu_deta(link, η)
    @inbounds @fastmath for j in eachindex(g)
        g[j] = scale * de.eta_gradient_buffer[j]
    end
    return g
end

# Symbol-based convenience wrappers that provide the public API
# These convert symbols to indices and call the indexed versions

# Helper function for symbol-to-index conversion
@inline function _symbols_to_indices!(var_indices::AbstractVector{Int}, evaluator_vars::Vector{Symbol}, requested_vars::AbstractVector{Symbol})
    @assert length(var_indices) == length(requested_vars)
    @inbounds for i in eachindex(requested_vars)
        var = requested_vars[i]
        idx = findfirst(==(var), evaluator_vars)
        idx === nothing && throw(ArgumentError("Variable $(var) not found in derivative evaluator"))
        var_indices[i] = idx
    end
    return var_indices
end

"""
    marginal_effects_eta_ad!(g, de, beta, row, vars) -> g
    marginal_effects_eta_ad!(g, de, beta, row) -> g

High-performance AD-based η-scale marginal effects computation.
Direct function call eliminates backend dispatch overhead.
"""
@inline function marginal_effects_eta_ad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    vars::AbstractVector{Symbol}
)
    @assert length(g) == length(vars)
    # Convert symbols to indices
    var_indices = Vector{Int}(undef, length(vars))
    _symbols_to_indices!(var_indices, de.vars, vars)
    # Direct call to indexed AD implementation
    return marginal_effects_eta_ad!(g, de, beta, row, var_indices)
end

# All variables convenience version
@inline function marginal_effects_eta_ad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int
)
    @assert length(g) == length(de.vars)
    var_indices = collect(1:length(de.vars))
    return marginal_effects_eta_ad!(g, de, beta, row, var_indices)
end

"""
    marginal_effects_eta_fd!(g, de, beta, row, vars) -> g
    marginal_effects_eta_fd!(g, de, beta, row) -> g

High-performance finite-difference η-scale marginal effects computation.
Zero allocations after warmup, direct function call.
"""
@inline function marginal_effects_eta_fd!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    vars::AbstractVector{Symbol}
)
    @assert length(g) == length(vars)
    # Convert symbols to indices
    var_indices = Vector{Int}(undef, length(vars))
    _symbols_to_indices!(var_indices, de.vars, vars)
    # Direct call to indexed FD implementation
    return marginal_effects_eta_fd!(g, de, beta, row, var_indices)
end

# All variables convenience version
@inline function marginal_effects_eta_fd!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int
)
    @assert length(g) == length(de.vars)
    var_indices = collect(1:length(de.vars))
    return marginal_effects_eta_fd!(g, de, beta, row, var_indices)
end

"""
    marginal_effects_mu_ad!(g, de, beta, row, vars, link) -> g
    marginal_effects_mu_ad!(g, de, beta, row, vars) -> g
    marginal_effects_mu_ad!(g, de, beta, row, link) -> g
    marginal_effects_mu_ad!(g, de, beta, row) -> g

High-performance AD-based μ-scale marginal effects computation.
Direct function call with concrete link type for optimal performance.
"""
# With explicit variables and link function
@inline function marginal_effects_mu_ad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    vars::AbstractVector{Symbol},
    link::GLM.Link
)
    @assert length(g) == length(vars)
    # Convert symbols to indices
    var_indices = Vector{Int}(undef, length(vars))
    _symbols_to_indices!(var_indices, de.vars, vars)
    # Direct call to indexed AD implementation
    return marginal_effects_mu_ad!(g, de, beta, row, var_indices, link)
end

# With explicit variables, identity link default
@inline function marginal_effects_mu_ad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    vars::AbstractVector{Symbol}
)
    return marginal_effects_mu_ad!(g, de, beta, row, vars, GLM.IdentityLink())
end

# All variables with explicit link function
@inline function marginal_effects_mu_ad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    link::GLM.Link
)
    @assert length(g) == length(de.vars)
    var_indices = collect(1:length(de.vars))
    return marginal_effects_mu_ad!(g, de, beta, row, var_indices, link)
end

# All variables, identity link default
@inline function marginal_effects_mu_ad!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int
)
    return marginal_effects_mu_ad!(g, de, beta, row, GLM.IdentityLink())
end

"""
    marginal_effects_mu_fd!(g, de, beta, row, vars, link) -> g
    marginal_effects_mu_fd!(g, de, beta, row, vars) -> g
    marginal_effects_mu_fd!(g, de, beta, row, link) -> g
    marginal_effects_mu_fd!(g, de, beta, row) -> g

High-performance finite-difference μ-scale marginal effects computation.
Zero allocations, direct function call with concrete link type.
"""
# With explicit variables and link function
@inline function marginal_effects_mu_fd!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    vars::AbstractVector{Symbol},
    link::GLM.Link
)
    @assert length(g) == length(vars)
    # Convert symbols to indices
    var_indices = Vector{Int}(undef, length(vars))
    _symbols_to_indices!(var_indices, de.vars, vars)
    # Direct call to indexed FD implementation
    return marginal_effects_mu_fd!(g, de, beta, row, var_indices, link)
end

# With explicit variables, identity link default
@inline function marginal_effects_mu_fd!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    vars::AbstractVector{Symbol}
)
    return marginal_effects_mu_fd!(g, de, beta, row, vars, GLM.IdentityLink())
end

# All variables with explicit link function
@inline function marginal_effects_mu_fd!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int,
    link::GLM.Link
)
    @assert length(g) == length(de.vars)
    var_indices = collect(1:length(de.vars))
    return marginal_effects_mu_fd!(g, de, beta, row, var_indices, link)
end

# All variables, identity link default
@inline function marginal_effects_mu_fd!(
    g::AbstractVector{Float64},
    de::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int
)
    return marginal_effects_mu_fd!(g, de, beta, row, GLM.IdentityLink())
end

# Parameter gradient function (uses variable index for zero-allocation performance)
function me_mu_grad_beta!(
    gβ::Vector{Float64},
    de::DerivativeEvaluator,
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
    fd_jacobian_column!(de.fd_yplus, de, row, var_idx)  # Now fd_yplus contains J_k

    # Step 4: Compute J_k' * β (scalar)
    Jk_dot_beta = dot(de.fd_yplus, β)

    # Step 5: Apply chain rule formula: gβ = g'(η) * J_k + (J_k' * β) * g''(η) * X_row
    # xrow_buffer contains X_row, fd_yplus contains J_k
    @inbounds @fastmath for i in eachindex(gβ)
        gβ[i] = g_prime * de.fd_yplus[i] + Jk_dot_beta * g_double_prime * de.xrow_buffer[i]
    end

    return gβ
end


# Generic dispatcher functions for backend selection

"""
    marginal_effects_eta!(g, evaluator, beta, row; backend=:ad) -> g

Generic dispatcher for η-scale marginal effects with backend selection.
Dispatches to `marginal_effects_eta_ad!` or `marginal_effects_eta_fd!` based on backend.
"""
@inline function marginal_effects_eta!(
    g::AbstractVector{Float64},
    evaluator::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    backend::Symbol=:ad
)
    if backend === :ad
        return marginal_effects_eta_ad!(g, evaluator, beta, row)
    elseif backend === :fd
        return marginal_effects_eta_fd!(g, evaluator, beta, row)
    else
        throw(ArgumentError("Unsupported backend: $backend. Use :ad or :fd"))
    end
end

"""
    marginal_effects_mu!(g, evaluator, beta, row; link=IdentityLink(), backend=:ad) -> g

Generic dispatcher for μ-scale marginal effects with backend selection.
Dispatches to `marginal_effects_mu_ad!` or `marginal_effects_mu_fd!` based on backend.
"""
@inline function marginal_effects_mu!(
    g::AbstractVector{Float64},
    evaluator::DerivativeEvaluator,
    beta::AbstractVector{<:Real},
    row::Int;
    link=GLM.IdentityLink(),
    backend::Symbol=:ad
)
    if backend === :ad
        return marginal_effects_mu_ad!(g, evaluator, beta, row, link)
    elseif backend === :fd
        return marginal_effects_mu_fd!(g, evaluator, beta, row, link)
    else
        throw(ArgumentError("Unsupported backend: $backend. Use :ad or :fd"))
    end
end
