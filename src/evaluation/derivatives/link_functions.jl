# link_functions.jl - GLM link function derivatives for marginal effects

# Helper functions for link function derivatives
@inline _σ(x) = inv(1 + exp(-x))
const _INV_SQRT_2PI = 0.3989422804014327  # 1 / sqrt(2π)

"""
Link function derivative implementations: dμ/dη for various GLM link functions
"""

@inline function _dmu_deta(link::GLM.IdentityLink, η::Real)
    return 1.0
end

@inline function _dmu_deta(link::GLM.LogLink, η::Real)
    return exp(η)  # μ = exp(η)
end

@inline function _dmu_deta(link::GLM.LogitLink, η::Real)
    # σ'(η) = σ(η)(1 - σ(η)) = e/(1 + e)² with e = exp(-|η|) (symmetric in η).
    # Forming σ(1 - σ) directly underflows to exactly 0 once σ rounds to 1 (η ≳ 37).
    e = exp(-abs(η))
    return e / (1 + e)^2
end

# Additional GLM links
@inline function _dmu_deta(link::GLM.ProbitLink, η::Real)
    # μ = Φ(η); dμ/dη = φ(η)
    return _INV_SQRT_2PI * exp(-0.5 * η^2)
end

@inline function _dmu_deta(link::GLM.CloglogLink, η::Real)
    # μ = 1 - exp(-exp(η)); dμ/dη = exp(η) * exp(-exp(η))
    return exp(η - exp(η))
end

@inline function _dmu_deta(link::GLM.PowerLink, η::Real)
    # η = μ^λ (λ ≠ 0), μ = η^(1/λ); λ = 0 is the log link
    λ = link.λ
    λ == 0 && return exp(η)
    p = inv(λ)
    return p * η^(p - 1)
end

@inline function _dmu_deta(link::GLM.NegativeBinomialLink, η::Real)
    # η = log(μ / (μ + θ)) (η < 0), μ = θu/(1 - u) with u = exp(η)
    # dμ/dη = θu/(1 - u)²; 1 - u = -expm1(η) keeps precision as η → 0⁻.
    # (GLM.mueta for this link returns μ, not dμ/dη, so it is not used here.)
    u = exp(η)
    d = -expm1(η)
    return link.θ * u / d^2
end

@inline function _dmu_deta(link::GLM.CauchitLink, η::Real)
    # μ = (1/π) * atan(η) + 1/2; dμ/dη = 1 / (π * (1 + η^2))
    return inv(pi * (1 + η^2))
end

@inline function _dmu_deta(link::GLM.InverseLink, η::Real)
    # μ = 1/η; dμ/dη = -1 / η^2
    return -inv(η^2)
end

@inline function _dmu_deta(link::GLM.SqrtLink, η::Real)
    # μ = η^2; dμ/dη = 2η
    return 2 * η
end

# Some GLM variants include an inverse-square link (η = 1/μ^2)
if isdefined(GLM, :InverseSquareLink)
    @inline function _dmu_deta(link::GLM.InverseSquareLink, η::Real)
        # μ = η^(-1/2); dμ/dη = -(1/2) * η^(-3/2)
        # Domain: η > 0 (required for real-valued square roots)
        η > 0 || throw(DomainError(η, "InverseSquareLink requires η > 0"))
        return -0.5 * η^(-1.5)
    end
end

"""
Second derivatives of link functions: d²μ/dη² for variance calculations
"""

@inline function _d2mu_deta2(link::GLM.IdentityLink, η::Real)
    return 0.0  # Linear function has zero second derivative
end

@inline function _d2mu_deta2(link::GLM.LogLink, η::Real)
    return exp(η)  # d²/dη²[exp(η)] = exp(η)
end

@inline function _d2mu_deta2(link::GLM.LogitLink, η::Real)
    # μ = σ(η), d²μ/dη² = σ(1-σ)(1-2σ). With e = exp(-|η|):
    # σ(1-σ) = e/(1+e)² and 1 - 2σ = -sign(η)(1-e)/(1+e), so
    # d²μ/dη² = -sign(η) e(1-e)/(1+e)³, with 1 - e = -expm1(-|η|).
    # The direct form underflows to 0 for η ≳ 37 and loses precision near η = 0.
    a = abs(η)
    e = exp(-a)
    v = e * (-expm1(-a)) / (1 + e)^3
    return η > 0 ? -v : v
end

@inline function _d2mu_deta2(link::GLM.ProbitLink, η::Real)
    # μ = Φ(η), dμ/dη = φ(η), d²μ/dη² = -η * φ(η)
    return -η * _INV_SQRT_2PI * exp(-0.5 * η^2)
end

@inline function _d2mu_deta2(link::GLM.CloglogLink, η::Real)
    # μ = 1 - exp(-exp(η)), dμ/dη = exp(η - exp(η))
    # d²μ/dη² = exp(η - exp(η)) (1 - exp(η)) = -exp(η - exp(η)) expm1(η)
    if η <= 0
        return exp(η - exp(η)) * (-expm1(η))
    else
        # Work in log space: log(expm1(η)) = η + log(-expm1(-η)). The product form gives
        # 0 * -Inf = NaN once exp(η) overflows (η > 709) and loses precision before that.
        return -exp(2η - exp(η) + log(-expm1(-η)))
    end
end

@inline function _d2mu_deta2(link::GLM.CauchitLink, η::Real)
    # μ = (1/π) * atan(η) + 1/2, dμ/dη = 1 / (π * (1 + η²))
    # d²μ/dη² = -2η / (π * (1 + η²)²), divided in two steps so (1 + η²)² cannot overflow
    s = 1 + η^2
    return -2 * (η / s) / (pi * s)
end

@inline function _d2mu_deta2(link::GLM.InverseLink, η::Real)
    # μ = 1/η, dμ/dη = -1/η², d²μ/dη² = 2/η³
    return 2 * inv(η^3)
end

@inline function _d2mu_deta2(link::GLM.SqrtLink, η::Real)
    # μ = η², dμ/dη = 2η, d²μ/dη² = 2
    return 2.0
end

@inline function _d2mu_deta2(link::GLM.PowerLink, η::Real)
    # μ = η^(1/λ): d²μ/dη² = (1/λ)(1/λ - 1) η^(1/λ - 2); λ = 0 is the log link
    λ = link.λ
    λ == 0 && return exp(η)
    p = inv(λ)
    return p * (p - 1) * η^(p - 2)
end

@inline function _d2mu_deta2(link::GLM.NegativeBinomialLink, η::Real)
    # μ = θu/(1 - u), u = exp(η): d²μ/dη² = θu(1 + u)/(1 - u)³
    u = exp(η)
    d = -expm1(η)
    return link.θ * u * (1 + u) / d^3
end

if isdefined(GLM, :InverseSquareLink)
    @inline function _d2mu_deta2(link::GLM.InverseSquareLink, η::Real)
        # μ = η^(-1/2), dμ/dη = -(1/2) * η^(-3/2), d²μ/dη² = (3/4) * η^(-5/2)
        # Domain: η > 0 (required for real-valued square roots)
        η > 0 || throw(DomainError(η, "InverseSquareLink requires η > 0"))
        return 0.75 * η^(-2.5)
    end
end

# Generic fallback for other `GLM.Link` subtypes (including user-defined links): derive both
# derivatives from `GLM.linkinv` by forward-mode AD, which is exact up to floating-point
# evaluation of `linkinv` itself. `GLM.mueta` is not used because it is not guaranteed to
# be dμ/dη for every link (GLM's NegativeBinomialLink `mueta` returns μ).
@inline _dmu_deta(link::GLM.Link, η::Real) =
    ForwardDiff.derivative(t -> GLM.linkinv(link, t), η)

@inline _d2mu_deta2(link::GLM.Link, η::Real) =
    ForwardDiff.derivative(e -> ForwardDiff.derivative(t -> GLM.linkinv(link, t), e), η)

"""
    link_derivatives(link::GLM.Link, η::Real) -> (μ, dμ/dη, d²μ/dη²)

Inverse link and its first two derivatives with respect to the linear predictor, for every
`GLM.Link` (analytic for the standard links, forward-mode AD of `GLM.linkinv` otherwise).
This is the public interface to the link derivatives used for response-scale marginal
effects and their delta-method gradients; the underscore helpers `_dmu_deta` and
`_d2mu_deta2` remain exported for compatibility.

```julia
μ, dμ, d²μ = link_derivatives(LogitLink(), 0.3)
```
"""
@inline link_derivatives(link::GLM.Link, η::Real) =
    (GLM.linkinv(link, η), _dmu_deta(link, η), _d2mu_deta2(link, η))
