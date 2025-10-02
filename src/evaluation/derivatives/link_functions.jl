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
    μ = _σ(η)
    return μ * (1 - μ)  # σ'(η)
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
    # μ = σ(η), dμ/dη = σ(η)(1-σ(η)), d²μ/dη² = σ(η)(1-σ(η))(1-2σ(η))
    μ = _σ(η)
    return μ * (1 - μ) * (1 - 2*μ)
end

@inline function _d2mu_deta2(link::GLM.ProbitLink, η::Real)
    # μ = Φ(η), dμ/dη = φ(η), d²μ/dη² = -η * φ(η)
    return -η * _INV_SQRT_2PI * exp(-0.5 * η^2)
end

@inline function _d2mu_deta2(link::GLM.CloglogLink, η::Real)
    # μ = 1 - exp(-exp(η)), dμ/dη = exp(η) * exp(-exp(η))
    # d²μ/dη² = exp(η) * exp(-exp(η)) * (1 - exp(η))
    exp_eta = exp(η)
    exp_neg_exp_eta = exp(-exp_eta)
    return exp_eta * exp_neg_exp_eta * (1 - exp_eta)
end

@inline function _d2mu_deta2(link::GLM.CauchitLink, η::Real)
    # μ = (1/π) * atan(η) + 1/2, dμ/dη = 1 / (π * (1 + η²))
    # d²μ/dη² = -2η / (π * (1 + η²)²)
    denom = pi * (1 + η^2)
    return -2*η / (denom * (1 + η^2))
end

@inline function _d2mu_deta2(link::GLM.InverseLink, η::Real)
    # μ = 1/η, dμ/dη = -1/η², d²μ/dη² = 2/η³
    return 2 * inv(η^3)
end

@inline function _d2mu_deta2(link::GLM.SqrtLink, η::Real)
    # μ = η², dμ/dη = 2η, d²μ/dη² = 2
    return 2.0
end

if isdefined(GLM, :InverseSquareLink)
    @inline function _d2mu_deta2(link::GLM.InverseSquareLink, η::Real)
        # μ = η^(-1/2), dμ/dη = -(1/2) * η^(-3/2), d²μ/dη² = (3/4) * η^(-5/2)
        # Domain: η > 0 (required for real-valued square roots)
        η > 0 || throw(DomainError(η, "InverseSquareLink requires η > 0"))
        return 0.75 * η^(-2.5)
    end
end