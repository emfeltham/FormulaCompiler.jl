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
        return -0.5 * η^(-1.5)
    end
end