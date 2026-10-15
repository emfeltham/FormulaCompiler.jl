# test_link_derivatives.jl
# Regression tests for the link-function derivative helpers `_dmu_deta` and `_d2mu_deta2`.
#
# Before 1.4.3:
# - LogitLink: dμ/dη and d²μ/dη² were formed from σ(η)(1 - σ(η)), which is exactly 0 once
#   σ(η) rounds to 1 (η ≳ 37), although the true value is ≈ exp(-|η|) (4.2e-18 at η = 40).
# - CloglogLink: d²μ/dη² was 0 * -Inf = NaN for η > 709.
# - PowerLink and NegativeBinomialLink had no methods (MethodError).
#
# References are closed-form expressions evaluated in 256-bit BigFloat, compared by relative
# error (exact match required where the reference underflows to 0 in Float64).

using Test
using GLM, ForwardDiff
using FormulaCompiler
using FormulaCompiler: _dmu_deta, _d2mu_deta2

# Closed-form (d1, d2) for μ = linkinv(η), valid for BigFloat
# For large |η| evaluate the logistic forms symmetrically so BigFloat keeps the tiny terms
function ref_derivs_logit(η)
    a = abs(η); e = exp(-a)
    d1 = e / (1 + e)^2
    d2 = -sign(η) * e * (1 - e) / (1 + e)^3
    return (d1, d2)
end
ref_derivs(::ProbitLink, η) = (φ = exp(-η^2 / 2) / sqrt(2 * big(pi)); (φ, -η * φ))
ref_derivs(::CloglogLink, η) = (d1 = exp(η - exp(η)); (d1, d1 * (1 - exp(η))))
ref_derivs(::CauchitLink, η) = (1 / (big(pi) * (1 + η^2)), -2η / (big(pi) * (1 + η^2)^2))
ref_derivs(::LogLink, η) = (exp(η), exp(η))
ref_derivs(::IdentityLink, η) = (one(η), zero(η))
ref_derivs(::InverseLink, η) = (-1 / η^2, 2 / η^3)
ref_derivs(::SqrtLink, η) = (2η, big(2.0))
ref_derivs(::InverseSquareLink, η) = (-η^(-big(1.5)) / 2, 3 * η^(-big(2.5)) / 4)
function ref_derivs(l::PowerLink, η)
    l.λ == 0 && return (exp(η), exp(η))
    p = 1 / big(l.λ)
    return (p * η^(p - 1), p * (p - 1) * η^(p - 2))
end
ref_derivs(l::NegativeBinomialLink, η) =
    (u = exp(η); θ = big(l.θ); (θ * u / (1 - u)^2, θ * u * (1 + u) / (1 - u)^3))

function relerr(v::Float64, ref::BigFloat)
    r = Float64(ref)
    v == r && return 0.0
    isfinite(v) || return Inf
    return abs(v - ref) / max(abs(ref), big(floatmin(Float64)))
end

# A user-defined link (μ = η³) exercises the generic AD fallback
struct CubeLink <: GLM.Link end
GLM.linkinv(::CubeLink, η::Real) = η^3

@testset "Link derivative helpers (N1)" begin
    setprecision(BigFloat, 256) do
        ηs = (-800.0, -709.5, -100.0, -40.0, -37.0, -10.0, -1.0, -1e-3, -1e-8, 0.0,
              1e-8, 1e-3, 0.3, 1.0, 10.0, 37.0, 40.0, 100.0, 700.0, 709.5, 710.0, 800.0)
        links = (LogitLink(), ProbitLink(), CloglogLink(), CauchitLink(), LogLink(),
                 IdentityLink(), InverseLink(), SqrtLink(), InverseSquareLink(),
                 PowerLink(0.5), PowerLink(1 / 3), PowerLink(-1.0), PowerLink(2.0), PowerLink(0.0),
                 NegativeBinomialLink(2.0), NegativeBinomialLink(0.5))
        positive_only = Union{InverseLink, InverseSquareLink, SqrtLink}
        for L in links, η in ηs
            (L isa positive_only && η <= 0) && continue
            (L isa PowerLink && L.λ != 0 && η <= 0) && continue
            (L isa NegativeBinomialLink && η >= 0) && continue
            (L isa LogLink && η > 709) && continue        # exp overflows; Inf is correct
            (L isa PowerLink && L.λ == 0 && η > 709) && continue
            b = big(η)
            r1, r2 = L isa LogitLink ? ref_derivs_logit(b) : ref_derivs(L, b)
            v1 = _dmu_deta(L, η)
            v2 = _d2mu_deta2(L, η)
            ok1 = relerr(v1, r1) < 1e-13
            ok2 = relerr(v2, r2) < 1e-13
            ok1 || @info "d1 mismatch" L η v1 Float64(r1)
            ok2 || @info "d2 mismatch" L η v2 Float64(r2)
            @test ok1
            @test ok2
        end
    end

    # Spot values from the review
    @test _dmu_deta(LogitLink(), 40.0) ≈ GLM.mueta(LogitLink(), 40.0) rtol = 1e-14
    @test _dmu_deta(LogitLink(), 40.0) > 0
    @test _d2mu_deta2(LogitLink(), 40.0) < 0
    @test !isnan(_d2mu_deta2(CloglogLink(), 710.0))
    @test _d2mu_deta2(CloglogLink(), 800.0) == 0.0
    @test _d2mu_deta2(LogitLink(), 0.0) == 0.0

    # Agreement with GLM.mueta where GLM's form is exact (moderate η)
    for L in (LogitLink(), ProbitLink(), CloglogLink(), CauchitLink(), PowerLink(0.5)), η in (0.2, 1.3, 3.0)
        @test _dmu_deta(L, η) ≈ GLM.mueta(L, η) rtol = 1e-12
    end

    # Generic fallback: derivatives of linkinv by AD
    @test _dmu_deta(CubeLink(), 2.0) ≈ 12.0
    @test _d2mu_deta2(CubeLink(), 2.0) ≈ 12.0

    # Scalar helpers stay allocation-free
    f(L, η) = _dmu_deta(L, η) + _d2mu_deta2(L, η)
    for L in (LogitLink(), CloglogLink(), PowerLink(0.5), NegativeBinomialLink(2.0))
        η = L isa NegativeBinomialLink ? -0.5 : 0.5
        f(L, η)
        @test (@allocated f(L, η)) == 0
    end
end
