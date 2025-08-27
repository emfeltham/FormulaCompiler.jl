# test_links.jl
# julia --project="." test/test_links.jl > test/test_links.txt 2>&1

using Test
using FormulaCompiler
using DataFrames, Tables, GLM, CategoricalArrays
using LinearAlgebra: dot

@testset "Link functions: marginal effects μ" begin
    n = 200
    df = DataFrame(y = randn(n), x = randn(n), z = abs.(randn(n)) .+ 0.1,
                   group3 = categorical(rand(["A","B","C"], n)))
    data = Tables.columntable(df)
    model = lm(@formula(y ~ 1 + x + z + x & group3), df)
    compiled = compile_formula(model, data)
    vars = [:x, :z]
    de = build_derivative_evaluator(compiled, data; vars=vars)
    β = coef(model)

    gη = Vector{Float64}(undef, length(vars))
    marginal_effects_eta!(gη, de, β, 3)

    links = Any[
        IdentityLink(), LogLink(), LogitLink(),
        ProbitLink(), CloglogLink(), CauchitLink(),
        InverseLink(), SqrtLink()
    ]

    for L in links
        gμ = Vector{Float64}(undef, length(vars))
        # Warm path and correctness (allocations are validated in test_allocations.jl)
        marginal_effects_mu!(gμ, de, β, 3; link=L)
        @test all(isfinite, gμ)
    end

    # Spot check scale agreement for a few links (Identity, Log, Logit)
    row = 5
    xrow = Vector{Float64}(undef, length(compiled))
    compiled(xrow, data, row)
    η = dot(β, xrow)
    # Recompute gη for this row
    gη = Vector{Float64}(undef, length(vars))
    marginal_effects_eta!(gη, de, β, row)

    # Identity: scale = 1
    gμ = Vector{Float64}(undef, length(vars))
    marginal_effects_mu!(gμ, de, β, row; link=IdentityLink())
    @test isapprox(gμ, gη; rtol=1e-8, atol=1e-10)

    # Log: scale = exp(η)
    marginal_effects_mu!(gμ, de, β, row; link=LogLink())
    @test isapprox(gμ, exp(η) .* gη; rtol=1e-8, atol=1e-10)

    # Logit: scale = σ(η)(1-σ(η))
    σ(x) = inv(1 + exp(-x))
    marginal_effects_mu!(gμ, de, β, row; link=LogitLink())
    @test isapprox(gμ, (σ(η)*(1-σ(η))) .* gη; rtol=1e-8, atol=1e-10)
end
