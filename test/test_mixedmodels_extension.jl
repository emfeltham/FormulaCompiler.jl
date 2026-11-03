# test_mixedmodels_extension.jl
# FC 1.5.0: MixedModels is a weak dependency. Its support (fixed_effects_form methods and
# skipping random-effects terms) lives in ext/MixedModelsExt.jl. These tests check that
# (1) FormulaCompiler loads and works for GLM without loading MixedModels (in a fresh
# process), and (2) with MixedModels loaded, the extension is active and compilation,
# derivatives and contrasts work for LMM/GLMM, including zerocorr and nested terms.

using Test, Random
using DataFrames, Tables, GLM, CategoricalArrays, StatsModels
using MixedModels
using Distributions: Bernoulli
using FormulaCompiler

const _MM_PKGID = Base.PkgId(Base.UUID("ff71e718-51f3-5ec2-a782-8ffcbfa3c316"), "MixedModels")

@testset "MixedModels package extension" begin
    @testset "FormulaCompiler without MixedModels (fresh process)" begin
        script = """
        using FormulaCompiler, GLM, DataFrames, Tables, CategoricalArrays
        mm = Base.PkgId(Base.UUID("ff71e718-51f3-5ec2-a782-8ffcbfa3c316"), "MixedModels")
        haskey(Base.loaded_modules, mm) && error("MixedModels was loaded by FormulaCompiler")
        Base.get_extension(FormulaCompiler, :MixedModelsExt) === nothing || error("extension loaded")
        df = DataFrame(y = rand(50) .< 0.5, x = randn(50), z = rand(50) .+ 1,
                       g = categorical(rand(["a", "b", "c"], 50)))
        m = glm(@formula(y ~ x * g + log(z)), df, Binomial(), LogitLink())
        d = Tables.columntable(df)
        c = compile_formula(m, d)
        all(modelrow(c, d, i) ≈ modelmatrix(m)[i, :] for i in 1:50) || error("modelrow")
        FormulaCompiler.fixed_effects_form(m) == formula(m) || error("fixed_effects_form")
        J1 = derivative_modelrow(derivativeevaluator(:ad, c, d, [:x, :z]), 3)
        J2 = derivative_modelrow(derivativeevaluator(:fd, c, d, [:x, :z]), 3)
        isapprox(J1, J2; rtol = 1e-6) || error("derivatives")
        ce = contrastevaluator(c, d, [:g])
        Δ = zeros(length(c)); contrast_modelrow!(Δ, ce, 1, :g, "a", "c")
        x1 = modelrow(c, merge(d, (g = categorical(fill("c", 50); levels = levels(d.g)),)), 1)
        x0 = modelrow(c, merge(d, (g = categorical(fill("a", 50); levels = levels(d.g)),)), 1)
        Δ ≈ x1 .- x0 || error("contrast")
        haskey(Base.loaded_modules, mm) && error("MixedModels loaded during use")
        println("NO_MIXEDMODELS_OK")
        """
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
        out = read(cmd, String)
        @test occursin("NO_MIXEDMODELS_OK", out)
    end

    @testset "extension active with MixedModels loaded" begin
        ext = Base.get_extension(FormulaCompiler, :MixedModelsExt)
        @test ext !== nothing
        m_lmm = which(FormulaCompiler.fixed_effects_form, (LinearMixedModel{Float64},))
        @test m_lmm.module === ext
        m_glmm = which(FormulaCompiler.fixed_effects_form, (GeneralizedLinearMixedModel{Float64, Bernoulli{Float64}},))
        @test m_glmm.module === ext
        # generic fallback stays in FormulaCompiler for models without random effects
        @test which(FormulaCompiler.fixed_effects_form, (Any,)).module === FormulaCompiler
    end

    @testset "LMM/GLMM: rows, derivatives and contrasts (zerocorr, nested)" begin
        rng = MersenneTwister(2026)
        n = 300
        d = DataFrame(x = randn(rng, n), z = rand(rng, n) .+ 0.5,
                      g = categorical(rand(rng, ["a", "b", "c"], n)),
                      s = categorical(rand(rng, 1:20, n)), t = categorical(rand(rng, 1:5, n)))
        d.y = d.x .+ 0.5 .* d.z .+ randn(rng, n)
        d.yb = rand(rng, n) .< 0.5
        data = Tables.columntable(d)
        models = [
            fit(MixedModel, @formula(y ~ 1 + x * g + log(z) + (1 + x | s)), d; progress = false),
            fit(MixedModel, @formula(y ~ 0 + x + g + log(z) + zerocorr(1 + x | s) + (1 | t)), d; progress = false),
            fit(MixedModel, @formula(y ~ x * g + log(z) + (1 | s / t)), d; progress = false),
            fit(MixedModel, @formula(yb ~ x + g + log(z) + (1 | s)), d, Bernoulli(); progress = false),
        ]
        for m in models
            fe = FormulaCompiler.fixed_effects_form(m)
            @test !any(t -> t isa MixedModels.AbstractReTerm, fe.rhs isa Tuple ? fe.rhs : (fe.rhs,))
            c = compile_formula(m, data)
            @test length(c) == size(m.X, 2)
            @test maximum(maximum(abs.(modelrow(c, data, i) .- m.X[i, :])) for i in 1:n) < 1e-12
            # cached path (modelrow with a model) agrees too
            @test modelrow(m, data, 9) ≈ m.X[9, :]

            de_ad = derivativeevaluator(:ad, c, data, [:x, :z])
            de_fd = derivativeevaluator(:fd, c, data, [:x, :z])
            J = Matrix{Float64}(undef, length(c), 2)
            for row in (1, 50, n)
                derivative_modelrow!(J, de_ad, row)
                @test J ≈ derivative_modelrow(de_fd, row) rtol = 1e-6 atol = 1e-8
            end
            derivative_modelrow!(J, de_ad, 2)
            @test (@allocated derivative_modelrow!(J, de_ad, 3)) == 0

            ce = contrastevaluator(c, data, [:g])
            Δ = zeros(length(c))
            contrast_modelrow!(Δ, ce, 4, :g, "a", "b")
            lv = levels(d.g)
            xb = modelrow(c, merge(data, (g = categorical(fill("b", n); levels = lv),)), 4)
            xa = modelrow(c, merge(data, (g = categorical(fill("a", n); levels = lv),)), 4)
            @test Δ ≈ xb .- xa
        end
    end
end
