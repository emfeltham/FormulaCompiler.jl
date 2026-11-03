# test_public_api.jl
# The public API added in 1.5.0 so that Margins.jl uses no FormulaCompiler internals:
# fixed_effects_form (model-class hook), with_private_scratch, extract_variable_names,
# AbstractDerivativeEvaluator, derivative_variables, derivative_backend, contrast_endpoints
# and baseline_level.

using Test
using DataFrames, Tables, GLM, StatsModels, CategoricalArrays, Random
using LinearAlgebra: dot
using FormulaCompiler

# A model type whose formula is not the formula of its model matrix: the hook must be used
struct _HookModel{M}
    inner::M
end
FormulaCompiler.fixed_effects_form(m::_HookModel) = StatsModels.formula(m.inner)

@testset "Public API (1.5.0)" begin
    Random.seed!(20261019)
    n = 60
    df = DataFrame(y = randn(n), x = randn(n), z = rand(n) .+ 0.5,
                   g = categorical(rand(["A", "B", "C"], n)), s = rand(["p", "q", "r"], n),
                   b = rand(Bool, n))
    data = Tables.columntable(df)
    model = lm(@formula(y ~ x * g + log(z) + s + b), df)
    compiled = compile_formula(model, data)
    X = modelmatrix(model)

    @testset "exported" begin
        for name in (:fixed_effects_form, :with_private_scratch, :extract_variable_names,
                     :AbstractDerivativeEvaluator, :derivative_variables, :derivative_backend,
                     :contrast_endpoints, :baseline_level)
            @test Base.isexported(FormulaCompiler, name)
        end
    end

    @testset "fixed_effects_form: model-class hook" begin
        @test fixed_effects_form(model) == StatsModels.formula(model)
        # A method added by another package is what compile_formula compiles
        hm = _HookModel(model)
        ch = compile_formula(hm, data)
        out = Vector{Float64}(undef, length(ch))
        ch(out, data, 5)
        @test out == X[5, :]
    end

    @testset "with_private_scratch" begin
        c2 = with_private_scratch(compiled)
        @test typeof(c2) === typeof(compiled)
        @test c2.ops === compiled.ops                 # operations shared
        @test c2.scratch !== compiled.scratch         # scratch private
        o1 = Vector{Float64}(undef, length(compiled)); o2 = similar(o1)
        for i in (1, 17, n)
            compiled(o1, data, i); c2(o2, data, i)
            @test o1 == o2 == X[i, :]
        end
        # Concurrent evaluation of private copies of one compiled object
        rows = repeat(1:n, 50)
        ok = Threads.Atomic{Bool}(true)
        Threads.@threads for t in 1:4
            c = with_private_scratch(compiled)
            out = Vector{Float64}(undef, length(c))
            for i in rows
                c(out, data, i)
                out == view(X, i, :) || (ok[] = false)
            end
        end
        @test ok[]
    end

    @testset "extract_variable_names" begin
        @test extract_variable_names(compiled) == Set([:x, :g, :z, :s, :b])
        @test :y ∉ extract_variable_names(compiled)
    end

    @testset "derivative evaluator accessors" begin
        for backend in (:ad, :fd)
            de = derivativeevaluator(backend, compiled, data, [:z, :x])
            @test de isa AbstractDerivativeEvaluator
            @test derivative_backend(de) === backend
            @test derivative_variables(de) == [:z, :x]          # Jacobian column order
            @test length(de) == length(compiled)
            # Column j of the Jacobian is the derivative with respect to variable j
            J = derivative_modelrow(de, 3)
            @test size(J) == (length(de), length(derivative_variables(de)))
            de_x = derivativeevaluator(backend, compiled, data, [:x])
            @test J[:, 2] ≈ derivative_modelrow(de_x, 3)[:, 1]
            # derivativeevaluator is a function, not a type (it was a Union type before 1.5)
            @test derivativeevaluator isa Function
            @test !(derivativeevaluator isa Type)
            @test de isa AbstractDerivativeEvaluator
            # The backend round-trips through derivativeevaluator
            @test derivative_backend(derivativeevaluator(derivative_backend(de), compiled, data,
                                                         derivative_variables(de))) === backend
        end
    end

    @testset "contrast_endpoints" begin
        ce = contrastevaluator(compiled, data, [:g, :s])
        Δ = Vector{Float64}(undef, length(compiled))
        contrast_modelrow!(Δ, ce, 4, :g, "A", "C")
        x₀, x₁ = contrast_endpoints(ce)
        @test length(x₀) == length(x₁) == length(ce)
        @test Δ == x₁ .- x₀
        r₀ = similar(Δ); r₁ = similar(Δ)
        level_modelrow!(r₀, ce, 4, :g, "A"); level_modelrow!(r₁, ce, 4, :g, "C")
        @test x₀ == r₀ && x₁ == r₁
        # Same buffers on every call (no copy), overwritten by the next contrast
        @test contrast_endpoints(ce)[1] === x₀
        contrast_modelrow!(Δ, ce, 9, :s, "p", "r")
        level_modelrow!(r₁, ce, 9, :s, "r")
        @test x₁ == r₁
        # Usable as level_modelrow! output (the documented scratch use)
        level_modelrow!(x₀, ce, 2, :g, "B")
        level_modelrow!(x₁, ce, 2, :g, "C")
        contrast_modelrow!(Δ, ce, 2, :g, "B", "C")
        @test Δ == x₁ .- x₀
        function endpoints_allocs(ce)
            contrast_endpoints(ce)
            return @allocated contrast_endpoints(ce)
        end
        @test endpoints_allocs(ce) == 0
    end

    @testset "baseline_level" begin
        @test baseline_level(model, :g) == "A"
        @test baseline_level(model, :s) == "p"
        m2 = lm(@formula(y ~ x + g + s), df;
                contrasts = Dict(:g => DummyCoding(base = "B"), :s => EffectsCoding(base = "r")))
        @test baseline_level(m2, :g) == "B"
        @test baseline_level(m2, :s) == "r"
        c2 = compile_formula(m2, data)
        @test categorical_levels(c2)[:g][1] == "A"      # not the baseline
        m3 = lm(@formula(y ~ x + g), df; contrasts = Dict(:g => DummyCoding(levels = ["C", "A", "B"])))
        @test baseline_level(m3, :g) == "C"
        @test_throws ArgumentError baseline_level(model, :x)           # not categorical
        @test_throws ArgumentError baseline_level(lm(@formula(y ~ x + x & g), df), :g)  # interaction only
        @test FormulaCompiler._get_baseline_level(m2, :g) == "B"       # former name kept
    end
end
