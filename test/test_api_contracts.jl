# test_api_contracts.jl
# Public API contracts.
#
# Every exported name must be defined. Twelve names (marginal_effects_eta!, delta_method_se,
# ...) stayed in the export list after their implementations moved to Margins.jl in v2.0,
# so `using FormulaCompiler` advertised functions that did not exist.

using Test
using DataFrames, Tables, GLM, StatsModels
using FormulaCompiler

@testset "API contracts" begin
    undefined = [n for n in names(FormulaCompiler) if !isdefined(FormulaCompiler, n)]
    @test isempty(undefined)

    # derivative_modelrow: allocating counterpart of derivative_modelrow!
    df = DataFrame(y = randn(30), x = randn(30), z = rand(30) .+ 0.5)
    model = lm(@formula(y ~ x * z + log(z)), df)
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    for backend in (:ad, :fd)
        de = derivativeevaluator(backend, compiled, data, [:x, :z])
        J = Matrix{Float64}(undef, length(compiled), 2)
        derivative_modelrow!(J, de, 4)
        @test derivative_modelrow(de, 4) == J
    end

    # Compiled formulas must reject a too-short output buffer. They used to skip the
    # out-of-range writes silently, leaving stale values in the caller's buffer.
    out_short = zeros(length(compiled) - 1)
    @test_throws DimensionMismatch compiled(out_short, data, 1)
    out = zeros(length(compiled))
    compiled(out, data, 3)
    @test out ≈ modelmatrix(model)[3, :]
    out_long = fill(NaN, length(compiled) + 2)    # longer buffers remain allowed
    compiled(out_long, data, 3)
    @test out_long[1:length(compiled)] ≈ modelmatrix(model)[3, :]

    # Mixture validators: public, used by the CategoricalMixture constructor (and re-exported
    # by Margins, which previously kept its own copies)
    @test validate_mixture_weights([0.25, 0.75]) === nothing
    @test_throws ArgumentError validate_mixture_weights([0.3, 0.8])
    @test_throws ArgumentError validate_mixture_weights([-0.2, 1.2])
    @test validate_mixture_levels(["a", "b"]) === nothing
    @test_throws ArgumentError validate_mixture_levels(String[])
    @test_throws ArgumentError validate_mixture_levels(["a", "a"])
    @test_throws ArgumentError mix("a" => 0.5, "a" => 0.5)
    @test_throws ArgumentError mix("a" => 0.5, "b" => 0.6)

    # derivative_modelrow!(J, x, de, row): Jacobian plus the model row, for both backends and
    # for AD chunk mode (> 12 variables), where the row comes from the last chunk's duals
    function jx_allocs(J, x, de, r)
        derivative_modelrow!(J, x, de, r)
        return @allocated derivative_modelrow!(J, x, de, r)
    end
    dfw = DataFrame([Symbol("v$i") => randn(40) for i in 1:14]); dfw.y = randn(40)
    fw = term(:y) ~ sum(term(Symbol("v$i")) for i in 1:14) + term(:v1) & term(:v2)
    dw = Tables.columntable(dfw); cw = compile_formula(lm(fw, dfw), dw)
    for (cc, dd, vs) in ((compiled, data, [:x, :z]), (cw, dw, [Symbol("v$i") for i in 1:14]))
        for backend in (:ad, :fd)
            de = derivativeevaluator(backend, cc, dd, vs)
            J = Matrix{Float64}(undef, length(cc), length(vs)); J0 = similar(J)
            x = Vector{Float64}(undef, length(cc))
            derivative_modelrow!(J0, de, 5)
            derivative_modelrow!(J, x, de, 5)
            @test J == J0
            @test x ≈ modelrow(cc, dd, 5)
            @test jx_allocs(J, x, de, 6) == 0
            @test_throws DimensionMismatch derivative_modelrow!(J, zeros(length(cc) - 1), de, 5)
        end
    end

    # link_derivatives: (μ, dμ/dη, d²μ/dη²) consistent with the individual primitives
    for link in (IdentityLink(), LogLink(), LogitLink(), ProbitLink(), CloglogLink(),
                 InverseLink(), NegativeBinomialLink(2.0)), η in (-1.3, -0.2)
        @test link_derivatives(link, η) ==
              (GLM.linkinv(link, η), FormulaCompiler._dmu_deta(link, η), FormulaCompiler._d2mu_deta2(link, η))
    end
end
