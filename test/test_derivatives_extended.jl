using Test
using FormulaCompiler
using DataFrames, Tables, GLM, MixedModels, CategoricalArrays

@testset "Derivatives Extended: GLM(Logit) and MixedModels" begin
    # Data
    n = 300
    df = DataFrame(
        y = rand([0, 1], n),
        x = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group3 = categorical(rand(["A", "B", "C"], n)),
        g = categorical(rand(1:20, n)),
    )
    data = Tables.columntable(df)

    # GLM (Logit)
    glm_model = glm(@formula(y ~ 1 + x + z + x & group3), df, Binomial(), LogitLink())
    compiled_glm = compile_formula(glm_model, data)
    vars = [:x, :z]
    de_glm = build_derivative_evaluator(compiled_glm, data; vars=vars)
    J = Matrix{Float64}(undef, length(compiled_glm), length(vars))
    derivative_modelrow!(J, de_glm, 3)  # warm path
    allocs = @allocated derivative_modelrow!(J, de_glm, 4)
    @test allocs == 0
    # FD compare
    J_fd = similar(J)
    derivative_modelrow_fd!(J_fd, compiled_glm, data, 4; vars=vars)
    @test isapprox(J, J_fd; rtol=1e-6, atol=1e-8)

    # MixedModels (fixed effects only)
    mm = fit(MixedModel, @formula(y ~ 1 + x + z + (1|g)), df; progress=false)
    compiled_mm = compile_formula(mm, data)
    de_mm = build_derivative_evaluator(compiled_mm, data; vars=vars)
    Jmm = Matrix{Float64}(undef, length(compiled_mm), length(vars))
    derivative_modelrow!(Jmm, de_mm, 2)
    allocs2 = @allocated derivative_modelrow!(Jmm, de_mm, 3)
    @test allocs2 == 0
    Jmm_fd = similar(Jmm)
    derivative_modelrow_fd!(Jmm_fd, compiled_mm, data, 3; vars=vars)
    @test isapprox(Jmm, Jmm_fd; rtol=1e-6, atol=1e-8)
end
