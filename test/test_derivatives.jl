# test_derivatives.jl
# julia --project="." test/test_derivatives.jl > test/test_derivatives.txt 2>&1
# Correctness tests for derivatives

using Test
using FormulaCompiler
using DataFrames, Tables, GLM, MixedModels, CategoricalArrays


@testset "Derivatives: ForwardDiff and FD fallback" begin
    # Data and model
    n = 300
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group3 = categorical(rand(["A", "B", "C"], n)),
    )
    data = Tables.columntable(df)
    model = lm(@formula(y ~ 1 + x + z + x & group3), df)
    compiled = compile_formula(model, data)

    # Choose continuous vars
    vars = [:x, :z]

    # Build derivative evaluator (ForwardDiff)
    de = build_derivative_evaluator(compiled, data; vars=vars)
    J = Matrix{Float64}(undef, length(compiled), length(vars))

    # Warmup to trigger Dual-typed caches
    derivative_modelrow!(J, de, 1)
    derivative_modelrow!(J, de, 2)

    # FD fallback comparison (standalone FD for correctness baseline)
    J_fd = similar(J)
    derivative_modelrow_fd!(J_fd, compiled, data, 3; vars=vars)
    @test isapprox(J, J_fd; rtol=1e-6, atol=1e-8)

    # Discrete contrast: swap group level at row
    Δ = Vector{Float64}(undef, length(compiled))
    row = 5
    contrast_modelrow!(Δ, compiled, data, row; var=:group3, from="A", to="B")
    # Validate against manual override with OverrideVector
    # Pass raw override values; create_override_data will wrap appropriately
    data_from = FormulaCompiler.create_override_data(data, Dict{Symbol,Any}(:group3 => "A"))
    data_to   = FormulaCompiler.create_override_data(data, Dict{Symbol,Any}(:group3 => "B"))
    y_from = modelrow(compiled, data_from, row)
    y_to = modelrow(compiled, data_to, row)
    @test isapprox(Δ, y_to .- y_from; rtol=0, atol=0)

    # Marginal effects: η = Xβ
    β = coef(model)
    gη = Vector{Float64}(undef, length(vars))
    marginal_effects_eta!(gη, de, β, row)
    # Check consistency with J' * β
    Jrow = Matrix{Float64}(undef, length(compiled), length(vars))
    derivative_modelrow!(Jrow, de, row)
    gη2 = transpose(Jrow) * β
    @test isapprox(gη, gη2; rtol=0, atol=0)
end

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
    # Note: AD Jacobian allocation caps are environment-dependent (see DERIVATIVE_PLAN.md).
    de_glm = build_derivative_evaluator(compiled_glm, data; vars=vars)
    J = Matrix{Float64}(undef, length(compiled_glm), length(vars))
    derivative_modelrow!(J, de_glm, 3)  # warm path
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
    Jmm_fd = similar(Jmm)
    derivative_modelrow_fd!(Jmm_fd, compiled_mm, data, 3; vars=vars)
    @test isapprox(Jmm, Jmm_fd; rtol=1e-6, atol=1e-8)
end
