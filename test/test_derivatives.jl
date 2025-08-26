# test_derivatives.jl
# julia --project="." test/test_derivatives.jl > test/test_derivatives.txt 2>&1

using Test
using FormulaCompiler
using DataFrames, Tables, GLM, CategoricalArrays
using BenchmarkTools

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

    # Strict allocation check after warmup - only ForwardDiff internals should allocate
    allocs = @allocated derivative_modelrow!(J, de, 3)
    @test allocs <= 112  # Tightened from 256 to reflect ForwardDiff internal minimum

    # FD fallback comparison
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
