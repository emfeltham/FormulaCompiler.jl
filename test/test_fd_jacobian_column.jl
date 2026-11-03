# test_fd_jacobian_column.jl
# fd_jacobian_column! (one column of the FD Jacobian) and FD accuracy at extreme variable scales.
# Moved from Margins' test/primitives/ (test_derivatives.jl and
# test_derivatives_log_profile_regression.jl): these test FormulaCompiler functions only.

using Test, Random
using DataFrames, Tables, GLM, CategoricalArrays
using FormulaCompiler

@testset "fd_jacobian_column!" begin
    Random.seed!(06515)

    @testset "equals the Jacobian's column" begin
        n = 200
        df = DataFrame(y = randn(n), x = randn(n), z = abs.(randn(n)) .+ 0.1,
                       group3 = categorical(rand(["A", "B", "C"], n)))
        data = Tables.columntable(df)
        compiled = compile_formula(lm(@formula(y ~ 1 + x + z + x & group3), df), data)
        vars = [:x, :z]
        de_ad = derivativeevaluator(:ad, compiled, data, vars)
        de_fd = derivativeevaluator(:fd, compiled, data, vars)
        J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
        J_fd = similar(J_ad)
        derivative_modelrow!(J_ad, de_ad, 5)
        derivative_modelrow!(J_fd, de_fd, 5)
        for i in eachindex(vars)
            Jk = Vector{Float64}(undef, length(compiled))
            fd_jacobian_column!(Jk, de_fd, 5, i)
            @test isapprox(Jk, J_ad[:, i]; rtol = 1e-6, atol = 1e-8)
            @test Jk == J_fd[:, i]
        end
    end

    # Regression: row 1 of a log(x) model (an earlier row-1 initialization bug)
    @testset "log(x) at row 1 equals [0, 1/x]" begin
        n = 500
        df = DataFrame(x = rand(n) .+ 10.0)
        df.y = 0.5 .* log.(df.x) .+ 0.1 .* randn(n)
        data = Tables.columntable(df)
        compiled = compile_formula(lm(@formula(y ~ log(x)), df), data)
        de_fd = derivativeevaluator(:fd, compiled, data, [:x])
        Jk = Vector{Float64}(undef, length(compiled))
        fd_jacobian_column!(Jk, de_fd, 1, 1)
        @test Jk[1] == 0.0
        @test Jk[2] ≈ 1 / df.x[1] rtol = 1e-8
    end
end

@testset "FD at extreme variable scales (1e-6, 1e6) matches AD" begin
    Random.seed!(06515)
    n = 100
    df = DataFrame(y = randn(n), x_tiny = randn(n) * 1e-6, x_large = randn(n) * 1e6,
                   x_normal = randn(n),
                   group4 = categorical(rand(["A", "B", "C", "D"], n)),
                   group2 = categorical(rand(["X", "Y"], n)))
    data = Tables.columntable(df)
    model = lm(@formula(y ~ 1 + x_tiny + x_large + x_normal + x_tiny & group4 +
                        x_normal & group2 + x_large & group4), df)
    compiled = compile_formula(model, data)
    vars = [:x_tiny, :x_large, :x_normal]
    de_ad = derivativeevaluator(:ad, compiled, data, vars)
    de_fd = derivativeevaluator(:fd, compiled, data, vars)
    J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
    J_fd = similar(J_ad)
    for row in (1, 10, 15, 25, 50, 75, 90)
        derivative_modelrow!(J_ad, de_ad, row)
        derivative_modelrow!(J_fd, de_fd, row)
        @test isapprox(J_fd, J_ad; rtol = 1e-3, atol = 1e-6)
    end
end
