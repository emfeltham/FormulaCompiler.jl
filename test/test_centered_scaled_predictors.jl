# Comprehensive CenteredTerm / ScaledTerm correctness tests
# Tests that FormulaCompiler produces identical results to GLM modelmatrix for Center() and Scale()

using FormulaCompiler
using StandardizedPredictors
using Test
using BenchmarkTools
using GLM
using DataFrames
using Tables
using CategoricalArrays
using Random
using Statistics
using FormulaCompiler: derivativeevaluator_fd, derivativeevaluator_ad, NumericCounterfactualVector

@testset "CenteredTerm and ScaledTerm Correctness Tests" begin

    # Create test data with known properties
    Random.seed!(42)
    n = 50
    df = DataFrame(
        y = randn(n),
        x = randn(n) * 3.0 .+ 5.0,  # Mean ≈ 5, std ≈ 3
        z = randn(n) * 2.0 .+ 1.0,  # Mean ≈ 1, std ≈ 2
        w = randn(n) * 1.5 .+ 8.0,  # Mean ≈ 8, std ≈ 1.5
        group = categorical(rand(["A", "B", "C"], n))
    )

    # ========== CenteredTerm tests ==========

    @testset "CenteredTerm: Single Variable" begin
        model_raw = lm(@formula(y ~ x), df)
        model_cen = lm(@formula(y ~ x), df, contrasts=Dict(:x => Center()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_cen = compile_formula(model_cen, data)

        X_raw = modelmatrix(model_raw)
        X_cen = modelmatrix(model_cen)

        output_raw = Vector{Float64}(undef, 2)
        output_cen = Vector{Float64}(undef, 2)

        for i in 1:nrow(df)
            compiled_raw(output_raw, data, i)
            compiled_cen(output_cen, data, i)

            @test isapprox(output_raw, X_raw[i, :], rtol=1e-12)
            @test isapprox(output_cen, X_cen[i, :], rtol=1e-12)

            @test output_raw[1] ≈ output_cen[1] ≈ 1.0
        end

        # Verify centering happened
        x_cen_values = X_cen[:, 2]
        @test abs(mean(x_cen_values)) < 1e-10  # Centered should have mean ≈ 0
        @test !isapprox(std(x_cen_values), 1.0, atol=0.1)  # But std NOT ≈ 1 (unlike zscore)
        @test std(x_cen_values) ≈ std(df.x) rtol=1e-10  # Std unchanged by centering
    end

    @testset "CenteredTerm: Multiple Variables" begin
        model_none = lm(@formula(y ~ x + z + w), df)
        model_xz = lm(@formula(y ~ x + z + w), df, contrasts=Dict(:x => Center(), :z => Center()))

        data = Tables.columntable(df)
        compiled_none = compile_formula(model_none, data)
        compiled_xz = compile_formula(model_xz, data)

        X_none = modelmatrix(model_none)
        X_xz = modelmatrix(model_xz)

        output_none = Vector{Float64}(undef, 4)
        output_xz = Vector{Float64}(undef, 4)

        for i in 1:min(10, nrow(df))
            compiled_none(output_none, data, i)
            compiled_xz(output_xz, data, i)

            @test isapprox(output_none, X_none[i, :], rtol=1e-12)
            @test isapprox(output_xz, X_xz[i, :], rtol=1e-12)
        end

        # x and z centered, w raw
        @test abs(mean(X_xz[:, 2])) < 1e-10
        @test abs(mean(X_xz[:, 3])) < 1e-10
        @test X_xz[:, 4] ≈ df.w
    end

    @testset "CenteredTerm: Interactions" begin
        model_raw = lm(@formula(y ~ x * group), df)
        model_cen = lm(@formula(y ~ x * group), df, contrasts=Dict(:x => Center()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_cen = compile_formula(model_cen, data)

        X_raw = modelmatrix(model_raw)
        X_cen = modelmatrix(model_cen)

        output_raw = Vector{Float64}(undef, size(X_raw, 2))
        output_cen = Vector{Float64}(undef, size(X_cen, 2))

        for i in 1:min(10, nrow(df))
            compiled_raw(output_raw, data, i)
            compiled_cen(output_cen, data, i)

            @test isapprox(output_raw, X_raw[i, :], rtol=1e-12)
            @test isapprox(output_cen, X_cen[i, :], rtol=1e-12)
        end

        @test size(X_raw) == size(X_cen)
        @test !isapprox(X_raw[:, 2], X_cen[:, 2])  # x main effect different
    end

    # ========== ScaledTerm tests ==========

    @testset "ScaledTerm: Single Variable" begin
        model_raw = lm(@formula(y ~ x), df)
        model_scl = lm(@formula(y ~ x), df, contrasts=Dict(:x => Scale()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_scl = compile_formula(model_scl, data)

        X_raw = modelmatrix(model_raw)
        X_scl = modelmatrix(model_scl)

        output_raw = Vector{Float64}(undef, 2)
        output_scl = Vector{Float64}(undef, 2)

        for i in 1:nrow(df)
            compiled_raw(output_raw, data, i)
            compiled_scl(output_scl, data, i)

            @test isapprox(output_raw, X_raw[i, :], rtol=1e-12)
            @test isapprox(output_scl, X_scl[i, :], rtol=1e-12)

            @test output_raw[1] ≈ output_scl[1] ≈ 1.0
        end

        # Verify scaling happened
        x_scl_values = X_scl[:, 2]
        @test !isapprox(mean(x_scl_values), 0.0, atol=0.1)  # Mean NOT ≈ 0 (unlike zscore)
        @test abs(std(x_scl_values) - 1.0) < 0.1  # Std ≈ 1 after scaling
    end

    @testset "ScaledTerm: Multiple Variables" begin
        model_none = lm(@formula(y ~ x + z + w), df)
        model_xz = lm(@formula(y ~ x + z + w), df, contrasts=Dict(:x => Scale(), :z => Scale()))

        data = Tables.columntable(df)
        compiled_none = compile_formula(model_none, data)
        compiled_xz = compile_formula(model_xz, data)

        X_none = modelmatrix(model_none)
        X_xz = modelmatrix(model_xz)

        output_none = Vector{Float64}(undef, 4)
        output_xz = Vector{Float64}(undef, 4)

        for i in 1:min(10, nrow(df))
            compiled_none(output_none, data, i)
            compiled_xz(output_xz, data, i)

            @test isapprox(output_none, X_none[i, :], rtol=1e-12)
            @test isapprox(output_xz, X_xz[i, :], rtol=1e-12)
        end

        # w should be raw
        @test X_xz[:, 4] ≈ df.w
    end

    @testset "ScaledTerm: Interactions" begin
        model_raw = lm(@formula(y ~ x * group), df)
        model_scl = lm(@formula(y ~ x * group), df, contrasts=Dict(:x => Scale()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_scl = compile_formula(model_scl, data)

        X_raw = modelmatrix(model_raw)
        X_scl = modelmatrix(model_scl)

        output_raw = Vector{Float64}(undef, size(X_raw, 2))
        output_scl = Vector{Float64}(undef, size(X_scl, 2))

        for i in 1:min(10, nrow(df))
            compiled_raw(output_raw, data, i)
            compiled_scl(output_scl, data, i)

            @test isapprox(output_raw, X_raw[i, :], rtol=1e-12)
            @test isapprox(output_scl, X_scl[i, :], rtol=1e-12)
        end

        @test size(X_raw) == size(X_scl)
        @test !isapprox(X_raw[:, 2], X_scl[:, 2])  # x main effect different
    end

    # ========== Mixed usage tests ==========

    @testset "Mixed: Center + Scale + ZScore" begin
        model = lm(@formula(y ~ x + z + w), df,
                    contrasts=Dict(:x => Center(), :z => Scale(), :w => ZScore()))

        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        X = modelmatrix(model)

        output = Vector{Float64}(undef, 4)

        for i in 1:nrow(df)
            compiled(output, data, i)
            @test isapprox(output, X[i, :], rtol=1e-12)
        end

        # x centered: mean ≈ 0, std preserved
        @test abs(mean(X[:, 2])) < 1e-10
        @test std(X[:, 2]) ≈ std(df.x) rtol=1e-10

        # z scaled: std ≈ 1, mean not 0
        @test abs(std(X[:, 3]) - 1.0) < 0.1

        # w z-scored: mean ≈ 0, std ≈ 1
        @test abs(mean(X[:, 4])) < 1e-10
        @test abs(std(X[:, 4]) - 1.0) < 1e-10
    end

    # ========== Performance tests ==========

    @testset "Performance: Zero Allocations" begin
        model_cen = lm(@formula(y ~ x + z), df, contrasts=Dict(:x => Center(), :z => Center()))
        model_scl = lm(@formula(y ~ x + z), df, contrasts=Dict(:x => Scale(), :z => Scale()))

        data = Tables.columntable(df)
        compiled_cen = compile_formula(model_cen, data)
        compiled_scl = compile_formula(model_scl, data)

        output_cen = Vector{Float64}(undef, length(compiled_cen))
        output_scl = Vector{Float64}(undef, length(compiled_scl))

        b_cen = @benchmark $compiled_cen($output_cen, $data, 1) samples=200 evals=1
        @test minimum(b_cen.memory) == 0

        b_scl = @benchmark $compiled_scl($output_scl, $data, 1) samples=200 evals=1
        @test minimum(b_scl.memory) == 0
    end

    # ========== Derivative tests ==========

    @testset "Derivatives: CenteredTerm Scale Validation" begin
        # For Center(): only centering, no scaling
        # Derivative w.r.t. raw x should be identical to raw model
        model_raw = lm(@formula(y ~ x), df)
        model_cen = lm(@formula(y ~ x), df, contrasts=Dict(:x => Center()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_cen = compile_formula(model_cen, data)

        β₁_raw = coef(model_raw)[2]
        β₁_cen = coef(model_cen)[2]

        # For centering only: β_cen should equal β_raw (centering doesn't change slope)
        @test β₁_raw ≈ β₁_cen rtol=1e-10

        # Test FD backend
        de_raw_fd = derivativeevaluator_fd(compiled_raw, data, [:x])
        de_cen_fd = derivativeevaluator_fd(compiled_cen, data, [:x])

        J_raw_fd = Matrix{Float64}(undef, length(compiled_raw), 1)
        J_cen_fd = Matrix{Float64}(undef, length(compiled_cen), 1)

        FormulaCompiler.derivative_modelrow!(J_raw_fd, de_raw_fd, 1)
        FormulaCompiler.derivative_modelrow!(J_cen_fd, de_cen_fd, 1)

        g_raw_fd = (J_raw_fd' * coef(model_raw))[1]
        g_cen_fd = (J_cen_fd' * coef(model_cen))[1]

        # Both should give same derivative (centering has scale=1, no chain rule factor)
        @test g_raw_fd ≈ g_cen_fd rtol=1e-10

        # Test AD backend
        de_raw_ad = derivativeevaluator_ad(compiled_raw, data, [:x])
        de_cen_ad = derivativeevaluator_ad(compiled_cen, data, [:x])

        J_raw_ad = Matrix{Float64}(undef, length(compiled_raw), 1)
        J_cen_ad = Matrix{Float64}(undef, length(compiled_cen), 1)

        FormulaCompiler.derivative_modelrow!(J_raw_ad, de_raw_ad, 1)
        FormulaCompiler.derivative_modelrow!(J_cen_ad, de_cen_ad, 1)

        g_raw_ad = (J_raw_ad' * coef(model_raw))[1]
        g_cen_ad = (J_cen_ad' * coef(model_cen))[1]

        @test g_raw_ad ≈ g_cen_ad rtol=1e-10
        @test g_raw_fd ≈ g_raw_ad rtol=1e-10
    end

    @testset "Derivatives: ScaledTerm Scale Validation" begin
        # For Scale(): only scaling, derivative includes 1/scale factor
        model_raw = lm(@formula(y ~ x), df)
        model_scl = lm(@formula(y ~ x), df, contrasts=Dict(:x => Scale()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_scl = compile_formula(model_scl, data)

        x_std_dev = std(df.x)
        β₁_raw = coef(model_raw)[2]
        β₁_scl = coef(model_scl)[2]

        # For scaling: β_scl = β_raw * σ, so β_scl / σ = β_raw
        @test β₁_scl / x_std_dev ≈ β₁_raw rtol=1e-10

        # Test FD backend
        de_raw_fd = derivativeevaluator_fd(compiled_raw, data, [:x])
        de_scl_fd = derivativeevaluator_fd(compiled_scl, data, [:x])

        J_raw_fd = Matrix{Float64}(undef, length(compiled_raw), 1)
        J_scl_fd = Matrix{Float64}(undef, length(compiled_scl), 1)

        FormulaCompiler.derivative_modelrow!(J_raw_fd, de_raw_fd, 1)
        FormulaCompiler.derivative_modelrow!(J_scl_fd, de_scl_fd, 1)

        g_raw_fd = (J_raw_fd' * coef(model_raw))[1]
        g_scl_fd = (J_scl_fd' * coef(model_scl))[1]

        # Both should give same derivative on raw scale
        @test g_raw_fd ≈ g_scl_fd rtol=1e-10

        # Test AD backend
        de_raw_ad = derivativeevaluator_ad(compiled_raw, data, [:x])
        de_scl_ad = derivativeevaluator_ad(compiled_scl, data, [:x])

        J_raw_ad = Matrix{Float64}(undef, length(compiled_raw), 1)
        J_scl_ad = Matrix{Float64}(undef, length(compiled_scl), 1)

        FormulaCompiler.derivative_modelrow!(J_raw_ad, de_raw_ad, 1)
        FormulaCompiler.derivative_modelrow!(J_scl_ad, de_scl_ad, 1)

        g_raw_ad = (J_raw_ad' * coef(model_raw))[1]
        g_scl_ad = (J_scl_ad' * coef(model_scl))[1]

        @test g_raw_ad ≈ g_scl_ad rtol=1e-10
        @test g_raw_fd ≈ g_raw_ad rtol=1e-10
    end

    @testset "Derivatives: Multi-Variable Mixed" begin
        model_raw = lm(@formula(y ~ x + z), df)
        model_mix = lm(@formula(y ~ x + z), df, contrasts=Dict(:x => Center(), :z => Scale()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_mix = compile_formula(model_mix, data)

        de_raw = derivativeevaluator_fd(compiled_raw, data, [:x, :z])
        de_mix = derivativeevaluator_fd(compiled_mix, data, [:x, :z])

        J_raw = Matrix{Float64}(undef, length(compiled_raw), 2)
        J_mix = Matrix{Float64}(undef, length(compiled_mix), 2)

        FormulaCompiler.derivative_modelrow!(J_raw, de_raw, 1)
        FormulaCompiler.derivative_modelrow!(J_mix, de_mix, 1)

        g_raw = J_raw' * coef(model_raw)
        g_mix = J_mix' * coef(model_mix)

        # Both models should give same marginal effects on raw scale
        @test g_raw[1] ≈ g_mix[1] rtol=1e-10  # x: centering only, same derivative
        @test g_raw[2] ≈ g_mix[2] rtol=1e-10  # z: scaling, chain rule applies
    end

    @testset "Derivatives: AD vs FD Consistency" begin
        model_cen = lm(@formula(y ~ x + z), df, contrasts=Dict(:x => Center()))
        model_scl = lm(@formula(y ~ x + z), df, contrasts=Dict(:x => Scale()))

        data = Tables.columntable(df)

        for (label, model) in [("Center", model_cen), ("Scale", model_scl)]
            compiled = compile_formula(model, data)

            de_fd = derivativeevaluator_fd(compiled, data, [:x, :z])
            de_ad = derivativeevaluator_ad(compiled, data, [:x, :z])

            J_fd = Matrix{Float64}(undef, length(compiled), 2)
            J_ad = Matrix{Float64}(undef, length(compiled), 2)

            FormulaCompiler.derivative_modelrow!(J_fd, de_fd, 1)
            FormulaCompiler.derivative_modelrow!(J_ad, de_ad, 1)

            @test isapprox(J_fd, J_ad, rtol=1e-6)
        end
    end

    @testset "Scenario Integration" begin
        model = lm(@formula(y ~ x + z), df, contrasts=Dict(:x => Center()))
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)

        cf_x = NumericCounterfactualVector{Float64}(data.x, 1, 10.0)
        cf_data = merge(data, (x = cf_x,))

        output_baseline = Vector{Float64}(undef, 3)
        output_scenario = Vector{Float64}(undef, 3)

        compiled(output_baseline, data, 1)
        compiled(output_scenario, cf_data, 1)

        @test !isapprox(output_baseline, output_scenario)
        @test all(isfinite.(output_scenario))
    end

end
