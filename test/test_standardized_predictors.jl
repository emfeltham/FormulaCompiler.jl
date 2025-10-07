# Comprehensive StandardizedPredictors.jl correctness tests
# Tests that FormulaCompiler produces identical results to GLM modelmatrix for standardized predictors

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

@testset "StandardizedPredictors Correctness Tests" begin

    # Create test data with known properties for better validation
    Random.seed!(06515)  # For reproducible tests
    n = 50
    df = DataFrame(
        y = randn(n),
        x = randn(n) * 3.0 .+ 5.0,  # Mean ≈ 5, std ≈ 3
        z = randn(n) * 2.0 .+ 1.0,  # Mean ≈ 1, std ≈ 2
        w = randn(n) * 1.5 .+ 8.0,  # Mean ≈ 8, std ≈ 1.5
        group = categorical(rand(["A", "B", "C"], n))
    )

    @testset "Single Variable Standardization" begin
        # Compare same model with and without standardization
        model_raw = lm(@formula(y ~ x), df)
        model_std = lm(@formula(y ~ x), df, contrasts=Dict(:x => ZScore()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_std = compile_formula(model_std, data)

        X_raw = modelmatrix(model_raw)
        X_std = modelmatrix(model_std)

        output_raw = Vector{Float64}(undef, 2)
        output_std = Vector{Float64}(undef, 2)

        # Test all rows for exact correctness
        for i in 1:nrow(df)
            compiled_raw(output_raw, data, i)
            compiled_std(output_std, data, i)

            # FormulaCompiler must match GLM exactly
            @test isapprox(output_raw, X_raw[i, :], rtol=1e-12)
            @test isapprox(output_std, X_std[i, :], rtol=1e-12)

            # Intercepts should be identical
            @test output_raw[1] ≈ output_std[1] ≈ 1.0
        end

        # Verify standardization actually happened
        x_raw_values = X_raw[:, 2]
        x_std_values = X_std[:, 2]

        @test !isapprox(x_raw_values, x_std_values)  # Should be different
        @test abs(mean(x_std_values)) < 1e-10       # Standardized should have mean ≈ 0
        @test abs(std(x_std_values) - 1.0) < 1e-10  # Standardized should have std ≈ 1
    end

    @testset "Multiple Variable Standardization" begin
        # Compare models with different standardization combinations
        model_none = lm(@formula(y ~ x + z + w), df)
        model_x = lm(@formula(y ~ x + z + w), df, contrasts=Dict(:x => ZScore()))
        model_xz = lm(@formula(y ~ x + z + w), df, contrasts=Dict(:x => ZScore(), :z => ZScore()))
        model_all = lm(@formula(y ~ x + z + w), df, contrasts=Dict(:x => ZScore(), :z => ZScore(), :w => ZScore()))

        data = Tables.columntable(df)
        compiled_none = compile_formula(model_none, data)
        compiled_x = compile_formula(model_x, data)
        compiled_xz = compile_formula(model_xz, data)
        compiled_all = compile_formula(model_all, data)

        X_none = modelmatrix(model_none)
        X_x = modelmatrix(model_x)
        X_xz = modelmatrix(model_xz)
        X_all = modelmatrix(model_all)

        output_none = Vector{Float64}(undef, 4)
        output_x = Vector{Float64}(undef, 4)
        output_xz = Vector{Float64}(undef, 4)
        output_all = Vector{Float64}(undef, 4)

        # Test correctness for first few rows
        for i in 1:min(10, nrow(df))
            compiled_none(output_none, data, i)
            compiled_x(output_x, data, i)
            compiled_xz(output_xz, data, i)
            compiled_all(output_all, data, i)

            # Each must match corresponding GLM modelmatrix exactly
            @test isapprox(output_none, X_none[i, :], rtol=1e-12)
            @test isapprox(output_x, X_x[i, :], rtol=1e-12)
            @test isapprox(output_xz, X_xz[i, :], rtol=1e-12)
            @test isapprox(output_all, X_all[i, :], rtol=1e-12)
        end

        # Verify progressive standardization
        # Intercept should always be 1.0
        @test all(X_none[:, 1] .≈ 1.0)
        @test all(X_x[:, 1] .≈ 1.0)
        @test all(X_xz[:, 1] .≈ 1.0)
        @test all(X_all[:, 1] .≈ 1.0)

        # Check that only specified variables are standardized
        @test X_none[:, 2] ≈ df.x                    # x raw
        @test abs(mean(X_x[:, 2])) < 1e-10           # x standardized
        @test abs(mean(X_xz[:, 2])) < 1e-10          # x standardized
        @test abs(mean(X_all[:, 2])) < 1e-10         # x standardized

        @test X_none[:, 3] ≈ df.z                    # z raw
        @test X_x[:, 3] ≈ df.z                       # z still raw
        @test abs(mean(X_xz[:, 3])) < 1e-10          # z standardized
        @test abs(mean(X_all[:, 3])) < 1e-10         # z standardized

        @test X_none[:, 4] ≈ df.w                    # w raw
        @test X_x[:, 4] ≈ df.w                       # w still raw
        @test X_xz[:, 4] ≈ df.w                      # w still raw
        @test abs(mean(X_all[:, 4])) < 1e-10         # w standardized
    end

    @testset "Interactions with Standardization" begin
        # Test standardized variables in interactions
        model_raw = lm(@formula(y ~ x * group), df)
        model_std = lm(@formula(y ~ x * group), df, contrasts=Dict(:x => ZScore()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_std = compile_formula(model_std, data)

        X_raw = modelmatrix(model_raw)
        X_std = modelmatrix(model_std)

        output_raw = Vector{Float64}(undef, size(X_raw, 2))
        output_std = Vector{Float64}(undef, size(X_std, 2))

        # Test correctness for interactions
        for i in 1:min(10, nrow(df))
            compiled_raw(output_raw, data, i)
            compiled_std(output_std, data, i)

            @test isapprox(output_raw, X_raw[i, :], rtol=1e-12)
            @test isapprox(output_std, X_std[i, :], rtol=1e-12)
        end

        # Verify interaction structure
        @test size(X_raw) == size(X_std)  # Same structure

        # Main effect of x should be different (raw vs standardized)
        @test !isapprox(X_raw[:, 2], X_std[:, 2])

        # Group contrasts should be identical (not standardized)
        # Find group columns (non-intercept, non-x main effect)
        group_cols_raw = X_raw[:, 3:end-2]  # Assuming 2 interaction cols at end
        group_cols_std = X_std[:, 3:end-2]
        @test isapprox(group_cols_raw, group_cols_std)

        # Interaction effects should be different (because x component is standardized)
        interaction_cols_raw = X_raw[:, end-1:end]
        interaction_cols_std = X_std[:, end-1:end]
        @test !isapprox(interaction_cols_raw, interaction_cols_std)
    end

    @testset "Complex Formulas with Standardization" begin
        # Test with functions and multiple interactions
        model_raw = lm(@formula(y ~ x + log(abs(z) + 1) + x * group), df)
        model_std = lm(@formula(y ~ x + log(abs(z) + 1) + x * group), df,
                      contrasts=Dict(:x => ZScore()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_std = compile_formula(model_std, data)

        X_raw = modelmatrix(model_raw)
        X_std = modelmatrix(model_std)

        output_raw = Vector{Float64}(undef, size(X_raw, 2))
        output_std = Vector{Float64}(undef, size(X_std, 2))

        # Test correctness
        for i in 1:min(5, nrow(df))
            compiled_raw(output_raw, data, i)
            compiled_std(output_std, data, i)

            @test isapprox(output_raw, X_raw[i, :], rtol=1e-12)
            @test isapprox(output_std, X_std[i, :], rtol=1e-12)
        end

        # Verify that log(z) term is identical (z not standardized)
        log_col_raw = 3  # Assuming column order: intercept, x, log(z), group..., interactions...
        log_col_std = 3
        @test X_raw[:, log_col_raw] ≈ X_std[:, log_col_std]

        # Verify x main effect is different
        @test !isapprox(X_raw[:, 2], X_std[:, 2])
    end

    @testset "Edge Cases" begin
        # Test with near-constant variable (small variance)
        df_const = copy(df)
        df_const.x = df_const.x .* 0.001 .+ 100.0  # Very small variance, large mean

        model_const = lm(@formula(y ~ x), df_const, contrasts=Dict(:x => ZScore()))
        data_const = Tables.columntable(df_const)
        compiled_const = compile_formula(model_const, data_const)

        X_const = modelmatrix(model_const)
        output_const = Vector{Float64}(undef, 2)

        # Should still work correctly
        for i in 1:5
            compiled_const(output_const, data_const, i)
            @test isapprox(output_const, X_const[i, :], rtol=1e-12)
        end

        # Test with extreme values
        df_extreme = copy(df)
        df_extreme.x[1] = 1e6  # Very large value
        df_extreme.x[2] = -1e6 # Very negative value

        model_extreme = lm(@formula(y ~ x), df_extreme, contrasts=Dict(:x => ZScore()))
        data_extreme = Tables.columntable(df_extreme)
        compiled_extreme = compile_formula(model_extreme, data_extreme)

        X_extreme = modelmatrix(model_extreme)
        output_extreme = Vector{Float64}(undef, 2)

        # Should handle extreme values correctly
        for i in 1:5
            compiled_extreme(output_extreme, data_extreme, i)
            @test isapprox(output_extreme, X_extreme[i, :], rtol=1e-12)
        end
    end

    @testset "Performance Validation" begin
        # Ensure standardization doesn't break zero-allocation promise
        model = lm(@formula(y ~ x + z), df, contrasts=Dict(:x => ZScore(), :z => ZScore()))
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        output = Vector{Float64}(undef, length(compiled))

        # Should be zero allocation
        b = @benchmark $compiled($output, $data, 1) samples=200 evals=1
        @test minimum(b.memory) == 0

        # Should be reasonably fast
        @test minimum(b.times) < 1_000  # Less than 1μs
    end

    @testset "Scenario Integration" begin
        # Test that scenarios work correctly with standardized variables
        model = lm(@formula(y ~ x + z), df, contrasts=Dict(:x => ZScore()))
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)

        # Create counterfactual with raw override values (should be standardized internally)
        cf_x = NumericCounterfactualVector{Float64}(data.x, 1, 10.0)
        cf_z = NumericCounterfactualVector{Float64}(data.z, 1, 3.0)
        cf_data = merge(data, (x = cf_x, z = cf_z))

        output_baseline = Vector{Float64}(undef, 3)
        output_scenario = Vector{Float64}(undef, 3)

        compiled(output_baseline, data, 1)
        compiled(output_scenario, cf_data, 1)

        # Results should be different (override took effect)
        @test !isapprox(output_baseline, output_scenario)

        # Values should be finite and reasonable
        @test all(isfinite.(output_scenario))
        @test abs(output_scenario[2]) < 10  # Standardized x shouldn't be too extreme
    end

    @testset "Derivative Integration" begin
        # Test that derivatives work with standardized variables
        model = lm(@formula(y ~ x + z), df, contrasts=Dict(:x => ZScore(), :z => ZScore()))
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)

        # Build derivative evaluators for standardized variables
        de_fd = derivativeevaluator_fd(compiled, data, [:x, :z])
        de_ad = derivativeevaluator_ad(compiled, data, [:x, :z])

        J_fd = Matrix{Float64}(undef, length(compiled), 2)
        J_ad = Matrix{Float64}(undef, length(compiled), 2)

        # Test both backends - compute Jacobian
        FormulaCompiler.derivative_modelrow!(J_fd, de_fd, 1)
        FormulaCompiler.derivative_modelrow!(J_ad, de_ad, 1)

        # Both should give finite, reasonable results
        @test all(isfinite.(J_fd))
        @test all(isfinite.(J_ad))

        # AD and FD should agree (within tolerance)
        @test isapprox(J_fd, J_ad, rtol=1e-6)

        # Derivatives should be non-zero (assuming non-zero coefficients)
        @test any(abs.(J_fd) .> 1e-6)
    end

    @testset "Derivative Scale Validation" begin
        # Critical test: Verify derivatives are on ORIGINAL (raw) scale, not standardized scale
        # This validates the automatic back-transformation via chain rule

        # Simple linear model: y = β₀ + β₁·x
        model_raw = lm(@formula(y ~ x), df)
        model_std = lm(@formula(y ~ x), df, contrasts=Dict(:x => ZScore()))

        data = Tables.columntable(df)
        compiled_raw = compile_formula(model_raw, data)
        compiled_std = compile_formula(model_std, data)

        # Get standardization parameters
        x_mean = mean(df.x)
        x_std_dev = std(df.x)

        # Coefficients
        β₁_raw = coef(model_raw)[2]  # Raw model coefficient
        β₁_std = coef(model_std)[2]  # Standardized model coefficient

        # For a linear model:
        # - Raw model: ∂η/∂x_raw = β₁_raw
        # - Standardized model: ∂η/∂x_std = β₁_std, but ∂η/∂x_raw = β₁_std / σ
        # Both should give the same derivative w.r.t. raw x

        # Test FD backend
        de_raw_fd = derivativeevaluator_fd(compiled_raw, data, [:x])
        de_std_fd = derivativeevaluator_fd(compiled_std, data, [:x])

        J_raw_fd = Matrix{Float64}(undef, length(compiled_raw), 1)
        J_std_fd = Matrix{Float64}(undef, length(compiled_std), 1)

        FormulaCompiler.derivative_modelrow!(J_raw_fd, de_raw_fd, 1)
        FormulaCompiler.derivative_modelrow!(J_std_fd, de_std_fd, 1)

        # Compute marginal effects: g = J' * β
        g_raw_fd = (J_raw_fd' * coef(model_raw))[1]
        g_std_fd = (J_std_fd' * coef(model_std))[1]

        # Theoretical derivatives on raw scale
        theoretical_raw = β₁_raw               # ∂η/∂x_raw for raw model
        theoretical_std = β₁_std / x_std_dev   # ∂η/∂x_raw for standardized model

        # Validate computed derivatives match theoretical values
        @test g_raw_fd ≈ theoretical_raw rtol=1e-10
        @test g_std_fd ≈ theoretical_std rtol=1e-10

        # CRITICAL TEST: Both should give same derivative (both on raw scale)
        @test g_raw_fd ≈ g_std_fd rtol=1e-10

        # Test AD backend
        de_raw_ad = derivativeevaluator_ad(compiled_raw, data, [:x])
        de_std_ad = derivativeevaluator_ad(compiled_std, data, [:x])

        J_raw_ad = Matrix{Float64}(undef, length(compiled_raw), 1)
        J_std_ad = Matrix{Float64}(undef, length(compiled_std), 1)

        FormulaCompiler.derivative_modelrow!(J_raw_ad, de_raw_ad, 1)
        FormulaCompiler.derivative_modelrow!(J_std_ad, de_std_ad, 1)

        g_raw_ad = (J_raw_ad' * coef(model_raw))[1]
        g_std_ad = (J_std_ad' * coef(model_std))[1]

        # AD should also produce raw-scale derivatives
        @test g_raw_ad ≈ theoretical_raw rtol=1e-10
        @test g_std_ad ≈ theoretical_std rtol=1e-10
        @test g_raw_ad ≈ g_std_ad rtol=1e-10

        # FD and AD should agree for both models
        @test g_raw_fd ≈ g_raw_ad rtol=1e-10
        @test g_std_fd ≈ g_std_ad rtol=1e-10

        # Multi-variable test
        model_multi_raw = lm(@formula(y ~ x + z), df)
        model_multi_std = lm(@formula(y ~ x + z), df, contrasts=Dict(:x => ZScore(), :z => ZScore()))

        compiled_multi_raw = compile_formula(model_multi_raw, data)
        compiled_multi_std = compile_formula(model_multi_std, data)

        de_multi_raw = derivativeevaluator_fd(compiled_multi_raw, data, [:x, :z])
        de_multi_std = derivativeevaluator_fd(compiled_multi_std, data, [:x, :z])

        J_multi_raw = Matrix{Float64}(undef, length(compiled_multi_raw), 2)
        J_multi_std = Matrix{Float64}(undef, length(compiled_multi_std), 2)

        FormulaCompiler.derivative_modelrow!(J_multi_raw, de_multi_raw, 1)
        FormulaCompiler.derivative_modelrow!(J_multi_std, de_multi_std, 1)

        g_multi_raw = J_multi_raw' * coef(model_multi_raw)
        g_multi_std = J_multi_std' * coef(model_multi_std)

        # Coefficients for multi-variable model
        β_x_raw = coef(model_multi_raw)[2]
        β_z_raw = coef(model_multi_raw)[3]
        β_x_std = coef(model_multi_std)[2]
        β_z_std = coef(model_multi_std)[3]

        z_std_dev = std(df.z)

        # Each derivative should be on raw scale
        @test g_multi_raw[1] ≈ β_x_raw rtol=1e-10
        @test g_multi_raw[2] ≈ β_z_raw rtol=1e-10
        @test g_multi_std[1] ≈ β_x_std / x_std_dev rtol=1e-10
        @test g_multi_std[2] ≈ β_z_std / z_std_dev rtol=1e-10

        # Raw and standardized models should give same derivatives
        @test g_multi_raw[1] ≈ g_multi_std[1] rtol=1e-10
        @test g_multi_raw[2] ≈ g_multi_std[2] rtol=1e-10
    end

end