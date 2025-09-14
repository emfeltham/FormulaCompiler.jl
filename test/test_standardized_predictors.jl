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

@testset "StandardizedPredictors Correctness Tests" begin

    # Create test data with known properties for better validation
    Random.seed!(42)  # For reproducible tests
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

            # Standardized values should be different from raw (except by coincidence)
            if abs(df.x[i] - mean(df.x)) > 0.01  # Avoid near-mean values
                @test abs(output_raw[2] - output_std[2]) > 0.1
            end
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

        # Create scenario with raw override values (should be standardized internally)
        scenario = create_scenario("test", data; x = 10.0, z = 3.0)

        output_baseline = Vector{Float64}(undef, 3)
        output_scenario = Vector{Float64}(undef, 3)

        compiled(output_baseline, data, 1)
        compiled(output_scenario, scenario.data, 1)

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

        # Build derivative evaluator for standardized variables
        de = FormulaCompiler.build_derivative_evaluator(compiled, data; vars=[:x, :z])
        g_fd = Vector{Float64}(undef, 2)
        g_ad = Vector{Float64}(undef, 2)

        # Test both backends
        FormulaCompiler.marginal_effects_eta!(g_fd, de, coef(model), 1; backend=:fd)
        FormulaCompiler.marginal_effects_eta!(g_ad, de, coef(model), 1; backend=:ad)

        # Both should give finite, reasonable results
        @test all(isfinite.(g_fd))
        @test all(isfinite.(g_ad))

        # AD and FD should agree (within tolerance)
        @test isapprox(g_fd, g_ad, rtol=1e-6)

        # Derivatives should be non-zero (assuming non-zero coefficients)
        @test any(abs.(g_fd) .> 1e-6)
    end

end