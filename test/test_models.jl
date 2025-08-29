# test_models.jl
# Comprehensive correctness testing for all supported model types
# Tests formula compilation correctness against modelmatrix and modelrow functionality

using Test
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays, MixedModels
using StatsModels, BenchmarkTools
# Test helpers are included from test/support/testing_utilities.jl via runtests.jl
using Random

@testset "Model Correctness Tests" begin
    Random.seed!(08540)
    
    # Setup test data
    n = 500
    df = make_test_data(; n)
    data = Tables.columntable(df)
    
    @testset "Linear Models (LM)" begin
        for fx in test_formulas.lm
            @testset "$(fx.name)" begin
                model = lm(fx.formula, df)
                @test test_model_correctness(model, data, n)
                
                # Additional structural checks
                compiled = compile_formula(model, data)
                @test length(compiled) == size(modelmatrix(model), 2)
                @test length(compiled) > 0
            end
        end
    end
    
    @testset "Generalized Linear Models (GLM)" begin
        for fx in test_formulas.glm
            @testset "$(fx.name)" begin
                model = glm(fx.formula, df, fx.distribution, fx.link)
                @test test_model_correctness(model, data, n)
                
                # Additional structural checks
                compiled = compile_formula(model, data)
                @test length(compiled) == size(modelmatrix(model), 2)
                @test length(compiled) > 0
            end
        end
    end
    
    @testset "Linear Mixed Models (LMM)" begin
        for fx in test_formulas.lmm
            @testset "$(fx.name)" begin
                model = fit(MixedModel, fx.formula, df; progress = false)
                @test test_model_correctness(model, data, n)
                
                # Additional structural checks
                compiled = compile_formula(model, data)
                @test length(compiled) == size(modelmatrix(model), 2)
                @test length(compiled) > 0
            end
        end
    end
    
    @testset "Generalized Linear Mixed Models (GLMM)" begin
        for fx in test_formulas.glmm
            @testset "$(fx.name)" begin
                model = fit(MixedModel, fx.formula, df, fx.distribution, fx.link; progress = false)
                @test test_model_correctness(model, data, n)
                
                # Additional structural checks
                compiled = compile_formula(model, data)
                @test length(compiled) == size(modelmatrix(model), 2)
                @test length(compiled) > 0
            end
        end
    end
    
    @testset "Complex Interaction Correctness" begin
        # Test complex interactions across different model types
        @testset "Complex LM interaction" begin
            formula = @formula(continuous_response ~ x * y * group3 + log(z) * group4)
            model = lm(formula, df)
            @test test_model_correctness(model, data, n)
        end
        
        @testset "Complex GLM interaction" begin
            formula = @formula(logistic_response ~ x * y * group3 + log(z) * group4)
            model = glm(formula, df, Binomial(), LogitLink())
            @test test_model_correctness(model, data, n)
        end
        
        @testset "Complex LMM interaction" begin
            formula = @formula(continuous_response ~ x * y * group3 + log(z) * group4 + (1|subject))
            model = fit(MixedModel, formula, df; progress = false)
            @test test_model_correctness(model, data, n)
        end
    end
    
    @testset "Integer Continuous Variables" begin
        # Test with integer continuous variables to catch type issues
        n_int = 200
        df_int = DataFrame(
            y = randn(n_int),
            int_x = rand(1:100, n_int),  # Integer continuous variable
            int_age = rand(18:80, n_int),  # Age as integer
            int_score = rand(0:1000, n_int),  # Test score as integer
            group = categorical(rand(["A", "B", "C"], n_int)),
            float_z = randn(n_int)  # Mix with float for comparison
        )
        data_int = Tables.columntable(df_int)
        
        @testset "Simple integer variable" begin
            model = lm(@formula(y ~ int_x), df_int)
            @test test_model_correctness(model, data_int, n_int)
            
            # Test compilation works
            compiled = compile_formula(model, data_int)
            @test length(compiled) == 2  # Intercept + int_x
            
            # Test evaluation
            output = Vector{Float64}(undef, length(compiled))
            compiled(output, data_int, 1)
            @test length(output) == 2
            @test all(isfinite.(output))
        end
        
        @testset "Integer with interactions" begin
            model = lm(@formula(y ~ int_x * group), df_int)
            @test test_model_correctness(model, data_int, n_int)
            
            compiled = compile_formula(model, data_int)
            @test length(compiled) > 2  # Should have interaction terms
        end
        
        @testset "Multiple integer variables" begin
            model = lm(@formula(y ~ int_x + int_age + int_score), df_int)
            @test test_model_correctness(model, data_int, n_int)
            
            compiled = compile_formula(model, data_int)
            @test length(compiled) == 4  # Intercept + 3 integer variables
        end
        
        @testset "Integer with function transformation" begin
            model = lm(@formula(y ~ log(int_score + 1)), df_int)  # +1 to avoid log(0)
            @test test_model_correctness(model, data_int, n_int)
            
            compiled = compile_formula(model, data_int)
            @test length(compiled) == 2  # Intercept + log(int_score + 1)
        end
        
        @testset "Mixed integer and float" begin
            model = lm(@formula(y ~ int_x * float_z + int_age), df_int)
            @test test_model_correctness(model, data_int, n_int)
            
            compiled = compile_formula(model, data_int)
            @test length(compiled) == 5  # Intercept + int_x + float_z + int_x:float_z + int_age
        end
        
        @testset "GLM with integer variables" begin
            # Create binary response for logistic regression
            df_int.binary_y = rand([0, 1], n_int)
            data_int = Tables.columntable(df_int)
            
            model = glm(@formula(binary_y ~ int_x + int_age), df_int, Binomial(), LogitLink())
            @test test_model_correctness(model, data_int, n_int)
        end
    end
    
    @testset "ModelRow Interface Tests" begin
        # Test ModelRowEvaluator interface
        model = lm(@formula(continuous_response ~ x * group3 + log(z)), df)
        compiled = compile_formula(model, data)
        expected_matrix = modelmatrix(model)
        
        @testset "ModelRowEvaluator creation and usage" begin
            # Test constructor with model (not compiled formula)
            evaluator = ModelRowEvaluator(model, data)
            
            # Test evaluation with different rows
            for test_row in [1, 25, 100, n]
                result = evaluator(test_row)
                expected = expected_matrix[test_row, :]
                @test isapprox(result, expected, rtol=1e-12)
            end
            
            # Test in-place evaluation
            output_buffer = Vector{Float64}(undef, length(compiled))
            for test_row in [1, 50, n]
                evaluator(output_buffer, test_row)
                expected = expected_matrix[test_row, :]
                @test isapprox(output_buffer, expected, rtol=1e-12)
            end
        end
        
        @testset "Batch evaluation correctness" begin
            # Test batch row evaluation
            test_rows = [1, 10, 25, 50, 100]
            result_matrix = modelrow(compiled, data, test_rows)
            
            @test size(result_matrix) == (length(test_rows), length(compiled))
            
            for (i, row_idx) in enumerate(test_rows)
                expected = expected_matrix[row_idx, :]
                @test isapprox(result_matrix[i, :], expected, rtol=1e-12)
            end
        end
    end
    
    @testset "Scenario Integration Correctness" begin
        # Test scenarios with correctness verification
        model = lm(@formula(continuous_response ~ x * group3), df)
        compiled = compile_formula(model, data)
        expected_matrix = modelmatrix(model)
        
        @testset "Original data scenario" begin
            scenario_original = create_scenario("original", data)
            
            for test_row in [1, 25, 100]
                result = modelrow(compiled, scenario_original, test_row)
                expected = expected_matrix[test_row, :]
                @test isapprox(result, expected, rtol=1e-12)
            end
        end
        
        @testset "Modified data scenario" begin
            scenario_modified = create_scenario("modified", data; x = 5.0, group3 = "A")
            
            # Test that modified scenario produces different results
            original_result = modelrow(compiled, data, 1)
            modified_result = modelrow(compiled, scenario_modified, 1)
            
            # Results should be different (x and group3 were modified)
            @test !isapprox(original_result, modified_result, rtol=1e-6)
            
            # But should still be valid model matrix row
            @test length(modified_result) == length(compiled)
            @test all(isfinite.(modified_result))
        end
    end
    
    @testset "Edge Case Correctness" begin
        @testset "Single row datasets" begin
            # Test with minimal dataset
            # Note: Single row with categorical will fail because StatsModels
            # needs at least 2 levels to compute contrasts
            small_df = df[1:1, :]
            small_data = Tables.columntable(small_df)
            
            # Use formula without categorical variables for single-row test
            model = lm(@formula(continuous_response ~ x + y), small_df)
            compiled = compile_formula(model, small_data)
            
            result = modelrow(compiled, small_data, 1)
            expected = modelmatrix(model)[1, :]
            @test isapprox(result, expected, rtol=1e-10)
        end
        
        @testset "Extreme values correctness" begin
            # Test with some extreme values
            extreme_df = copy(df[1:10, :])
            extreme_df.x = [1e10, -1e10, 0.0, 1e-10, -1e-10, Inf, -Inf, 1e6, -1e6, 42.0]
            extreme_df.z = abs.(extreme_df.x) .+ 1e-6  # Ensure positive for log
            
            # Remove infinite values for log safety
            extreme_df.x[extreme_df.x .== Inf] .= 1e10
            extreme_df.x[extreme_df.x .== -Inf] .= -1e10
            
            extreme_data = Tables.columntable(extreme_df)
            
            model = lm(@formula(continuous_response ~ x + log(z)), extreme_df)
            compiled = compile_formula(model, extreme_data)
            expected_matrix = modelmatrix(model)
            
            for i in 1:nrow(extreme_df)
                if all(isfinite.(expected_matrix[i, :]))  # Skip rows with non-finite values
                    result = modelrow(compiled, extreme_data, i)
                    expected = expected_matrix[i, :]
                    @test isapprox(result, expected, rtol=1e-10)
                end
            end
        end
    end
end
