# test_models.jl
# Comprehensive correctness testing for all supported model types
# Tests formula compilation correctness against modelmatrix and modelrow functionality

using Test
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays, MixedModels
using StatsModels, BenchmarkTools
using FormulaCompiler: make_test_data, test_formulas
using Random

@testset "Model Correctness Tests" begin
    Random.seed!(08540)
    
    # Setup test data
    n = 500
    df = make_test_data(; n)
    data = Tables.columntable(df)
    
    # Helper function to test correctness against modelmatrix
    function test_model_correctness(model, model_name)
        compiled = compile_formula(model, data)
        output = Vector{Float64}(undef, length(compiled))
        expected_matrix = modelmatrix(model)
        
        # Test correctness against modelmatrix on multiple test rows
        test_rows = [1, 10, 50, 100, min(n, 250), n]
        for test_row in test_rows
            compiled(output, data, test_row)
            expected = expected_matrix[test_row, :]
            @test isapprox(output, expected, rtol=1e-12)
        end
        
        # Test modelrow functionality (allocating version)
        for test_row in [1, 25, n÷2, n]
            output_modelrow = modelrow(compiled, data, test_row)
            expected = expected_matrix[test_row, :]
            @test isapprox(output_modelrow, expected, rtol=1e-12)
        end
        
        # Test modelrow! functionality (in-place version)
        output_inplace = Vector{Float64}(undef, length(compiled))
        for test_row in [1, 10, n]
            modelrow!(output_inplace, compiled, data, test_row)
            expected = expected_matrix[test_row, :]
            @test isapprox(output_inplace, expected, rtol=1e-12)
        end
        
        return true
    end
    
    @testset "Linear Models (LM)" begin
        for fx in test_formulas.lm
            @testset "$(fx.name)" begin
                model = lm(fx.formula, df)
                @test test_model_correctness(model, fx.name)
                
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
                @test test_model_correctness(model, fx.name)
                
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
                @test test_model_correctness(model, fx.name)
                
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
                @test test_model_correctness(model, fx.name)
                
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
            @test test_model_correctness(model, "Complex LM")
        end
        
        @testset "Complex GLM interaction" begin
            formula = @formula(logistic_response ~ x * y * group3 + log(z) * group4)
            model = glm(formula, df, Binomial(), LogitLink())
            @test test_model_correctness(model, "Complex GLM")
        end
        
        @testset "Complex LMM interaction" begin
            formula = @formula(continuous_response ~ x * y * group3 + log(z) * group4 + (1|subject))
            model = fit(MixedModel, formula, df; progress = false)
            @test test_model_correctness(model, "Complex LMM")
        end
    end
    
    @testset "ModelRow Interface Tests" begin
        # Test ModelRowEvaluator interface
        model = lm(@formula(continuous_response ~ x * group3 + log(z)), df)
        compiled = compile_formula(model, data)
        expected_matrix = modelmatrix(model)
        
        @testset "ModelRowEvaluator creation and usage" begin
            # Test constructor with pre-allocated buffer
            evaluator = ModelRowEvaluator(compiled, data)
            
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
            small_df = df[1:1, :]
            small_data = Tables.columntable(small_df)
            
            model = lm(@formula(continuous_response ~ x + group3), small_df)
            compiled = compile_formula(model, small_data)
            
            result = modelrow(compiled, small_data, 1)
            expected = modelmatrix(model)[1, :]
            @test isapprox(result, expected, rtol=1e-12)
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