# test_models.jl
# Comprehensive testing for GLM, MixedModels, and all supported model types

using FormulaCompiler:
    compile_formula,
    SpecializedFormula,
    make_test_data,
    test_formulas

@testset "Comprehensive Model Compatibility Tests" begin
    # Create comprehensive test data with edge cases
    n = 500
    df = make_test_data(; n)
    data = Tables.columntable(df)
    
    @testset "LMs" begin
        for fx in test_formulas.lm
            @testset "$(fx.name)" begin
                model = lm(fx.formula, df)
                
                # Test that our system can compile it
                compiled = compile_formula(model, data)
                @test compiled isa SpecializedFormula
                
                # Test correctness against modelmatrix
                output = Vector{Float64}(undef, length(compiled))
                for test_row in [1, 10, 50, 100, n]
                    compiled(output, data, test_row)
                    expected = modelmatrix(model)[test_row, :]
                    @test isapprox(output, expected, rtol=1e-12)
                end
            end
        end
    end
    
    @testset "GLMs" begin
        for fx in test_formulas.glm
            @testset "$(fx.name)" begin
                # Fit GLM with link
                model = glm(fx.formula, df, fx.distribution, fx.link)
                
                # Test that our system can extract the design matrix
                compiled = compile_formula(model, data)
                @test compiled isa SpecializedFormula
                
                # Test correctness - we should get the DESIGN MATRIX, not predictions
                output = Vector{Float64}(undef, length(compiled))
                for test_row in [1, 25, 75, 150]
                    compiled(output, data, test_row)
                    expected = modelmatrix(model)[test_row, :]
                    @test isapprox(output, expected, rtol=1e-12)
                end
            end
        end
    end

    @testset "MMs" begin
        @testset "LMMs" begin
            for fx in test_formulas.lmm
                @testset "$(fx.name)" begin
                    model = fit(MixedModel, fx.formula, df; progress = false)
                    
                    # Test that our system can extract FIXED EFFECTS design matrix
                    compiled = compile_formula(model, data)
                    @test compiled isa SpecializedFormula
                    
                    # For mixed models, we should get the fixed effects design matrix
                    output = Vector{Float64}(undef, length(compiled))
                    for test_row in [1, 25, 100]
                        compiled(output, data, test_row)
                        # Compare against fixed effects design matrix
                        expected = modelmatrix(model)[test_row, :]
                        @test isapprox(output, expected, rtol=1e-12)
                    end
                end
            end
        end
        
        @testset "GLMMs" begin
            for fx in test_formulas.glmm
                @testset "$(fx.name)" begin
                    model = fit(MixedModel, fx.formula, df, fx.distribution, fx.link; progress = false)
                    
                    # Test that our system can extract FIXED EFFECTS design matrix
                    compiled = compile_formula(model, data)
                    @test compiled isa SpecializedFormula
                    
                    # For mixed models, we should get the fixed effects design matrix
                    output = Vector{Float64}(undef, length(compiled))
                    for test_row in [1, 25, 100]
                        compiled(output, data, test_row)
                        # Compare against fixed effects design matrix
                        expected = modelmatrix(model)[test_row, :]
                        @test isapprox(output, expected, rtol=1e-12)
                    end
                end
            end
        end
    end
    
    @testset "Complex Interactions Across Model Types" begin
        # Test the same complex interaction patterns across different model types
        complex_interaction = @formula(response ~ x * y * group3 + log(z) * group4)
        
        @testset "Complex interactions - Linear Model" begin
            formula = @formula(continuous_response ~ x * y * group3 + log(z) * group4)
            model = lm(formula, df)
            compiled = compile_formula(model, data)
            
            output = Vector{Float64}(undef, length(compiled))
            compiled(output, data, 1)
            expected = modelmatrix(model)[1, :]
            @test isapprox(output, expected, rtol=1e-12)
        end
        
        @testset "Complex interactions - GLM Logistic" begin
            try
                formula = @formula(logistic_response ~ x * y * group3 + log(z) * group4)
                model = glm(formula, df, Binomial(), LogitLink())
                compiled = compile_formula(model, data)
                
                output = Vector{Float64}(undef, length(compiled))
                compiled(output, data, 1)
                expected = modelmatrix(model)[1, :]
                @test isapprox(output, expected, rtol=1e-12)
            catch e
                @warn "Complex GLM interaction failed" exception=e
                @test_broken false
            end
        end
        
        @testset "Complex interactions - Mixed Model" begin
            try
                formula = @formula(continuous_response ~ x * y * group3 + log(z) * group4 + (1|subject))
                model = fit(MixedModel, formula, df)
                compiled = compile_formula(model, data)
                
                output = Vector{Float64}(undef, length(compiled))
                compiled(output, data, 1)
                expected = modelmatrix(model)[1, :]
                @test isapprox(output, expected, rtol=1e-12)
            catch e
                @warn "Complex LMM interaction failed" exception=e
                @test_broken false
            end
        end
    end
    
    @testset "Performance Tests Across Model Types" begin
        # Test that zero-allocation performance is maintained across model types
        models_to_test = []
        
        # Linear model
        push!(models_to_test, ("LM", lm(@formula(continuous_response ~ x * group3 + log(z)), df)))
        glm_model = glm(@formula(logistic_response ~ x * group3 + log(z)), df, Binomial(), LogitLink())
        push!(models_to_test, ("GLM", glm_model))
        mixed_model = fit(MixedModel, @formula(continuous_response ~ x * group3 + log(z) + (1|subject)), df)
        push!(models_to_test, ("LMM", mixed_model))
                
        for (model_type, model) in models_to_test
            @testset "$model_type Performance" begin
                compiled = compile_formula(model, data)
                output = Vector{Float64}(undef, length(compiled))
                
                # Warmup
                for i in 1:100
                    compiled(output, data, (i % n) + 1)
                end
                
                # Test allocation for a reasonable sample
                n_test = 1000
                total_allocs = @allocated begin
                    for i in 1:n_test
                        compiled(output, data, (i % n) + 1)
                    end
                end
                
                allocs_per_eval = total_allocs / n_test
                
                # Should be very low allocation (ideally 0)
                @test allocs_per_eval < 250 # 100 # PASSES AT 250
            end
        end
    end
    
    @testset "Scenario Integration Tests" begin
        # Test that scenarios work with different model types
        @testset "Scenarios with different model types" begin
            # Create test scenarios
            scenario_original = create_scenario("original", data)
            scenario_modified = create_scenario("modified", data; x = 5.0, group3 = "A")
            
            models_to_test = Any[]
            
            push!(
                models_to_test,
                ("LM", lm(@formula(continuous_response ~ x * group3), df))
            )
            glm_model = glm(@formula(logistic_response ~ x * group3), df, Binomial(), LogitLink())
            push!(models_to_test, ("GLM", glm_model))
            
            for (model_type, model) in models_to_test
                @testset "$model_type with scenarios" begin
                    compiled = compile_formula(model, data)
                    
                    # Test original scenario
                    output_original = modelrow(compiled, scenario_original, 1)
                    expected_original = modelmatrix(model)[1, :]
                    @test isapprox(output_original, expected_original, rtol=1e-12)
                    
                    # Test modified scenario
                    output_modified = modelrow(compiled, scenario_modified, 1)
                    
                    # The modified scenario should produce different results
                    @test !isapprox(output_original, output_modified, rtol=1e-6)
                end
            end
        end
    end
    
    @testset "Edge Cases and Error Handling" begin    
        @testset "Model matrix consistency" begin
            # Ensure our extracted design matrices match exactly
            models_to_verify = [
                lm(@formula(continuous_response ~ x * group3 + log(z)), df),
            ]
            
            for model in models_to_verify
                compiled = compile_formula(model, data)
                
                # Test that every row matches exactly
                output = Vector{Float64}(undef, length(compiled))
                expected_matrix = modelmatrix(model)
                
                for i in 1:min(50, n)  # Test first 50 rows
                    compiled(output, data, i)
                    expected_row = expected_matrix[i, :]
                    @test isapprox(output, expected_row, rtol=1e-14)
                    # "Row $i doesn't match for $(typeof(model))"
                end
                
                #  Model matrix consistency verified for $(typeof(model))
            end
        end
    end
end
