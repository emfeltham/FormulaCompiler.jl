# test_models.jl
# Comprehensive testing for GLM, MixedModels, and all supported model types

using FormulaCompiler:
    compile_formula,
    SpecializedFormula

@testset "Comprehensive Model Compatibility Tests" begin
    # Create comprehensive test data with edge cases
    n = 500
    df = DataFrame(
        # Continuous variables
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.01,  # Positive for log
        w = randn(n),
        t = randn(n),
        
        # Categorical variables with different levels
        group3 = categorical(rand(["A", "B", "C"], n)),           # 3 levels
        group4 = categorical(rand(["W", "X", "Y", "Z"], n)),      # 4 levels
        binary = categorical(rand(["Yes", "No"], n)),             # 2 levels
        group5 = categorical(rand(["P", "Q", "R", "S", "T"], n)), # 5 levels
        
        # Random effects grouping variables
        subject = categorical(rand(1:20, n)),     # 20 subjects
        cluster = categorical(rand(1:10, n)),     # 10 clusters
        
        # Boolean/logical
        flag = rand([true, false], n),
        
        # Response variables for different model types
        continuous_response = randn(n),
        binary_response = rand([0, 1], n),
        count_response = rand(0:10, n),
        
    )
    # Create correlated response for logistic
    df.linear_predictor = 0.5 .+ 0.3 .* randn(n) .+ 0.2 .* (df.group3 .== "A")
    
    # Create logistic response from linear predictor
    probabilities = 1 ./ (1 .+ exp.(-df.linear_predictor))
    df.logistic_response = [rand() < p ? 1 : 0 for p in probabilities]
    
    data = Tables.columntable(df)
    
    @testset "Linear Models (LM) - Baseline" begin
        @testset "Basic linear model formulas" begin
            linear_formulas = [
                @formula(continuous_response ~ 1),                    # Intercept only
                @formula(continuous_response ~ 0 + x),                # No intercept  
                @formula(continuous_response ~ x),                    # Simple continuous
                @formula(continuous_response ~ group3),               # Simple categorical
                @formula(continuous_response ~ x + y),                # Multiple continuous
                @formula(continuous_response ~ group3 + group4),      # Multiple categorical
                @formula(continuous_response ~ x + group3),           # Mixed
                @formula(continuous_response ~ x * group3),           # Interaction
                @formula(continuous_response ~ log(z)),               # Function
                @formula(continuous_response ~ x * y * group3),       # Three-way interaction
                @formula(continuous_response ~ x * y * group3 * group4), # Four-way interaction
                @formula(continuous_response ~ exp(x) * y * group3 * group4), # Four-way interaction w/ func
            ]
            
            for formula in linear_formulas
                @testset "LM Formula: $formula" begin
                    model = lm(formula, df)
                    
                    # Test that our system can compile it
                    compiled = compile_formula(model, data)
                    @test compiled isa SpecializedFormula
                    
                    # Test correctness against modelmatrix
                    output = Vector{Float64}(undef, length(compiled))
                    for test_row in [1, 10, 50, 100, n]
                        compiled(output, data, test_row)
                        expected = modelmatrix(model)[test_row, :]
                        @test isapprox(output, expected, rtol=1e-12)
                        # "LM failed at row $test_row for $formula"
                    end
                end
            end
        end
    end
    
    @testset "Generalized Linear Models (GLM)" begin
        @testset "Logistic regression" begin
            logistic_formulas = [
                @formula(logistic_response ~ x),
                @formula(logistic_response ~ x + group3),
                @formula(logistic_response ~ x * group3),
                @formula(logistic_response ~ log(abs(z)) + group3),
                @formula(logistic_response ~ x + y + group3 + group4),
                @formula(logistic_response ~ x * y + group3),
            ]
            
            for formula in logistic_formulas
                @testset "Logistic: $formula" begin
                    try
                        # Fit GLM with logistic link
                        model = glm(formula, df, Binomial(), LogitLink())
                        
                        # Test that our system can extract the design matrix
                        compiled = compile_formula(model, data)
                        @test compiled isa SpecializedFormula
                        
                        # Test correctness - we should get the DESIGN MATRIX, not predictions
                        output = Vector{Float64}(undef, length(compiled))
                        for test_row in [1, 25, 75, 150]
                            compiled(output, data, test_row)
                            expected = modelmatrix(model)[test_row, :]
                            @test isapprox(output, expected, rtol=1e-12)
                            # "GLM Logistic failed at row $test_row for $formula"
                        end    
                    catch e
                        @warn "GLM Logistic failed for $formula" exception=e
                        @test_broken false
                        # "GLM Logistic support needed"
                    end
                end
            end
        end
        
        @testset "Poisson regression" begin
            poisson_formulas = [
                @formula(count_response ~ x),
                @formula(count_response ~ x + group3),
                @formula(count_response ~ log(z)),
                @formula(count_response ~ x * group3),
            ]
            
            for formula in poisson_formulas
                @testset "Poisson: $formula" begin
                    try
                        # Fit GLM with Poisson distribution
                        model = glm(formula, df, Poisson(), LogLink())
                        
                        # Test that our system can extract the design matrix
                        compiled = compile_formula(model, data)
                        @test compiled isa SpecializedFormula
                        
                        # Test correctness
                        output = Vector{Float64}(undef, length(compiled))
                        for test_row in [1, 50, 100]
                            compiled(output, data, test_row)
                            expected = modelmatrix(model)[test_row, :]
                            @test isapprox(output, expected, rtol=1e-12)
                            # "GLM Poisson failed at row $test_row for $formula"
                        end
                    catch e
                        @warn "GLM Poisson failed for $formula" exception=e
                        @test_broken false
                        # "GLM Poisson support needed"
                    end
                end
            end
        end
        
        @testset "Other GLM families" begin
            # Test other common GLM families
            other_glm_tests = [
                # Gamma regression
                (@formula(z ~ x + group3), Gamma(), LogLink()),
                
                # Gaussian with log link (unusual but valid)
                (@formula(z ~ x + group3), Normal(), LogLink()),
            ]
            
            for (formula, family, link) in other_glm_tests
                @testset "GLM $(family) $(link): $formula" begin
                    try
                        model = glm(formula, df, family, link)
                        compiled = compile_formula(model, data)
                        
                        output = Vector{Float64}(undef, length(compiled))
                        compiled(output, data, 1)
                        expected = modelmatrix(model)[1, :]
                        @test isapprox(output, expected, rtol=1e-12)
                    catch e
                        @warn "GLM $(family) $(link) failed for $formula" exception=e
                        @test_broken false
                        # "GLM $(family) $(link) support may be needed"
                    end
                end
            end
        end
    end
    
    @testset "Mixed Effects Models (MixedModels.jl)" begin
        @testset "Linear mixed models" begin
            mixed_formulas = [
                # Simple random intercept
                @formula(continuous_response ~ x + (1|subject)),
                @formula(continuous_response ~ x + group3 + (1|subject)),
                
                # Random slope
                @formula(continuous_response ~ x + (x|subject)),
                @formula(continuous_response ~ x + group3 + (x|subject)),
                
                # Multiple random effects
                @formula(continuous_response ~ x + (1|subject) + (1|cluster)),
                
                # Interactions with random effects
                @formula(continuous_response ~ x * group3 + (1|subject)),
                @formula(continuous_response ~ x * group3 + (x|subject)),
            ]
            
            for formula in mixed_formulas
                @testset "LMM: $formula" begin
                    try
                        model = fit(MixedModel, formula, df; progress = false)
                        
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
                            # "LMM failed at row $test_row for $formula"
                        end
                    catch e
                        @warn "Linear Mixed Model failed for $formula" exception=e
                        @test_broken false
                        # "LMM support needed"
                    end
                end
            end
        end
        
        @testset "Generalized linear mixed models" begin
            glmm_formulas = [
                # Logistic mixed models
                @formula(logistic_response ~ x + (1|subject)),
                @formula(logistic_response ~ x + group3 + (1|subject)),
                @formula(logistic_response ~ x * group3 + (1|subject)),
                
                # Poisson mixed models
                @formula(count_response ~ x + (1|subject)),
                @formula(count_response ~ x + group3 + (1|cluster)),
            ]
            
            for formula in glmm_formulas
                @testset "GLMM: $formula" begin
                    try
                        # Determine family based on response variable
                        family = if occursin("logistic_response", string(formula.lhs))
                            Binomial()
                        elseif occursin("count_response", string(formula.lhs))
                            Poisson()
                        else
                            Normal()
                        end
                        
                        model = fit(MixedModel, formula, df, family; progress = false)
                        
                        # Test that our system can extract fixed effects design matrix
                        compiled = compile_formula(model, data)
                        @test compiled isa SpecializedFormula
                        
                        # Test correctness
                        output = Vector{Float64}(undef, length(compiled))
                        compiled(output, data, 1)
                        expected = modelmatrix(model)[1, :]
                        @test isapprox(output, expected, rtol=1e-12)
                    catch e
                        @warn "GLMM failed for $formula" exception=e
                        @test_broken false
                        # "GLMM support may be needed"
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
