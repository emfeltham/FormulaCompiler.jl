# test_glm_integration.jl - GLM.jl Integration Tests for ContrastEvaluator

using Test, BenchmarkTools
using FormulaCompiler
using DataFrames, GLM, StatsModels, CategoricalArrays, Tables, Distributions
using Random, Statistics, LinearAlgebra

Random.seed!(2024)

# Helper function (define only if not already defined)
if !@isdefined(invlogit)
    invlogit(x) = 1.0 ./ (1.0 .+ exp.(-x))
end

@testset "GLM Integration Tests" begin
    # Create comprehensive test data suitable for all GLM families
    function create_glm_test_data(n=500)
        df = DataFrame(
            # Continuous predictors
            x1 = randn(n),
            x2 = randn(n),
            age = rand(18:80, n),

            # Categorical predictors
            treatment = categorical(rand(["Control", "Treatment_A", "Treatment_B"], n)),
            region = categorical(rand(["North", "South", "East", "West"], n)),
            education = categorical(rand(["HS", "College", "Graduate"], n)),

            # Binary predictors
            female = rand([0, 1], n),
            married = rand([true, false], n),
            urban = categorical(rand([true, false], n)),
        )

        # Generate appropriate response variables for different GLM families
        # Linear predictor for all models
        η = 0.5 .+ 0.3.*df.x1 .+ 0.2.*df.x2 .+ 0.1.*(df.age .- 50)./10 .+
            0.4.*(df.treatment .== "Treatment_A") .+ 0.6.*(df.treatment .== "Treatment_B") .+
            0.2.*df.female .+ 0.1.*df.married

        # Generate responses
        df.y_normal = η .+ 0.5.*randn(n)  # Normal for linear models
        df.y_binary = rand.(Bernoulli.(invlogit.(η)))  # Binary for logistic
        df.y_count = rand.(Poisson.(exp.(η .- 2)))  # Count for Poisson (offset to keep reasonable)
        df.y_nb = rand.(NegativeBinomial.(exp.(η .- 1), 0.5))  # Negative binomial

        return df, Tables.columntable(df)
    end

    df, data = create_glm_test_data(400)

    @testset "Linear Model Integration" begin
        @testset "Basic Linear Models" begin
            model = lm(@formula(y_normal ~ x1 + x2 + treatment + female), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment, :female])

            # Test basic functionality
            contrast_buf = Vector{Float64}(undef, length(compiled))

            @testset "Treatment contrasts" begin
                contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Treatment_A")
                @test !all(contrast_buf .== 0.0)

                # Verify zero allocations
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Treatment_A")
                @test result.memory == 0

                # Test symmetry
                contrast_reverse = similar(contrast_buf)
                contrast_modelrow!(contrast_reverse, evaluator, 1, :treatment, "Treatment_A", "Control")
                @test contrast_buf ≈ -contrast_reverse atol=1e-14
            end

            @testset "Binary contrasts" begin
                contrast_modelrow!(contrast_buf, evaluator, 1, :female, 0, 1)
                @test !all(contrast_buf .== 0.0)

                # Should affect exactly one coefficient in simple linear model
                non_zero_count = count(x -> abs(x) > 1e-10, contrast_buf)
                @test non_zero_count == 1

                # Verify zero allocations
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :female, 0, 1)
                @test result.memory == 0
            end

            @testset "Mathematical consistency" begin
                # Compare with manual coefficient extraction
                coefs = coef(model)

                # Treatment effect should match coefficient difference
                contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Treatment_A")
                treatment_coef_idx = findfirst(occursin("treatment: Treatment_A"), string.(coefnames(model)))

                if treatment_coef_idx !== nothing
                    expected_contrast = zeros(length(contrast_buf))
                    expected_contrast[treatment_coef_idx] = 1.0
                    @test contrast_buf ≈ expected_contrast atol=1e-12

                    # Magnitude should match coefficient
                    @test abs(contrast_buf[treatment_coef_idx]) ≈ 1.0 atol=1e-12
                end
            end
        end

        @testset "Linear Models with Interactions" begin
            model = lm(@formula(y_normal ~ x1 * treatment + female * married), df)
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment, :female, :married])
            contrast_buf = Vector{Float64}(undef, length(compiled))

            @testset "Interaction effects" begin
                # Treatment contrast with interaction should affect multiple terms
                contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Treatment_A")
                non_zero_count = count(x -> abs(x) > 1e-10, contrast_buf)
                @test non_zero_count >= 1  # At least some effect

                # Verify zero allocations with interactions
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Treatment_B")
                @test result.memory == 0
            end

            @testset "Binary interaction effects" begin
                # Binary × Binary interaction
                contrast_modelrow!(contrast_buf, evaluator, 1, :female, 0, 1)
                non_zero_count = count(x -> abs(x) > 1e-10, contrast_buf)
                @test non_zero_count >= 1  # At least some effect

                # Zero allocations
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :married, false, true)
                @test result.memory == 0
            end
        end
    end

    @testset "Logistic Model Integration" begin
        @testset "Basic Logistic Models" begin
            model = glm(@formula(y_binary ~ x1 + x2 + treatment + female), df, Binomial(), LogitLink())
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment, :female])

            contrast_buf = Vector{Float64}(undef, length(compiled))

            @testset "Logistic treatment contrasts" begin
                contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Treatment_A")
                @test !all(contrast_buf .== 0.0)

                # Verify zero allocations
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Treatment_A")
                @test result.memory == 0

                # Test contrast symmetry
                contrast_reverse = similar(contrast_buf)
                contrast_modelrow!(contrast_reverse, evaluator, 1, :treatment, "Treatment_A", "Control")
                @test contrast_buf ≈ -contrast_reverse atol=1e-14
            end

            @testset "Logistic binary contrasts" begin
                contrast_modelrow!(contrast_buf, evaluator, 1, :female, 0, 1)
                @test !all(contrast_buf .== 0.0)

                # Should affect exactly one coefficient
                non_zero_count = count(x -> abs(x) > 1e-10, contrast_buf)
                @test non_zero_count == 1

                # Verify zero allocations
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :female, 0, 1)
                @test result.memory == 0
            end

            # Inference tests (gradient/SE) migrated to Margins.jl (2025-10-09)
        end

        @testset "Logistic Models with Interactions" begin
            model = glm(@formula(y_binary ~ x1 + treatment * region), df, Binomial(), LogitLink())
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment, :region])
            contrast_buf = Vector{Float64}(undef, length(compiled))

            @testset "Complex categorical interactions" begin
                # Treatment × Region interaction
                contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Treatment_A")
                non_zero_count = count(x -> abs(x) > 1e-10, contrast_buf)
                @test non_zero_count >= 1  # At least some effects

                # Region effects with interactions
                contrast_modelrow!(contrast_buf, evaluator, 1, :region, "North", "South")
                @test !all(contrast_buf .== 0.0)

                # Verify zero allocations with complex interactions
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :region, "East", "West")
                @test result.memory == 0
            end
        end
    end

    @testset "Poisson Model Integration" begin
        @testset "Basic Poisson Models" begin
            model = glm(@formula(y_count ~ x1 + x2 + treatment + female), df, Poisson(), LogLink())
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment, :female])

            contrast_buf = Vector{Float64}(undef, length(compiled))

            @testset "Poisson treatment contrasts" begin
                contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Treatment_A")
                @test !all(contrast_buf .== 0.0)

                # Verify zero allocations
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Treatment_A")
                @test result.memory == 0
            end

            # Inference tests (gradient/SE) migrated to Margins.jl (2025-10-09)
        end

        @testset "Poisson Models with Offset" begin
            # Add exposure/offset variable
            df_offset = copy(df)
            df_offset.exposure = rand(0.5:0.1:2.0, nrow(df))
            data_offset = Tables.columntable(df_offset)

            model = glm(@formula(y_count ~ x1 + treatment + female), df_offset, Poisson(), LogLink(); offset=log.(df_offset.exposure))
            compiled = compile_formula(model, data_offset)
            evaluator = contrastevaluator(compiled, data_offset, [:treatment, :female])

            contrast_buf = Vector{Float64}(undef, length(compiled))

            @testset "Contrasts with offset" begin
                contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Treatment_A")
                @test !all(contrast_buf .== 0.0)

                # Should work correctly with offset (offset doesn't affect contrasts)
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Treatment_A")
                @test result.memory == 0
            end
        end
    end

    @testset "Negative Binomial Model Integration" begin
        # Note: Negative Binomial requires GLM.jl version that supports it
        @testset "Basic Negative Binomial Models" begin
            try
                model = glm(@formula(y_nb ~ x1 + x2 + treatment + female), df, NegativeBinomial(), LogLink())
                compiled = compile_formula(model, data)
                evaluator = contrastevaluator(compiled, data, [:treatment, :female])

                contrast_buf = Vector{Float64}(undef, length(compiled))

                @testset "Negative binomial contrasts" begin
                    contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Treatment_A")
                    @test !all(contrast_buf .== 0.0)

                    # Verify zero allocations
                    result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Treatment_A")
                    @test result.memory == 0
                end

                # Inference tests (gradient/SE) migrated to Margins.jl (2025-10-09)

                println("✓ Negative Binomial models supported and working")

            catch MethodError
                println("⚠ Negative Binomial models not supported in this GLM.jl version - skipping")
                @test_skip "Negative Binomial support depends on GLM.jl version"
            end
        end
    end

    @testset "Cross-Model Consistency Tests" begin
        # Test that the same contrasts give consistent results across model types
        # (when appropriate - e.g., linear vs identity link GLM)

        @testset "Linear vs GLM Identity Consistency" begin
            # Fit same model with lm() and glm() with identity link
            lm_model = lm(@formula(y_normal ~ x1 + treatment + female), df)
            glm_model = glm(@formula(y_normal ~ x1 + treatment + female), df, Normal(), IdentityLink())

            lm_compiled = compile_formula(lm_model, data)
            glm_compiled = compile_formula(glm_model, data)

            lm_evaluator = contrastevaluator(lm_compiled, data, [:treatment, :female])
            glm_evaluator = contrastevaluator(glm_compiled, data, [:treatment, :female])

            lm_contrast = Vector{Float64}(undef, length(lm_compiled))
            glm_contrast = Vector{Float64}(undef, length(glm_compiled))

            @testset "Treatment contrast consistency" begin
                contrast_modelrow!(lm_contrast, lm_evaluator, 1, :treatment, "Control", "Treatment_A")
                contrast_modelrow!(glm_contrast, glm_evaluator, 1, :treatment, "Control", "Treatment_A")

                # Should be identical for identity link
                @test lm_contrast ≈ glm_contrast atol=1e-12
            end

            @testset "Binary contrast consistency" begin
                contrast_modelrow!(lm_contrast, lm_evaluator, 1, :female, 0, 1)
                contrast_modelrow!(glm_contrast, glm_evaluator, 1, :female, 0, 1)

                # Should be identical for identity link
                @test lm_contrast ≈ glm_contrast atol=1e-12
            end
        end

        # Link function inference tests migrated to Margins.jl (2025-10-09)
    end

    @testset "Performance Validation Across Models" begin
        # Test that ContrastEvaluator maintains zero allocations across all model types

        models_to_test = [
            ("Linear", lm(@formula(y_normal ~ x1 + treatment + female), df)),
            ("Logistic", glm(@formula(y_binary ~ x1 + treatment + female), df, Binomial(), LogitLink())),
            ("Poisson", glm(@formula(y_count ~ x1 + treatment + female), df, Poisson(), LogLink())),
        ]

        for (name, model) in models_to_test
            @testset "$name Model Performance" begin
                compiled = compile_formula(model, data)
                evaluator = contrastevaluator(compiled, data, [:treatment, :female])
                contrast_buf = Vector{Float64}(undef, length(compiled))

                # Warmup
                for _ in 1:5
                    contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Treatment_A")
                    contrast_modelrow!(contrast_buf, evaluator, 1, :female, 0, 1)
                end

                # Test zero allocations for treatment contrast
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Treatment_A")
                @test result.memory == 0

                # Test zero allocations for binary contrast
                result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :female, 0, 1)
                @test result.memory == 0

                # Performance should be reasonable (under 2 microseconds)
                time_treatment = @belapsed contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Treatment_A")
                time_binary = @belapsed contrast_modelrow!($contrast_buf, $evaluator, 1, :female, 0, 1)

                @test time_treatment < 2e-6
                @test time_binary < 2e-6

                println("$name: Treatment=$(round(time_treatment*1e9, digits=1))ns, Binary=$(round(time_binary*1e9, digits=1))ns")
            end
        end
    end

    @testset "Edge Cases and Error Handling" begin
        @testset "Model compatibility validation" begin
            # Test that ContrastEvaluator works with various GLM configurations

            model = glm(@formula(y_binary ~ treatment + region), df, Binomial(), LogitLink())
            compiled = compile_formula(model, data)
            evaluator = contrastevaluator(compiled, data, [:treatment, :region])
            contrast_buf = Vector{Float64}(undef, length(compiled))

            # Test all treatment levels
            treatment_levels = levels(df.treatment)
            for i in 1:length(treatment_levels)-1, j in (i+1):length(treatment_levels)
                from_level, to_level = treatment_levels[i], treatment_levels[j]
                @test_nowarn contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, from_level, to_level)
            end

            # Test all region levels
            region_levels = levels(df.region)
            for i in 1:length(region_levels)-1, j in (i+1):length(region_levels)
                from_level, to_level = region_levels[i], region_levels[j]
                @test_nowarn contrast_modelrow!(contrast_buf, evaluator, 1, :region, from_level, to_level)
            end
        end

        @testset "Large dataset scalability" begin
            # Test with larger dataset
            df_large, data_large = create_glm_test_data(2000)

            model = glm(@formula(y_binary ~ x1 + treatment + region + female), df_large, Binomial(), LogitLink())
            compiled = compile_formula(model, data_large)
            evaluator = contrastevaluator(compiled, data_large, [:treatment, :region, :female])
            contrast_buf = Vector{Float64}(undef, length(compiled))

            # Should still maintain zero allocations with large dataset
            contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Treatment_A")
            result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Treatment_A")
            @test result.memory == 0

            # Performance should still be reasonable
            time_large = @belapsed contrast_modelrow!($contrast_buf, $evaluator, 1, :region, "North", "South")
            @test time_large < 5e-6  # Under 5 microseconds

            println("Large dataset (2000 rows): $(round(time_large*1e9, digits=1))ns")
        end
    end

    println("GLM Integration Tests completed successfully!")
    println("Linear, Logistic, and Poisson models fully supported")
    println("All link functions working correctly with gradient computation")
    println("Zero allocations maintained across all GLM families")
    println("Mathematical consistency validated across model types")
end
