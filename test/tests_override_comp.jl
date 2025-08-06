# tests_override_comp.jl

using Revise
using Test
using BenchmarkTools, Profile
using FormulaCompiler

using Statistics
using DataFrames, GLM, Tables, CategoricalArrays, Random
using StatsModels, StandardizedPredictors
using MixedModels
using BenchmarkTools

# Set consistent random seed for reproducible tests
Random.seed!(06515)

@testset "Comprehensive Override Tests - Functions & Interactions" begin
    
    # Setup test data
    df, data = test_data(n=200)
    
    @testset "1. Simple Function Override - log(continuous)" begin
        # Test basic function with continuous override
        fx = @formula(response ~ x + log(z))
        model = lm(fx, df)
        
        test_cases = [
            ("log_small", Dict(:z => 0.1)),    # log(0.1) = -2.3
            ("log_one", Dict(:z => 1.0)),      # log(1.0) = 0.0
            ("log_large", Dict(:z => 10.0)),   # log(10.0) = 2.3
        ]
        
        for (name, overrides) in test_cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test multiple rows
                for row_idx in [1, 25, 50]
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, scenario.data, row_idx)
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10
                    
                    # Verify log computation specifically
                    expected_log = log(overrides[:z])
                    @test output[3] ≈ expected_log atol=1e-10  # log(z) is 3rd column
                end
            end
        end
    end
    
    @testset "2. Multiple Function Override - exp(x) + sqrt(z)" begin
        # Test multiple functions with overrides
        fx = @formula(response ~ exp(x) + sqrt(z) + y)
        model = lm(fx, df)
        
        test_cases = [
            ("both_small", Dict(:x => -1.0, :z => 0.25)),  # exp(-1)≈0.37, sqrt(0.25)=0.5
            ("both_large", Dict(:x => 2.0, :z => 9.0)),    # exp(2)≈7.39, sqrt(9)=3.0
            ("mixed", Dict(:x => 0.0, :z => 1.0)),         # exp(0)=1.0, sqrt(1)=1.0
        ]
        
        for (name, overrides) in test_cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness
                for row_idx in [1, 100]
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, scenario.data, row_idx)
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10
                end
            end
        end
    end
    
    @testset "3. Nested Function Override - log(sqrt(z))" begin
        # Test nested functions with override
        fx = @formula(response ~ x + log(sqrt(z)) + y)
        model = lm(fx, df)
        
        test_cases = [
            ("nested_small", Dict(:z => 0.01)),  # log(sqrt(0.01)) = log(0.1) ≈ -2.3
            ("nested_one", Dict(:z => 1.0)),     # log(sqrt(1.0)) = log(1.0) = 0.0
            ("nested_large", Dict(:z => 100.0)), # log(sqrt(100)) = log(10) ≈ 2.3
        ]
        
        for (name, overrides) in test_cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness
                output = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(output, scenario.data, 1)
                @test output ≈ ref_mm[1, :] atol=1e-10
                
                # Verify nested computation
                expected_nested = log(sqrt(overrides[:z]))
                @test output[3] ≈ expected_nested atol=1e-10
            end
        end
    end
    
    @testset "4. Simple Interaction Override - x * group2" begin
        # Test continuous × categorical interaction
        fx = @formula(response ~ x + group2 + x * group2)
        model = lm(fx, df)
        
        test_cases = [
            ("x_zero_Z", Dict(:x => 0.0, :group2 => "Z")),
            ("x_pos_M", Dict(:x => 2.0, :group2 => "M")),
            ("x_neg_L", Dict(:x => -1.5, :group2 => "L")),
        ]
        
        for (name, overrides) in test_cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness
                for row_idx in [1, 50, 100]
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, scenario.data, row_idx)
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10
                end
                
                # All rows should be identical (all variables overridden)
                out1 = Vector{Float64}(undef, size(ref_mm, 2))
                out2 = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(out1, scenario.data, 1)
                compiled(out2, scenario.data, 100)
                @test out1 ≈ out2 atol=1e-12
            end
        end
    end
    
    @testset "5. Categorical × Categorical Interaction - group2 * group3" begin
        # Test categorical × categorical interaction
        fx = @formula(response ~ group2 + group3 + group2 * group3)
        model = lm(fx, df)
        
        test_cases = [
            ("ref_ref", Dict(:group2 => "Z", :group3 => "A")),  # Both reference levels
            ("non_ref", Dict(:group2 => "M", :group3 => "B")),  # Both non-reference
            ("mixed", Dict(:group2 => "L", :group3 => "C")),    # Different levels
        ]
        
        for (name, overrides) in test_cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness
                output = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(output, scenario.data, 1)
                @test output ≈ ref_mm[1, :] atol=1e-10
                
                # All rows identical
                out1 = Vector{Float64}(undef, size(ref_mm, 2))
                out2 = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(out1, scenario.data, 1)
                compiled(out2, scenario.data, 150)
                @test out1 ≈ out2 atol=1e-12
            end
        end
    end
    
    @testset "6. Function + Interaction - log(x) * group2" begin
        # Test function × categorical interaction
        fx = @formula(response ~ x + log(abs(x)) + group2 + log(abs(x)) * group2)
        # Note: Need positive x for log
        model = lm(fx, df)  # Make x positive
        
        cases = [
            ("log_small_Z", Dict(:x => 0.1, :group2 => "Z")),  # log(0.1) * Z contrasts
            ("log_one_M", Dict(:x => 1.0, :group2 => "M")),    # log(1) = 0 * M contrasts  
            ("log_large_L", Dict(:x => 10.0, :group2 => "L")), # log(10) * L contrasts
        ]
        
        for (name, overrides) in cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference - need positive x data
                ref_df = copy(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness
                output = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(output, scenario.data, 1)
                @test output ≈ ref_mm[1, :] atol=1e-10
            end
        end
    end
    
    @testset "7. Three-Way Interaction - x * group2 * group3" begin
        # Test three-way interaction with overrides
        fx = @formula(response ~ x * group2 * group3)
        model = lm(fx, df)
        
        test_cases = [
            ("all_ref", Dict(:x => 1.0, :group2 => "Z", :group3 => "A")),
            ("x_only", Dict(:x => 2.0, :group2 => "Z", :group3 => "A")),
            ("all_non_ref", Dict(:x => -1.0, :group2 => "M", :group3 => "B")),
        ]
        
        for (name, overrides) in test_cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness
                output = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(output, scenario.data, 1)
                @test output ≈ ref_mm[1, :] atol=1e-10
                
                # All rows identical (all variables overridden)
                out1 = Vector{Float64}(undef, size(ref_mm, 2))
                out2 = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(out1, scenario.data, 1)
                compiled(out2, scenario.data, 75)
                @test out1 ≈ out2 atol=1e-12
            end
        end
    end
    
    @testset "8. Complex Mixed Formula - Functions + Interactions + Multiple Categoricals" begin
        # Test very complex formula with everything
        fx = @formula(response ~ x + log(z) + group2 + group3 + x * group2 + log(z) * group3 + group2 * group3)
        model = lm(fx, df)
        
        test_cases = [
            ("complex_1", Dict(:x => 1.0, :z => 2.0, :group2 => "M", :group3 => "B")),
            ("complex_2", Dict(:x => -0.5, :z => 0.5, :group2 => "L", :group3 => "C")),
            ("complex_3", Dict(:x => 0.0, :z => 1.0, :group2 => "Z", :group3 => "A")),
        ]
        
        for (name, overrides) in test_cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness
                output = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(output, scenario.data, 1)
                @test output ≈ ref_mm[1, :] atol=1e-10
                
                # Test consistency across rows
                out1 = Vector{Float64}(undef, size(ref_mm, 2))
                out2 = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(out1, scenario.data, 1)
                compiled(out2, scenario.data, 200)
                @test out1 ≈ out2 atol=1e-12
            end
        end
    end
    
    @testset "9. Partial Overrides - Some Variables Fixed, Others Vary" begin
        # Test scenarios where only some variables are overridden
        fx = @formula(response ~ x + y + group2 + x * group2)
        model = lm(fx, df)
        
        test_cases = [
            ("x_only", Dict(:x => 2.0)),                    # Only x fixed
            ("group_only", Dict(:group2 => "M")),           # Only categorical fixed
            ("interaction_vars", Dict(:x => 1.5, :group2 => "L")),  # Both interaction variables
        ]
        
        for (name, overrides) in test_cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness for multiple rows
                for row_idx in [1, 50, 100, 150, 200]
                    output = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(output, scenario.data, row_idx)
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10
                end
                
                # If only some variables overridden, rows should differ in non-overridden parts
                if length(overrides) < 4  # Not all variables overridden
                    out1 = Vector{Float64}(undef, size(ref_mm, 2))
                    out2 = Vector{Float64}(undef, size(ref_mm, 2))
                    compiled(out1, scenario.data, 1)
                    compiled(out2, scenario.data, 50)
                    
                    # Check that overridden variables produce same values
                    for (col, val) in overrides
                        if col == :x
                            @test out1[2] ≈ out2[2] atol=1e-12  # x coefficient position
                        end
                        # Add more specific checks as needed
                    end
                end
            end
        end
    end
    
    @testset "10. Edge Cases - Extreme Values and Multiple Functions" begin
        # Test edge cases with extreme values and multiple functions
        fx = @formula(response ~ log(z) + sqrt(abs(x)) + exp(w) + group2)
        # Note: This includes abs() to handle negative x values in sqrt
        model = lm(fx, df)
        
        test_cases = [
            ("extreme_small", Dict(:z => 1e-6, :x => -100.0, :w => -10.0, :group2 => "Z")),
            ("extreme_large", Dict(:z => 1000.0, :x => 100.0, :w => 5.0, :group2 => "M")),
            ("zeros", Dict(:z => 1.0, :x => 0.0, :w => 0.0, :group2 => "L")),
            ("mixed_extreme", Dict(:z => 0.001, :x => 25.0, :w => -5.0, :group2 => "M")),
        ]
        
        for (name, overrides) in test_cases
            @testset "$name" begin
                scenario = create_scenario(name, data, overrides)
                compiled = compile_formula(model, scenario.data)
                
                # Reference
                ref_df = DataFrame(df)
                for (col, val) in overrides
                    ref_df[!, col] .= val
                end
                ref_mm = modelmatrix(model.mf; data = ref_df)
                
                # Test correctness
                output = Vector{Float64}(undef, size(ref_mm, 2))
                compiled(output, scenario.data, 1)
                @test output ≈ ref_mm[1, :] atol=1e-8  # Slightly more tolerance for extreme values
                
                # Verify specific function computations
                z_val = overrides[:z]
                x_val = overrides[:x] 
                w_val = overrides[:w]
                
                @test output[2] ≈ log(z_val) atol=1e-8         # log(z)
                @test output[3] ≈ sqrt(abs(x_val)) atol=1e-8   # sqrt(abs(x))
                @test output[4] ≈ exp(w_val) atol=1e-8         # exp(w)
                
                # Test that extreme values don't cause numerical issues
                @test all(isfinite, output)
            end
        end
    end
 
end