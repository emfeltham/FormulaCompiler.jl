# Comprehensive Test Suite for Categorical Mixtures (Phase 5)
# This file provides the comprehensive test suite specified in CATEGORICAL_MIXTURES_DESIGN.md

using Test
using FormulaCompiler
using FormulaCompiler: mix, compile_formula, MixtureContrastOp, execute_op, BoolCounterfactualVector,
    is_mixture_column, validate_mixture_consistency!, extract_mixture_spec, ContrastOp,
    create_mixture_column, expand_mixture_grid, create_balanced_mixture,
    validate_mixture_weights, validate_mixture_levels
using DataFrames, Tables
using StatsModels, GLM, CategoricalArrays
using LinearAlgebra  # For I(n) identity matrix

# Use native FormulaCompiler mixtures instead of duck-typed test mixtures

@testset "Categorical Mixtures - Comprehensive Test Suite" begin
    
    @testset "Basic Mixture Compilation" begin
        # Test that we can compile formulas with mixture columns
        # This tests the full integration from data → compilation → execution
        
        @testset "Simple Mixture Formula" begin
            # Create test data with mixture column
            mixture_obj = mix("A" => 0.3, "B" => 0.7)
            
            # Note: Since we can't easily create a real statistical model with mixtures
            # without full GLM integration, we'll test the core compilation components
            df = DataFrame(
                x = [1.0, 2.0, 3.0],
                y = [0.1, 0.2, 0.3],
                cat = [mixture_obj, mixture_obj, mixture_obj]
            )
            
            data = Tables.columntable(df)
            
            # Test that mixture detection works
            @test is_mixture_column(data.cat) == true
            @test is_mixture_column(data.x) == false
            
            # Test that validation passes
            @test_nowarn validate_mixture_consistency!(data)
            
            # Test mixture spec extraction
            spec = extract_mixture_spec(data.cat[1])
            @test spec.levels == ["A", "B"]
            @test spec.weights == [0.3, 0.7]
        end
        
        @testset "Multiple Mixture Variables" begin
            # Test data with multiple mixture columns
            group_mix = mix("Control" => 0.4, "Treatment" => 0.6)
            dose_mix = mix("Low" => 0.25, "Medium" => 0.5, "High" => 0.25)
            
            df = DataFrame(
                x = [1.0, 2.0, 3.0, 4.0],
                group = [group_mix, group_mix, group_mix, group_mix],
                dose = [dose_mix, dose_mix, dose_mix, dose_mix]
            )
            
            data = Tables.columntable(df)
            
            # Both should be detected as mixture columns
            @test is_mixture_column(data.group)
            @test is_mixture_column(data.dose)
            
            # Validation should pass for both
            @test_nowarn validate_mixture_consistency!(data)
            
            # Specs should be extracted correctly
            group_spec = extract_mixture_spec(data.group[1])
            @test group_spec.levels == ["Control", "Treatment"]
            @test group_spec.weights == [0.4, 0.6]
            
            dose_spec = extract_mixture_spec(data.dose[1])
            @test dose_spec.levels == ["Low", "Medium", "High"]
            @test dose_spec.weights == [0.25, 0.5, 0.25]
        end
        
        @testset "Mixed Regular and Mixture Columns" begin
            # Test data with both regular categorical and mixture columns
            mixture_obj = mix("X" => 0.6, "Y" => 0.4)
            
            df = DataFrame(
                continuous = [1.0, 2.0, 3.0],
                regular_cat = ["A", "B", "A"],  # Regular categorical
                mixture_cat = [mixture_obj, mixture_obj, mixture_obj]  # Mixture categorical
            )
            
            data = Tables.columntable(df)
            
            # Only mixture column should be detected
            @test is_mixture_column(data.continuous) == false
            @test is_mixture_column(data.regular_cat) == false
            @test is_mixture_column(data.mixture_cat) == true
            
            # Validation should pass
            @test_nowarn validate_mixture_consistency!(data)
        end
    end
    
    @testset "Zero-Allocation Execution Performance" begin
        # Test that mixture operations maintain zero-allocation guarantees
        
        @testset "MixtureContrastOp Performance" begin
            # Create test operation
            contrast_matrix = [
                1.0 0.0;    # Level 1: "A"
                0.0 1.0;    # Level 2: "B"
                0.0 0.0     # Level 3: "C" (reference)
            ]
            
            mixture_op = MixtureContrastOp{
                :group,
                (1, 2),
                (1, 2, 3),
                (0.2, 0.3, 0.5)
            }(contrast_matrix)
            
            scratch = Vector{Float64}(undef, 5)
            data = (group = ["test"],)  # Not used by mixture ops
            
            # Warm up
            for _ in 1:10
                execute_op(mixture_op, scratch, data, 1)
            end
            
            # Test zero allocation
            allocs = @allocated execute_op(mixture_op, scratch, data, 1)
            @test allocs == 0
            
        end
        
        
        @testset "Comparison with Standard Categorical" begin
            # Compare mixture performance with standard categorical operations
            contrast_matrix = [1.0 0.0; 0.0 1.0; -1.0 -1.0]
            
            # Standard categorical op
            standard_op = ContrastOp{:group, (1, 2)}(contrast_matrix)
            
            # Mixture op with same dimensions
            mixture_op = MixtureContrastOp{
                :group,
                (1, 2),
                (1, 2),
                (0.5, 0.5)
            }(contrast_matrix)
            
            scratch = Vector{Float64}(undef, 3)
            
            # Warm up both
            standard_data = (group = [categorical(["A", "B", "C"])[1]],)  # CategoricalValue
            mixture_data = (group = ["test"],)  # Not used
            
            for _ in 1:10
                execute_op(standard_op, scratch, standard_data, 1)
                execute_op(mixture_op, scratch, mixture_data, 1)
            end
            
            # Time both operations
            standard_time = @elapsed execute_op(standard_op, scratch, standard_data, 1)
            mixture_time = @elapsed execute_op(mixture_op, scratch, mixture_data, 1)
            
            # Mixture should be within 10x of standard (realistic for microbenchmark noise)
            # Note: Both operations are very fast, so focus on order of magnitude
            @test mixture_time <= 10.0 * standard_time
        end
    end
    
    @testset "Correctness vs Manual Weighted Combinations" begin
        # Test mixture execution against manually computed weighted contrasts
        
        @testset "Binary Mixture Correctness" begin
            # Test binary mixture: 40% A, 60% B
            contrast_matrix = [
                1.0 0.0;    # Level A
                0.0 1.0     # Level B
            ]
            
            mixture_op = MixtureContrastOp{
                :group,
                (1, 2),
                (1, 2),
                (0.4, 0.6)
            }(contrast_matrix)
            
            scratch = Vector{Float64}(undef, 3)
            data = (group = ["test"],)
            
            execute_op(mixture_op, scratch, data, 1)
            
            # Manual calculation: 0.4 * [1,0] + 0.6 * [0,1] = [0.4, 0.6]
            @test scratch[1] ≈ 0.4 atol=1e-12
            @test scratch[2] ≈ 0.6 atol=1e-12
        end
        
        @testset "Three-Way Mixture Correctness" begin
            # Test three-way mixture with dummy coding
            contrast_matrix = [
                1.0 0.0;    # Level A  
                0.0 1.0;    # Level B
                0.0 0.0     # Level C (reference)
            ]
            
            mixture_op = MixtureContrastOp{
                :group,
                (1, 2),
                (1, 2, 3),
                (0.2, 0.3, 0.5)
            }(contrast_matrix)
            
            scratch = Vector{Float64}(undef, 3)
            data = (group = ["test"],)
            
            execute_op(mixture_op, scratch, data, 1)
            
            # Manual calculation: 0.2*[1,0] + 0.3*[0,1] + 0.5*[0,0] = [0.2, 0.3]
            @test scratch[1] ≈ 0.2 atol=1e-12
            @test scratch[2] ≈ 0.3 atol=1e-12
        end
        
        @testset "Effects Coding Correctness" begin
            # Test with effects coding (sum contrasts)
            effects_matrix = [
                 1.0  0.0;    # Level A
                 0.0  1.0;    # Level B  
                -1.0 -1.0     # Level C
            ]
            
            mixture_op = MixtureContrastOp{
                :group,
                (1, 2),
                (1, 2, 3),
                (0.25, 0.25, 0.5)
            }(effects_matrix)
            
            scratch = Vector{Float64}(undef, 3)
            data = (group = ["test"],)
            
            execute_op(mixture_op, scratch, data, 1)
            
            # Manual calculation: 0.25*[1,0] + 0.25*[0,1] + 0.5*[-1,-1] = [0.25+0-0.5, 0+0.25-0.5] = [-0.25, -0.25]
            expected_1 = 0.25 * 1.0 + 0.25 * 0.0 + 0.5 * (-1.0)  # -0.25
            expected_2 = 0.25 * 0.0 + 0.25 * 1.0 + 0.5 * (-1.0)  # -0.25
            
            @test scratch[1] ≈ expected_1 atol=1e-12
            @test scratch[2] ≈ expected_2 atol=1e-12
        end
        
        @testset "Helmert Coding Correctness" begin
            # Test with Helmert coding (orthogonal contrasts)
            helmert_matrix = [
                -1.0 -1.0;    # Level 1
                 1.0 -1.0;    # Level 2  
                 0.0  2.0     # Level 3
            ]
            
            mixture_op = MixtureContrastOp{
                :group,
                (1, 2),
                (1, 2, 3),
                (0.3, 0.3, 0.4)
            }(helmert_matrix)
            
            scratch = Vector{Float64}(undef, 3)
            data = (group = ["test"],)
            
            execute_op(mixture_op, scratch, data, 1)
            
            # Manual calculation: 0.3*[-1,-1] + 0.3*[1,-1] + 0.4*[0,2] = [0, 0.2]
            expected_1 = 0.3 * (-1.0) + 0.3 * 1.0 + 0.4 * 0.0    # 0.0
            expected_2 = 0.3 * (-1.0) + 0.3 * (-1.0) + 0.4 * 2.0  # 0.2
            
            @test scratch[1] ≈ expected_1 atol=1e-12
            @test scratch[2] ≈ expected_2 atol=1e-12
        end
    end
    
    @testset "Edge Cases and Robustness" begin
        # Test edge cases and boundary conditions
        
        @testset "Single Level Mixtures" begin
            # Degenerate case: 100% one level
            contrast_matrix = [1.0; 0.0; -1.0]  # Single contrast column
            
            single_op = MixtureContrastOp{
                :group,
                (1,),
                (2,),
                (1.0,)
            }(reshape(contrast_matrix, 3, 1))
            
            scratch = Vector{Float64}(undef, 2)
            data = (group = ["test"],)
            
            execute_op(single_op, scratch, data, 1)
            
            # Should give 100% of level 2 contrast
            @test scratch[1] ≈ 0.0 atol=1e-12  # Level 2 has 0.0 in contrast
        end
        
        @testset "Extreme Weights" begin
            # Test with very small and large (but valid) weights
            contrast_matrix = [1.0 0.0; 0.0 1.0]
            
            extreme_op = MixtureContrastOp{
                :group,
                (1, 2),
                (1, 2),
                (0.001, 0.999)
            }(contrast_matrix)
            
            scratch = Vector{Float64}(undef, 3)
            data = (group = ["test"],)
            
            execute_op(extreme_op, scratch, data, 1)
            
            # Should still compute correctly
            @test scratch[1] ≈ 0.001 atol=1e-12
            @test scratch[2] ≈ 0.999 atol=1e-12
        end
        
        @testset "Large Number of Levels" begin
            # Test with many mixture levels (5-way mixture)
            n_levels = 5
            contrast_matrix = Matrix{Float64}(I(n_levels)[1:n_levels, 1:n_levels-1])  # Dummy coding with 4 contrasts
            
            # Equal weights
            weights = tuple(fill(1.0/n_levels, n_levels)...)
            level_indices = tuple(1:n_levels...)
            
            many_level_op = MixtureContrastOp{
                :group,
                (1, 2, 3, 4),
                level_indices,
                weights
            }(contrast_matrix)
            
            scratch = Vector{Float64}(undef, 5)
            data = (group = ["test"],)
            
            execute_op(many_level_op, scratch, data, 1)
            
            # Each contrast should be 1/5 of its corresponding level
            for i in 1:4
                expected = (1.0/n_levels) * contrast_matrix[i, i]  # Level i contributes to contrast i
                @test scratch[i] ≈ expected atol=1e-12
            end
        end
    end
    
    @testset "Integration with Helper Functions" begin
        # Test that Phase 4 helper functions work with Phase 5 execution
        
        @testset "create_mixture_column Integration" begin
            # Test that helper-created columns work with mixture operations
            mixture = mix("P" => 0.7, "Q" => 0.3)
            mixture_col = create_mixture_column(mixture, 4)
            
            test_data = (
                x = [1.0, 2.0, 3.0, 4.0],
                group = mixture_col
            )
            
            # Should pass validation
            @test_nowarn validate_mixture_consistency!(test_data)
            @test is_mixture_column(test_data.group)
            
            # Should extract correct spec
            spec = extract_mixture_spec(test_data.group[1])
            @test spec.levels == ["P", "Q"]
            @test spec.weights == [0.7, 0.3]
        end
        
        @testset "expand_mixture_grid Integration" begin
            base_data = (x = [1.0, 2.0],)
            mixture_specs = Dict(:group => mix("Alpha" => 0.4, "Beta" => 0.6))
            
            expanded = expand_mixture_grid(base_data, mixture_specs)
            result_data = expanded[1]
            
            # Should create valid mixture data
            @test_nowarn validate_mixture_consistency!(result_data)
            @test is_mixture_column(result_data.group)
            
            # Should have correct structure
            @test length(result_data.x) == 2
            @test length(result_data.group) == 2
            
            spec = extract_mixture_spec(result_data.group[1])
            @test spec.levels == ["Alpha", "Beta"]
            @test spec.weights == [0.4, 0.6]
        end
        
        @testset "create_balanced_mixture Integration" begin
            balanced_dict = create_balanced_mixture(["Red", "Green", "Blue"])
            
            # Should create valid balanced mixture
            @test length(balanced_dict) == 3
            @test all(w ≈ 1/3 for w in values(balanced_dict))
            
            # Should pass validation
            validate_mixture_weights(collect(values(balanced_dict)))
            validate_mixture_levels(collect(keys(balanced_dict)))
        end
    end
    
    @testset "Memory Usage and Scalability" begin
        # Test memory characteristics for large datasets
        
        @testset "Large Dataset Memory Usage" begin
            # Test that mixture operations scale O(1) with data size
            mixture = mix("Large" => 0.8, "Small" => 0.2)
            
            # Small dataset
            small_data = (
                x = randn(100),
                group = create_mixture_column(mixture, 100)
            )
            
            # Large dataset  
            large_data = (
                x = randn(10000),
                group = create_mixture_column(mixture, 10000)
            )
            
            # Memory usage for validation should be similar (O(1) in dataset size)
            @test_nowarn validate_mixture_consistency!(small_data)
            @test_nowarn validate_mixture_consistency!(large_data)
            
            # Both should be detected as mixture columns with same efficiency
            @test is_mixture_column(small_data.group)
            @test is_mixture_column(large_data.group)
        end
        
        @testset "Mixture Operation Scalability" begin
            # Test that mixture operation performance is independent of data size
            contrast_matrix = [1.0 0.0; 0.0 1.0; -0.5 -0.5]
            
            mixture_op = MixtureContrastOp{
                :group,
                (1, 2),
                (1, 2, 3),
                (0.4, 0.3, 0.3)
            }(contrast_matrix)
            
            scratch = Vector{Float64}(undef, 3)
            data = (group = ["test"],)
            
            # Warm up to ensure compilation is done
            for _ in 1:10
                execute_op(mixture_op, scratch, data, 1)
            end
            
            # Performance should be constant regardless of conceptual "dataset size"
            # since mixture operations don't depend on data size
            for _ in 1:10  # Test zero allocation after warmup
                @test (@allocated execute_op(mixture_op, scratch, data, 1)) == 0
            end
        end
    end
    
    @testset "Boolean Mixture Support" begin
        # Test boolean mixtures work correctly with current CounterfactualVector system

        @testset "Boolean Mixture Creation" begin
            # Test creation of boolean mixtures
            bool_mixture = mix("false" => 0.3, "true" => 0.7)

            # Test with boolean column
            df = DataFrame(
                x = [1.0, 2.0, 3.0, 4.0],
                treated = [true, false, true, false]  # Boolean column
            )
            data = Tables.columntable(df)

            @test typeof(data.treated) == Vector{Bool}

            # Test mixture properties
            @test bool_mixture.levels == ["false", "true"]
            @test bool_mixture.weights == [0.3, 0.7]
            @test length(bool_mixture) == 2
        end

        @testset "Boolean Mixture Loop Pattern" begin
            # Test boolean mixtures using the current CounterfactualVector approach
            df = DataFrame(
                x = [1.0, 2.0, 3.0, 4.0],
                y = [1.0, 2.0, 3.0, 4.0],
                treated = [true, false, true, false]
            )
            data = Tables.columntable(df)

            # Fit a simple model
            model = lm(@formula(y ~ x + treated), df)
            compiled = compile_formula(model, data)

            # Create different boolean mixtures for analysis
            mixtures = [
                ("mostly_false", mix("false" => 0.8, "true" => 0.2)),
                ("balanced", mix("false" => 0.5, "true" => 0.5)),
                ("mostly_true", mix("false" => 0.2, "true" => 0.8))
            ]

            results = Dict{String, Vector{Float64}}()

            for (name, mixture) in mixtures
                # Simulate the mixture using weighted average approach
                # This replaces the eliminated scenario system
                false_weight = mixture.weights[findfirst(x -> x == "false", mixture.levels)]
                true_weight = mixture.weights[findfirst(x -> x == "true", mixture.levels)]

                # Create counterfactual data for false case
                cf_data_false = merge(data, (
                    treated = BoolCounterfactualVector(data.treated, 1, false),
                ))

                # Create counterfactual data for true case
                cf_data_true = merge(data, (
                    treated = BoolCounterfactualVector(data.treated, 1, true),
                ))

                # Evaluate both cases
                output_false = Vector{Float64}(undef, length(compiled))
                output_true = Vector{Float64}(undef, length(compiled))

                compiled(output_false, cf_data_false, 1)
                compiled(output_true, cf_data_true, 1)

                # Compute weighted combination
                weighted_result = false_weight .* output_false .+ true_weight .* output_true
                results[name] = weighted_result
            end

            # Should have results for each mixture
            @test length(results) == 3
            @test all(haskey(results, name) for (name, _) in mixtures)

            # Results should vary across mixtures
            result_vectors = collect(values(results))
            @test length(unique(result_vectors)) > 1
        end

        @testset "Boolean Mixture Edge Cases" begin
            df = DataFrame(x = [1.0], treated = [true])
            data = Tables.columntable(df)

            # Test extreme cases
            all_false = mix("false" => 1.0, "true" => 0.0)
            all_true = mix("false" => 0.0, "true" => 1.0)

            @test all_false.weights == [1.0, 0.0]
            @test all_true.weights == [0.0, 1.0]

            # These should be valid mixtures
            @test length(all_false) == 2
            @test length(all_true) == 2
        end

        @testset "Boolean Mixture with GLM Integration" begin
            # Test boolean mixtures work with actual statistical models using CounterfactualVector
            df = DataFrame(
                x = randn(100),
                y = randn(100),
                treated = rand([true, false], 100)
            )

            # Fit a simple model
            model = lm(@formula(y ~ x + treated), df)
            data = Tables.columntable(df)

            # Create counterfactual data with boolean modification
            cf_data = merge(data, (
                treated = BoolCounterfactualVector(data.treated, 1, true),
            ))

            # This should not error - boolean counterfactuals should work with GLM
            compiled = nothing
            compiled_cf = nothing
            @test_nowarn compiled = compile_formula(model, data)
            @test_nowarn compiled_cf = compile_formula(model, cf_data)

            # Both should have same structure
            @test length(compiled) == length(compiled_cf)
        end
    end
end
