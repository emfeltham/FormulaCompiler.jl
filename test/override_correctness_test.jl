# scenario_test_generator.jl
# Generate test scenarios for override correctness based on formula cases

# scenario_test_example.jl
# Example of testing one scenario comprehensively

using Test
using DataFrames
using GLM
using StatsModels
using CategoricalArrays
using Tables
using Random

"""
    test_mixed_realistic_scenario(df, data, formula)

Test the "mixed_realistic" scenario - a common use case with partial overrides.
This function demonstrates the complete testing pattern for a single scenario.
"""
function test_mixed_realistic_scenario(df, data, fx)
    @testset "Mixed Realistic Scenario" begin
        # Define the scenario
        scenario_name = "mixed_realistic"
        overrides = Dict(
            :x => 1.5,
            :y => -0.5,
            :group3 => "B",
            :binary => "Yes"
        )
        
        # Fit the original model
        model = lm(fx, df)
        original_mm = modelmatrix(model)
        n_cols = size(original_mm, 2)
        
        # Create the scenario
        scenario = create_scenario(scenario_name, data, overrides)
        
        # Compile the specialized formula
        compiled = compile_formula(model, scenario.data)
        
        # Create reference DataFrame with overrides applied
        ref_df = DataFrame(df)
        for (col, val) in overrides
            ref_df[!, col] .= val
        end
        
        # Fit reference model to get expected model matrix
        ref_model = lm(fx, ref_df)
        ref_mm = modelmatrix(ref_model)
        
        @testset "Row-by-row correctness" begin
            # Test multiple rows throughout the dataset
            test_rows = [1, 10, 25, 50, 75, 100]
            
            for row_idx in test_rows
                if row_idx <= nrow(df)
                    output = Vector{Float64}(undef, n_cols)
                    compiled(output, scenario.data, row_idx)
                    
                    @test output ≈ ref_mm[row_idx, :] atol=1e-10 rtol=1e-10
                end
            end
        end
        
        @testset "Override values applied correctly" begin
            # Verify that the override values are actually being used
            
            # Test row 1 specifically
            output = Vector{Float64}(undef, n_cols)
            compiled(output, scenario.data, 1)
            
            # Compare with manually constructed expected values
            # This requires knowledge of the formula structure
            
            # For example, if formula includes x as a main effect:
            # The coefficient for x should reflect the override value of 1.5
            
            # Create a test DataFrame with just one row to verify
            test_df = DataFrame(
                x = [1.5],
                y = [-0.5],
                z = df.z[1:1],  # Keep original
                w = df.w[1:1],  # Keep original
                t = df.t[1:1],  # Keep original
                group2 = df.group2[1:1],  # Keep original
                group3 = ["B"],
                group4 = df.group4[1:1],  # Keep original
                binary = ["Yes"],
                group5 = df.group5[1:1],  # Keep original
                response = [0.0]
            )

            # Make categoricals
            for col in [:group2, :group3, :group4, :binary, :group5]
                test_df[!, col] = categorical(test_df[!, col])
            end
            
            test_model = lm(fx, test_df)
            test_mm = modelmatrix(test_model)
            
            # The output should match this single-row test
            @test output ≈ test_mm[1, :] atol=1e-10 rtol=1e-10
        end
        
        @testset "Performance" begin

            output = Vector{Float64}(undef, n_cols)
            compiled(output, scenario.data, 1)

            # Test for zero allocations
            b = @benchmark compiled($output, $scenario.data, $1)
            @test b.allocs == 0
            @test b.memory == 0

            # Warmup
            output = Vector{Float64}(undef, n_cols)
            for _ in 1:100
                compiled(output, scenario.data, 1)
            end
            
            # Timing test
            elapsed = @elapsed begin
                for _ in 1:1000
                    compiled(output, scenario.data, 1)
                end
            end
            
            # performance info
            avg_time_μs = (elapsed / 1000) * 1e6
            @test avg_time_μs < 100  # Should be fast
            
        end
        
        @testset "Comparison with reference" begin
            # Comprehensive comparison across all rows
            max_error = 0.0
            total_error = 0.0
            
            for row_idx in 1:min(100, nrow(df))
                output = Vector{Float64}(undef, n_cols)
                compiled(output, scenario.data, row_idx)
                
                error = maximum(abs.(output .- ref_mm[row_idx, :]))
                max_error = max(max_error, error)
                total_error += error
            end
            
            avg_error = total_error / min(100, nrow(df))
            
            @test max_error < 1e-8
            @test avg_error < 1e-10
        end
    end
end

"""
    demonstrate_scenario_testing()

Demonstrate testing the mixed_realistic scenario with various formulas.
"""
function demonstrate_scenario_testing()
    # Create test data
    test_data(; n = 100)
    
    data = Tables.columntable(df)
    
    # Test with different formulas
    test_formulas = [
        @formula(response ~ x + y),
        @formula(response ~ x + y + group3 + binary),
        @formula(response ~ x * group3),
        @formula(response ~ x + y + group3 * binary),
        @formula(response ~ log(z) + x * y)
    ]
    
    (i, fx) = collect(enumerate(test_formulas))[1]
    for (i, fx) in enumerate(test_formulas)
        println("\n" * "="^60)
        println("Testing Formula $i: $fx")
        println("="^60)
        
        test_mixed_realistic_scenario(df, data, fx)
    end
end

Random.seed!(08540)

# zero allocation fails
demonstrate_scenario_testing()

# fx = basic[3]
# test_mixed_realistic_scenario(df, data, fx)
