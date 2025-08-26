# test_position_mapping.jl
# Formal test suite for position mapping functionality in the compilation system

using Test
using BenchmarkTools
using FormulaCompiler
using Statistics
using DataFrames, GLM, Tables, CategoricalArrays, Random
using StatsModels
using FormulaCompiler: test_data, test_formula_correctness, test_allocation_performance

@testset "Position Mapping Tests" begin
    Random.seed!(06515)
    
    # Setup test data
    df, data = test_data(n=200)
    
    @testset "Basic Position Mapping" begin
        @testset "Simple continuous variable" begin
            formula = @formula(response ~ x)
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            @test length(compiled) == 2  # Intercept + x
        end
        
        @testset "Simple categorical variable" begin
            formula = @formula(response ~ group3)
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            @test length(compiled) == 3  # Intercept + 2 contrast columns for 3-level factor
        end
        
        @testset "Multiple continuous variables" begin
            formula = @formula(response ~ x + y)
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            @test length(compiled) == 3  # Intercept + x + y
        end
    end
    
    @testset "Interaction Position Mapping" begin
        @testset "Simple 2-way continuous interaction" begin
            formula = @formula(response ~ x * y)
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            @test length(compiled) == 4  # Intercept + x + y + x:y
        end
        
        @testset "Continuous × Categorical interaction" begin
            formula = @formula(response ~ x * group3)
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            @test length(compiled) == 6  # Intercept + x + group3 (2 cols) + x:group3 (2 cols)
        end
        
        @testset "3-way interaction" begin
            formula = @formula(response ~ x * y * group3)
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            # Intercept + x + y + group3 (2) + x:y + x:group3 (2) + y:group3 (2) + x:y:group3 (2)
            @test length(compiled) == 12
        end
        
        @testset "4-way interaction (Kronecker ordering test)" begin
            formula = @formula(response ~ x * y * group3 * group4)  
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            # This tests the Kronecker product ordering fix
            @test length(compiled) > 20  # Complex interaction should have many terms
        end
    end
    
    @testset "Function Position Mapping" begin
        @testset "Simple function" begin
            formula = @formula(response ~ log(z))
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            @test length(compiled) == 2  # Intercept + log(z)
        end
        
        @testset "Function × Categorical interaction" begin
            formula = @formula(response ~ log(z) * group4)
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            @test length(compiled) == 8  # Intercept + log(z) + group4 (3) + log(z):group4 (3)
        end
    end
    
    @testset "Complex Formula Position Mapping" begin
        @testset "Complex mixed formula" begin
            formula = @formula(response ~ x * y * group3 + log(z) * group4)
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            # Should handle both 3-way interactions and function interactions correctly
            @test length(compiled) > 15  # Complex formula with many terms
        end
    end
    
    @testset "Edge Cases" begin
        @testset "Intercept only" begin
            formula = @formula(response ~ 1)
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            @test length(compiled) == 1  # Only intercept
        end
        
        @testset "No intercept" begin
            formula = @formula(response ~ 0 + x)
            compiled, output = test_formula_correctness(formula, df, data)
            allocs = test_allocation_performance(compiled, output, data)
            
            @test length(compiled) == 1  # Only x
        end
    end
    
    @testset "Position Mapping Consistency" begin
        @testset "Same formula produces same positions" begin
            formula = @formula(response ~ x * group3)
            
            # Compile twice
            model1 = fit(LinearModel, formula, df)
            compiled1 = compile_formula(model1, data)
            
            model2 = fit(LinearModel, formula, df)  
            compiled2 = compile_formula(model2, data)
            
            # Should produce identical position mappings
            @test length(compiled1) == length(compiled2)
            
            # Test that they produce identical outputs
            output1 = Vector{Float64}(undef, length(compiled1))
            output2 = Vector{Float64}(undef, length(compiled2))
            
            for test_row in [1, 5, 10, 25]
                compiled1(output1, data, test_row)
                compiled2(output2, data, test_row)
                @test output1 ≈ output2
            end
        end
    end
    
    @testset "Performance Characteristics" begin
        @testset "Timing benchmarks" begin
            # Test that compiled formulas are fast
            formula = @formula(response ~ x * group3)
            model = fit(LinearModel, formula, df)
            compiled = compile_formula(model, data)
            output = Vector{Float64}(undef, length(compiled))
            
            # Warmup
            for i in 1:10
                compiled(output, data, 1)
            end
            
            # Benchmark
            benchmark_result = @benchmark $compiled($output, $data, 1) samples=1000
            median_time_ns = median(benchmark_result.times)
            
            # Should be very fast (sub-microsecond for simple formulas)
            @test median_time_ns < 1_000_000  # Less than 1ms (very conservative)
            
            # For simple formulas, should be much faster
            if !occursin("interaction", string(formula))
                @test median_time_ns < 100_000  # Less than 100μs for simple formulas
            end
        end
    end
end