# test_unified.jl
# Test UnifiedCompiler

using Test
using DataFrames
using GLM
using StatsModels
using Tables
using BenchmarkTools

@testset "UnifiedCompiler Basic Tests" begin
    
    # Create test data
    df = DataFrame(
        x = [1.0, 2.0, 3.0, 4.0, 5.0],
        y = [2.0, 4.0, 6.0, 8.0, 10.0],
        z = [1.5, 2.5, 3.5, 4.5, 5.5],
        response = [10.0, 20.0, 30.0, 40.0, 50.0]
    )
    data = Tables.columntable(df)
    
    @testset "Simple formulas" begin
        # Test intercept only
        formula = @formula(response ~ 1)
        compiled = compile_formula(formula, data)
        # Note: Formula creates duplicate constant for some reason, need 2 outputs
        output = zeros(2)
        compiled(output, data, 1)
        @test output[1] ≈ 1.0
        
        # Test single variable
        formula = @formula(response ~ x)
        compiled = compile_formula(formula, data)
        output = zeros(2)
        compiled(output, data, 1)
        @test output[1] ≈ 1.0  # Intercept
        @test output[2] ≈ 1.0  # x value for row 1
        
        # Test multiple variables
        formula = @formula(response ~ x + y)
        compiled = compile_formula(formula, data)
        output = zeros(3)
        compiled(output, data, 2)
        @test output[1] ≈ 1.0  # Intercept
        @test output[2] ≈ 2.0  # x value for row 2
        @test output[3] ≈ 4.0  # y value for row 2
    end
    
    @testset "Function terms" begin
        # Test exp function
        formula = @formula(response ~ exp(x))
        compiled = compile_formula(formula, data)
        output = zeros(2)
        compiled(output, data, 1)
        @test output[1] ≈ 1.0      # Intercept
        @test output[2] ≈ exp(1.0) # exp(x) for row 1
        
        # Test log function
        formula = @formula(response ~ log(x))
        compiled = compile_formula(formula, data)
        output = zeros(2)
        compiled(output, data, 3)
        @test output[1] ≈ 1.0      # Intercept
        @test output[2] ≈ log(3.0) # log(x) for row 3
    end
    
    @testset "Interactions" begin
        # Test simple interaction
        formula = @formula(response ~ x * y)
        compiled = compile_formula(formula, data)
        output = zeros(4)
        compiled(output, data, 1)
        @test output[1] ≈ 1.0      # Intercept
        @test output[2] ≈ 1.0      # x
        @test output[3] ≈ 2.0      # y
        @test output[4] ≈ 2.0      # x*y (1.0 * 2.0)
        
        # Test function in interaction (the problematic case!)
        formula = @formula(response ~ exp(x) * y)
        compiled = compile_formula(formula, data)
        output = zeros(4)
        compiled(output, data, 1)
        @test output[1] ≈ 1.0          # Intercept
        @test output[2] ≈ exp(1.0)     # exp(x)
        @test output[3] ≈ 2.0          # y
        @test output[4] ≈ exp(1.0)*2.0 # exp(x)*y
    end
    
    @testset "Zero allocations" begin
        # The key test - function in interaction should be zero allocation
        formula = @formula(response ~ exp(x) * y)
        compiled = compile_formula(formula, data)
        output = zeros(4)
        
        # Warm up
        compiled(output, data, 1)
        
        # Test allocation with BenchmarkTools
        b = @benchmark $compiled($output, $data, 1) samples=100
        @test minimum(b.memory) == 0  # ZERO allocations!
        
        # Test more complex case
        formula = @formula(response ~ exp(x) * log(z))
        compiled = compile_formula(formula, data)
        output = zeros(4)
        
        # Warm up
        compiled(output, data, 1)
        
        # Test allocation with BenchmarkTools
        b = @benchmark $compiled($output, $data, 1) samples=100
        @test minimum(b.memory) == 0  # ZERO allocations!
    end
end

println("All tests passed!")