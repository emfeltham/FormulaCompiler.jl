# Simplified StandardizedPredictors.jl integration test
# Verifies that ZScoredTerm integration works without double-transformation

using FormulaCompiler
using StandardizedPredictors
using Test
using GLM
using DataFrames
using Tables
using CategoricalArrays

@testset "StandardizedPredictors Integration" begin
    
    # Create simple test data
    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n) * 5.0 .+ 10.0,
        group = categorical(rand(["A", "B"], n))
    )
    
    @testset "Basic Integration Test" begin
        # Fit model with standardization
        model = lm(@formula(y ~ x + group), df, contrasts=Dict(:x => ZScore()))
        data = Tables.columntable(df)
        
        # Should compile successfully (tests ZScoredTerm handling)
        compiled = compile_formula(model, data)
        @test compiled isa FormulaCompiler.UnifiedCompiled
        
        # Should evaluate successfully
        output = Vector{Float64}(undef, length(compiled))
        compiled(output, data, 1)
        @test all(isfinite.(output))
        
        # Should be zero allocation
        @test @allocated(compiled(output, data, 1)) == 0
    end
    
    @testset "Scenario Integration" begin
        model = lm(@formula(y ~ x + group), df, contrasts=Dict(:x => ZScore()))
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        output = Vector{Float64}(undef, length(compiled))
        
        # Create scenario with override
        # Note: The value should be in the standardized scale if we want predictable results
        scenario = create_scenario("test", data; x = 0.0, group = "A")
        
        # Should evaluate without error
        compiled(output, scenario.data, 1)
        @test all(isfinite.(output))
        
        # Should be zero allocation  
        @test @allocated(compiled(output, scenario.data, 1)) == 0
    end
    
    @testset "Derivative Integration" begin
        model = lm(@formula(y ~ x + group), df, contrasts=Dict(:x => ZScore()))
        data = Tables.columntable(df)
        compiled = compile_formula(model, data)
        
        # Build derivative evaluator
        de = FormulaCompiler.build_derivative_evaluator(compiled, data; vars=[:x])
        g = Vector{Float64}(undef, 1)
        
        # Should compute derivatives without error
        FormulaCompiler.marginal_effects_eta!(g, de, coef(model), 1; backend=:fd)
        @test length(g) == 1
        @test isfinite(g[1])
    end
    
    @testset "Multiple Standardized Variables" begin
        # Add another continuous variable to test multiple standardizations
        df2 = DataFrame(
            y = randn(n),
            x = randn(n) * 5.0 .+ 10.0,
            z = randn(n) * 2.0 .+ 3.0,
            group = categorical(rand(["A", "B"], n))
        )
        
        model = lm(@formula(y ~ x + z + group), df2, 
                  contrasts=Dict(:x => ZScore(), :z => ZScore()))
        data = Tables.columntable(df2)
        
        # Should handle multiple ZScoredTerms
        compiled = compile_formula(model, data)
        output = Vector{Float64}(undef, length(compiled))
        
        compiled(output, data, 1)
        @test all(isfinite.(output))
        @test length(output) == 4  # intercept, x, z, group_B
        
        # Test with scenarios
        scenario = create_scenario("multi", data; x = 0.0, z = 0.0)
        compiled(output, scenario.data, 1)
        @test all(isfinite.(output))
    end
    
    @testset "Complex Functions with Standardization" begin
        # Test complex formula with functions and interactions combined with standardization
        # y ~ x^2 + log(z) + log(z) * x^2 with ZScore standardization
        df3 = DataFrame(
            y = randn(n),
            x = abs.(randn(n)) .+ 1.0,  # Ensure positive for x^2
            z = abs.(randn(n)) .+ 1.0   # Ensure positive for log(z)
        )
        
        # Fit complex model with standardization
        model = lm(@formula(y ~ x^2 + log(z) + log(z) * x^2), df3,
                  contrasts=Dict(:x => ZScore(), :z => ZScore()))
        data = Tables.columntable(df3)
        
        # Should compile complex formula with ZScoredTerms
        compiled = compile_formula(model, data)
        output = Vector{Float64}(undef, length(compiled))
        
        # Should evaluate without error
        compiled(output, data, 1)
        @test all(isfinite.(output))
        @test length(output) == 4  # intercept, x^2, log(z), log(z)*x^2
        
        # Should work with scenarios
        scenario = create_scenario("complex", data; x = 2.0, z = 1.0)  # Will be standardized
        compiled(output, scenario.data, 1)
        @test all(isfinite.(output))
        
        # Should maintain zero allocation
        @test @allocated(compiled(output, data, 1)) == 0
    end
    
end