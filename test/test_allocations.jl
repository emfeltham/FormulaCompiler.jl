# test_allocations.jl
# SMOKE TEST: Quick zero-allocation verification for model compilation
#
# Purpose: Fast sanity check that basic model compilation completes without errors
# Scope: Tests compile_formula() for LM, GLM, LMM, GLMM models
# Note: For comprehensive allocation testing, see test_derivative_allocations.jl
#
# julia --project="test" test/test_allocations.jl > test/test_allocations.txt 2>&1

using Test
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays, MixedModels
using StatsModels, BenchmarkTools
# Test helpers are included from test/support/testing_utilities.jl via runtests.jl
using Random

@testset "Zero Allocation Survey" begin
    Random.seed!(10115)
    
    # Setup test data
    n = 500
    df = make_test_data(; n)
    data = Tables.columntable(df)
    
    @testset "Linear Models (LM)" begin
        for fx in test_formulas.lm
            @testset "$(fx.name)" begin
                model = lm(fx.formula, df)
                memory_bytes, time_ns = test_zero_allocation(model, data)
                
                # Additional sanity checks
                @test length(compile_formula(model, data)) > 0
                @test time_ns > 0
            end
        end
    end
    
    fx = test_formulas.glm[5]

    @testset "Generalized Linear Models (GLM)" begin
        for fx in test_formulas.glm
            @testset "$(fx.name)" begin
                model = glm(fx.formula, df, fx.distribution, fx.link)
                memory_bytes, time_ns = test_zero_allocation(model, data)
                
                # Additional sanity checks
                @test length(compile_formula(model, data)) > 0
                @test time_ns > 0
            end
        end
    end
    
    @testset "Linear Mixed Models (LMM)" begin
        for fx in test_formulas.lmm
            @testset "$(fx.name)" begin
                model = fit(MixedModel, fx.formula, df; progress = false)
                memory_bytes, time_ns = test_zero_allocation(model, data)
                
                # Additional sanity checks
                @test length(compile_formula(model, data)) > 0
                @test time_ns > 0
            end
        end
    end
    
    @testset "Generalized Linear Mixed Models (GLMM)" begin
        for fx in test_formulas.glmm
            @testset "$(fx.name)" begin
                model = fit(MixedModel, fx.formula, df, fx.distribution, fx.link; progress = false)
                memory_bytes, time_ns = test_zero_allocation(model, data)
                
                # Additional sanity checks
                @test length(compile_formula(model, data)) > 0
                @test time_ns > 0
            end
        end
    end
end
