# test_allocations.jl
# Formal test suite for zero-allocation verification

# julia --project="." test/test_allocations.jl > test/test_allocations.txt 2>&1

using Test
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays, MixedModels
using StatsModels, BenchmarkTools
using FormulaCompiler: make_test_data, test_formulas
using Random

# Helper function to test allocation with proper warmup
function test_zero_allocation(model, data)
    compiled = compile_formula(model, data)
    buffer = Vector{Float64}(undef, length(compiled))
    
    # Extensive warmup to ensure compilation is complete
    for _ in 1:100
        compiled(buffer, data, 1)
    end
    
    # Benchmark for accurate allocation measurement
    benchmark_result = @benchmark $compiled($buffer, $data, 1) samples=1000 seconds=2
    memory_bytes = minimum(benchmark_result.memory)
    
    @test memory_bytes == 0
    
    return memory_bytes, minimum(benchmark_result.times)
end

@testset "Zero Allocation Survey" begin
    Random.seed!(08540)
    
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
