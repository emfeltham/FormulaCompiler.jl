# test_modelrow_allocations.jl
# Formal allocation tests for modelrow! function
# julia --project=test test/test_modelrow_allocations.jl

using Test
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays, MixedModels
using StatsModels, BenchmarkTools
using Random

# Include test utilities
include("support/testing_utilities.jl")

@testset "modelrow! Allocation Tests" begin
    Random.seed!(12345)
    
    # Setup test data
    n = 500
    df = make_test_data(; n)
    data = Tables.columntable(df)
    
    @testset "Single Call Allocations" begin
        @testset "Linear Models" begin
            for (i, fx) in enumerate(test_formulas.lm[1:5])  # Test first 5 formulas
                @testset "$(fx.name)" begin
                    model = lm(fx.formula, df)
                    
                    # Test direct compiled() call (baseline)
                    memory_bytes, _ = test_zero_allocation(model, data)
                    @test memory_bytes == 0
                    
                    # Test modelrow! call (should also be zero)
                    memory_bytes_mr, _ = test_modelrow_zero_allocation(model, data)
                    @test memory_bytes_mr == 0
                end
            end
        end
        
        @testset "GLM Models" begin
            for (i, fx) in enumerate(test_formulas.glm[1:3])  # Test first 3 GLM formulas
                @testset "$(fx.name)" begin
                    model = glm(fx.formula, df, fx.distribution, fx.link)
                    
                    # Test direct compiled() call (baseline)  
                    memory_bytes, _ = test_zero_allocation(model, data)
                    @test memory_bytes == 0
                    
                    # Test modelrow! call (should also be zero)
                    memory_bytes_mr, _ = test_modelrow_zero_allocation(model, data)
                    @test memory_bytes_mr == 0
                end
            end
        end
    end
    
    @testset "Loop Allocations" begin
        @testset "modelrow! in Loops" begin
            # Test simple linear model
            model = lm(@formula(continuous_response ~ x + group3), df)
            
            # Test loop allocation (should be zero)
            total_bytes, per_call_bytes, _ = test_modelrow_loop_allocation(model, data; n_calls=1000)
            @test total_bytes == 0
            @test per_call_bytes == 0.0
        end
        
        @testset "Complex Formula in Loop" begin
            # Test complex interaction
            model = lm(@formula(continuous_response ~ x * y * group3 + log(abs(z))), df)
            
            # Test loop allocation (should be zero)
            total_bytes, per_call_bytes, _ = test_modelrow_loop_allocation(model, data; n_calls=500)
            @test total_bytes == 0
            @test per_call_bytes == 0.0
        end
    end
    
    @testset "Struct Field Access Allocations" begin
        @testset "Engine Pattern Detection" begin
            model = lm(@formula(continuous_response ~ x + group3), df)
            
            # Test struct field access pattern (expected to allocate)
            total_bytes, per_call_bytes, _ = test_struct_field_allocation(model, data; n_calls=1000)
            
            # This should allocate (due to struct field access)
            @test total_bytes > 0
            @test per_call_bytes > 0
        end
        
        @testset "Multiple Models Engine Pattern" begin
            # Test with different model types
            models = [
                lm(@formula(continuous_response ~ x), df),
                lm(@formula(continuous_response ~ x + group3), df),
                lm(@formula(continuous_response ~ x * group3), df)
            ]
            
            for (i, model) in enumerate(models)
                total_bytes, per_call_bytes, _ = test_struct_field_allocation(model, data; n_calls=500)
                @test total_bytes > 0  # Should allocate
                @test per_call_bytes > 0  # Should have per-call overhead
            end
        end
    end
    
    @testset "Performance Comparison" begin
        @testset "Direct vs modelrow!" begin
            model = lm(@formula(continuous_response ~ x + y + group3), df)
            compiled = compile_formula(model, data)
            buffer = Vector{Float64}(undef, length(compiled))
            
            # Warmup
            for _ in 1:100
                compiled(buffer, data, 1)
                modelrow!(buffer, compiled, data, 1)
            end
            
            # Benchmark both approaches
            b_direct = @benchmark $compiled($buffer, $data, 1) samples=1000
            b_modelrow = @benchmark modelrow!($buffer, $compiled, $data, 1) samples=1000
            
            # Both should be zero allocation
            @test minimum(b_direct.memory) == 0
            @test minimum(b_modelrow.memory) == 0
            
            # Performance should be reasonable (modelrow! shouldn't be much slower)
            direct_time = minimum(b_direct.times)
            modelrow_time = minimum(b_modelrow.times)
            overhead_ratio = modelrow_time / direct_time
            
            @test overhead_ratio < 5.0  # Should be less than 5x slower
        end
    end
    
    @testset "Integration with Existing Tests" begin
        @testset "Consistency Check" begin
            # Use same test data as existing tests
            Random.seed!(10115)  # Same seed as test_allocations.jl
            n = 500
            df = make_test_data(; n)
            data = Tables.columntable(df)
            
            # Test that our modelrow! tests are consistent with existing zero-allocation tests
            for fx in test_formulas.lm[1:3]
                model = lm(fx.formula, df)
                
                # Both should show zero allocation
                mem_direct, _ = test_zero_allocation(model, data)
                mem_modelrow, _ = test_modelrow_zero_allocation(model, data)
                
                @test mem_direct == mem_modelrow == 0
            end
        end
    end
end