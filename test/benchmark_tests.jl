###############################################################################
# Benchmark Tests (Optional)
###############################################################################

@testset "Performance Benchmarks" begin
    # These tests are for performance monitoring, not strict pass/fail
    
    df, data = create_test_data()
    
    @testset "Speed Benchmarks" begin
        # Test a few representative formulas for timing
        benchmark_cases = [
            (@formula(y ~ x), "simple"),
            (@formula(y ~ x * group), "medium"),
            (@formula(y ~ x * z * group * cat2a), "complex"),
        ]
        
        for (formula, complexity) in benchmark_cases
            model = lm(formula, df)
            compiled = compile_formula(model)
            row_vec = Vector{Float64}(undef, length(compiled))
            
            # Warmup
            for _ in 1:100
                compiled(row_vec, data, 1)
            end
            
            # Time single evaluation
            times = Float64[]
            for _ in 1:1000
                t = @elapsed compiled(row_vec, data, 1)
                push!(times, t)
            end
            
            avg_time_ns = mean(times) * 1e9
            min_time_ns = minimum(times) * 1e9
            
            @info "Timing benchmark" formula=formula complexity=complexity avg_time_ns=avg_time_ns min_time_ns=min_time_ns
            
            # Aspirational performance targets
            if avg_time_ns < 100
                @info "🎉 Excellent performance: $(round(avg_time_ns, digits=1)) ns average"
            elseif avg_time_ns < 500
                @info "✅ Good performance: $(round(avg_time_ns, digits=1)) ns average"
            else
                @info "⚠️  Slower than target: $(round(avg_time_ns, digits=1)) ns average"
            end
        end
    end
end

###############################################################################
# Test Runner Function
###############################################################################

"""
Run all tests with summary reporting.
"""
function run_all_tests()
    @info "Starting EfficientModelMatrices.jl test suite..."
    
    # Set up test environment
    original_seed = Random.seed!()
    
    try
        # Run the test suite
        Test.@testset "Complete Test Suite" begin
            # The @testset above will run automatically when this file is included
        end
        
        @info "✅ Test suite completed successfully!"
        
    catch e
        @error "❌ Test suite failed" exception=e
        rethrow(e)
    finally
        # Restore random seed
        Random.seed!(original_seed)
    end
end
