# test/test_performance.jl
# Performance and allocation tests

@testset "Performance Tests" begin
    
    Random.seed!(42)
    
    # Create test data of various sizes
    small_df = DataFrame(
        x = randn(100),
        y = randn(100),
        z = abs.(randn(100)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 100))
    )
    
    medium_df = DataFrame(
        x = randn(10_000),
        y = randn(10_000),
        z = abs.(randn(10_000)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 10_000))
    )
    
    large_df = DataFrame(
        x = randn(100_000),
        y = randn(100_000),
        z = abs.(randn(100_000)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 100_000))
    )
    
    @testset "Compilation Performance" begin
        # Test that compilation time is reasonable
        
        # Simple model
        simple_model = lm(@formula(y ~ x), small_df)
        compile_time = @elapsed compiled = compile_formula(simple_model)
        @test compile_time < 0.1  # Should compile in under 100ms
        
        # Complex model
        complex_model = lm(@formula(y ~ x * group + log(z) + x^2), small_df)
        compile_time_complex = @elapsed compiled_complex = compile_formula(complex_model)
        @test compile_time_complex < 0.5  # Should compile in under 500ms
        
        # Very complex model
        very_complex = lm(@formula(y ~ x * group * z + log(z) + sqrt(abs(x)) + x^2 + x^3), small_df)
        compile_time_very_complex = @elapsed compiled_very_complex = compile_formula(very_complex)
        @test compile_time_very_complex < 2.0  # Should compile in under 2s
        
        # Test that compilation is cached
        cached_time = @elapsed compile_formula(simple_model)
        @test cached_time < compile_time / 10  # Should be much faster second time
    end
    
    @testset "Zero Allocation Evaluation" begin
        # Test that evaluation has zero allocations
        
        model = lm(@formula(y ~ x * group + log(z)), small_df)
        compiled = compile_formula(model)
        data = Tables.columntable(small_df)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        # Test single evaluation
        allocs = @allocated compiled(row_vec, data, 1)
        @test allocs == 0
        
        # Test multiple evaluations
        for i in 1:10
            allocs = @allocated compiled(row_vec, data, i)
            @test allocs == 0
        end
        
        # Test with different model complexities
        simple_model = lm(@formula(y ~ x), small_df)
        simple_compiled = compile_formula(simple_model)
        simple_row_vec = Vector{Float64}(undef, length(simple_compiled))
        
        allocs_simple = @allocated simple_compiled(simple_row_vec, data, 1)
        @test allocs_simple == 0
        
        # Test with very complex model
        complex_model = lm(@formula(y ~ x * group * z + log(z) + x^2), small_df)
        complex_compiled = compile_formula(complex_model)
        complex_row_vec = Vector{Float64}(undef, length(complex_compiled))
        
        allocs_complex = @allocated complex_compiled(complex_row_vec, data, 1)
        @test allocs_complex == 0
    end
    
    @testset "Evaluation Speed" begin
        # Test that evaluation is fast
        
        model = lm(@formula(y ~ x * group + log(z)), small_df)
        compiled = compile_formula(model)
        data = Tables.columntable(small_df)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        # Single evaluation should be very fast
        eval_time = @elapsed compiled(row_vec, data, 1)
        @test eval_time < 0.001  # Should be under 1ms
        
        # Test with BenchmarkTools for more precise timing
        benchmark_result = @benchmark $compiled($row_vec, $data, 1)
        @test median(benchmark_result.times) < 500  # Should be under 500ns
        
        # Test batch evaluation performance
        n_evals = 10_000
        batch_time = @elapsed begin
            for i in 1:n_evals
                compiled(row_vec, data, (i-1) % nrow(small_df) + 1)
            end
        end
        
        avg_time_per_eval = batch_time / n_evals
        @test avg_time_per_eval < 0.0001  # Should be under 0.1ms per eval
    end
    
    @testset "Memory Usage" begin
        # Test memory usage patterns
        
        # Test that compiled formulas don't hold unnecessary references
        model = lm(@formula(y ~ x * group + log(z)), small_df)
        compiled = compile_formula(model)
        
        # Should be able to garbage collect the original model
        model = nothing
        GC.gc()
        
        # Compiled formula should still work
        data = Tables.columntable(small_df)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        @test all(isfinite.(row_vec))
        
        # Test that evaluator storage doesn't cause memory leaks
        evaluator = extract_root_evaluator(compiled)
        @test evaluator isa AbstractEvaluator
        
        # Test with many compilations
        initial_memory = GC.gc(); GC.gc()
        
        for i in 1:100
            temp_model = lm(@formula(y ~ x + z), small_df)
            temp_compiled = compile_formula(temp_model)
            temp_row_vec = Vector{Float64}(undef, length(temp_compiled))
            temp_compiled(temp_row_vec, data, 1)
        end
        
        final_memory = GC.gc(); GC.gc()
        # Memory usage shouldn't grow excessively
    end
    
    @testset "Scaling Performance" begin
        # Test performance scales appropriately with data size
        
        dfs = [small_df, medium_df, large_df]
        model_formula = @formula(y ~ x * group + log(z))
        
        compile_times = Float64[]
        eval_times = Float64[]
        
        for df in dfs
            # Compilation time (should be independent of data size)
            compile_time = @elapsed begin
                model = lm(model_formula, df)
                compiled = compile_formula(model)
            end
            push!(compile_times, compile_time)
            
            # Evaluation time (should be constant regardless of data size)
            model = lm(model_formula, df)
            compiled = compile_formula(model)
            data = Tables.columntable(df)
            row_vec = Vector{Float64}(undef, length(compiled))
            
            eval_time = @elapsed compiled(row_vec, data, 1)
            push!(eval_times, eval_time)
        end
        
        # Compilation times should be similar regardless of data size
        @test maximum(compile_times) / minimum(compile_times) < 3.0
        
        # Evaluation times should be very similar (constant time)
        @test maximum(eval_times) / minimum(eval_times) < 2.0
        @test all(eval_times .< 0.001)  # All should be under 1ms
    end
    
    @testset "Comparative Performance" begin
        # Compare performance against modelmatrix
        
        model = lm(@formula(y ~ x * group + log(z)), small_df)
        compiled = compile_formula(model)
        data = Tables.columntable(small_df)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        # Test single row performance
        mm_time = @elapsed begin
            mm = modelmatrix(model)
            row_result = mm[1, :]
        end
        
        compiled_time = @elapsed compiled(row_vec, data, 1)
        
        # Compiled version should be faster for single rows
        @test compiled_time < mm_time / 10  # At least 10x faster
        
        # Test multiple row performance
        n_rows = 1000
        
        mm_multi_time = @elapsed begin
            mm = modelmatrix(model)
            for i in 1:n_rows
                row_result = mm[i, :]
            end
        end
        
        compiled_multi_time = @elapsed begin
            for i in 1:n_rows
                compiled(row_vec, data, i)
            end
        end
        
        # For many rows, compiled should still be competitive
        @test compiled_multi_time < mm_multi_time * 2  # At most 2x slower
    end
    
    @testset "ModelRow Interface Performance" begin
        # Test performance of different interfaces
        
        model = lm(@formula(y ~ x * group + log(z)), small_df)
        compiled = compile_formula(model)
        data = Tables.columntable(small_df)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        # Pre-compiled version (should be fastest)
        precompiled_time = @elapsed modelrow!(row_vec, compiled, data, 1)
        precompiled_allocs = @allocated modelrow!(row_vec, compiled, data, 1)
        
        # Cached version
        cached_time = @elapsed modelrow!(row_vec, model, data, 1; cache=true)
        cached_allocs = @allocated modelrow!(row_vec, model, data, 1; cache=true)
        
        # Non-cached version
        non_cached_time = @elapsed modelrow!(row_vec, model, data, 1; cache=false)
        non_cached_allocs = @allocated modelrow!(row_vec, model, data, 1; cache=false)
        
        # Allocating version
        allocating_time = @elapsed modelrow(model, data, 1)
        allocating_allocs = @allocated modelrow(model, data, 1)
        
        # Performance hierarchy
        @test precompiled_time < cached_time
        @test cached_time < non_cached_time
        @test precompiled_allocs == 0
        @test cached_allocs < non_cached_allocs
        @test allocating_allocs > 0
        
        # Object-based interface
        evaluator = ModelRowEvaluator(model, small_df)
        object_time = @elapsed evaluator(1)
        object_allocs = @allocated evaluator(1)
        
        @test object_allocs == 0
        @test object_time < 0.001
    end
    
    @testset "Complex Formula Performance" begin
        # Test performance with increasingly complex formulas
        
        formulas = [
            @formula(y ~ x),
            @formula(y ~ x + z),
            @formula(y ~ x * group),
            @formula(y ~ x * group + log(z)),
            @formula(y ~ x * group * z),
            @formula(y ~ x * group + log(z) + x^2 + sqrt(abs(z))),
            @formula(y ~ x * group * z + log(z) + x^2 + x^3 + sin(x) + cos(z))
        ]
        
        data = Tables.columntable(small_df)
        
        compile_times = Float64[]
        eval_times = Float64[]
        
        for formula in formulas
            model = lm(formula, small_df)
            
            # Compilation time
            compile_time = @elapsed compiled = compile_formula(model)
            push!(compile_times, compile_time)
            
            # Evaluation time
            row_vec = Vector{Float64}(undef, length(compiled))
            eval_time = @elapsed compiled(row_vec, data, 1)
            push!(eval_times, eval_time)
            
            # Should maintain zero allocations
            allocs = @allocated compiled(row_vec, data, 1)
            @test allocs == 0
        end
        
        # Compilation time should grow reasonably with complexity
        @test compile_times[end] < compile_times[1] * 100  # Less than 100x increase
        
        # Evaluation time should remain low
        @test all(eval_times .< 0.001)  # All under 1ms
        
        # Complex formulas shouldn't be too much slower
        @test eval_times[end] < eval_times[1] * 10  # Less than 10x increase
    end
    
    @testset "Concurrent Performance" begin
        # Test thread safety and concurrent performance
        
        model = lm(@formula(y ~ x * group + log(z)), small_df)
        compiled = compile_formula(model)
        data = Tables.columntable(small_df)
        
        # Test that multiple threads can use same compiled formula
        if Threads.nthreads() > 1
            results = Vector{Vector{Float64}}(undef, Threads.nthreads())
            
            Threads.@threads for i in 1:Threads.nthreads()
                row_vec = Vector{Float64}(undef, length(compiled))
                compiled(row_vec, data, i)
                results[i] = copy(row_vec)
            end
            
            # All results should be valid
            for result in results
                @test all(isfinite.(result))
                @test length(result) == length(compiled)
            end
        end
    end
    
    @testset "Regression Performance" begin
        # Test against known performance benchmarks
        
        # Simple model should be very fast
        simple_model = lm(@formula(y ~ x), small_df)
        simple_compiled = compile_formula(simple_model)
        simple_data = Tables.columntable(small_df)
        simple_row_vec = Vector{Float64}(undef, length(simple_compiled))
        
        simple_benchmark = @benchmark $simple_compiled($simple_row_vec, $simple_data, 1)
        @test median(simple_benchmark.times) < 100  # Under 100ns
        
        # Complex model should still be reasonable
        complex_model = lm(@formula(y ~ x * group + log(z) + x^2), small_df)
        complex_compiled = compile_formula(complex_model)
        complex_row_vec = Vector{Float64}(undef, length(complex_compiled))
        
        complex_benchmark = @benchmark $complex_compiled($complex_row_vec, $simple_data, 1)
        @test median(complex_benchmark.times) < 1000  # Under 1μs
        
        # Performance should be consistent
        @test std(simple_benchmark.times) < median(simple_benchmark.times)
        @test std(complex_benchmark.times) < median(complex_benchmark.times)
    end
    
end
