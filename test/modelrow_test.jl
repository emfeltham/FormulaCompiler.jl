###############################################################################
# INTEGRATION EXAMPLE
###############################################################################

"""
Example showing the progression from convenient to high-performance usage.
"""
function performance_example()
    # Setup
    # using GLM, DataFrames, Tables, BenchmarkTools
    
    df = DataFrame(
        x = randn(1000), 
        y = randn(1000), 
        group = categorical(rand(["A", "B", "C"], 1000))
    )
    model = lm(@formula(y ~ x^2 * group), df)
    data = Tables.columntable(df)
    
    println("=== Performance Progression ===")
    
    # Level 1: Standard StatsModels (slow)
    println("1. Standard approach:")
    mm = modelmatrix(model)
    @btime $mm[1, :]  # ~1-10 μs, allocates
    
    # Level 2: Convenient modelrow! (fast, auto-caching)
    println("2. Convenient modelrow!:")
    row_vec = Vector{Float64}(undef, size(mm, 2))
    @btime modelrow!($row_vec, $model, $data, 1)  # ~100-500ns after first call
    
    # Level 3: Pre-compiled (fastest)
    println("3. Pre-compiled:")
    compiled = compile_formula(model)
    @btime $compiled($row_vec, $data, 1)  # ~50-100ns, zero allocations
    
    # Level 4: Batch processing
    println("4. Batch processing:")
    matrix = Matrix{Float64}(undef, 100, length(compiled))
    @btime modelrow!($matrix, $model, $data, 1:100)
    
    return compiled
end

###############################################################################
# TESTING INTERFACE
###############################################################################

"""
    test_modelrow_interface()

Test the modelrow! interface for correctness and performance.
"""
function test_modelrow_interface()
    println("=== Testing modelrow! Interface ===")
    
    Random.seed!(42)
    df = DataFrame(
        x = randn(100),
        y = randn(100), 
        z = abs.(randn(100)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 100))
    )
    
    data = Tables.columntable(df)
    
    # Test various model types
    models = [
        lm(@formula(y ~ x + log(z) + group), df),
        lm(@formula(y ~ x * group + x^2), df),
        lm(@formula(y ~ x + log(z) * group + (x > 0)), df)
    ]
    
    results = []
    
    for (i, model) in enumerate(models)
        println("\n--- Test $i ---")
        
        try
            # Test pre-compiled approach
            compiled = compile_formula(model)
            row_vec = Vector{Float64}(undef, length(compiled))
            
            # Test zero-allocation call
            modelrow!(row_vec, compiled, data, 1)
            
            # Verify against model matrix
            mm = modelmatrix(model)
            expected = mm[1, :]
            error = maximum(abs.(row_vec .- expected))
            
            if error < 1e-12
                println("✅ Pre-compiled: PASSED (error = $error)")
                
                # Test allocations
                allocs = @allocated modelrow!(row_vec, compiled, data, 1)
                println("  Allocations: $allocs bytes")
                
                # Test convenience method
                clear_model_cache!()
                modelrow!(row_vec, model, data, 1; cache=true)
                error2 = maximum(abs.(row_vec .- expected))
                
                if error2 < 1e-12
                    println("✅ Convenience cached: PASSED (error = $error2)")
                    push!(results, (i, true, allocs))
                else
                    println("❌ Convenience cached: FAILED (error = $error2)")
                    push!(results, (i, false, error2))
                end
            else
                println("❌ Pre-compiled: FAILED (error = $error)")
                push!(results, (i, false, error))
            end
            
        catch e
            println("❌ EXCEPTION: $e")
            push!(results, (i, false, "exception"))
        end
    end
    
    # Summary
    successful = sum(r[2] for r in results)
    println("\n" * "="^50)
    println("RESULTS: $successful/$(length(models)) passed")
    
    if successful == length(models)
        println("🎉 ALL INTERFACE TESTS PASSED!")
    end
    
    return results
end
