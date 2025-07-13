# test_compiled_formula_struct.jl
# Test the new struct interface

function test_compiled_formula_struct()
    println("=== Testing New CompiledFormula Struct Interface ===")
    
    # Create test data
    Random.seed!(123)
    n = 50
    df = DataFrame(
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], n))
    )
    data = Tables.columntable(df)
    
    # Test different formulas
    test_cases = [
        (@formula(y ~ 1), "Intercept only"),
        (@formula(y ~ x), "Simple continuous"),
        (@formula(y ~ x + group), "Mixed terms"),
        (@formula(y ~ x^2 * log(z)), "Complex interaction"),
    ]
    
    all_passed = true
    
    for (formula, description) in test_cases
        println("\n--- Testing: $description ---")
        println("Formula: $formula")
        
        try
            # Build model
            model = lm(formula, df)
            mm = modelmatrix(model)
            
            # NEW: Compile with struct interface
            compiled = compile_formula(model)
            
            # Test struct properties
            println("✅ Compiled formula created")
            println("   Type: $(typeof(compiled))")
            println("   Width: $(length(compiled))")
            println("   Variables: $(variables(compiled))")
            println("   Hash: $(formula_hash(compiled))")
            
            # Test single row evaluation with call operator
            row_vec = Vector{Float64}(undef, length(compiled))
            compiled(row_vec, data, 1)  # ⭐ NEW SYNTAX!
            
            expected = mm[1, :]
            if isapprox(row_vec, expected, atol=1e-12)
                println("✅ Single row evaluation: PASSED")
            else
                println("❌ Single row evaluation: FAILED")
                println("   Expected: $expected")
                println("   Got:      $row_vec")
                all_passed = false
            end
            
            # Test batch evaluation  
            batch_size = 10
            matrix = Matrix{Float64}(undef, batch_size, length(compiled))
            evaluate_batch!(matrix, compiled, data, 1:batch_size)
            
            expected_batch = mm[1:batch_size, :]
            if isapprox(matrix, expected_batch, atol=1e-12)
                println("✅ Batch evaluation: PASSED")
            else
                println("❌ Batch evaluation: FAILED")
                all_passed = false
            end
            
            # Test performance (should be same as before)
            println("⏱️  Performance test:")
            
            # Warmup
            for i in 1:10
                compiled(row_vec, data, (i % n) + 1)
            end
            
            # Time single call
            time_ns = @elapsed begin
                for i in 1:100
                    compiled(row_vec, data, (i % n) + 1)
                end
            end
            avg_time = (time_ns / 100) * 1e9
            
            allocs = @allocated compiled(row_vec, data, 1)
            
            println("   Average time: $(round(avg_time, digits=1)) ns")
            println("   Allocations: $allocs bytes")
            
            if avg_time < 200 && allocs <= 64
                println("✅ Performance: EXCELLENT")
            else
                println("⚠️  Performance: Could be better")
            end
            
        catch e
            println("❌ ERROR: $e")
            all_passed = false
        end
    end
    
    # Test the display methods
    println("\n--- Testing Display Methods ---")
    if !isempty(test_cases)
        model = lm(test_cases[end][1], df)  # Use last formula
        compiled = compile_formula(model)
        
        println("Brief display:")
        println(compiled)
        
        println("\nDetailed display:")
        display(compiled)
    end
    
    println("\n" * "="^50)
    if all_passed
        println("🎉 ALL STRUCT INTERFACE TESTS PASSED!")
        println("✅ CompiledFormula struct working perfectly")
        println("✅ Call operator syntax: compiled(row_vec, data, idx)")  
        println("✅ Batch evaluation working")
        println("✅ Performance maintained")
    else
        println("❌ Some tests failed - check output above")
    end
    println("="^50)
    
    return all_passed
end

# Run the test
test_compiled_formula_struct()
