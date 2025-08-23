###############################################################################
# DEBUGGING AND VALIDATION
###############################################################################

"""
    trace_recursive_execution(categorical_data::Tuple, description="")

Debug function to trace how the recursive execution unfolds.
"""
function trace_recursive_execution(categorical_data::Tuple, description="")
    println("Tracing recursive execution: $description")
    println("  Input tuple type: $(typeof(categorical_data))")
    println("  Tuple length: $(length(categorical_data))")
    
    if length(categorical_data) == 0
        println("  → Base case: empty tuple")
    else
        println("  → Recursive case:")
        println("    First element type: $(typeof(categorical_data[1]))")
        if length(categorical_data) > 1
            println("    Remaining elements: $(typeof(Base.tail(categorical_data)))")
        else
            println("    Remaining elements: (none - will hit base case)")
        end
    end
    
    return nothing
end

"""
    benchmark_recursive_categorical(formula, df, data; n_iterations=1000)

Benchmark the recursive categorical execution approach.
"""
function benchmark_recursive_categorical(formula, df, data; n_iterations=1000)
    println("Benchmarking recursive categorical execution...")
    println("Formula: $formula")
    
    # Compile the formula
    model = fit(LinearModel, formula, df)
    compiled = compile_formula(model, data)
    output = Vector{Float64}(undef, length(compiled))
    
    println("Compiled type: $(typeof(compiled.data.categorical))")
    
    # Trace the recursion structure
    trace_recursive_execution(compiled.data.categorical, "Compiled formula")
    
    # Warmup
    for _ in 1:20
        compiled(output, data, 1)
    end
    
    # Benchmark allocation
    alloc = @allocated begin
        for i in 1:n_iterations
            row_idx = ((i - 1) % length(data.x)) + 1
            compiled(output, data, row_idx)
        end
    end
    
    avg_alloc = alloc / n_iterations
    
    println("Performance results:")
    println("  Iterations: $n_iterations")
    println("  Total allocations: $alloc bytes")
    println("  Average per call: $avg_alloc bytes")
    
    if avg_alloc == 0
        println("  🎯 PERFECT: Zero allocations achieved!")
    elseif avg_alloc <= 32
        println("  ✅ EXCELLENT: ≤32 bytes per call")
    elseif avg_alloc <= 64
        println("  ✅ GOOD: ≤64 bytes per call")
    else
        println("  ⚠️  NEEDS WORK: >64 bytes per call")
    end
    
    # Test correctness
    test_output1 = Vector{Float64}(undef, length(compiled))
    test_output2 = Vector{Float64}(undef, length(compiled))
    
    compiled(test_output1, data, 1)
    compiled(test_output2, data, 5)
    
    println("  Sample outputs look reasonable: $(all(isfinite, test_output1) && all(isfinite, test_output2))")
    
    return avg_alloc
end

"""
    test_recursive_approach()

Test the recursive approach on various categorical configurations.
"""
function test_recursive_approach()
    println("="^60)
    println("TESTING RECURSIVE CATEGORICAL EXECUTION")
    println("="^60)
    
    # Create test data
    n = 100
    df = DataFrame(
        x = randn(n),
        group3 = categorical(rand(["A", "B", "C"], n)),           
        group4 = categorical(rand(["W", "X", "Y", "Z"], n)),      
        binary = categorical(rand(["Yes", "No"], n)),             
        response = randn(n)
    )
    data = Tables.columntable(df)
    
    # Test cases
    test_cases = [
        (@formula(response ~ 1), "No categoricals"),
        (@formula(response ~ group3), "1 categorical"),
        (@formula(response ~ group3 + group4), "2 categoricals"),
        (@formula(response ~ group3 + group4 + binary), "3 categoricals"),
        (@formula(response ~ x + group3 + group4 + binary), "3 categoricals + continuous"),
    ]
    
    results = []
    for (formula, description) in test_cases
        println("\n" * "-"^40)
        result = benchmark_recursive_categorical(formula, df, data, n_iterations=100)
        push!(results, (description, result))
    end
    
    # Summary
    println("\n" * "="^40)
    println("RECURSIVE APPROACH SUMMARY")
    println("="^40)
    
    for (description, result) in results
        status = result == 0 ? "🎯" : result <= 32 ? "✅" : "⚠️"
        println("$status $description: $result bytes per call")
    end
    
    # Check if the 3-categorical problem is solved
    three_cat_result = results[findfirst(r -> r[1] == "3 categoricals", results)][2]
    if three_cat_result == 0
        println("\n🎉 SUCCESS: 3-categorical allocation problem solved!")
    elseif three_cat_result < 100
        println("\n✅ MAJOR IMPROVEMENT: 3-categorical allocations greatly reduced")
    else
        println("\n⚠️  PARTIAL: 3-categorical allocations reduced but not eliminated")
    end
    
    return results
end
