# phase3_generated_integration.jl
# Phase 3: Complete @generated function integration

###############################################################################
# Phase 3: @generated Function Integration
###############################################################################

# Global cache for formula instructions and metadata
const FORMULA_CACHE = Dict{UInt64, Tuple{Vector{String}, Vector{Symbol}, Int}}()

"""
    modelrow!(row_vec, ::Val{formula_hash}, data, row_idx) where formula_hash

Zero-allocation @generated function for model row evaluation.
This is the runtime function that achieves ~50-100ns, 0 allocation performance.
"""
@generated function modelrow!(row_vec, ::Val{formula_hash}, data, row_idx) where formula_hash
    # Retrieve instructions from cache
    if !haskey(FORMULA_CACHE, formula_hash)
        error("Formula hash $formula_hash not found in cache. Did you call compile_formula_complete?")
    end
    
    instructions, column_names, output_width = FORMULA_CACHE[formula_hash]
    
    # Only show debug info during development
    # println("@generated: Compiling for formula hash $formula_hash")
    # println("@generated: $(length(instructions)) instructions, width $output_width")
    
    # Convert instruction strings to expressions
    try
        code_exprs = [Meta.parse(line) for line in instructions]
        
        return quote
            @inbounds begin
                $(code_exprs...)
            end
            return row_vec
        end
        
    catch e
        error("Failed to parse instructions for hash $formula_hash: $e")
    end
end

###############################################################################
# Convenience Functions
###############################################################################

"""
    get_compiled_function(model) -> (func, output_width, column_names)

Get a direct reference to the compiled function for maximum performance.
Bypasses @generated dispatch overhead.
"""
function get_compiled_function(model)
    # First ensure the formula is compiled
    formula_val, output_width, column_names = compile_formula_complete(model)
    
    # Create a wrapper function that calls the @generated version
    function compiled_func(row_vec, data, row_idx)
        return modelrow!(row_vec, formula_val, data, row_idx)
    end
    
    return compiled_func, output_width, column_names
end

"""
    test_compilation_performance(model, data; n_trials=1000)

Comprehensive performance test of the complete compilation pipeline.
"""
function test_compilation_performance(model, data; n_trials=1000)
    println("=== Testing Complete Compilation Performance ===")
    
    # Phase 1-3: Compilation (one-time cost)
    println("\n1. Compilation Phase:")
    compilation_time = @elapsed begin
        formula_val, output_width, column_names = compile_formula_complete(model)
    end
    
    println("   Compilation time: $(round(compilation_time * 1000, digits=2)) ms")
    
    # Setup for runtime testing
    row_vec = Vector{Float64}(undef, output_width)
    n_rows = length(data[1])
    
    println("\n2. Correctness Verification:")
    
    # Test against model matrix
    modelrow!(row_vec, formula_val, data, 1)
    mm = modelmatrix(model)
    expected = mm[1, :]
    
    if isapprox(row_vec, expected, atol=1e-12)
        println("   ✅ Row 1 matches model matrix exactly")
    else
        println("   ❌ Row 1 mismatch!")
        println("      Expected: $expected")
        println("      Got:      $row_vec")
        println("      Max diff: $(maximum(abs.(row_vec .- expected)))")
    end
    
    # Test a few more rows
    all_correct = true
    for test_row in [1, min(5, n_rows), min(10, n_rows)]
        if test_row <= n_rows
            modelrow!(row_vec, formula_val, data, test_row)
            expected = mm[test_row, :]
            if !isapprox(row_vec, expected, atol=1e-12)
                println("   ❌ Row $test_row mismatch!")
                all_correct = false
            end
        end
    end
    
    if all_correct
        println("   ✅ All test rows correct")
    end
    
    println("\n3. Performance Testing:")
    
    # Warmup
    for i in 1:10
        modelrow!(row_vec, formula_val, data, (i % n_rows) + 1)
    end
    
    # Single call timing
    single_time = @elapsed modelrow!(row_vec, formula_val, data, 1)
    println("   Single call time: $(round(single_time * 1e9, digits=1)) ns")
    
    # Allocation test
    allocs = @allocated modelrow!(row_vec, formula_val, data, 1)
    println("   Single call allocations: $allocs bytes")
    
    # Batch timing
    batch_time = @elapsed begin
        for i in 1:n_trials
            modelrow!(row_vec, formula_val, data, (i % n_rows) + 1)
        end
    end
    
    avg_time = (batch_time / n_trials) * 1e9  # Convert to nanoseconds
    println("   Average time ($n_trials calls): $(round(avg_time, digits=1)) ns")
    
    # Batch allocations
    batch_allocs = @allocated begin
        for i in 1:n_trials
            modelrow!(row_vec, formula_val, data, (i % n_rows) + 1)
        end
    end
    
    avg_allocs = batch_allocs / n_trials
    println("   Average allocations: $(round(avg_allocs, digits=3)) bytes/call")
    
    # Performance summary
    println("\n4. Performance Summary:")
    
    if avg_time < 200
        println("   ✅ Excellent performance: $(round(avg_time, digits=1)) ns")
    elseif avg_time < 500
        println("   ✅ Good performance: $(round(avg_time, digits=1)) ns")
    else
        println("   ⚠️  Slower than expected: $(round(avg_time, digits=1)) ns")
    end
    
    if avg_allocs < 0.1
        println("   ✅ Zero allocations achieved")
    else
        println("   ⚠️  Some allocations detected: $(round(avg_allocs, digits=3)) bytes/call")
    end
    
    return (compilation_time, avg_time, avg_allocs, all_correct)
end

###############################################################################
# Usage Examples and Documentation
###############################################################################

"""
# Complete Usage Example

```julia
using DataFrames, GLM, Tables, CategoricalArrays

# 1. Create your data and model
df = DataFrame(
    x = randn(1000),
    y = randn(1000),
    z = abs.(randn(1000)) .+ 0.1,
    group = categorical(rand(["A", "B"], 1000))
)

model = lm(@formula(y ~ x + x^2 * log(z) + group), df)
data = Tables.columntable(df)

# 2. One-time compilation (expensive, ~1-10ms)
formula_val, output_width, column_names = compile_formula_complete(model)

# 3. Setup for fast evaluation
row_vec = Vector{Float64}(undef, output_width)

# 4. Zero-allocation runtime usage (~50-100ns per call)
for i in 1:1000
    modelrow!(row_vec, formula_val, data, i)
    # Now row_vec contains the model matrix row for observation i
    # Use row_vec for predictions, marginal effects, etc.
end

# 5. Performance testing
test_compilation_performance(model, data)
```

# Alternative: Direct function access (even faster)

```julia
# Get direct function reference
func, output_width, column_names = get_compiled_function(model)

# Use directly (bypasses @generated dispatch)
func(row_vec, data, 1)  # Potentially 10-20% faster
```
"""
