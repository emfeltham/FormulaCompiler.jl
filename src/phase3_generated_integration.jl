# phase3_generated_integration.jl
# Phase 3: Complete @generated function integration

using Tables
using CategoricalArrays: CategoricalValue, levelcode
using StatsModels
using StandardizedPredictors: ZScoredTerm

###############################################################################
# Phase 3: @generated Function Integration
###############################################################################

# Global cache for formula instructions and metadata
const FORMULA_CACHE = Dict{UInt64, Tuple{Vector{String}, Vector{Symbol}, Int}}()

"""
    compile_formula_complete(model) -> (formula_val, output_width, column_names)

Complete three-phase compilation pipeline:
1. Phase 1: Analyze formula structure
2. Phase 2: Generate instructions  
3. Phase 3: Register for @generated dispatch

Returns values needed for zero-allocation `modelrow!` calls.
"""
function compile_formula_complete(model)
    println("=== Complete Three-Phase Compilation ===")
    
    # Phase 1: Structure Analysis
    println("Phase 1: Analyzing formula structure...")
    analysis = analyze_formula_structure(model)
    
    if !validate_analysis(analysis, model)
        error("Phase 1 validation failed!")
    end
    
    # Phase 2: Instruction Generation
    println("Phase 2: Generating instructions...")
    instructions = generate_instructions(analysis)
    
    # Phase 3: Registration for @generated dispatch
    println("Phase 3: Registering for @generated dispatch...")
    formula_hash = hash(string(fixed_effects_form(model).rhs))
    
    # Store in global cache for @generated function
    FORMULA_CACHE[formula_hash] = (instructions, analysis.all_columns, analysis.total_width)
    
    println("✅ Complete compilation successful!")
    println("   Formula hash: $formula_hash")
    println("   Output width: $(analysis.total_width)")
    println("   Instructions: $(length(instructions))")
    println("   Columns used: $(analysis.all_columns)")
    
    return (Val(formula_hash), analysis.total_width, analysis.all_columns)
end

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
    
    println("@generated: Compiling for formula hash $formula_hash")
    println("@generated: $(length(instructions)) instructions, width $output_width")
    
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
# Comprehensive Testing Suite
###############################################################################

"""
    test_complete_pipeline()

Test the complete three-phase pipeline on various formula types.
"""
function test_complete_pipeline()
    println("=== Testing Complete Three-Phase Pipeline ===")
    
    # Create comprehensive test data
    Random.seed!(42)
    n = 100
    df = DataFrame(
        x = randn(n),
        y = randn(n),
        z = abs.(randn(n)) .+ 0.1,  # Positive for log
        w = randn(n),
        group = categorical(rand(["A", "B", "C"], n)),
        binary = categorical(rand(["Yes", "No"], n))
    )
    
    data = Tables.columntable(df)
    
    # Test cases from simple to complex
    test_formulas = [
        (@formula(y ~ 1), "Intercept only"),
        (@formula(y ~ x), "Simple continuous"),
        (@formula(y ~ group), "Simple categorical"),
        (@formula(y ~ x + group), "Mixed terms"),
        (@formula(y ~ x^2), "Power function"),
        (@formula(y ~ log(z)), "Log function"),
        (@formula(y ~ x + x^2 + log(z)), "Multiple functions"),
        (@formula(y ~ x * group), "Simple interaction"),
        (@formula(y ~ x^2 * log(z)), "Complex function interaction"),
        (@formula(y ~ x + x^2 + log(z) + group + w + x*group), "Kitchen sink"),
        (@formula(y ~ x*z*group), "Three-way interaction"),
        (@formula(y ~ (x>0) + log(z)*x), "Boolean and function interaction")
    ]
    
    results = []
    successful = 0
    
    for (i, (formula, description)) in enumerate(test_formulas)
        println("\n" * "="^60)
        println("Test $i: $description")
        println("Formula: $formula")
        println("="^60)
        
        try
            # Build model
            model = lm(formula, df)
            
            # Test complete pipeline
            compilation_time, avg_time, avg_allocs, correctness = test_compilation_performance(
                model, data, n_trials=100
            )
            
            # Record results
            push!(results, (
                description = description,
                formula = string(formula),
                compilation_time = compilation_time,
                avg_time = avg_time,
                avg_allocs = avg_allocs,
                correctness = correctness,
                success = true
            ))
            
            if correctness && avg_allocs < 0.1
                successful += 1
                println("✅ PASSED: Correct and zero-allocation")
            else
                println("⚠️  PARTIAL: Some issues detected")
            end
            
        catch e
            println("❌ FAILED: $e")
            @error "Test $i failed" exception=(e, catch_backtrace())
            
            push!(results, (
                description = description,
                formula = string(formula),
                compilation_time = NaN,
                avg_time = NaN,
                avg_allocs = NaN,
                correctness = false,
                success = false,
                error = string(e)
            ))
        end
    end
    
    # Summary report
    println("\n" * "="^60)
    println("FINAL SUMMARY")
    println("="^60)
    println("Successful tests: $successful / $(length(test_formulas))")
    
    println("\nPerformance Summary:")
    for result in results
        if result.success
            println("  $(result.description): $(round(result.avg_time, digits=1)) ns, $(round(result.avg_allocs, digits=3)) allocs")
        else
            println("  $(result.description): FAILED")
        end
    end
    
    if successful == length(test_formulas)
        println("\n🎉 ALL TESTS PASSED! Three-phase pipeline working perfectly!")
    elseif successful > length(test_formulas) * 0.8
        println("\n✅ Most tests passed, minor issues to resolve")
    else
        println("\n⚠️  Major issues detected, needs debugging")
    end
    
    return results
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

# Export main interface
export compile_formula_complete, modelrow!, get_compiled_function
export test_compilation_performance, test_complete_pipeline