
using Revise
using Test
using BenchmarkTools, Profile
using FormulaCompiler

using Statistics
using DataFrames, GLM, Tables, CategoricalArrays, Random
using StatsModels, StandardizedPredictors
using MixedModels
using BenchmarkTools

using FormulaCompiler:
    compile_term
    
# Set consistent random seed for reproducible tests
Random.seed!(06515)

using FormulaCompiler:
    compile_function_term, compile_matrix_term,
    ScratchAllocator
using FormulaCompiler:
    compile_formula_specialized, show_specialized_info


###############################################################################
# TESTING AND VALIDATION
###############################################################################

"""
    test_continuous_specialization(formula, df, data; n_iterations=1000)

Test continuous variable specialization against current implementation.
"""
function test_continuous_specialization(formula, df, data; n_iterations=1000)
    println("Testing continuous variable specialization...")
    
    # Compile both versions
    model = fit(LinearModel, formula, df)
    current_compiled = compile_formula(model, data)
    specialized_compiled = compile_formula_specialized(model, data)
    
    println("Formula: $formula")
    println("Output width: $(length(current_compiled))")
    println("Specialized type: $(typeof(specialized_compiled))")
    
    # Test correctness
    output_current = Vector{Float64}(undef, length(current_compiled))
    output_specialized = Vector{Float64}(undef, length(specialized_compiled))
    
    # Test several rows
    for test_row in [1, 5, 10, 50]
        fill!(output_current, NaN)
        fill!(output_specialized, NaN)
        
        current_compiled(output_current, data, test_row)
        specialized_compiled(output_specialized, data, test_row)
        
        if !isapprox(output_current, output_specialized, rtol=1e-14)
            error("Results differ at row $test_row: $output_current vs $output_specialized")
        end
    end
    println("✅ Correctness test passed")
    
    # Performance comparison
    println("\nPerformance comparison:")
    
    # Warmup
    for i in 1:10
        current_compiled(output_current, data, 1)
        specialized_compiled(output_specialized, data, 1)
    end
    
    # Benchmark current implementation
    current_allocs = @allocated begin
        for i in 1:100
            row_idx = ((i - 1) % length(data.x)) + 1
            current_compiled(output_current, data, row_idx)
        end
    end
    current_allocs_per_call = current_allocs / 100
    
    # Benchmark specialized implementation  
    specialized_allocs = @allocated begin
        for i in 1:100
            row_idx = ((i - 1) % length(data.x)) + 1
            specialized_compiled(output_specialized, data, row_idx)
        end
    end
    specialized_allocs_per_call = specialized_allocs / 100
    
    println("Current implementation: $(current_allocs_per_call) bytes per call")
    println("Specialized implementation: $(specialized_allocs_per_call) bytes per call")
    
    if specialized_allocs_per_call == 0
        println("🎉 ZERO ALLOCATIONS ACHIEVED!")
    elseif specialized_allocs_per_call < current_allocs_per_call
        reduction = (1 - specialized_allocs_per_call / current_allocs_per_call) * 100
        println("📈 $(round(reduction, digits=1))% allocation reduction")
    else
        println("⚠️  Specialized implementation has more allocations")
    end
    
    println("\nTiming comparison:")
    print("Current: ")
    @btime $current_compiled($output_current, $data, 1)
    
    print("Specialized: ")
    @btime $specialized_compiled($output_specialized, $data, 1)
    
    return specialized_compiled
end

"""
    run_step1_tests()

Run comprehensive tests for Step 1 implementation.
"""
function run_step1_tests()
    
    n = 200
    df = DataFrame(
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.01,
        w = randn(n),
        t = randn(n),
        response = randn(n)
    )
    data = Tables.columntable(df)
    
    println("="^60)
    println("STEP 1 TESTING: CONTINUOUS VARIABLES ONLY")
    println("="^60)
    
    # Test formulas (continuous only)
    test_formulas = [
        @formula(response ~ 1),           # Intercept only (should be handled as constant)
        @formula(response ~ x),           # Single continuous
        @formula(response ~ x + y),       # Multiple continuous
        @formula(response ~ x + y + z),   # More continuous variables
        @formula(response ~ x + y + z + w + t),  # Many continuous variables
    ]
    
    for (i, formula) in enumerate(test_formulas)
        println("\n--- Test $i: $formula ---")
        try
            test_continuous_specialization(formula, df, data)
            println("✅ Test $i passed")
        catch e
            if occursin("only supports continuous", string(e))
                println("⏭️  Test $i skipped (contains non-continuous operations)")
            else
                println("❌ Test $i failed: $e")
                rethrow(e)
            end
        end
    end
    
    println("\n" * "="^60)
    println("STEP 1 TESTING COMPLETE")
    println("="^60)
end

# Run the comprehensive test suite
run_step1_tests()

# Or test a specific formula

n = 100
df = DataFrame(x = randn(n), y = randn(n), response = randn(n))
data = Tables.columntable(df)

specialized = test_continuous_specialization(@formula(response ~ x + y), df, data)
show_specialized_info(specialized)