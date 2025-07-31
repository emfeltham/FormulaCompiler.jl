# test_step3_function_support.jl

using Revise
using Test
using BenchmarkTools, Profile
using FormulaCompiler

using Statistics
using DataFrames, GLM, Tables, CategoricalArrays, Random
using StatsModels, StandardizedPredictors
using MixedModels
using BenchmarkTools
    
# Set consistent random seed for reproducible tests
Random.seed!(06515)

    ScratchAllocator
using FormulaCompiler:
    compile_formula_specialized, show_specialized_info

###############################################################################
# COMPREHENSIVE TESTING FUNCTIONS
###############################################################################

"""
    test_comprehensive_specialization(formula, df, data; n_iterations=1000)

Test comprehensive specialization (constants + continuous + categorical + functions) against current implementation.
"""
function test_comprehensive_specialization(formula, df, data; n_iterations=1000)
    println("Testing comprehensive specialization (constants + continuous + categorical + functions)...")
    
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
    for test_row in [1, 5, 10, 25, 50]
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
    run_step3_tests()

Run comprehensive tests for Step 3 implementation.
"""
function run_step3_tests()
    # Create test data with all variable types
    
    n = 200
    df = DataFrame(
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.01,  # Positive for log
        w = randn(n),
        t = randn(n),
        group3 = categorical(rand(["A", "B", "C"], n)),           
        group4 = categorical(rand(["W", "X", "Y", "Z"], n)),      
        binary = categorical(rand(["Yes", "No"], n)),             
        group5 = categorical(rand(["P", "Q", "R", "S", "T"], n)), 
        response = randn(n)
    )
    data = Tables.columntable(df)
    
    println("="^60)
    println("STEP 3 TESTING: CONSTANTS + CONTINUOUS + CATEGORICAL + FUNCTIONS")
    println("="^60)
    
    # Test formulas (no interactions yet)
    test_formulas = [
        @formula(response ~ 1),                          # Constants only
        @formula(response ~ x),                          # Continuous only  
        @formula(response ~ group3),                     # Categorical only
        @formula(response ~ log(z)),                     # Simple function
        @formula(response ~ x + log(z)),                 # Continuous + function
        @formula(response ~ log(z) + group3),            # Function + categorical
        @formula(response ~ x + log(z) + group3),        # All three types
        @formula(response ~ log(z) + exp(w)),            # Multiple functions
        @formula(response ~ log(z) + exp(w) + sin(x)),   # Many functions
        @formula(response ~ x + y + log(z) + group3 + group4), # Complex mix
        @formula(response ~ log(exp(z))),                # Nested functions
        @formula(response ~ log(abs(x) + abs(y))),       # Function with multiple args
        @formula(response ~ x^2 + log(z) + sqrt(abs(w))), # Complex functions
    ]
    
    for (i, formula) in enumerate(test_formulas)
        println("\n--- Test $i: $formula ---")
        try
            test_comprehensive_specialization(formula, df, data)
            println("✅ Test $i passed")
        catch e
            if occursin("only supports constants, continuous, categorical, and functions", string(e))
                println("⏭️  Test $i skipped (contains interactions)")
            else
                println("❌ Test $i failed: $e")
                rethrow(e)
            end
        end
    end
    
    println("\n" * "="^60)
    println("STEP 3 TESTING COMPLETE")
    println("="^60)
end

# Run the comprehensive test suite
run_step3_tests()

###############################################################################
# LINEAR COMPREHENSIVE TESTING FUNCTIONS
###############################################################################

"""
    test_linear_comprehensive_specialization(formula, df, data; n_iterations=1000)

Test linear comprehensive specialization against current implementation.
"""
function test_linear_comprehensive_specialization(formula, df, data; n_iterations=1000)
    println("Testing linear comprehensive specialization (zero-allocation functions)...")
    
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
    for test_row in [1, 5, 10, 25, 50]
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
    run_step3_polish_tests()

Run comprehensive tests for Step 3 Polish (linear function execution).
"""
function run_step3_polish_tests()
    
    n = 200
    df = DataFrame(
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.01,  # Positive for log
        w = randn(n),
        t = randn(n),
        group3 = categorical(rand(["A", "B", "C"], n)),           
        group4 = categorical(rand(["W", "X", "Y", "Z"], n)),      
        binary = categorical(rand(["Yes", "No"], n)),             
        group5 = categorical(rand(["P", "Q", "R", "S", "T"], n)), 
        response = randn(n)
    )
    data = Tables.columntable(df)
    
    println("="^70)
    println("STEP 3 POLISH TESTING: LINEAR ZERO-ALLOCATION FUNCTIONS")
    println("="^70)
    
    # Test formulas focusing on function operations
    test_formulas = [
        @formula(response ~ 1),                          # No functions (baseline)
        @formula(response ~ x),                          # No functions (baseline)
        @formula(response ~ log(z)),                     # Simple function
        @formula(response ~ x + log(z)),                 # Mixed
        @formula(response ~ log(z) + exp(w)),            # Multiple functions
        @formula(response ~ log(z) + group3),            # Function + categorical
        @formula(response ~ x + log(z) + group3),        # All operation types
        @formula(response ~ log(exp(z))),                # Nested functions
        @formula(response ~ x^2),                        # Binary function
        @formula(response ~ abs(x) + abs(y)),            # Multiple unary functions
        @formula(response ~ log(abs(x))),                 # Nested unary functions
        @formula(response ~ x + y + log(z) + exp(w) + group3), # Complex mix
    ]
    
    for (i, formula) in enumerate(test_formulas)
        println("\n--- Test $i: $formula ---")
        try
            test_linear_comprehensive_specialization(formula, df, data)
            println("✅ Test $i passed")
        catch e
            if occursin("not yet supported", string(e))
                println("⏭️  Test $i skipped (feature not yet implemented: $e)")
            else
                println("❌ Test $i failed: $e")
                rethrow(e)
            end
        end
    end
    
    println("\n" * "="^70)
    println("STEP 3 POLISH TESTING COMPLETE")
    println("="^70)
end

using FormulaCompiler: compile_formula_specialized


run_step3_polish_tests()