# test_step2_categorical_support.jl

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
# ENHANCED TESTING FUNCTIONS
###############################################################################

"""
    test_enhanced_specialization(formula, df, data; n_iterations=1000)

Test enhanced specialization (constants + continuous + categorical) against current implementation.
"""
function test_enhanced_specialization(formula, df, data; n_iterations=1000)
    println("Testing enhanced specialization (constants + continuous + categorical)...")
    
    # Compile both versions
    model = fit(LinearModel, formula, df)
    current_compiled = compile_formula(model, data)
    specialized_compiled = compile_formula_specialized_enhanced(model, data)
    
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
    run_step2_tests()

Run comprehensive tests for Step 2 implementation.
"""
function run_step2_tests()
    
    n = 200
    df = DataFrame(
        x = randn(n),
        y = randn(n), 
        z = abs.(randn(n)) .+ 0.01,
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
    println("STEP 2 TESTING: CONSTANTS + CONTINUOUS + CATEGORICAL")
    println("="^60)
    
    # Test formulas (no functions or interactions yet)
    test_formulas = [
        @formula(response ~ 1),                      # Constants only
        @formula(response ~ x),                      # Continuous only  
        @formula(response ~ group3),                 # Categorical only
        @formula(response ~ x + group3),             # Continuous + categorical
        @formula(response ~ x + y + group3),         # Multiple continuous + categorical
        @formula(response ~ group3 + group4),        # Multiple categorical
        @formula(response ~ x + y + group3 + binary), # Mixed: continuous + multiple categorical
        @formula(response ~ x + y + z + group3 + group4 + binary), # Many variables
    ]
    
    for (i, formula) in enumerate(test_formulas)
        println("\n--- Test $i: $formula ---")
        try
            test_enhanced_specialization(formula, df, data)
            println("✅ Test $i passed")
        catch e
            if occursin("only supports constants, continuous, and categorical", string(e))
                println("⏭️  Test $i skipped (contains functions/interactions)")
            else
                println("❌ Test $i failed: $e")
                rethrow(e)
            end
        end
    end
    
    println("\n" * "="^60)
    println("STEP 2 TESTING COMPLETE")
    println("="^60)
end

using FormulaCompiler:
    compile_formula_specialized_enhanced

# Run the comprehensive test suite
run_step2_tests()