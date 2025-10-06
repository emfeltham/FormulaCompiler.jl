# test_loop_allocations.jl
# Comprehensive test for FormulaCompiler loop allocation behavior
# 
# Usage:
#   julia --project=. test/test_loop_allocations.jl
#   
# This test validates Margins.jl's claims about FormulaCompiler allocations
# and tracks progress on the zero-allocation loop performance goal.

using Pkg
Pkg.activate(".")

using FormulaCompiler
using GLM, DataFrames, Tables, CategoricalArrays
using Test

println("=" ^ 60)
println("FormulaCompiler Loop Allocation Test")
println("=" ^ 60)
println("Purpose: Validate zero-allocation claims for loop usage patterns")
println("Context: Testing Margins.jl's allocation analysis claims")
println()

# Test configuration
const N_ROWS = 1000
const N_TEST_CALLS = 1000
const EXPECTED_ALLOCATION_PER_CALL = 0.0  # Goal: zero allocations

# Create test data with proper categorical handling
function create_test_data(n=N_ROWS)
    return DataFrame(
        y = randn(n),
        x = randn(n),
        age = rand(18:80, n),
        group = categorical(rand(["A", "B"], n))
    )
end

# Test suite
function run_allocation_tests()
    println("Creating test data ($N_ROWS rows)...")
    df = create_test_data()
    model = lm(@formula(y ~ x + age + group), df)
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    output = Vector{Float64}(undef, length(compiled))
    
    println("Setup complete")
    println("   - Model: $(length(compiled)) parameters")
    println("   - Compiled type: $(typeof(compiled))")
    println()
    
    # Warmup phase
    println("Warmup phase...")
    for i in 1:10
        compiled(output, data, 1)
        modelrow!(output, compiled, data, 1)
    end
    println("Warmup complete")
    println()
    
    # Test results storage
    results = Dict{String, Float64}()
    
    # Test 1: Single direct compiled() call
    println("Test 1: Single compiled() call (post-warmup)")
    allocs_single = @allocated compiled(output, data, 1)
    results["single_compiled"] = allocs_single
    status_single = allocs_single == 0 ? "PASS" : "FAIL"
    println("   Result: $allocs_single bytes $status_single")
    println()
    
    # Test 2: Loop of direct compiled() calls
    println("Test 2: Loop of direct compiled() calls ($N_TEST_CALLS iterations)")
    function test_compiled_loop(compiled, output, data, n_calls)
        for i in 1:n_calls
            row = ((i-1) % N_ROWS) + 1
            compiled(output, data, row)
        end
    end
    
    allocs_loop = @allocated test_compiled_loop(compiled, output, data, N_TEST_CALLS)
    allocs_per_call = allocs_loop / N_TEST_CALLS
    results["loop_compiled"] = allocs_per_call
    status_loop = allocs_per_call == EXPECTED_ALLOCATION_PER_CALL ? "PASS" : "FAIL"
    println("   Total allocation: $allocs_loop bytes")
    println("   Per call: $allocs_per_call bytes $status_loop")
    println()
    
    # Test 3: Loop of modelrow!() calls (pre-compiled)
    println("Test 3: Loop of modelrow!(compiled) calls ($N_TEST_CALLS iterations)")
    function test_modelrow_loop(output, compiled, data, n_calls)
        for i in 1:n_calls
            row = ((i-1) % N_ROWS) + 1
            modelrow!(output, compiled, data, row)
        end
    end
    
    allocs_modelrow_loop = @allocated test_modelrow_loop(output, compiled, data, N_TEST_CALLS)
    allocs_modelrow_per_call = allocs_modelrow_loop / N_TEST_CALLS
    results["loop_modelrow_precompiled"] = allocs_modelrow_per_call
    status_modelrow = allocs_modelrow_per_call == EXPECTED_ALLOCATION_PER_CALL ? "PASS" : "FAIL"
    println("   Total allocation: $allocs_modelrow_loop bytes")
    println("   Per call: $allocs_modelrow_per_call bytes $status_modelrow")
    println("   Overhead vs direct: $(allocs_modelrow_per_call - allocs_per_call) bytes/call")
    println()
    
    # Test 4: Loop of modelrow!(model) calls (with cache lookup)
    println("Test 4: Loop of modelrow!(model) calls with caching ($N_TEST_CALLS iterations)")
    function test_modelrow_model_loop(output, model, data, n_calls)
        for i in 1:n_calls
            row = ((i-1) % N_ROWS) + 1
            modelrow!(output, model, data, row; cache=true)
        end
    end
    
    allocs_model_loop = @allocated test_modelrow_model_loop(output, model, data, N_TEST_CALLS)
    allocs_model_per_call = allocs_model_loop / N_TEST_CALLS
    results["loop_modelrow_cached"] = allocs_model_per_call
    println("   Total allocation: $allocs_model_loop bytes")
    println("   Per call: $allocs_model_per_call bytes")
    println("   Cache overhead: $(allocs_model_per_call - allocs_modelrow_per_call) bytes/call")
    println()
    
    # Test 5: Derivative functions if available
    if isdefined(FormulaCompiler, :marginal_effects_eta!) && isdefined(FormulaCompiler, :derivativeevaluator)
        println("Test 5: marginal_effects_eta! loop ($N_TEST_CALLS iterations)")
        try
            vars = [:x, :age]
            de_fd = derivativeevaluator(:fd, compiled, data, vars)
            g = Vector{Float64}(undef, length(vars))
            coefs = coef(model)

            function test_me_loop(g, de_fd, coefs, n_calls)
                for i in 1:n_calls
                    row = ((i-1) % N_ROWS) + 1
                    marginal_effects_eta!(g, de_fd, coefs, row)
                end
            end
            
            allocs_me_loop = @allocated test_me_loop(g, de_fd, coefs, N_TEST_CALLS)
            allocs_me_per_call = allocs_me_loop / N_TEST_CALLS
            results["loop_marginal_effects"] = allocs_me_per_call
            status_me = allocs_me_per_call == EXPECTED_ALLOCATION_PER_CALL ? "PASS" : "FAIL"
            println("   Total allocation: $allocs_me_loop bytes")
            println("   Per call: $allocs_me_per_call bytes $status_me")
        catch e
            println("   Error testing marginal_effects_eta!: $e")
            results["loop_marginal_effects"] = NaN
        end
        println()
    else
        println("Test 5: marginal_effects_eta! not available in this version")
        results["loop_marginal_effects"] = NaN
        println()
    end
    
    return results
end

# Summary and analysis
function summarize_results(results)
    println("=" ^ 60)
    println("ALLOCATION TEST SUMMARY")
    println("=" ^ 60)
    
    # Zero-allocation status
    all_zero = true
    core_loops_zero = true
    
    println("Zero-Allocation Status:")
    for (test, allocation) in results
        if test in ["loop_compiled", "loop_modelrow_precompiled", "loop_marginal_effects"]
            status = isnan(allocation) ? "N/A" : (allocation == 0.0 ? "ZERO" : "ALLOCATES")
            println("   $test: $(isnan(allocation) ? "N/A" : string(allocation)) bytes/call $status")
            if !isnan(allocation) && allocation > 0
                all_zero = false
                if test in ["loop_compiled"]
                    core_loops_zero = false
                end
            end
        end
    end
    
    println()
    println("Analysis:")
    
    if haskey(results, "loop_compiled") && results["loop_compiled"] > 0
        println("CORE ISSUE: Direct compiled() calls allocate $(results["loop_compiled"]) bytes/call in loops")
        println("   This confirms Margins.jl's findings - FormulaCompiler is NOT zero-allocation")
        println("   Root cause likely: fill!(scratch, zero(T)) in execution.jl:91")
    end
    
    if haskey(results, "loop_modelrow_precompiled") && haskey(results, "loop_compiled")
        overhead = results["loop_modelrow_precompiled"] - results["loop_compiled"]
        if overhead > 0
            println("WARNING: WRAPPER OVERHEAD: modelrow! adds $overhead bytes/call overhead")
            println("   Likely causes: @assert statements, function call overhead")
        end
    end
    
    if haskey(results, "loop_modelrow_cached") && haskey(results, "loop_modelrow_precompiled")
        cache_overhead = results["loop_modelrow_cached"] - results["loop_modelrow_precompiled"]
        if cache_overhead > 0
            println("CACHE OVERHEAD: Dictionary lookup adds $cache_overhead bytes/call")
        end
    end
    
    println()
    println("VERDICT:")
    if core_loops_zero
        println("FormulaCompiler achieves zero-allocation loop performance")
        println("   Zero-allocation claims are ACCURATE")
    else
        println("FormulaCompiler does NOT achieve zero-allocation loop performance")  
        println("   Margins.jl's analysis is CORRECT")
        println("   Zero-allocation claims are MISLEADING for practical usage")
    end
    
    println()
    println("Recommendations:")
    if !core_loops_zero
        println("   1. Fix core allocation in src/compilation/execution.jl:91")
        println("      Replace: fill!(scratch, zero(T))")
        println("      With:    fill!(scratch, 0.0)")
        println("   2. Run this test again to verify fix")
        println("   3. See ALLOCATION_LOOP_FIX.md for detailed implementation plan")
    else
        println("   Core loop performance is optimal")
        println("   Consider optimizing wrapper functions if needed")
    end
    
    return all_zero, core_loops_zero
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running FormulaCompiler allocation tests...")
    println()
    
    results = run_allocation_tests()
    all_zero, core_zero = summarize_results(results)
    
    # Exit with appropriate code
    if core_zero
        println("\nSUCCESS: Core loop performance verified!")
        exit(0)
    else
        println("\nFAILURE: Loop allocations detected - fix needed")
        exit(1)
    end
end