# test_position_mapping.jl
# Tests position mapping functionality in the compilation system

using Test
using BenchmarkTools
using FormulaCompiler
using Statistics
using DataFrames, GLM, Tables, CategoricalArrays, Random
using StatsModels
using FormulaCompiler: test_correctness, test_data

# Set consistent random seed for reproducible tests
Random.seed!(06515)

###############################################################################
# TESTING AND VALIDATION
###############################################################################

"""
    test_position_mapping(formula, df, data)

Test the compilation system's position mapping functionality.
"""
function test_position_mapping(formula, df, data)
    println("="^70)
    println("TESTING POSITION MAPPING")
    println("Formula: $formula")
    println("="^70)
    
    # Compile formula
    model = fit(LinearModel, formula, df)
    compiled = compile_formula(model, data)
    
    # Test correctness against modelmatrix - this is critical
    output_compiled = Vector{Float64}(undef, length(compiled))
    mm = modelmatrix(model)
    
    println("🔍 Testing correctness against modelmatrix on multiple rows...")
    correctness_passed = true
    
    for test_row in [1, 2, 5, 10, 25, 50]
        if test_row > size(mm, 1)
            continue  # Skip if test row exceeds data size
        end
        
        fill!(output_compiled, NaN)
        
        compiled(output_compiled, data, test_row)
        expected_row = mm[test_row, :]
        
        if !isapprox(output_compiled, expected_row, rtol=1e-12)
            println("❌ CORRECTNESS ERROR at row $test_row:")
            println("  Expected: $expected_row")
            println("  Actual:   $output_compiled")
            println("  Max diff: $(maximum(abs.(output_compiled .- expected_row)))")
            correctness_passed = false
            break
        end
    end
    
    if correctness_passed
        println("✅ Correctness test passed for all test rows")
    else
        println("❌ Correctness test FAILED - stopping here")
        return false
    end
    
    # Test allocation performance
    println("\n📊 Testing allocation performance...")
    
    # Warmup
    for i in 1:10
        compiled(output_compiled, data, 1)
    end
    
    # Benchmark allocation performance
    compiled_allocs = @allocated begin
        for i in 1:100
            row_idx = ((i-1) % size(mm, 1)) + 1
            compiled(output_compiled, data, row_idx)
        end
    end
    
    allocs_per_call = compiled_allocs / 100
    
    println("Compilation allocations: $(allocs_per_call) bytes per call")
    
    if allocs_per_call == 0
        println("🎉 ZERO ALLOCATIONS ACHIEVED!")
    elseif allocs_per_call <= 32
        println("🏆 EXCELLENT: Very low allocation (≤32 bytes per call)!")
    elseif allocs_per_call <= 96
        println("🥇 VERY GOOD: Low allocation (≤96 bytes per call)!")
    elseif allocs_per_call <= 200
        println("🥈 GOOD: Moderate allocation (≤200 bytes per call)")
    else
        println("⚠️  High allocation: $(allocs_per_call) bytes per call")
    end
    
    # Timing benchmark
    println("\n⏱️  Timing benchmark:")
    print("Compiled formula: ")
    @btime $compiled($output_compiled, $data, 1)
    
    return true
end

"""
    run_comprehensive_position_mapping_tests()

Run comprehensive tests on all formula types.
"""
function run_comprehensive_position_mapping_tests()
    # Create test data
    df, data = test_data(n=200)
    
    println("🚀 COMPREHENSIVE POSITION MAPPING TESTING")
    println("="^80)
    
    # Test formulas progressively
    test_formulas = [
        (@formula(response ~ x), "Simple continuous"),
        (@formula(response ~ group3), "Simple categorical"),
        (@formula(response ~ x + y), "Multiple continuous"),
        (@formula(response ~ x * y), "Simple 2-way continuous interaction"),
        (@formula(response ~ x * group3), "Continuous × Categorical"),  
        (@formula(response ~ log(z)), "Simple function"),
        (@formula(response ~ log(z) * group4), "Function × Categorical"),
        (@formula(response ~ x * y * group3), "3-way interaction"),
        (@formula(response ~ x * y * group3 + log(z) * group4), "Complex formula - Complete test"),
    ]
    
    all_passed = true
    
    for (i, (formula, description)) in enumerate(test_formulas)
        println("\n$(i). $description")
        println("   Formula: $formula")
        println("   " * "-"^60)
        
        try
            success = test_position_mapping(formula, df, data)
            if success
                println("   ✅ Test $i passed")
            else
                println("   ❌ Test $i FAILED")
                all_passed = false
                break  # Stop on first failure to debug
            end
        catch e
            println("   ❌ Test $i FAILED with exception: $e")
            all_passed = false
            break
        end
    end
    
    println("\n" * "="^80)
    if all_passed
        println("🎉 ALL POSITION MAPPING TESTS PASSED!")
        println("The position mapping system is working correctly.")
    else
        println("❌ SOME TESTS FAILED")
        println("Need to debug the position mapping implementation.")
    end
    println("="^80)
    
    return all_passed
end

# Run the tests when this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_comprehensive_position_mapping_tests()
end