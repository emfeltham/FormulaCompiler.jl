# test_position_mapping.jl

# test/runtests.jl
# Main test runner for FormulaCompiler.jl

using BenchmarkTools, Profile
using FormulaCompiler

using Statistics
using DataFrames, GLM, Tables, CategoricalArrays, Random
using StatsModels, StandardizedPredictors
    
# Set consistent random seed for reproducible tests
Random.seed!(06515)

using FormulaCompiler:
    compile_formula_specialized

###############################################################################
# TESTING AND VALIDATION
###############################################################################

"""
    test_position_mapping_approach(formula, df, data)

Test the all-in position mapping approach.
"""
function test_position_mapping_approach(formula, df, data)
    println("="^70)
    println("TESTING ALL-IN POSITION MAPPING APPROACH")
    println("Formula: $formula")
    println("="^70)
    
    # Compile both versions
    model = fit(LinearModel, formula, df)
    current_compiled = compile_formula(model, data)
    specialized_compiled = compile_formula_specialized(model, data)
    
    # Test correctness first - this is critical
    output_current = Vector{Float64}(undef, length(current_compiled))
    output_specialized = Vector{Float64}(undef, length(specialized_compiled))
    
    println("🔍 Testing correctness on multiple rows...")
    correctness_passed = true
    
    for test_row in [1, 2, 5, 10, 25, 50]
        fill!(output_current, NaN)
        fill!(output_specialized, NaN)
        
        current_compiled(output_current, data, test_row)
        specialized_compiled(output_specialized, data, test_row)
        
        if !isapprox(output_current, output_specialized, rtol=1e-14)
            println("❌ CORRECTNESS ERROR at row $test_row:")
            println("  Expected: $output_current")
            println("  Actual:   $output_specialized")
            println("  Max diff: $(maximum(abs.(output_current .- output_specialized)))")
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
    
    # Test performance
    println("\n📊 Testing performance...")
    
    # Warmup
    for i in 1:10
        current_compiled(output_current, data, 1)
        specialized_compiled(output_specialized, data, 1)
    end
    
    # Benchmark both implementations
    current_allocs = @allocated begin
        for i in 1:100
            current_compiled(output_current, data, ((i-1) % length(data.x)) + 1)
        end
    end
    
    specialized_allocs = @allocated begin
        for i in 1:100
            specialized_compiled(output_specialized, data, ((i-1) % length(data.x)) + 1)
        end
    end
    
    current_per_call = current_allocs / 100
    specialized_per_call = specialized_allocs / 100
    
    println("Current implementation: $(current_per_call) bytes per call")
    println("Position mapping implementation: $(specialized_per_call) bytes per call")
    
    if specialized_per_call < current_per_call
        reduction = (1 - specialized_per_call / current_per_call) * 100
        println("📈 $(round(reduction, digits=1))% allocation reduction achieved")
        
        if specialized_per_call == 0
            println("🎉 ZERO ALLOCATIONS ACHIEVED!")
        elseif reduction >= 85
            println("🏆 EXCELLENT: ≥85% reduction target achieved!")
        elseif reduction >= 75
            println("🥇 VERY GOOD: ≥75% reduction achieved!")
        elseif reduction >= 50
            println("🥈 GOOD: ≥50% reduction achieved!")
        end
    else
        println("⚠️  No improvement achieved")
    end
    
    # Timing comparison
    println("\n⏱️  Timing comparison:")
    print("Current: ")
    @btime $current_compiled($output_current, $data, 1)
    
    print("Position mapping: ")
    @btime $specialized_compiled($output_specialized, $data, 1)
    
    return true
end

"""
    run_comprehensive_position_mapping_tests()

Run comprehensive tests on all formula types.
"""
function run_comprehensive_position_mapping_tests()
    # Create test data
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
        response = randn(n)
    )
    data = Tables.columntable(df)
    
    println("🚀 COMPREHENSIVE POSITION MAPPING TESTING")
    println("="^80)
    
    # Test formulas progressively
    test_formulas = [
        (@formula(response ~ x * y), "Simple 2-way continuous"),
        (@formula(response ~ x * group3), "Continuous × Categorical"),  
        (@formula(response ~ log(z) * group4), "Function × Categorical - THE CRITICAL TEST"),
        (@formula(response ~ x * y * group3), "3-way interaction"),
        (@formula(response ~ x * y * group3 + log(z) * group4), "TARGET FORMULA - Complete test"),
    ]
    
    all_passed = true
    
    for (i, (formula, description)) in enumerate(test_formulas)
        println("\n$(i). $description")
        println("   Formula: $formula")
        println("   " * "-"^60)
        
        success = test_position_mapping_approach(formula, df, data)
        if success
            println("   ✅ Test $i passed")
        else
            println("   ❌ Test $i FAILED")
            all_passed = false
            break  # Stop on first failure to debug
        end
    end
    
    println("\n" * "="^80)
    if all_passed
        println("🎉 ALL POSITION MAPPING TESTS PASSED!")
        println("The all-in position mapping approach is working correctly.")
    else
        println("❌ SOME TESTS FAILED")
        println("Need to debug the position mapping implementation.")
    end
    println("="^80)
    
    return all_passed
end

run_comprehensive_position_mapping_tests()