# test_modelrow!.jl

using BenchmarkTools, Test, Profile
using FormulaCompiler

using Statistics
using DataFrames, GLM, Tables, CategoricalArrays, Random
using StatsModels, StandardizedPredictors

using FormulaCompiler:
    compile_formula_specialized,
    SpecializedFormula,
    ModelRowEvaluator


# Set consistent random seed for reproducible tests
Random.seed!(06515)

###############################################################################
# SIMPLE TESTING FUNCTION
###############################################################################

"""
    test_clean_modelrow_system()

Test the clean specialized-only modelrow! system.
"""
function test_clean_modelrow_system()
    println("🧪 TESTING CLEAN SPECIALIZED MODELROW! SYSTEM")
    println("="^60)
    
    # Simple test data
    df = DataFrame(
        x = [1.0, 2.0, 3.0],
        y = [4.0, 5.0, 6.0],
        group = categorical(["A", "B", "A"])
    )
    data = Tables.columntable(df)
    
    # Simple test case
    formula = @formula(y ~ x + group)
    println("Testing formula: $formula")
    
    try
        # Create model and get expected results
        model = lm(formula, df)
        expected_matrix = modelmatrix(model)
        println("Expected matrix size: $(size(expected_matrix))")
        println("Expected row 1: $(expected_matrix[1, :])")
        
        # Test 1: Direct specialized formula
        println("\n1. Testing direct specialized formula:")
        specialized = compile_formula(model, data)
        println("   Specialized formula type: $(typeof(specialized))")
        println("   Formula length: $(length(specialized))")
        
        output = Vector{Float64}(undef, length(specialized))
        println("   Output vector size: $(length(output))")
        
        # Try evaluation
        specialized(output, data, 1)
        println("   Result: $output")
        
        correct = isapprox(output, expected_matrix[1, :], rtol=1e-12)
        println("   Correct: $correct")
        
        if !correct
            println("   Expected: $(expected_matrix[1, :])")
            println("   Got:      $output")
            println("   Diff:     $(output .- expected_matrix[1, :])")
        end
        
        # Test allocations
        specialized(output, data, 1)  # Warmup
        allocs = @allocated specialized(output, data, 1)
        println("   Allocations: $allocs bytes")
        
        # Test 2: ModelRowEvaluator
        println("\n2. Testing ModelRowEvaluator:")
        evaluator = ModelRowEvaluator(model, df)
        println("   Evaluator type: $(typeof(evaluator))")
        
        result = evaluator(1)
        println("   Result: $result")
        
        correct2 = isapprox(result, expected_matrix[1, :], rtol=1e-12)
        println("   Correct: $correct2")
        
        # Test allocations
        evaluator(1)  # Warmup
        allocs2 = @allocated evaluator(1)
        println("   Allocations: $allocs2 bytes")
        
        # Test 3: Convenience function
        println("\n3. Testing convenience modelrow! function:")
        output3 = Vector{Float64}(undef, length(specialized))
        
        modelrow!(output3, model, data, 1; cache=false)
        println("   Result: $output3")
        
        correct3 = isapprox(output3, expected_matrix[1, :], rtol=1e-12)
        println("   Correct: $correct3")
        
        # Test allocations
        modelrow!(output3, model, data, 1; cache=false)  # Warmup
        allocs3 = @allocated modelrow!(output3, model, data, 1; cache=false)
        println("   Allocations: $allocs3 bytes")
        
        # Overall assessment
        println("\n📊 OVERALL ASSESSMENT:")
        if correct && correct2 && correct3
            println("   ✅ All correctness tests passed")
        else
            println("   ❌ Some correctness tests failed")
        end
        
        if allocs <= 100 && allocs2 <= 100 && allocs3 <= 200
            println("   ✅ Allocation performance good")
        else
            println("   ⚠️  Some allocation overhead detected")
        end
        
    catch e
        println("❌ Test failed with error: $e")
        println("Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
    
    return true
end

###############################################################################

test_clean_modelrow_system();

test_clean_modelrow_system()
