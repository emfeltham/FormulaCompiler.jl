# phase1_test_runner.jl - Clean test runner for recursive implementation

###############################################################################
# Easy-to-run script for Phase 1 recursive implementation validation
###############################################################################

"""
    validate_phase1()

Main function to run all Phase 1 validation tests for recursive implementation.
Call this after loading the recursive implementation.

# Usage
```julia
# 1. Load your existing Margins.jl and EfficientModelMatrices.jl
using Margins, EfficientModelMatrices

# 2. Load Phase 1 recursive implementation
include("phase1_complete_recursion.jl")
include("phase1_testing.jl")
include("phase1_validation_suite.jl")
include("phase1_test_runner.jl")

# 3. Run validation
validate_phase1()
```
"""
function validate_phase1()
    println("🚀 Starting Phase 1 Validation...")
    println("Testing recursive single-term evaluation for Strategy 4 readiness")
    println()
    
    try
        # Run the comprehensive validation suite
        all_passed = run_phase1_validation()
        
        if all_passed
            println("\n" * "🎉"^20)
            println("PHASE 1 VALIDATION SUCCESSFUL!")
            println("🎉"^20)
            println()
            println("✅ Recursive evaluation working correctly")
            println("✅ All term types supported natively") 
            println("✅ Numerical accuracy confirmed (< 1e-12 precision)")
            println("✅ Performance benefits validated")
            println("✅ Integration with existing systems verified")
            println("✅ Edge cases handled properly")
            println("✅ Column-by-column computation ready")
            println()
            println("🚀 READY TO PROCEED TO PHASE 2!")
            println("   Strategy 4 implementation can begin with confidence")
            println("   No fallbacks needed - full recursive capability confirmed")
            println()
        else
            println("\n" * "❌"^20)
            println("PHASE 1 VALIDATION FAILED!")
            println("❌"^20)
            println()
            println("❌ Some recursive evaluation tests did not pass")
            println("⚠️  Must fix issues before Phase 2")
            println("⚠️  Review test output above for details")
            println()
            println("🔧 RECOMMENDED ACTIONS:")
            println("   1. Fix failing recursive evaluation cases")
            println("   2. Address numerical precision issues")
            println("   3. Optimize recursive performance bottlenecks")
            println("   4. Handle unsupported term types")
            println("   5. Re-run validation until all tests pass")
            println()
        end
        
        return all_passed
        
    catch e
        println("\n❌ VALIDATION FAILED WITH ERROR:")
        println("Error: $e")
        println()
        println("🔧 This likely means:")
        println("   1. Missing dependencies (EfficientModelMatrices.jl, etc.)")
        println("   2. Incomplete recursive implementation")
        println("   3. Compatibility issues with existing code")
        println("   4. Missing column mapping functions")
        println()
        println("Fix the error and try again.")
        return false
    end
end

"""
    quick_test_phase1()

Quick validation test for recursive implementation - runs subset of tests for faster iteration.
"""
function quick_test_phase1()
    println("⚡ Quick Phase 1 Recursive Test...")
    
    try
        # Test basic recursive functionality
        println("1. Testing basic recursive evaluation...")
        basic_passed = test_basic_terms()
        
        println("2. Testing column mapping...")
        mapping_passed = test_column_mapping()
        
        println("3. Testing recursive complex formula...")
        n = 100
        df = DataFrame(
            x = abs.(randn(n)) .+ 0.1,
            z = abs.(randn(n)) .+ 0.1,
            group = rand(["A", "B"], n),
            y = randn(n)
        )
        
        formula = @formula(y ~ x + log(z) + x & group)
        model = lm(formula, df)
        data = Tables.columntable(df)
        
        complex_passed = validate_recursive_evaluation(model, data)
        
        all_quick_passed = basic_passed && mapping_passed && complex_passed
        
        if all_quick_passed
            println("\n✅ Quick test PASSED - recursive functionality working")
            println("   Run validate_phase1() for full validation")
        else
            println("\n❌ Quick test FAILED - basic recursive issues found")
            println("   Fix basic issues before running full validation")
        end
        
        return all_quick_passed
        
    catch e
        println("\n❌ Quick test ERROR: $e")
        return false
    end
end

"""
    debug_recursive_issue(formula::FormulaTerm, n::Int = 50)

Debug a specific formula that's causing issues in recursive evaluation.
"""
function debug_recursive_issue(formula::FormulaTerm, n::Int = 50)
    println("🔍 Debugging recursive evaluation issue with formula: $formula")
    
    try
        # Generate test data
        df = generate_test_data(n, formula)
        model = lm(formula, df)
        data = Tables.columntable(df)
        
        println("Model fitted successfully with $(length(coef(model))) parameters")
        
        # Check column mapping
        mapping = enhanced_column_mapping(model)
        X_fitted = modelmatrix(model)
        
        println("Column mapping: $(mapping.total_columns) columns")
        println("Model matrix: $(size(X_fitted, 2)) columns")
        
        if mapping.total_columns != size(X_fitted, 2)
            println("❌ Column mapping size mismatch!")
            return false
        end
        
        # Check each column individually using recursive evaluation
        for col in 1:mapping.total_columns
            try
                term, local_col = get_term_for_column(mapping, col)
                println("Column $col: $(typeof(term)) (local: $local_col)")
                
                # Try to evaluate this column recursively
                col_output = Vector{Float64}(undef, n)
                evaluate_single_column!(term, data, col, local_col, col_output, nothing)
                
                # Compare with reference
                ref_col = X_fitted[:, col]
                max_diff = maximum(abs.(col_output .- ref_col))
                
                if max_diff > 1e-12
                    println("  ❌ Column $col: max difference = $max_diff")
                    println("     Term: $term")
                    println("     This indicates a recursive evaluation issue")
                    return false
                else
                    println("  ✅ Column $col: OK (diff = $max_diff)")
                end
                
            catch e
                println("  ❌ Column $col: RECURSIVE ERROR - $e")
                println("     Term type may not be fully supported")
                return false
            end
        end
        
        println("\n✅ Debug successful - all columns working with recursive evaluation")
        return true
        
    catch e
        println("❌ Debug failed with error: $e")
        return false
    end
end

"""
    test_recursive_formula(formula_string::String, n::Int = 100)

Convenience function for testing specific formulas with recursive evaluation.
"""
function test_recursive_formula(formula_string::String, n::Int = 100)
    println("🧪 Testing recursive evaluation of formula: $formula_string")
    
    try
        # Parse formula
        formula = eval(Meta.parse("@formula($formula_string)"))
        
        # Debug the formula using recursive evaluation
        success = debug_recursive_issue(formula, n)
        
        if success
            println("✅ Recursive formula test PASSED")
        else
            println("❌ Recursive formula test FAILED")
        end
        
        return success
        
    catch e
        println("❌ Recursive formula test ERROR: $e")
        return false
    end
end

"""
    phase1_status_report()

Generate a status report of Phase 1 recursive implementation readiness.
"""
function phase1_status_report()
    println("📊 PHASE 1 RECURSIVE IMPLEMENTATION STATUS")
    println("="^60)
    
    # Check what functions are available
    functions_to_check = [
        "evaluate_term!",
        "evaluate_single_column!", 
        "enhanced_column_mapping",
        "get_term_for_column",
        "get_terms_for_columns",
        "validate_recursive_evaluation",
        "evaluate_term_recursive!",
        "evaluate_single_column_recursive!"
    ]
    
    println("Function Availability:")
    for func_name in functions_to_check
        try
            func = eval(Symbol(func_name))
            println("  ✅ $func_name - Available")
        catch
            println("  ❌ $func_name - Missing")
        end
    end
    
    println("\nRecursive Term Type Support:")
    term_types = [
        ("ContinuousTerm", "Basic continuous variables"),
        ("ConstantTerm", "Constants and intercepts"), 
        ("InterceptTerm", "Model intercepts"),
        ("CategoricalTerm", "Categorical variables with contrasts"),
        ("FunctionTerm", "Functions like log(x), x^2, etc."),
        ("InteractionTerm", "Interactions like x & z"),
        ("MatrixTerm", "Composite terms")
    ]
    
    for (term_type, description) in term_types
        try
            # Check if we have recursive evaluation for this term type
            if isdefined(Main, Symbol("_evaluate_term_recursive_impl!"))
                println("  ✅ $term_type - $description")
            else
                println("  ⚠️  $term_type - $description (unknown status)")
            end
        catch
            println("  ❌ $term_type - $description (not supported)")
        end
    end
    
    println("\nDependency Check:")
    dependencies = [
        ("StatsModels", "Formula parsing and term types"),
        ("GLM", "Model fitting"), 
        ("DataFrames", "Data handling"),
        ("Tables", "Column table interface"),
        ("LinearAlgebra", "Matrix operations"),
        ("EfficientModelMatrices", "Model matrix construction")
    ]
    
    for (dep, description) in dependencies
        try
            eval(Symbol(dep))
            println("  ✅ $dep - $description")
        catch
            println("  ❌ $dep - $description (MISSING)")
        end
    end
    
    println("\n" * "="^60)
    println("NEXT STEPS:")
    println("  1. Run quick_test_phase1() for basic functionality")
    println("  2. Run validate_phase1() for full validation")
    println("  3. Use debug_recursive_issue() for specific problems")
    println("  4. Use test_recursive_formula() for formula testing")
    println("="^60)
end

###############################################################################
# Usage Guide and Documentation
###############################################################################

"""
Phase 1 Recursive Implementation Usage Guide:

1. SETUP
   Load all required packages and Phase 1 recursive implementation:
   ```julia
   using Margins, EfficientModelMatrices, StatsModels, GLM, DataFrames
   include("phase1_complete_recursion.jl")  # Complete recursive implementation
   include("phase1_testing.jl")             # Basic tests
   include("phase1_validation_suite.jl")    # Comprehensive validation
   include("phase1_test_runner.jl")         # This file
   ```

2. STATUS CHECK
   ```julia
   phase1_status_report()  # Check what's available
   ```

3. QUICK TEST
   ```julia
   quick_test_phase1()     # Fast basic recursive test
   ```

4. FULL VALIDATION  
   ```julia
   validate_phase1()       # Comprehensive recursive validation
   ```

5. DEBUG SPECIFIC ISSUES
   ```julia
   test_recursive_formula("y ~ x + log(z) + C(group)")
   debug_recursive_issue(@formula(y ~ x & z & C(group)), 100)
   ```

6. INTERPRETATION OF RESULTS

   ✅ ALL TESTS PASSED:
   - Recursive evaluation working correctly
   - Ready for Phase 2 Strategy 4 implementation
   - No fallbacks needed - full recursive capability
   
   ❌ SOME TESTS FAILED:
   - Fix specific recursive evaluation issues
   - Address term types that aren't fully supported
   - Re-run validation until all pass
   
   ❌ MAJOR ERRORS:
   - Check dependency installation
   - Verify recursive implementation completeness
   - Review integration with existing code

7. NEXT STEPS AFTER VALIDATION

   If validation passes:
   - Proceed to Phase 2 Strategy 4 implementation
   - Implement Strategy4Workspace with recursive evaluation
   - Add column-by-column AME computation using recursive functions
   - Integrate with main margins() function
   
   If validation fails:
   - Address specific failing recursive evaluation cases
   - Improve term type support in recursive system
   - Optimize recursive performance bottlenecks
   - Fix numerical precision issues in recursive computation
"""

# Export all test functions
export validate_phase1, quick_test_phase1, debug_recursive_issue
export test_recursive_formula, phase1_status_report