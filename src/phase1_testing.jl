# phase1_testing.jl - Testing for recursive single-term evaluation

###############################################################################
# Comprehensive testing for Phase 1 recursive implementation
###############################################################################

using Test
using DataFrames
using StatsModels
using GLM
using Tables

# All functions now come from phase1_complete_recursion.jl

###############################################################################
# Test Suite 1: Basic Recursive Term Evaluation
###############################################################################

"""
    test_basic_terms()

Test recursive evaluation of simple term types.
"""
function test_basic_terms()
    println("Testing basic recursive term evaluation...")
    
    # Create simple test data
    n = 10
    data = (
        x = collect(1.0:n),
        z = collect(2.0:2.0:2n),
        group = repeat(["A", "B"], n÷2)
    )
    
    @testset "Basic Terms" begin
        # Test ContinuousTerm
        term = ContinuousTerm(:x)
        output = Matrix{Float64}(undef, n, 1)
        evaluate_term!(term, data, output, nothing)
        @test output[:, 1] == data.x
        
        # Test single column version
        col_output = Vector{Float64}(undef, n)
        evaluate_single_column!(term, data, 1, 1, col_output, nothing)
        @test col_output == data.x
        
        # Test ConstantTerm
        term = ConstantTerm(5.0)
        output = Matrix{Float64}(undef, n, 1)
        evaluate_term!(term, data, output, nothing)
        @test all(output[:, 1] .== 5.0)
        
        # Test InterceptTerm
        term = InterceptTerm{true}()
        output = Matrix{Float64}(undef, n, 1)
        evaluate_term!(term, data, output, nothing)
        @test all(output[:, 1] .== 1.0)
        
        println("  ✅ Basic terms passed")
    end
end

###############################################################################
# Test Suite 2: Column Mapping with Recursive System
###############################################################################

"""
    test_column_mapping()

Test column mapping functionality with recursive evaluation.
"""
function test_column_mapping()
    println("Testing column mapping with recursive system...")
    
    n = 50
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        group = rand(["A", "B", "C"], n),
        y = randn(n)
    )
    
    @testset "Column Mapping" begin
        # Test simple formula
        formula = @formula(y ~ x + z)
        model = lm(formula, df)
        mapping = enhanced_column_mapping(model)
        
        @test mapping.total_columns == 3  # intercept + x + z
        
        # Test term extraction
        term1, local1 = get_term_for_column(mapping, 1)  # Should be intercept
        term2, local2 = get_term_for_column(mapping, 2)  # Should be x
        term3, local3 = get_term_for_column(mapping, 3)  # Should be z
        
        @test term1 isa InterceptTerm
        @test term2 isa ContinuousTerm && term2.sym == :x
        @test term3 isa ContinuousTerm && term3.sym == :z
        @test local1 == local2 == local3 == 1
        
        # Test multiple column extraction
        terms_map = get_terms_for_columns(mapping, [2, 3])
        @test length(terms_map) == 2
        
        println("  ✅ Column mapping passed")
    end
end

###############################################################################
# Test Suite 3: Recursive Function Terms
###############################################################################

"""
    test_function_terms()

Test recursive evaluation of function terms.
"""
function test_function_terms()
    println("Testing recursive function terms...")
    
    n = 20
    df = DataFrame(
        x = abs.(randn(n)) .+ 0.1,  # Ensure positive for log
        z = abs.(randn(n)) .+ 0.1,  # Ensure positive for sqrt
        y = randn(n)
    )
    
    @testset "Function Terms" begin
        # Test log function
        formula = @formula(y ~ log(x))
        model = lm(formula, df)
        data = Tables.columntable(df)
        
        is_valid = validate_recursive_evaluation(model, data)
        @test is_valid
        
        # Test polynomial  
        formula = @formula(y ~ x + x^2)
        model = lm(formula, df)
        data = Tables.columntable(df)
        
        is_valid = validate_recursive_evaluation(model, data)
        @test is_valid
        
        # Test two-argument function
        formula = @formula(y ~ x + z)
        model = lm(formula, df)
        data = Tables.columntable(df)
        
        is_valid = validate_recursive_evaluation(model, data)
        @test is_valid
        
        println("  ✅ Function terms passed")
    end
end

###############################################################################
# Test Suite 4: Recursive Categorical Terms
###############################################################################

"""
    test_categorical_terms()

Test recursive evaluation of categorical terms.
"""
function test_categorical_terms()
    println("Testing recursive categorical terms...")
    
    n = 30
    df = DataFrame(
        x = randn(n),
        group = rand(["A", "B", "C"], n),
        y = randn(n)
    )
    
    @testset "Categorical Terms" begin
        formula = @formula(y ~ x + group)
        model = lm(formula, df)
        data = Tables.columntable(df)
        
        # Validate recursive evaluation
        is_valid = validate_recursive_evaluation(model, data)
        @test is_valid
        
        # Test column extraction for categorical term
        mapping = enhanced_column_mapping(model)
        
        # Find categorical term columns (should be columns 3 and 4 for 3-level factor)
        categorical_cols = Int[]
        for col in 1:mapping.total_columns
            term, _ = get_term_for_column(mapping, col)
            if term isa CategoricalTerm
                push!(categorical_cols, col)
            end
        end
        
        @test length(categorical_cols) == 2  # 3-level factor has 2 dummy columns
        
        println("  ✅ Categorical terms passed")
    end
end

###############################################################################
# Test Suite 5: Recursive Complex Formulas
###############################################################################

"""
    test_complex_formulas()

Test recursive evaluation on complex formulas with interactions and nested functions.
"""
function test_complex_formulas()
    println("Testing recursive complex formulas...")
    
    n = 40
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        w = abs.(randn(n)) .+ 0.1,  # Ensure positive for log/sqrt
        group = rand(["A", "B"], n),
        time = rand(1:5, n),
        y = randn(n)
    )
    
    complex_formulas = [
        @formula(y ~ x + z + C(group)),
        @formula(y ~ x + log(w) + C(group)),
        @formula(y ~ x + z + w + C(group) + C(time)),
        @formula(y ~ x + x^2 + C(group)),
        @formula(y ~ log(w) + sqrt(w) + C(group)),
        @formula(y ~ x & z),
        @formula(y ~ x & C(group)),
        @formula(y ~ log(w) & C(group)),
    ]
    
    @testset "Complex Formulas" begin
        for (i, formula) in enumerate(complex_formulas)
            println("  Testing complex formula $i: $formula")
            
            try
                model = lm(formula, df)
                data = Tables.columntable(df)
                
                # Validate recursive evaluation matches full evaluation
                is_valid = validate_recursive_evaluation(model, data)
                @test is_valid
                
                # Test column mapping completeness
                mapping = enhanced_column_mapping(model)
                X_fitted = modelmatrix(model)
                @test mapping.total_columns == size(X_fitted, 2)
                
                # Test that we can extract all columns recursively
                for col in 1:mapping.total_columns
                    term, local_col = get_term_for_column(mapping, col)
                    @test 1 <= local_col <= width(term)
                    
                    # Test actual recursive evaluation
                    col_output = Vector{Float64}(undef, nrow(df))
                    evaluate_single_column!(term, data, col, local_col, col_output, nothing)
                    
                    # Should match fitted model matrix
                    max_diff = maximum(abs.(col_output .- X_fitted[:, col]))
                    @test max_diff < 1e-12
                end
                
            catch e
                @error "Complex formula $i failed" exception=e
                @test false  # Force test failure
            end
        end
        
        println("  ✅ Complex formulas passed")
    end
end

###############################################################################
# Test Suite 6: Performance Baseline for Recursive System
###############################################################################

"""
    test_performance_baseline()

Establish baseline performance measurements for recursive single-term evaluation.
"""
function test_performance_baseline()
    println("Performance baseline testing for recursive system...")
    
    # Create moderately large dataset
    n = 1000
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        w = abs.(randn(n)) .+ 0.1,
        group = rand(["A", "B", "C", "D"], n),
        y = randn(n)
    )
    
    formula = @formula(y ~ x + z + w + log(w) + C(group))
    model = lm(formula, df)
    data = Tables.columntable(df)
    
    # Time full model matrix construction
    ipm = InplaceModeler(model, n)
    p = width(fixed_effects_form(model).rhs)
    X_full = Matrix{Float64}(undef, n, p)
    
    full_time = @elapsed begin
        for _ in 1:100
            modelmatrix!(ipm, data, X_full)
        end
    end
    
    # Time recursive single-term evaluation for one column
    mapping = enhanced_column_mapping(model)
    term, local_col = get_term_for_column(mapping, 2)  # Second column (likely x)
    col_output = Vector{Float64}(undef, n)
    
    recursive_time = @elapsed begin
        for _ in 1:100
            evaluate_single_column!(term, data, 2, local_col, col_output, nothing)
        end
    end
    
    println("  Full matrix (100 iterations): $(round(full_time*1000, digits=2)) ms")
    println("  Recursive single column (100 iterations): $(round(recursive_time*1000, digits=2)) ms")
    println("  Speedup ratio: $(round(full_time/recursive_time, digits=2))x")
    
    # Validation
    @test recursive_time < full_time  # Single column should be faster
    @test validate_recursive_evaluation(model, data)  # Should still be accurate
    
    println("  ✅ Performance baseline established")
end

###############################################################################
# Test Suite 7: Edge Cases for Recursive System
###############################################################################

"""
    test_edge_cases()

Test edge cases and error handling for recursive evaluation.
"""
function test_edge_cases()
    println("Testing edge cases for recursive system...")
    
    @testset "Edge Cases" begin
        # Test with minimal data
        n = 2
        df = DataFrame(x = [1.0, 2.0], y = [1.0, 2.0])
        formula = @formula(y ~ x)
        model = lm(formula, df)
        data = Tables.columntable(df)
        
        @test validate_recursive_evaluation(model, data)
        
        # Test column index out of bounds
        mapping = enhanced_column_mapping(model)
        @test_throws Exception get_term_for_column(mapping, 10)
        
        # Test with intercept-only model
        formula = @formula(y ~ 1)
        model = lm(formula, df)
        data = Tables.columntable(df)
        @test validate_recursive_evaluation(model, data)
        
        # Test complex nested functions
        n = 20
        df = DataFrame(
            x = abs.(randn(n)) .+ 0.1,
            y = randn(n)
        )
        formula = @formula(y ~ log(sqrt(x)))
        model = lm(formula, df)
        data = Tables.columntable(df)
        @test validate_recursive_evaluation(model, data)
        
        println("  ✅ Edge cases passed")
    end
end

###############################################################################
# Main Test Runner
###############################################################################

"""
    run_phase1_tests()

Run all Phase 1 tests for recursive implementation and report results.
"""
function run_phase1_tests()
    println("="^70)
    println("PHASE 1 TESTING: Recursive Single-Term Evaluation")
    println("="^70)
    
    test_results = Dict{String, Bool}()
    
    try
        println("\n1. Basic Terms...")
        test_basic_terms()
        test_results["Basic Terms"] = true
        
        println("\n2. Column Mapping...")
        test_column_mapping()
        test_results["Column Mapping"] = true
        
        println("\n3. Function Terms...")
        test_function_terms()
        test_results["Function Terms"] = true
        
        println("\n4. Categorical Terms...")
        test_categorical_terms()
        test_results["Categorical Terms"] = true
        
        println("\n5. Complex Formulas...")
        test_complex_formulas()
        test_results["Complex Formulas"] = true
        
        println("\n6. Performance Baseline...")
        test_performance_baseline()
        test_results["Performance"] = true
        
        println("\n7. Edge Cases...")
        test_edge_cases()
        test_results["Edge Cases"] = true
        
    catch e
        println("\n❌ TESTING FAILED with error: $e")
        return false
    end
    
    # Report results
    println("\n" * "="^70)
    println("PHASE 1 TEST RESULTS:")
    println("="^70)
    
    all_passed = true
    for (test_name, passed) in test_results
        status = passed ? "✅ PASSED" : "❌ FAILED"
        println("  $test_name: $status")
        all_passed &= passed
    end
    
    println("\n" * "="^70)
    if all_passed
        println("🎉 ALL PHASE 1 TESTS PASSED!")
        println("✅ Recursive evaluation working correctly")
        println("✅ Ready to proceed to Phase 2 implementation")
    else
        println("❌ SOME TESTS FAILED - Fix issues before proceeding")
    end
    println("="^70)
    
    return all_passed
end

# Export test functions
export run_phase1_tests
export test_basic_terms, test_column_mapping, test_function_terms
export test_categorical_terms, test_complex_formulas, test_performance_baseline
export test_edge_cases
