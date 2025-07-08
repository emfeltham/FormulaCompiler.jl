# phase1_validation_suite.jl - Comprehensive validation for recursive implementation

###############################################################################
# Comprehensive validation suite to ensure Phase 1 recursive implementation
# is ready for Phase 2 Strategy 4
###############################################################################

using Test
using DataFrames
using StatsModels
using GLM
using Tables
using BenchmarkTools
using Statistics
using LinearAlgebra

# All functions come from phase1_complete_recursion.jl

###############################################################################
# Test Suite 1: Integration with EfficientModelMatrices.jl
###############################################################################

"""
    test_efficient_model_matrices_integration()

Test that recursive evaluation integrates correctly with the existing
EfficientModelMatrices.jl infrastructure.
"""
function test_efficient_model_matrices_integration()
    println("="^70)
    println("TEST 1: EfficientModelMatrices.jl Integration")
    println("="^70)
    
    # Create realistic test scenarios
    test_scenarios = [
        ("Simple linear", 100, @formula(y ~ x + z)),
        ("With categorical", 150, @formula(y ~ x + z + group)),
        ("With interactions", 200, @formula(y ~ x + z + x & z + group)),
        ("Function terms", 120, @formula(y ~ x + log(z) + sqrt(w))),
        ("Complex mixed", 180, @formula(y ~ x + log(z) + x & group + z^2)),
    ]
    
    all_passed = true
    
    for (name, n, formula) in test_scenarios
        println("\nTesting: $name")
        println("Formula: $formula")
        
        try
            # Generate appropriate test data
            df = generate_test_data(n, formula)
            
            # Fit model
            model = lm(formula, df)
            data = Tables.columntable(df)
            
            println("  Parameters: $(length(coef(model)))")
            
            # Test 1: Column mapping works
            mapping = enhanced_column_mapping(model)
            X_fitted = modelmatrix(model)
            
            if mapping.total_columns != size(X_fitted, 2)
                println("  ❌ Column mapping mismatch")
                all_passed = false
                continue
            end
            
            # Test 2: Can extract all terms
            terms_found = 0
            for col in 1:mapping.total_columns
                try
                    term, local_col = get_term_for_column(mapping, col)
                    terms_found += 1
                catch e
                    println("  ❌ Failed to extract term for column $col: $e")
                    all_passed = false
                end
            end
            
            if terms_found != mapping.total_columns
                println("  ❌ Could only extract $terms_found/$(mapping.total_columns) terms")
                all_passed = false
                continue
            end
            
            # Test 3: Recursive evaluation matches existing system
            is_accurate = validate_recursive_evaluation(model, data)
            if !is_accurate
                println("  ❌ Recursive evaluation failed accuracy test")
                all_passed = false
                continue
            end
            
            # Test 4: All term types natively supported
            unsupported_terms = 0
            for col in 1:mapping.total_columns
                term, local_col = get_term_for_column(mapping, col)
                if !is_recursively_supported(term)
                    println("  ⚠️  Column $col: unsupported term type $(typeof(term))")
                    unsupported_terms += 1
                end
            end
            
            if unsupported_terms > 0
                println("  ⚠️  $unsupported_terms terms may need optimization")
            end
            
            println("  ✅ PASSED - Integration successful")
            
        catch e
            println("  ❌ FAILED with error: $e")
            all_passed = false
        end
    end
    
    if all_passed
        println("\n🎉 EfficientModelMatrices.jl integration: ALL TESTS PASSED")
    else
        println("\n❌ EfficientModelMatrices.jl integration: SOME TESTS FAILED")
    end
    
    return all_passed
end

"""
    extract_formula_vars(formula::FormulaTerm)

Extract variable symbols from a formula using proper StatsModels functions.
"""
function extract_formula_vars(formula::FormulaTerm)
    vars = Set{Symbol}()
    
    # Get response variable
    if isa(formula.lhs, Term)
        push!(vars, formula.lhs.sym)
    end
    
    # Get predictor variables from RHS - handle different RHS types explicitly
    rhs = formula.rhs
    if isa(rhs, Tuple)
        # RHS is a tuple of terms (like (x, z) for y ~ x + z)
        for term in rhs
            _collect_vars_from_term!(vars, term)
        end
    else
        # RHS is a single term
        _collect_vars_from_term!(vars, rhs)
    end
    
    return collect(vars)
end

"""
    _collect_vars_from_term!(vars::Set{Symbol}, term::AbstractTerm)

Recursively collect variable symbols from a term.
"""
function _collect_vars_from_term!(vars::Set{Symbol}, term::AbstractTerm)
    if isa(term, Term)
        push!(vars, term.sym)
    elseif isa(term, ContinuousTerm)
        push!(vars, term.sym)
    elseif isa(term, CategoricalTerm)
        push!(vars, term.sym)
    elseif isa(term, FunctionTerm)
        for arg in term.args
            _collect_vars_from_term!(vars, arg)
        end
    elseif isa(term, InteractionTerm)
        for subterm in term.terms
            _collect_vars_from_term!(vars, subterm)
        end
    elseif isa(term, MatrixTerm)
        for subterm in term.terms
            _collect_vars_from_term!(vars, subterm)
        end
    elseif isa(term, Tuple)
        for subterm in term
            _collect_vars_from_term!(vars, subterm)
        end
    end
    # ConstantTerm and InterceptTerm don't contribute variables
end

"""
    generate_test_data(n::Int, formula::FormulaTerm)

Generate appropriate test data for a given formula.
"""
function generate_test_data(n::Int, formula::FormulaTerm)
    # Extract variable names from formula
    vars = extract_formula_vars(formula)
    
    df = DataFrame()
    
    # Get response variable name
    response_var = formula.lhs.sym
    
    # Add response variable
    df[!, response_var] = randn(n)
    
    # Add predictor variables based on what's in the formula
    for var in vars
        if var == response_var
            continue  # Skip response
        elseif String(var) in ["group", "category", "treatment", "region"]
            # Categorical variables
            df[!, var] = categorical(rand(["A", "B", "C"], n))
        elseif String(var) in ["x", "z", "w", "age", "income"]
            # Continuous variables - ensure positive for log/sqrt
            df[!, var] = abs.(randn(n)) .+ 0.1
        else
            # Default: continuous, positive
            df[!, var] = abs.(randn(n)) .+ 0.1
        end
    end
    
    return df
end

"""
    is_recursively_supported(term::AbstractTerm) -> Bool

Check if a term type is fully supported by recursive implementation.
"""
function is_recursively_supported(term::AbstractTerm)
    return term isa Union{
        ContinuousTerm,
        ConstantTerm, 
        InterceptTerm,
        Term,
        CategoricalTerm,
        FunctionTerm,
        InteractionTerm,
        MatrixTerm
    }
end

###############################################################################
# Test Suite 2: Numerical Precision Validation
###############################################################################

"""
    test_numerical_precision()

Rigorous testing of numerical precision for recursive evaluation.
"""
function test_numerical_precision()
    println("\n" * "="^70)
    println("TEST 2: Numerical Precision Validation")
    println("="^70)
    
    precision_targets = [1e-15, 1e-12, 1e-10]
    
    formulas_to_test = [
        @formula(y ~ x + z),
        @formula(y ~ x + z + x & z),
        @formula(y ~ log(x) + sqrt(z)),
        @formula(y ~ x + z + group + x & group),
        @formula(y ~ log(x + 1) + x^2 + sqrt(z)),
    ]
    
    all_passed = true
    max_differences = Float64[]
    
    println("\nTesting precision across $(length(formulas_to_test)) formulas...")
    
    for formula in formulas_to_test
        n = 200
        df = generate_test_data(n, formula)
        model = lm(formula, df)
        data = Tables.columntable(df)
        
        # Build reference matrix using existing system
        ipm = InplaceModeler(model, n)
        p = width(fixed_effects_form(model).rhs)
        X_reference = Matrix{Float64}(undef, n, p)
        modelmatrix!(ipm, data, X_reference)
        
        # Build matrix using recursive system
        mapping = enhanced_column_mapping(model)
        max_diff_this_formula = 0.0
        
        for col in 1:p
            term, local_col = get_term_for_column(mapping, col)
            
            # Evaluate using recursive system
            col_recursive = Vector{Float64}(undef, n)
            evaluate_single_column!(term, data, col, local_col, col_recursive, nothing)
            
            # Compare
            col_reference = X_reference[:, col]
            diff = maximum(abs.(col_recursive .- col_reference))
            max_diff_this_formula = max(max_diff_this_formula, diff)
        end
        
        push!(max_differences, max_diff_this_formula)
        
        if max_diff_this_formula > precision_targets[2]  # Use 1e-12 as threshold
            println("  ❌ Formula $(formula): max diff = $max_diff_this_formula")
            all_passed = false
        else
            println("  ✅ Formula $(formula): max diff = $max_diff_this_formula")
        end
    end
    
    println("\nPrecision Summary:")
    println("  Maximum difference across all tests: $(maximum(max_differences))")
    println("  Average difference: $(mean(max_differences))")
    println("  Target precision: $(precision_targets[2])")
    
    if all_passed
        println("\n🎉 Numerical precision: ALL TESTS PASSED")
    else
        println("\n❌ Numerical precision: SOME TESTS FAILED")
    end
    
    return all_passed
end

###############################################################################
# Test Suite 3: Performance Reality Check
###############################################################################

"""
    test_performance_reality()

Test that theoretical performance benefits are achievable with recursive evaluation.
"""
function test_performance_reality()
    println("\n" * "="^70)
    println("TEST 3: Performance Reality Check")
    println("="^70)
    
    # Create scenarios with different efficiency ratios
    performance_scenarios = [
        ("High efficiency (panel data)", 1000, 
         @formula(y ~ x + z + individual + time), 
         Dict(:individual => 1:100, :time => 1:10)),
        ("Medium efficiency (interactions)", 500, 
         @formula(y ~ x + z + w + x & z + x & w + z & w),
         Dict()),
        ("Low efficiency (simple model)", 300, 
         @formula(y ~ x + z + w + a + b),
         Dict()),
    ]
    
    all_passed = true
    
    for (name, n, formula, special_vars) in performance_scenarios
        println("\nTesting: $name")
        
        # Generate data
        df = DataFrame()
        df.y = randn(n)
        
        # Add variables based on formula and special requirements
        vars = extract_formula_vars(formula)
        response_var = formula.lhs.sym
        for var in vars
            if var == response_var
                continue
            elseif haskey(special_vars, var)
                # Special categorical with many levels
                levels = special_vars[var]
                df[!, var] = categorical(rand(levels, n))
            elseif String(var) in ["individual", "time", "group"]
                df[!, var] = categorical(rand(1:5, n))
            else
                df[!, var] = abs.(randn(n)) .+ 0.1
            end
        end
        
        model = lm(formula, df)
        data = Tables.columntable(df)
        
        println("  Model parameters: $(length(coef(model)))")
        
        # Analyze efficiency for a continuous variable
        continuous_vars = filter(v -> v != response_var && eltype(df[!, v]) <: Real, extract_formula_vars(formula))
        if isempty(continuous_vars)
            println("  ⚠️  No continuous variables to test")
            continue
        end
        
        test_var = first(continuous_vars)
        mapping = enhanced_column_mapping(model)
        affected_cols = get_all_variable_columns(mapping, test_var)
        efficiency_ratio = length(affected_cols) / mapping.total_columns
        
        println("  Variable $test_var affects $(length(affected_cols))/$(mapping.total_columns) columns ($(round(efficiency_ratio*100, digits=1))%)")
        
        # Benchmark current approach (full matrix reconstruction)
        ipm = InplaceModeler(model, n)
        X_temp = Matrix{Float64}(undef, n, mapping.total_columns)
        h = sqrt(eps(Float64))
        perturbed_data = merge(data, (test_var => data[test_var] .+ h,))
        
        current_time = @belapsed begin
            modelmatrix!($ipm, $perturbed_data, $X_temp)
        end
        
        # Benchmark recursive approach (column-by-column)
        recursive_time = @belapsed begin
            for col in $affected_cols
                term, local_col = get_term_for_column($mapping, col)
                col_output = Vector{Float64}(undef, $n)
                evaluate_single_column!(term, $perturbed_data, col, local_col, col_output, nothing)
            end
        end
        
        actual_speedup = current_time / recursive_time
        theoretical_speedup = mapping.total_columns / length(affected_cols)
        
        println("  Current approach: $(round(current_time*1000, digits=2)) ms")
        println("  Recursive approach: $(round(recursive_time*1000, digits=2)) ms") 
        println("  Actual speedup: $(round(actual_speedup, digits=2))x")
        println("  Theoretical speedup: $(round(theoretical_speedup, digits=2))x")
        println("  Efficiency: $(round(actual_speedup/theoretical_speedup*100, digits=1))% of theoretical")
        
        # Test passes if we get reasonable speedup
        efficiency_threshold = 0.3  # At least 30% of theoretical
        speedup_threshold = 1.2     # At least 20% improvement
        
        if actual_speedup >= speedup_threshold && actual_speedup >= theoretical_speedup * efficiency_threshold
            println("  ✅ PASSED - Performance benefits achieved")
        else
            println("  ❌ FAILED - Insufficient performance improvement")
            all_passed = false
        end
    end
    
    if all_passed
        println("\n🎉 Performance reality check: ALL TESTS PASSED")
    else
        println("\n❌ Performance reality check: SOME TESTS FAILED")
    end
    
    return all_passed
end

###############################################################################
# Test Suite 4: Edge Cases and Robustness
###############################################################################

"""
    test_edge_cases_robustness()

Test edge cases and robustness of recursive implementation.
"""
function test_edge_cases_robustness()
    println("\n" * "="^70)
    println("TEST 4: Edge Cases and Robustness")
    println("="^70)
    
    edge_cases = [
        ("Minimal data", 2, @formula(y ~ x)),
        ("Intercept only", 5, @formula(y ~ 1)),
        ("Many categories", 100, @formula(y ~ group), Dict(:group => 1:20)),
        ("High-order polynomial", 50, @formula(y ~ x + x^2 + x^3 + x^4)),
        ("Deep nesting", 30, @formula(y ~ log(sqrt(abs(x) + 1) + 1))),
        ("Complex interaction", 80, @formula(y ~ x & z & w & group)),
    ]
    
    all_passed = true
    
    for (name, n, formula, special_vars...) in edge_cases
        println("\nTesting edge case: $name")
        
        try
            # Generate data
            df = DataFrame()
            df.y = randn(n)
            
            vars = extract_formula_vars(formula)
            response_var = formula.lhs.sym
            for var in vars
                if var == response_var
                    continue
                elseif !isempty(special_vars) && haskey(first(special_vars), var)
                    levels = first(special_vars)[var]
                    df[!, var] = categorical(rand(levels, n))
                elseif String(var) == "group"
                    df[!, var] = categorical(rand(["A", "B", "C"], n))
                else
                    df[!, var] = abs.(randn(n)) .+ 0.1  # Ensure positive for functions
                end
            end
            
            model = lm(formula, df)
            data = Tables.columntable(df)
            
            # Test that recursive evaluation works
            is_valid = validate_recursive_evaluation(model, data)
            
            if is_valid
                println("  ✅ PASSED - $name handled correctly")
            else
                println("  ❌ FAILED - $name failed validation")
                all_passed = false
            end
            
        catch e
            println("  ❌ ERROR - $name threw exception: $e")
            all_passed = false
        end
    end
    
    if all_passed
        println("\n🎉 Edge cases and robustness: ALL TESTS PASSED")
    else
        println("\n❌ Edge cases and robustness: SOME TESTS FAILED")
    end
    
    return all_passed
end

###############################################################################
# Test Suite 5: Strategy 4 Readiness Check
###############################################################################

"""
    test_strategy4_readiness()

Final check that all components needed for Strategy 4 are working with recursive evaluation.
"""
function test_strategy4_readiness()
    println("\n" * "="^70)
    println("TEST 5: Strategy 4 Readiness Check")
    println("="^70)
    
    # Test realistic Strategy 4 scenario
    n = 500
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        treatment = rand([0, 1], n),
        individual = categorical(repeat(1:50, inner=10)),
        time = categorical(repeat(1:10, 50)),
        y = randn(n)
    )
    
    formula = @formula(y ~ x + z + treatment + individual + time + x & treatment)
    model = lm(formula, df)
    data = Tables.columntable(df)
    
    println("Strategy 4 simulation:")
    println("  Formula: $formula") 
    println("  Parameters: $(length(coef(model)))")
    
    # Get existing model matrix (Strategy 4 baseline)
    X_fitted = modelmatrix(model)
    
    # Test each continuous variable
    continuous_vars = [:x, :z, :treatment]
    mapping = enhanced_column_mapping(model)
    
    strategy4_ready = true
    
    for var in continuous_vars
        println("\nTesting Strategy 4 readiness for variable :$var")
        
        # Step 1: Identify affected columns
        affected_cols = get_all_variable_columns(mapping, var)
        efficiency = length(affected_cols) / mapping.total_columns
        
        println("  Affected columns: $(length(affected_cols))/$(mapping.total_columns) ($(round(efficiency*100, digits=1))%)")
        
        # Step 2: Test column-by-column computation using recursive evaluation
        h = sqrt(eps(Float64))
        perturbed_data = merge(data, (var => data[var] .+ h,))
        
        success_count = 0
        max_error = 0.0
        
        for col in affected_cols
            try
                # Extract baseline from fitted model matrix
                baseline_col = X_fitted[:, col]
                
                # Compute perturbed column using recursive evaluation
                term, local_col = get_term_for_column(mapping, col)
                perturbed_col = Vector{Float64}(undef, n)
                evaluate_single_column!(term, perturbed_data, col, local_col, perturbed_col, nothing)
                
                # Compute derivative
                derivative_col = (perturbed_col .- baseline_col) ./ h
                
                # Validate against full matrix approach
                ipm = InplaceModeler(model, n)
                X_full_perturbed = Matrix{Float64}(undef, n, mapping.total_columns)
                modelmatrix!(ipm, perturbed_data, X_full_perturbed)
                reference_derivative = (X_full_perturbed[:, col] .- baseline_col) ./ h
                
                error = maximum(abs.(derivative_col .- reference_derivative))
                max_error = max(max_error, error)
                
                if error < 1e-12
                    success_count += 1
                else
                    println("    ❌ Column $col: error = $error")
                end
                
            catch e
                println("    ❌ Column $col: failed with $e")
            end
        end
        
        success_rate = success_count / length(affected_cols)
        println("  Success rate: $success_count/$(length(affected_cols)) ($(round(success_rate*100, digits=1))%)")
        println("  Maximum error: $max_error")
        
        if success_rate == 1.0 && max_error < 1e-12
            println("  ✅ Strategy 4 ready for variable :$var")
        else
            println("  ❌ Strategy 4 NOT ready for variable :$var")
            strategy4_ready = false
        end
    end
    
    if strategy4_ready
        println("\n🚀 STRATEGY 4 READINESS: FULLY READY")
        println("✅ All recursive components working correctly for Phase 2")
        println("✅ Column-by-column computation validated")
        println("✅ Integration with existing model matrices confirmed")
    else
        println("\n❌ STRATEGY 4 READINESS: NOT READY")
        println("❌ Issues found that must be resolved before Phase 2")
    end
    
    return strategy4_ready
end

###############################################################################
# Master Test Runner
###############################################################################

"""
    run_phase1_validation()

Run all Phase 1 validation tests for recursive implementation and provide final readiness assessment.
"""
function run_phase1_validation()
    println("🧪 PHASE 1 COMPREHENSIVE VALIDATION")
    println("Testing recursive implementation readiness for Strategy 4")
    println("="^70)
    
    test_results = Dict{String, Bool}()
    
    # Run all test suites
    test_results["EfficientModelMatrices Integration"] = test_efficient_model_matrices_integration()
    test_results["Numerical Precision"] = test_numerical_precision()
    test_results["Performance Reality"] = test_performance_reality()
    test_results["Edge Cases & Robustness"] = test_edge_cases_robustness()
    test_results["Strategy 4 Readiness"] = test_strategy4_readiness()
    
    # Final assessment
    println("\n" * "="^70)
    println("PHASE 1 VALIDATION RESULTS")
    println("="^70)
    
    all_passed = true
    for (test_name, passed) in test_results
        status = passed ? "✅ PASSED" : "❌ FAILED"
        println("$status - $test_name")
        all_passed &= passed
    end
    
    println("\n" * "="^70)
    if all_passed
        println("🎉 PHASE 1 VALIDATION: ALL TESTS PASSED!")
        println("✅ Recursive single-term evaluation working correctly")
        println("✅ Numerical accuracy matches existing system")
        println("✅ Performance benefits confirmed")
        println("✅ Ready to proceed to Phase 2 implementation")
        println("✅ Full Strategy 4 capability confirmed - no fallbacks needed")
    else
        println("❌ PHASE 1 VALIDATION: SOME TESTS FAILED")
        println("⚠️  Must resolve issues before Phase 2 implementation")
        println("⚠️  Review test output above for specific problems")
        println("⚠️  Consider addressing failed areas or implementing fallbacks")
    end
    println("="^70)
    
    return all_passed
end

# Export validation functions
export run_phase1_validation
export test_efficient_model_matrices_integration, test_numerical_precision
export test_performance_reality, test_edge_cases_robustness, test_strategy4_readiness
export generate_test_data, is_recursively_supported, extract_formula_vars
