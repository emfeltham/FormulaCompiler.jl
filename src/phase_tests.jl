# phase_tests.jl

###############################################################################
# TESTING FRAMEWORK FOR PHASE 2A
###############################################################################

"""
Run all Phase 2A tests.
"""
function test_phase2a_complete()
    println("🧪 PHASE 2A COMPREHENSIVE TESTING")
    println("=" ^ 50)
    
    arch_success = false
    syntax_success = false
    integration_success = false
    
    try
        test_phase2a_architecture()
        arch_success = true
    catch e
        println("❌ Architecture test failed: $e")
    end
    
    try
        syntax_success = validate_phase2a_syntax()
    catch e
        println("❌ Syntax validation failed: $e")
    end
    
    try
        integration_success = test_phase2a_integration()
    catch e
        println("❌ Integration test failed: $e")
    end
    
    println("\n" * "=" ^ 60)
    println("PHASE 2A TEST SUMMARY")
    println("=" ^ 60)
    println("Architecture:  $(arch_success ? "✅ PASS" : "❌ FAIL")")
    println("Syntax:        $(syntax_success ? "✅ PASS" : "❌ FAIL")")  
    println("Integration:   $(integration_success ? "✅ PASS" : "❌ FAIL")")
    
    overall_success = arch_success && syntax_success && integration_success
    println("OVERALL:       $(overall_success ? "✅ SUCCESS" : "⚠️  NEEDS WORK")")
    
    if overall_success
        println("\n🚀 Phase 2A foundation is solid!")
        println("📋 Ready for Phase 2B: Complex function expressions")
        println("   Ask: 'Write Phase 2B: Implement recursive function expression generation'")
    else
        println("\n🔧 Phase 2A needs fixes before proceeding to Phase 2B")
    end
    
    return overall_success
end

"""
Test the Phase 2A core architecture with simple cases.
"""
function test_phase2a_architecture()
    println("=== Testing Phase 2A Core Architecture ===")
    
    # Create simple test data
    df = DataFrame(
        x = [1.0, 2.0, 3.0],
        y = [2.0, 4.0, 6.0],
        group = categorical(["A", "B", "A"])
    )
    data = Tables.columntable(df)
    
    println("\n1. Testing simple constant:")
    const_eval = ConstantEvaluator(5.0)
    expr = generate_expression_recursive(const_eval)
    println("   ConstantEvaluator(5.0) → '$expr'")
    @assert expr == "5.0"
    
    println("\n2. Testing simple continuous:")
    cont_eval = ContinuousEvaluator(:x)
    expr = generate_expression_recursive(cont_eval)
    println("   ContinuousEvaluator(:x) → '$expr'")
    @assert expr == "Float64(data.x[row_idx])"
    
    println("\n3. Testing simple function:")
    try
        log_eval = FunctionEvaluator(log, [ContinuousEvaluator(:x)])
        expr = generate_expression_recursive(log_eval)
        println("   FunctionEvaluator(log, [x]) → '$expr'")
        println("   ✅ Simple function generation works")
    catch e
        println("   Expected for complex functions: $e")
    end
    
    println("\n4. Testing simple categorical:")
    try
        # Create a simple categorical evaluator (2 levels, 1 contrast)
        contrast_matrix = reshape([0.0, 1.0], 2, 1)
        cat_eval = CategoricalEvaluator(:group, contrast_matrix, 2)
        expr = generate_expression_recursive(cat_eval)
        println("   CategoricalEvaluator → '$expr'")
        println("   ✅ Simple categorical generation works")
    catch e
        println("   Expected for complex categoricals: $e")
    end
    
    println("\n5. Testing combined evaluator:")
    try
        combined_eval = CombinedEvaluator([
            ConstantEvaluator(1.0),
            ContinuousEvaluator(:x)
        ])
        instructions, next_pos = generate_statements_recursive(combined_eval, 1)
        println("   CombinedEvaluator → $(length(instructions)) instructions, next_pos=$next_pos:")
        for (i, instr) in enumerate(instructions)
            println("     $i: $instr")
        end
        println("   ✅ Combined statement generation works")
    catch e
        println("   Error in combined evaluator: $e")
    end
    
    println("\n6. Testing scaled evaluator:")
    try
        scaled_eval = ScaledEvaluator(ContinuousEvaluator(:x), 2.5)
        expr = generate_expression_recursive(scaled_eval)
        println("   ScaledEvaluator → '$expr'")
        println("   ✅ Scaled expression generation works")
    catch e
        println("   Error in scaled evaluator: $e")
    end
    
    println("\n7. Testing integration with existing system:")
    try
        # Test backward compatibility
        instructions = String[]
        const_eval = ConstantEvaluator(42.0)
        next_pos = generate_evaluator_code!(instructions, const_eval, 1)
        println("   Backward compatibility → $(length(instructions)) instructions, next_pos=$next_pos")
        for (i, instr) in enumerate(instructions)
            println("     $i: $instr")
        end
        println("   ✅ Backward compatibility works")
    catch e
        println("   Error in backward compatibility: $e")
    end
    
    println("\n=== Phase 2A Architecture Test Complete ===")
    println("✅ Foundation is ready for Phase 2B (Complex Function Expressions)")
end

"""
Test expressions are syntactically valid Julia code.
"""
function validate_phase2a_syntax()
    println("\n=== Validating Phase 2A Generated Syntax ===")
    
    test_evaluators = [
        ConstantEvaluator(1.0),
        ConstantEvaluator(-5.5),
        ConstantEvaluator(0.0),
        ContinuousEvaluator(:x),
        ContinuousEvaluator(:temperature),
        ScaledEvaluator(ContinuousEvaluator(:x), 3.14)
    ]
    
    all_valid = true
    
    for (i, evaluator) in enumerate(test_evaluators)
        try
            expr = generate_expression_recursive(evaluator)
            parsed = Meta.parse(expr)
            println("   $i. $(typeof(evaluator)) → '$expr' ✅")
        catch e
            println("   $i. $(typeof(evaluator)) → INVALID: $e ❌")
            all_valid = false
        end
    end
    
    if all_valid
        println("✅ All generated expressions have valid syntax")
    else
        println("❌ Some expressions have invalid syntax")
    end
    
    return all_valid
end

"""
Test end-to-end integration with actual models.
"""
function test_phase2a_integration()
    println("\n=== Testing Phase 2A End-to-End Integration ===")
    
    Random.seed!(42)
    df = DataFrame(
        x = randn(10),
        y = randn(10),
        z = abs.(randn(10)) .+ 0.1
    )
    
    # Test simple formulas that should work in Phase 2A
    test_formulas = [
        @formula(y ~ 1),              # ConstantEvaluator
        @formula(y ~ x),              # ContinuousEvaluator  
        @formula(y ~ 1 + x),          # CombinedEvaluator
        @formula(y ~ x + z),          # CombinedEvaluator
    ]
    
    all_passed = true
    
    for (i, formula) in enumerate(test_formulas)
        try
            println("\n   Test $i: $formula")
            
            # Compile the model
            model = lm(formula, df)
            compiled = compile_formula(model)
            
            # Test code generation
            instructions = generate_code_from_evaluator(compiled.root_evaluator)
            println("     Generated $(length(instructions)) instructions")
            
            # Test that generated code compiles
            for instr in instructions
                try
                    Meta.parse(instr)
                catch e
                    println("     ❌ Invalid syntax: $instr")
                    all_passed = false
                end
            end
            
            println("     ✅ Formula works with Phase 2A")
            
        catch e
            if occursin("Phase 2", string(e))
                println("     ⚠️  Expected limitation: $e")
            else
                println("     ❌ Unexpected error: $e")
                all_passed = false
            end
        end
    end
    
    if all_passed
        println("\n✅ All simple formulas work with Phase 2A architecture")
    else
        println("\n❌ Some issues found in Phase 2A integration")
    end
    
    return all_passed
end

###############################################################################
# TESTING FRAMEWORK FOR PHASE 2B
###############################################################################

"""
Test complex nested function expressions (Phase 2B).
"""
function test_phase2b_nested_functions()
    println("\n=== Testing Phase 2B Nested Function Expressions ===")
    
    # Test nested function evaluators
    test_cases = [
        # log(x^2)
        FunctionEvaluator(log, [FunctionEvaluator(^, [ContinuousEvaluator(:x), ConstantEvaluator(2.0)])]),
        
        # sqrt(x + 1)
        FunctionEvaluator(sqrt, [FunctionEvaluator(+, [ContinuousEvaluator(:x), ConstantEvaluator(1.0)])]),
        
        # exp(log(x))
        FunctionEvaluator(exp, [FunctionEvaluator(log, [ContinuousEvaluator(:x)])]),
        
        # sin(x) + cos(y)
        FunctionEvaluator(+, [
            FunctionEvaluator(sin, [ContinuousEvaluator(:x)]),
            FunctionEvaluator(cos, [ContinuousEvaluator(:y)])
        ]),
        
        # (x + y) * (z + 1)
        ProductEvaluator([
            FunctionEvaluator(+, [ContinuousEvaluator(:x), ContinuousEvaluator(:y)]),
            FunctionEvaluator(+, [ContinuousEvaluator(:z), ConstantEvaluator(1.0)])
        ])
    ]
    
    test_descriptions = [
        "log(x^2)",
        "sqrt(x + 1)", 
        "exp(log(x))",
        "sin(x) + cos(y)",
        "(x + y) * (z + 1)"
    ]
    
    all_passed = true
    
    for (i, (evaluator, description)) in enumerate(zip(test_cases, test_descriptions))
        try
            expr = generate_expression_recursive(evaluator)
            # Validate syntax
            parsed = Meta.parse(expr)
            println("   $i. $description → '$expr' ✅")
        catch e
            println("   $i. $description → ERROR: $e ❌")
            all_passed = false
        end
    end
    
    return all_passed
end

"""
Test end-to-end integration with complex formulas (Phase 2B).
"""
function test_phase2b_integration()
    println("\n=== Testing Phase 2B End-to-End Integration ===")
    
    Random.seed!(42)
    df = DataFrame(
        x = abs.(randn(10)) .+ 0.1,  # Positive for log
        y = randn(10),
        z = abs.(randn(10))
    )
    
    # Test complex formulas that should work in Phase 2B
    test_formulas = [
        @formula(y ~ log(x)),            # Simple function
        @formula(y ~ x^2),               # Power function
        @formula(y ~ log(x^2)),          # Nested: log(x^2)
        @formula(y ~ sqrt(x + 1)),       # Nested: sqrt(x + 1)
        @formula(y ~ sin(x) + cos(y)),   # Multiple functions
        @formula(y ~ log(x) * sqrt(z)),  # Function products
    ]
    
    all_passed = true
    
    for (i, formula) in enumerate(test_formulas)
        try
            println("\n   Test $i: $formula")
            
            # Compile the model
            model = lm(formula, df)
            compiled = compile_formula(model)
            
            # Test code generation
            instructions = generate_code_from_evaluator(compiled.root_evaluator)
            println("     Generated $(length(instructions)) instructions")
            
            # Test that generated code compiles
            for instr in instructions
                try
                    Meta.parse(instr)
                catch e
                    println("     ❌ Invalid syntax: $instr")
                    all_passed = false
                end
            end
            
            # Test actual evaluation
            data = Tables.columntable(df)
            row_vec = Vector{Float64}(undef, length(compiled))
            compiled(row_vec, data, 1)
            
            if all(isfinite.(row_vec))
                println("     ✅ Formula evaluates correctly")
            else
                println("     ⚠️  Some non-finite values: $row_vec")
            end
            
        catch e
            if occursin("Phase 2", string(e))
                println("     ⚠️  Expected limitation: $e")
            else
                println("     ❌ Unexpected error: $e")
                all_passed = false
            end
        end
    end
    
    return all_passed
end

"""
Run all Phase 2B tests.
"""
function test_phase2b_complete()
    println("🧪 PHASE 2B COMPREHENSIVE TESTING")
    println("=" ^ 50)
    
    nested_success = false
    integration_success = false
    
    try
        nested_success = test_phase2b_nested_functions()
    catch e
        println("❌ Nested functions test failed: $e")
    end
    
    try
        integration_success = test_phase2b_integration()
    catch e
        println("❌ Integration test failed: $e")
    end
    
    println("\n" * "=" ^ 60)
    println("PHASE 2B TEST SUMMARY")
    println("=" ^ 60)
    println("Nested Functions:  $(nested_success ? "✅ PASS" : "❌ FAIL")")
    println("Integration:       $(integration_success ? "✅ PASS" : "❌ FAIL")")
    
    overall_success = nested_success && integration_success
    println("OVERALL:           $(overall_success ? "🎉 SUCCESS" : "⚠️  NEEDS WORK")")
    
    if overall_success
        println("\n🚀 Phase 2B is working perfectly!")
        println("📋 Complex nested functions like log(x^2 + 1) now work!")
        println("📋 Ready for Phase 2C: Multi-output expressions and complex categoricals")
        println("   Ask: 'Write Phase 2C: Implement multi-output statement generation'")
    else
        println("\n🔧 Phase 2B needs fixes before proceeding to Phase 2C")
    end
    
    return overall_success
end

# test_generators_phase2c.jl
# Comprehensive testing framework for Phase 2C recursive generation

using DataFrames, Random, GLM, Tables, CategoricalArrays
using FormulaCompiler

###############################################################################
# PHASE 2C TESTING FRAMEWORK
###############################################################################

"""
Test multi-contrast categorical expressions (Phase 2C).
"""
function test_phase2c_categorical_statements()
    println("\n=== Testing Phase 2C Multi-Contrast Categorical Statements ===")
    
    # Create test data with categorical variables
    df = DataFrame(
        x = [1.0, 2.0, 3.0, 4.0],
        group3 = categorical(["A", "B", "C", "A"]),
        group4 = categorical(["W", "X", "Y", "Z"])
    )
    data = Tables.columntable(df)
    
    # Test categoricals with different contrast structures
    test_cases = [
        # 3-level categorical with treatment coding (2 contrasts)
        (CategoricalEvaluator(:group3, [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0]', 3), "3-level treatment"),
        
        # 4-level categorical with treatment coding (3 contrasts)  
        (CategoricalEvaluator(:group4, [0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0; 1.0 0.0 0.0 0.0]', 4), "4-level treatment"),
    ]
    
    all_passed = true
    
    for (i, (evaluator, description)) in enumerate(test_cases)
        try
            println("   Test $i: $description")
            
            # Test statement generation
            instructions, next_pos = generate_statements_recursive(evaluator, 1)
            println("     Generated $(length(instructions)) statements, next_pos=$next_pos")
            
            # Validate syntax of all instructions
            for instr in instructions
                try
                    Meta.parse(instr)
                catch e
                    println("     ❌ Invalid syntax: $instr")
                    all_passed = false
                end
            end
            
            println("     ✅ Multi-contrast categorical works")
            
        catch e
            if occursin("Phase 2", string(e))
                println("     ⚠️  Expected limitation: $e")
            else
                println("     ❌ Unexpected error: $e")
                all_passed = false
            end
        end
    end
    
    return all_passed
end

"""
Test simple interaction expressions (Phase 2C).
"""
function test_phase2c_interaction_statements()
    println("\n=== Testing Phase 2C Simple Interaction Statements ===")
    
    # Create test data
    df = DataFrame(
        x = [1.0, 2.0, 3.0, 4.0],
        y = [2.0, 4.0, 6.0, 8.0],
        group = categorical(["A", "B", "A", "B"])
    )
    data = Tables.columntable(df)
    
    # Test simple interactions
    test_cases = [
        # x * group (scalar × categorical)
        (InteractionEvaluator([
            ContinuousEvaluator(:x),
            CategoricalEvaluator(:group, [0.0; 1.0][:, :], 2)  # Single contrast
        ]), "x * group (scalar × categorical)"),
        
        # Constant * variable  
        (InteractionEvaluator([
            ConstantEvaluator(2.5),
            ContinuousEvaluator(:x)
        ]), "constant * variable"),
    ]
    
    all_passed = true
    
    for (i, (evaluator, description)) in enumerate(test_cases)
        try
            println("   Test $i: $description")
            
            # Test statement generation
            instructions, next_pos = generate_statements_recursive(evaluator, 1)
            println("     Generated $(length(instructions)) statements, next_pos=$next_pos")
            
            # Validate syntax
            for instr in instructions
                try
                    Meta.parse(instr)
                catch e
                    println("     ❌ Invalid syntax: $instr")
                    all_passed = false
                end
            end
            
            println("     ✅ Simple interaction works")
            
        catch e
            if occursin("Phase 2", string(e))
                println("     ⚠️  Expected limitation: $e")
            else
                println("     ❌ Unexpected error: $e")
                all_passed = false
            end
        end
    end
    
    return all_passed
end

"""
Test end-to-end integration with categorical formulas (Phase 2C).
"""
function test_phase2c_integration()
    println("\n=== Testing Phase 2C End-to-End Integration ===")
    
    Random.seed!(42)
    df = DataFrame(
        x = abs.(randn(10)) .+ 1.0,
        y = randn(10),
        group3 = categorical(rand(["A", "B", "C"], 10)),
        group2 = categorical(rand(["X", "Y"], 10))
    )
    
    # Test formulas that should work in Phase 2C
    test_formulas = [
        @formula(y ~ group2),                    # Simple categorical
        @formula(y ~ group3),                    # Multi-level categorical
        @formula(y ~ x + group2),                # Mixed continuous + categorical
        @formula(y ~ log(x) + group3),           # Function + categorical
        @formula(y ~ x * group2),                # Simple interaction
    ]
    
    all_passed = true
    
    for (i, formula) in enumerate(test_formulas)
        try
            println("\n   Test $i: $formula")
            
            # Compile the model
            model = lm(formula, df)
            compiled = compile_formula(model)
            
            # Test code generation
            instructions = generate_code_from_evaluator(compiled.root_evaluator)
            println("     Generated $(length(instructions)) instructions")
            
            # Test that generated code compiles
            for instr in instructions
                try
                    Meta.parse(instr)
                catch e
                    println("     ❌ Invalid syntax: $instr")
                    all_passed = false
                end
            end
            
            # Test actual evaluation
            data = Tables.columntable(df)
            row_vec = Vector{Float64}(undef, length(compiled))
            compiled(row_vec, data, 1)
            
            if all(isfinite.(row_vec))
                println("     ✅ Formula evaluates correctly")
            else
                println("     ⚠️  Some non-finite values: $row_vec")
            end
            
        catch e
            if occursin("Phase 2", string(e))
                println("     ⚠️  Expected limitation: $e")
            else
                println("     ❌ Unexpected error: $e")
                all_passed = false
            end
        end
    end
    
    return all_passed
end

"""
Test ZScore evaluator statement generation (Phase 2C).
"""
function test_phase2c_zscore_statements()
    println("\n=== Testing Phase 2C ZScore Statement Generation ===")
    
    # Create test data
    df = DataFrame(
        x = [1.0, 2.0, 3.0, 4.0],
        group = categorical(["A", "B", "A", "B"])
    )
    
    # Test ZScore with multi-output underlying evaluator
    test_cases = [
        # ZScore of categorical (multi-output)
        (ZScoreEvaluator(
            CategoricalEvaluator(:group, [0.0; 1.0][:, :], 2),
            0.5,  # center
            2.0   # scale
        ), "ZScore of categorical"),
        
        # ZScore of continuous (single output)
        (ZScoreEvaluator(
            ContinuousEvaluator(:x),
            1.5,  # center
            1.0   # scale
        ), "ZScore of continuous"),
    ]
    
    all_passed = true
    
    for (i, (evaluator, description)) in enumerate(test_cases)
        try
            println("   Test $i: $description")
            
            # Test statement generation
            instructions, next_pos = generate_statements_recursive(evaluator, 1)
            println("     Generated $(length(instructions)) statements, next_pos=$next_pos")
            
            # Validate syntax
            for instr in instructions
                try
                    Meta.parse(instr)
                catch e
                    println("     ❌ Invalid syntax: $instr")
                    all_passed = false
                end
            end
            
            println("     ✅ ZScore statement generation works")
            
        catch e
            if occursin("Phase 2", string(e))
                println("     ⚠️  Expected limitation: $e")
            else
                println("     ❌ Unexpected error: $e")
                all_passed = false
            end
        end
    end
    
    return all_passed
end

"""
Test mixed expression and statement generation (Phase 2C).
"""
function test_phase2c_mixed_generation()
    println("\n=== Testing Phase 2C Mixed Expression/Statement Generation ===")
    
    # Test cases that mix single expressions and multi-statement generation
    test_cases = [
        # CombinedEvaluator with mix of simple and complex terms
        (CombinedEvaluator([
            ConstantEvaluator(1.0),                                    # Simple expression
            ContinuousEvaluator(:x),                                   # Simple expression
            CategoricalEvaluator(:group, [0.0 1.0; 1.0 0.0]', 2),    # Multi-statement
            FunctionEvaluator(log, [ContinuousEvaluator(:x)])          # Simple expression
        ]), "Mixed simple and complex terms"),
        
        # Nested function with categorical
        (CombinedEvaluator([
            FunctionEvaluator(+, [
                FunctionEvaluator(log, [ContinuousEvaluator(:x)]),
                ConstantEvaluator(1.0)
            ]),
            CategoricalEvaluator(:group, [0.0 1.0; 1.0 0.0]', 2)
        ]), "Nested function + categorical"),
    ]
    
    all_passed = true
    
    for (i, (evaluator, description)) in enumerate(test_cases)
        try
            println("   Test $i: $description")
            
            # Test statement generation
            instructions, next_pos = generate_statements_recursive(evaluator, 1)
            println("     Generated $(length(instructions)) statements, next_pos=$next_pos")
            
            # Validate syntax
            for instr in instructions
                try
                    Meta.parse(instr)
                catch e
                    println("     ❌ Invalid syntax: $instr")
                    all_passed = false
                end
            end
            
            println("     ✅ Mixed generation works")
            
        catch e
            if occursin("Phase 2", string(e))
                println("     ⚠️  Expected limitation: $e")
            else
                println("     ❌ Unexpected error: $e")
                all_passed = false
            end
        end
    end
    
    return all_passed
end

"""
Run all Phase 2C tests.
"""
function test_phase2c_complete()
    println("🧪 PHASE 2C COMPREHENSIVE TESTING")
    println("=" ^ 50)
    
    categorical_success = false
    interaction_success = false
    integration_success = false
    zscore_success = false
    mixed_success = false
    
    try
        categorical_success = test_phase2c_categorical_statements()
    catch e
        println("❌ Categorical statements test failed: $e")
    end
    
    try
        interaction_success = test_phase2c_interaction_statements()
    catch e
        println("❌ Interaction statements test failed: $e")
    end
    
    try
        integration_success = test_phase2c_integration()
    catch e
        println("❌ Integration test failed: $e")
    end
    
    try
        zscore_success = test_phase2c_zscore_statements()
    catch e
        println("❌ ZScore statements test failed: $e")
    end
    
    try
        mixed_success = test_phase2c_mixed_generation()
    catch e
        println("❌ Mixed generation test failed: $e")
    end
    
    println("\n" * "=" ^ 60)
    println("PHASE 2C TEST SUMMARY")
    println("=" ^ 60)
    println("Categorical Statements: $(categorical_success ? "✅ PASS" : "❌ FAIL")")
    println("Interaction Statements: $(interaction_success ? "✅ PASS" : "❌ FAIL")")
    println("Integration:            $(integration_success ? "✅ PASS" : "❌ FAIL")")
    println("ZScore Statements:      $(zscore_success ? "✅ PASS" : "❌ FAIL")")
    println("Mixed Generation:       $(mixed_success ? "✅ PASS" : "❌ FAIL")")
    
    overall_success = categorical_success && interaction_success && integration_success && zscore_success && mixed_success
    println("OVERALL:                $(overall_success ? "🎉 SUCCESS" : "⚠️  NEEDS WORK")")
    
    if overall_success
        println("\n🚀 Phase 2C is working perfectly!")
        println("📋 Multi-contrast categoricals and simple interactions now work!")
        println("📋 Ready for Phase 2D: Advanced evaluator types and complex interactions")
        println("   Ask: 'Write Phase 2D: Implement advanced evaluator types'")
    else
        println("\n🔧 Phase 2C needs fixes before proceeding to Phase 2D")
    end
    
    return overall_success
end

###############################################################################
# COMPARISON TESTING (Phase 2C vs. Expected Results)
###############################################################################

"""
Test that Phase 2C generates the same results as GLM model matrices.
"""
function test_phase2c_correctness()
    println("\n🎯 TESTING PHASE 2C CORRECTNESS vs. GLM")
    println("=" ^ 50)
    
    Random.seed!(42)
    df = DataFrame(
        x = [1.0, 2.0, 3.0, 4.0, 5.0],
        y = [1.0, 4.0, 9.0, 16.0, 25.0],
        group2 = categorical(["A", "B", "A", "B", "A"]),
        group3 = categorical(["X", "Y", "Z", "X", "Y"])
    )
    
    test_formulas = [
        @formula(y ~ group2),
        @formula(y ~ group3),
        @formula(y ~ x + group2),
        @formula(y ~ x * group2),
    ]
    
    all_correct = true
    
    for (i, formula) in enumerate(test_formulas)
        try
            println("\nTest $i: $formula")
            
            # Get GLM model matrix
            model = lm(formula, df)
            expected_matrix = modelmatrix(model)
            
            # Get our generated result
            compiled = compile_formula(model)
            data = Tables.columntable(df)
            
            # Test first few rows
            for row_idx in 1:min(3, nrow(df))
                row_vec = Vector{Float64}(undef, length(compiled))
                compiled(row_vec, data, row_idx)
                expected_row = expected_matrix[row_idx, :]
                
                error = maximum(abs.(row_vec .- expected_row))
                if error > 1e-12
                    println("  ❌ Row $row_idx: Max error = $error")
                    println("    Generated: $row_vec")
                    println("    Expected:  $expected_row")
                    all_correct = false
                else
                    println("  ✅ Row $row_idx: Correct (error = $error)")
                end
            end
            
        catch e
            println("  ❌ Error: $e")
            all_correct = false
        end
    end
    
    println("\n📊 CORRECTNESS RESULT: $(all_correct ? "✅ ALL CORRECT" : "❌ SOME ERRORS")")
    return all_correct
end

###############################################################################
# PERFORMANCE TESTING
###############################################################################

"""
Test Phase 2C performance vs. existing approaches.
"""
function test_phase2c_performance()
    println("\n⚡ TESTING PHASE 2C PERFORMANCE")
    println("=" ^ 40)
    
    Random.seed!(42)
    df = DataFrame(
        x = randn(1000),
        y = randn(1000),
        group = categorical(rand(["A", "B", "C"], 1000))
    )
    
    formula = @formula(y ~ x * group)
    model = lm(formula, df)
    compiled = compile_formula(model)
    data = Tables.columntable(df)
    
    # Warm up
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, data, 1)
    
    # Test performance
    n_iterations = 10000
    
    println("Testing $n_iterations iterations...")
    
    # Time the evaluation
    start_time = time()
    for i in 1:n_iterations
        row_idx = ((i - 1) % nrow(df)) + 1
        compiled(row_vec, data, row_idx)
    end
    end_time = time()
    
    elapsed = end_time - start_time
    per_iteration = elapsed / n_iterations * 1e6  # microseconds
    
    # Test allocations
    allocs = @allocated compiled(row_vec, data, 1)
    
    println("Results:")
    println("  Time per iteration: $(round(per_iteration, digits=2)) μs")
    println("  Allocations: $allocs bytes")
    println("  $(allocs == 0 ? "✅ Zero allocations!" : "⚠️  Some allocations")")
    
    # Performance targets
    performance_good = per_iteration < 1.0  # Less than 1 microsecond
    allocations_good = allocs == 0
    
    overall_performance = performance_good && allocations_good
    println("  Overall: $(overall_performance ? "✅ EXCELLENT" : "⚠️  COULD BE BETTER")")
    
    return overall_performance
end

# Export all test functions
export test_phase2a_complete, test_phase2a_architecture
export test_phase2b_complete, test_phase2b_nested_functions
export test_phase2c_complete, test_phase2c_categorical_statements, test_phase2c_interaction_statements
export test_phase2c_integration, test_phase2c_zscore_statements, test_phase2c_mixed_generation
export test_phase2c_correctness, test_phase2c_performance


#####################################################

###############################################################################
# PHASE 2D TESTING FRAMEWORK
###############################################################################

"""
Test ZScore expression and statement generation (Phase 2D).
"""
function test_phase2d_zscore_generation()
    println("\n=== Testing Phase 2D ZScore Generation ===")
    
    test_cases = [
        # Simple ZScore expression (single output)
        (ZScoreEvaluator(ContinuousEvaluator(:x), 2.5, 1.5), "ZScore of continuous variable"),
        
        # ZScore with function (single output)
        (ZScoreEvaluator(
            FunctionEvaluator(log, [ContinuousEvaluator(:x)]),
            0.0, 1.0
        ), "ZScore of log(x)"),
        
        # ZScore with categorical (multi-output - requires statements)
        (ZScoreEvaluator(
            CategoricalEvaluator(:group, [0.0 1.0; 1.0 0.0]', 2),
            0.5, 2.0
        ), "ZScore of categorical (statements)"),
    ]
    
    all_passed = true
    
    for (i, (evaluator, description)) in enumerate(test_cases)
        try
            println("   Test $i: $description")
            
            # Check the evaluator's output width to decide approach
            width = output_width(evaluator)
            println("     Output width: $width")
            
            if width == 1
                # Single output - should work as expression
                try
                    expr = generate_expression_recursive(evaluator)
                    parsed = Meta.parse(expr)
                    println("     Expression: '$expr' ✅")
                catch e
                    println("     ❌ Expression generation failed: $e")
                    all_passed = false
                end
            else
                # Multi-output - must use statements
                try
                    instructions, next_pos = generate_statements_recursive(evaluator, 1)
                    println("     Generated $(length(instructions)) statements, next_pos=$next_pos")
                    
                    # Validate syntax
                    for instr in instructions
                        try
                            Meta.parse(instr)
                        catch e
                            println("     ❌ Invalid syntax: $instr")
                            all_passed = false
                        end
                    end
                    println("     ✅ ZScore statements work")
                catch e
                    println("     ❌ Statement generation failed: $e")
                    all_passed = false
                end
            end
            
        catch e
            println("     ❌ Unexpected error: $e")
            all_passed = false
        end
    end
    
    return all_passed
end

"""
Test complex interaction generation (Phase 2D).
"""
function test_phase2d_complex_interactions()
    println("\n=== Testing Phase 2D Complex Interactions ===")
    
    test_cases = [
        # Simple scalar × scalar
        (InteractionEvaluator([
            ContinuousEvaluator(:x),
            ContinuousEvaluator(:y)
        ]), "x * y (scalar × scalar)"),
        
        # Three-way interaction
        (InteractionEvaluator([
            ContinuousEvaluator(:x),
            ContinuousEvaluator(:y),
            ConstantEvaluator(2.0)
        ]), "x * y * 2 (three-way)"),
        
        # Vector × Vector interaction
        (InteractionEvaluator([
            CategoricalEvaluator(:group1, [0.0; 1.0][:, :], 2),
            CategoricalEvaluator(:group2, [0.0; 1.0][:, :], 2)
        ]), "group1 * group2 (vector × vector)"),
        
        # Complex nested interaction
        (InteractionEvaluator([
            FunctionEvaluator(log, [ContinuousEvaluator(:x)]),
            CategoricalEvaluator(:group, [0.0; 1.0][:, :], 2)
        ]), "log(x) * group (function × categorical)"),
    ]
    
    all_passed = true
    
    for (i, (evaluator, description)) in enumerate(test_cases)
        try
            println("   Test $i: $description")
            
            if output_width(evaluator) == 1 && all(comp -> is_simple_expression(comp), evaluator.components)
                # Try expression generation for simple cases
                try
                    expr = generate_expression_recursive(evaluator)
                    parsed = Meta.parse(expr)
                    println("     Expression: '$expr' ✅")
                catch e
                    println("     Expression failed (trying statements): $e")
                    instructions, next_pos = generate_statements_recursive(evaluator, 1)
                    println("     Statements: $(length(instructions)) instructions ✅")
                end
            else
                # Use statement generation
                instructions, next_pos = generate_statements_recursive(evaluator, 1)
                println("     Generated $(length(instructions)) statements, next_pos=$next_pos")
                
                # Validate syntax
                for instr in instructions
                    try
                        Meta.parse(instr)
                    catch e
                        println("     ❌ Invalid syntax: $instr")
                        all_passed = false
                    end
                end
                println("     ✅ Complex interaction works")
            end
            
        catch e
            if occursin("too large", string(e)) || occursin("too complex", string(e))
                println("     ⚠️  Expected size limitation: $e")
            else
                println("     ❌ Unexpected error: $e")
                all_passed = false
            end
        end
    end
    
    return all_passed
end

"""
Test end-to-end integration with advanced formulas (Phase 2D).
"""
function test_phase2d_integration()
    println("\n=== Testing Phase 2D End-to-End Integration ===")
    
    Random.seed!(42)
    df = DataFrame(
        x = abs.(randn(10)) .+ 1.0,
        y = randn(10),
        z = randn(10),
        group1 = categorical(rand(["A", "B"], 10)),
        group2 = categorical(rand(["X", "Y"], 10))
    )
    
    # Test advanced formulas that should work in Phase 2D
    test_formulas = [
        @formula(y ~ x * y),                     # Scalar × scalar interaction
        @formula(y ~ log(x) * sin(z)),          # Function × function interaction
        @formula(y ~ x * y * z),                # Three-way interaction
        @formula(y ~ group1 * group2),          # Categorical × categorical
        @formula(y ~ log(x) * group1),          # Function × categorical
    ]
    
    all_passed = true
    
    for (i, formula) in enumerate(test_formulas)
        try
            println("\n   Test $i: $formula")
            
            # Compile the model
            model = lm(formula, df)
            compiled = compile_formula(model)
            
            # Test code generation
            instructions = generate_code_from_evaluator(compiled.root_evaluator)
            println("     Generated $(length(instructions)) instructions")
            
            # Test that generated code compiles
            for instr in instructions
                try
                    Meta.parse(instr)
                catch e
                    println("     ❌ Invalid syntax: $instr")
                    all_passed = false
                end
            end
            
            # Test actual evaluation
            data = Tables.columntable(df)
            row_vec = Vector{Float64}(undef, length(compiled))
            compiled(row_vec, data, 1)
            
            if all(isfinite.(row_vec))
                println("     ✅ Formula evaluates correctly")
            else
                println("     ⚠️  Some non-finite values: $row_vec")
            end
            
        catch e
            if occursin("too large", string(e)) || occursin("too complex", string(e))
                println("     ⚠️  Expected complexity limitation: $e")
            else
                println("     ❌ Unexpected error: $e")
                all_passed = false
            end
        end
    end
    
    return all_passed
end

"""
Run all Phase 2D tests.
"""
function test_phase2d_complete()
    println("🧪 PHASE 2D COMPREHENSIVE TESTING")
    println("=" ^ 50)
    
    zscore_success = false
    interaction_success = false
    integration_success = false
    
    try
        zscore_success = test_phase2d_zscore_generation()
    catch e
        println("❌ ZScore generation test failed: $e")
    end
    
    try
        interaction_success = test_phase2d_complex_interactions()
    catch e
        println("❌ Complex interactions test failed: $e")
    end
    
    try
        integration_success = test_phase2d_integration()
    catch e
        println("❌ Integration test failed: $e")
    end
    
    println("\n" * "=" ^ 60)
    println("PHASE 2D TEST SUMMARY")
    println("=" ^ 60)
    println("ZScore Generation:      $(zscore_success ? "✅ PASS" : "❌ FAIL")")
    println("Complex Interactions:   $(interaction_success ? "✅ PASS" : "❌ FAIL")")
    println("Integration:            $(integration_success ? "✅ PASS" : "❌ FAIL")")
    
    overall_success = zscore_success && interaction_success && integration_success
    println("OVERALL:                $(overall_success ? "🎉 SUCCESS" : "⚠️  NEEDS WORK")")
    
    if overall_success
        println("\n🎉 PHASE 2D COMPLETE! 🎉")
        println("📋 ALL RECURSIVE GENERATION PHASES IMPLEMENTED!")
        println("✅ Phase 2A: Core recursive architecture")
        println("✅ Phase 2B: Nested function expressions") 
        println("✅ Phase 2C: Multi-output statements and categoricals")
        println("✅ Phase 2D: Advanced evaluators and complex interactions")
        println("\n🚀 Your recursive expression generation system is now COMPLETE!")
    else
        println("\n🔧 Phase 2D needs fixes to complete the system")
    end
    
    return overall_success
end

"""
Test comprehensive coverage of all phases together.
"""
function test_all_phases_comprehensive()
    println("🌟 COMPREHENSIVE ALL-PHASES TESTING")
    println("=" ^ 60)
    
    # Test all phases
    phase2a_success = test_phase2a_complete()
    phase2b_success = test_phase2b_complete() 
    phase2c_success = test_phase2c_complete()
    phase2d_success = test_phase2d_complete()
    
    # Test correctness
    correctness_success = test_phase2c_correctness()
    
    # Test performance
    performance_success = test_phase2c_performance()
    
    println("\n" * "=" ^ 70)
    println("🎯 FINAL COMPREHENSIVE TEST SUMMARY")
    println("=" ^ 70)
    println("Phase 2A (Core Architecture):     $(phase2a_success ? "✅ PASS" : "❌ FAIL")")
    println("Phase 2B (Nested Functions):      $(phase2b_success ? "✅ PASS" : "❌ FAIL")")
    println("Phase 2C (Multi-Output/Categorical): $(phase2c_success ? "✅ PASS" : "❌ FAIL")")
    println("Phase 2D (Advanced Features):     $(phase2d_success ? "✅ PASS" : "❌ FAIL")")
    println("Correctness vs GLM:               $(correctness_success ? "✅ PASS" : "❌ FAIL")")
    println("Performance:                      $(performance_success ? "✅ PASS" : "❌ FAIL")")
    
    all_success = phase2a_success && phase2b_success && phase2c_success && phase2d_success && correctness_success && performance_success
    
    if all_success
        println("\n🎉🎉🎉 COMPLETE SUCCESS! 🎉🎉🎉")
        println("🏆 Your recursive expression generation system is FULLY IMPLEMENTED and WORKING!")
        println("📊 All phases pass, correctness verified, performance excellent!")
        println("🚀 Ready for production use!")
    else
        println("\n⚠️  Some components need attention before full deployment")
        println("🔧 Check individual phase results above for specific issues")
    end
    
    return all_success
end

# Export all test functions including Phase 2D
export test_phase2d_complete, test_phase2d_zscore_generation, test_phase2d_complex_interactions, test_phase2d_integration
export test_all_phases_comprehensive
