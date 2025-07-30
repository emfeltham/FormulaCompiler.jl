# testing.jl

###############################################################################
# TESTING FUNCTION
###############################################################################

"""
    test_self_contained_evaluators()

Comprehensive test of the self-contained evaluator system.
"""
function test_self_contained_evaluators()
    println("🧪 TESTING SELF-CONTAINED EVALUATORS WITH SCRATCH SPACE")
    println("=" ^ 60)
    
    # Create test data
    df = DataFrame(
        x = [1.0, 2.0, 3.0],
        y = [4.0, 5.0, 6.0],
        group = categorical(["A", "B", "A"]),
        z = [0.1, 0.2, 0.3]
    )
    data = Tables.columntable(df)
    
    test_cases = [
        (@formula(y ~ x), "Simple: y ~ x"),
        (@formula(y ~ x + group), "Mixed: y ~ x + group"),
        (@formula(y ~ x * group), "Interaction: y ~ x * group"),
        (@formula(y ~ log(z) + x), "Function: y ~ log(z) + x"),
    ]
    
    for (formula, description) in test_cases
        println("\n🔍 Testing: $description")
        println("   Formula: $formula")
        
        try
            # Create model and evaluator
            model = lm(formula, df)
            evaluator = compile_term(fixed_effects_form(model).rhs)
            
            println("   📊 Evaluator Analysis:")
            println("     Type: $(typeof(evaluator))")
            println("     Output width: $(output_width(evaluator))")
            println("     Positions: $(get_positions(evaluator))")
            println("     Max scratch needed: $(max_scratch_needed(evaluator))")
            
            # Test compilation
            compiled = compile_formula(model, data)
            println("   ✅ Compilation successful")
            println("     Compiled length: $(length(compiled))")
            println("     Scratch size: $(length(compiled.scratch_space))")
            
            # Test execution
            output = Vector{Float64}(undef, length(compiled))
            compiled(output, data, 1)
            
            # Compare with expected
            expected = modelmatrix(model)[1, :]
            if isapprox(output, expected, rtol=1e-12)
                println("   ✅ Execution correct")
                println("     Result: $output")
            else
                println("   ❌ Execution incorrect")
                println("     Got:      $output")
                println("     Expected: $expected")
            end
            
            # Test zero allocation
            allocs = @allocated compiled(output, data, 1)
            if allocs == 0
                println("   ✅ Zero allocation achieved!")
            else
                println("   ❌ Still allocating: $allocs bytes")
            end
            
        catch e
            println("   ❌ Test failed: $e")
            println("     Error type: $(typeof(e))")
        end
    end
    
    println("\n" ^ 1)
    println("🎯 SELF-CONTAINED EVALUATOR SYSTEM TEST COMPLETE")
end
