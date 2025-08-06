# debug_sqrt_derivative.jl
# Debug the sqrt derivative specifically - FIXED

using FormulaCompiler
using DataFrames, GLM, Tables

function debug_sqrt_derivative()
    println("🔍 Debugging Sqrt Derivative")
    println("=" ^ 40)
    
    # Create simple test case  
    df = DataFrame(y = [1.0, 2.0, 3.0], z = [1.0, 4.0, 9.0])
    model = lm(@formula(y ~ sqrt(z)), df)
    
    println("Formula: y ~ sqrt(z)")
    println("Data: z = [1.0, 4.0, 9.0]")
    println("Expected ∂√z/∂z: [1/(2√1), 1/(2√4), 1/(2√9)] = [0.5, 0.25, 0.167]")
    
    # Compile formula
    compiled = compile_formula(model, verbose=true)
    
    # Examine the evaluator structure
    root_evaluator = compiled.root_evaluator
    println("\nRoot evaluator type: $(typeof(root_evaluator))")
    
    if root_evaluator isa CombinedEvaluator
        println("Sub-evaluators:")
        for (i, sub) in enumerate(root_evaluator.sub_evaluators)
            println("  $i: $(typeof(sub))")
            if sub isa FunctionEvaluator
                println("     Function: $(sub.func)")
                println("     Args: $(length(sub.arg_evaluators))")
                for (j, arg) in enumerate(sub.arg_evaluators)
                    println("       Arg $j: $(typeof(arg))")
                    if arg isa ContinuousEvaluator
                        println("         Column: $(arg.column)")
                    end
                end
            end
        end
    end
    
    # Compile derivative
    println("\nCompiling derivative w.r.t. :z...")
    dz_compiled = compile_derivative_formula(compiled, :z, verbose=true)
    
    # Test what kind of derivative evaluator was created
    println("\nAnalyzing derivative computation:")
    if root_evaluator isa CombinedEvaluator
        for (i, sub) in enumerate(root_evaluator.sub_evaluators)
            if sub isa FunctionEvaluator && sub.func === sqrt
                println("Found sqrt function at position $i")
                
                # Test the derivative computation manually
                derivative_eval = compute_derivative_evaluator(sub, :z)
                println("Derivative evaluator type: $(typeof(derivative_eval))")
                
                if derivative_eval isa ChainRuleEvaluator
                    println("Chain rule derivative detected")
                    println("Inner evaluator: $(typeof(derivative_eval.inner_evaluator))")
                    println("Inner derivative: $(typeof(derivative_eval.inner_derivative))")
                    println("Original function: $(derivative_eval.original_func)")
                    
                    # Test the derivative function manually
                    test_values = [1.0, 4.0, 9.0]
                    println("Testing derivative function manually:")
                    for val in test_values
                        try
                            deriv_result = derivative_eval.derivative_func(val)
                            expected = 0.5 / sqrt(val)
                            println("  f'($val) = $deriv_result, expected = $expected")
                        catch e
                            println("  f'($val) = ERROR: $e")
                        end
                    end
                end
            end
        end
    end
    
    # Test actual evaluation
    println("\nTesting actual derivative evaluation:")
    data = Tables.columntable(df)
    deriv_vec = Vector{Float64}(undef, length(dz_compiled))
    
    for i in 1:nrow(df)
        modelrow!(deriv_vec, dz_compiled, data, i)
        z_val = df.z[i]
        expected = 0.5 / sqrt(z_val)
        actual = deriv_vec[2]  # Position after intercept
        error = abs(actual - expected)
        
        println("  Row $i: z=$z_val")
        println("    Got: $actual")
        println("    Expected: $expected") 
        println("    Error: $error")
        
        # Also test numerical derivative for comparison
        numerical = compute_numerical_derivative_simple(z_val)
        println("    Numerical: $numerical")
        println()
    end
    
    # Test the generated code - FIXED ACCESS TO HASH
    println("Generated derivative instructions:")
    try
        # Get the hash value correctly - it's the type parameter of Val
        hash_val = dz_compiled.formula_val
        if hash_val isa Val
            # For Val{H}, get H using typeof().parameters[1]
            hash_value = typeof(hash_val).parameters[1]
            if haskey(FormulaCompiler.DERIVATIVE_CACHE, hash_value)
                instructions, _, _ = FormulaCompiler.DERIVATIVE_CACHE[hash_value]
                for (i, instr) in enumerate(instructions)
                    println("  $i: $instr")
                end
            else
                println("  Hash $hash_value not found in DERIVATIVE_CACHE")
                println("  Available hashes: $(collect(keys(FormulaCompiler.DERIVATIVE_CACHE)))")
            end
        else
            println("  formula_val is not a Val type: $(typeof(hash_val))")
        end
    catch e
        println("  Error accessing generated code: $e")
    end
end

function compute_numerical_derivative_simple(z_val, ε=1e-8)
    f_plus = sqrt(z_val + ε)
    f_minus = sqrt(z_val - ε)
    return (f_plus - f_minus) / (2ε)
end

# Clear all caches to force recompilation
FormulaCompiler.clear_derivative_cache!()
empty!(FormulaCompiler.FORMULA_CACHE)

# Test the sqrt derivative again
debug_sqrt_derivative()