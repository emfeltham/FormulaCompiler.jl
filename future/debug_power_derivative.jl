# debug_power_derivative.jl
# Debug what's happening with the power derivative

using FormulaCompiler
using DataFrames, GLM, Tables

function debug_power_derivative()
    println("🔍 Debugging Power Derivative")
    println("=" ^ 40)
    
    # Clear caches
    FormulaCompiler.clear_derivative_cache!()
    empty!(FormulaCompiler.FORMULA_CACHE)
    
    # Create simple test case
    df = DataFrame(y = [4.0], x = [2.0])
    model = lm(@formula(y ~ x^2), df)
    
    println("Formula: y ~ x^2")
    println("Data: x = 2.0")
    println("Expected ∂(x²)/∂x at x=2: 2*2 = 4.0")
    
    # Compile formula
    compiled = compile_formula(model, verbose=true)
    println("\nCompiled formula width: $(length(compiled))")
    println("Column names: $(compiled.column_names)")
    
    # Get the root evaluator and examine its structure
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
                    if arg isa ConstantEvaluator
                        println("         Value: $(arg.value)")
                    elseif arg isa ContinuousEvaluator
                        println("         Column: $(arg.column)")
                    end
                end
            end
        end
    end
    
    # Compile derivative
    println("\nCompiling derivative w.r.t. :x...")
    dx_compiled = compile_derivative_formula(compiled, :x, verbose=true)
    
    # Look at what derivative evaluator was created
    println("\nDerivative evaluator created:")
    deriv_eval = dx_compiled.root_derivative_evaluator
    println("Type: $(typeof(deriv_eval))")
    println("Original evaluator: $(typeof(deriv_eval.original_evaluator))")
    println("Focal variable: $(deriv_eval.focal_variable)")
    println("Target width: $(deriv_eval.target_width)")
    
    # Test the derivative evaluator directly
    println("\nTesting derivative evaluator directly:")
    if deriv_eval.original_evaluator isa CombinedEvaluator
        for (i, sub) in enumerate(deriv_eval.original_evaluator.sub_evaluators)
            println("Sub-evaluator $i: $(typeof(sub))")
            sub_derivative = compute_derivative_evaluator(sub, :x)
            println("  Derivative: $(typeof(sub_derivative))")
            if sub_derivative isa ScaledEvaluator
                println("    Scale factor: $(sub_derivative.scale_factor)")
                println("    Inner evaluator: $(typeof(sub_derivative.evaluator))")
            elseif sub_derivative isa ConstantEvaluator
                println("    Value: $(sub_derivative.value)")
            end
        end
    end
    
    # Test evaluation
    data = Tables.columntable(df)
    deriv_vec = Vector{Float64}(undef, length(dx_compiled))
    
    println("\nEvaluating derivative:")
    modelrow!(deriv_vec, dx_compiled, data, 1)
    println("Result: $deriv_vec")
    println("Expected: [0.0, 4.0]")  # [intercept_deriv, x^2_deriv]
    
    # Check if the issue is in evaluation or in derivative computation
    println("\nChecking each position:")
    for i in 1:length(deriv_vec)
        println("  Position $i: $(deriv_vec[i])")
    end
end

debug_power_derivative()
