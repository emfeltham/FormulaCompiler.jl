# debug_interaction_derivative.jl
# Debug the x * w interaction derivative

using FormulaCompiler
using FormulaCompiler: contains_variable
using DataFrames, GLM, Tables

function debug_interaction_derivative()
    println("🔍 Debugging Interaction Derivative ∂(x * w)/∂x")
    println("=" ^ 50)
    
    # Create test case
    df = DataFrame(
        y = [1.0, 2.0, 3.0, 4.0],
        x = [1.0, 2.0, 3.0, 4.0], 
        w = [0.5, 1.5, 2.5, 3.5]
    )
    
    println("Formula: y ~ x * w")
    println("Data:")
    println("  x = $(df.x)")
    println("  w = $(df.w)")
    println("Expected ∂(x * w)/∂x = w = $(df.w)")
    
    # Clear caches
    FormulaCompiler.clear_derivative_cache!()
    empty!(FormulaCompiler.FORMULA_CACHE)
    
    # Compile formula
    model = lm(@formula(y ~ x * w), df)
    compiled = compile_formula(model, verbose=true)
    
    println("\nFormula structure:")
    println("Root evaluator: $(typeof(compiled.root_evaluator))")
    
    if compiled.root_evaluator isa CombinedEvaluator
        for (i, sub) in enumerate(compiled.root_evaluator.sub_evaluators)
            println("  $i: $(typeof(sub))")
            if sub isa InteractionEvaluator
                println("     Components: $(length(sub.components))")
                for (j, comp) in enumerate(sub.components)
                    println("       $j: $(typeof(comp))")
                    if comp isa ContinuousEvaluator
                        println("          Column: $(comp.column)")
                    end
                end
            elseif sub isa ContinuousEvaluator
                println("     Column: $(sub.column)")
            end
        end
    end
    
    # Compile derivative
    println("\nCompiling derivative w.r.t. :x...")
    dx_compiled = compile_derivative_formula(compiled, :x, verbose=true)
    
    # Test what the interaction derivative looks like
    println("\nAnalyzing interaction derivative:")
    if compiled.root_evaluator isa CombinedEvaluator
        for (i, sub) in enumerate(compiled.root_evaluator.sub_evaluators)
            if sub isa InteractionEvaluator
                println("Found interaction at position $i")
                
                interaction_derivative = compute_derivative_evaluator(sub, :x)
                println("Interaction derivative type: $(typeof(interaction_derivative))")
                
                # For x * w, ∂/∂x should give w
                if length(sub.components) == 2
                    comp1, comp2 = sub.components[1], sub.components[2]
                    println("Component 1: $(typeof(comp1)) - $(comp1 isa ContinuousEvaluator ? comp1.column : "")")
                    println("Component 2: $(typeof(comp2)) - $(comp2 isa ContinuousEvaluator ? comp2.column : "")")
                    
                    # Check which component depends on x
                    comp1_has_x = contains_variable(comp1, :x)
                    comp2_has_x = contains_variable(comp2, :x)
                    println("Component 1 depends on x: $comp1_has_x")
                    println("Component 2 depends on x: $comp2_has_x")
                    
                    if comp1_has_x && !comp2_has_x
                        println("Expected: ∂(x * w)/∂x = 1 * w = w")
                    elseif !comp1_has_x && comp2_has_x
                        println("Expected: ∂(w * x)/∂x = w * 1 = w")
                    end
                end
            end
        end
    end
    
    # Test actual evaluation
    println("\nTesting actual derivative evaluation:")
    data = Tables.columntable(df)
    deriv_vec = Vector{Float64}(undef, length(dx_compiled))
    
    for i in 1:nrow(df)
        fill!(deriv_vec, 999.0)  # Fill with sentinel value to see what gets written
        modelrow!(deriv_vec, dx_compiled, data, i)
        
        x_val = df.x[i]
        w_val = df.w[i]
        expected_deriv = [0.0, 1.0, 0.0, w_val]  # [∂intercept/∂x, ∂x/∂x, ∂w/∂x, ∂(x*w)/∂x]
        
        println("  Row $i: x=$x_val, w=$w_val")
        println("    Got:      $deriv_vec")
        println("    Expected: $expected_deriv")
        
        # Check each component
        for j in 1:length(deriv_vec)
            error = abs(deriv_vec[j] - expected_deriv[j])
            status = error < 1e-10 ? "✅" : "❌"
            println("    Position $j: $status got=$(deriv_vec[j]), expected=$(expected_deriv[j]), error=$error")
        end
        println()
    end
    
    # Check generated instructions
    println("Generated derivative instructions:")
    hash_val = dx_compiled.formula_val
    hash_value = typeof(hash_val).parameters[1]
    
    if haskey(FormulaCompiler.DERIVATIVE_CACHE, hash_value)
        instructions, _, _ = FormulaCompiler.DERIVATIVE_CACHE[hash_value]
        for (i, instr) in enumerate(instructions)
            println("  $i: $instr")
        end
    end
    
    # Test numerical derivative for comparison
    println("\nNumerical derivative verification:")
    for i in 1:min(2, nrow(df))
        numerical_deriv = compute_numerical_interaction_derivative(compiled, :x, data, i)
        analytical_deriv = deriv_vec
        modelrow!(analytical_deriv, dx_compiled, data, i)
        
        println("  Row $i:")
        println("    Numerical:  $numerical_deriv")
        println("    Analytical: $analytical_deriv")
        
        max_error = maximum(abs.(numerical_deriv .- analytical_deriv))
        println("    Max error:  $max_error")
    end
end

function compute_numerical_interaction_derivative(compiled, focal_var, data, row_idx, ε=1e-8)
    # Get current value
    current_val = Float64(data[focal_var][row_idx])
    
    # Create modified data
    original_column = collect(data[focal_var])
    
    # Evaluate at x + ε
    modified_plus = copy(original_column)
    modified_plus[row_idx] = current_val + ε
    data_plus = merge(data, (focal_var => modified_plus,))
    
    result_plus = Vector{Float64}(undef, length(compiled))
    modelrow!(result_plus, compiled, data_plus, row_idx)
    
    # Evaluate at x - ε
    modified_minus = copy(original_column)
    modified_minus[row_idx] = current_val - ε
    data_minus = merge(data, (focal_var => modified_minus,))
    
    result_minus = Vector{Float64}(undef, length(compiled))
    modelrow!(result_minus, compiled, data_minus, row_idx)
    
    # Central difference
    return (result_plus .- result_minus) ./ (2ε)
end

# Run the debug
debug_interaction_derivative()

function debug_product_evaluator(evaluator::ProductEvaluator, focal_variable::Symbol)
    println("DEBUG: Checking if ProductEvaluator is zero")
    println("  Focal variable: $focal_variable")
    println("  Number of components: $(length(evaluator.components))")
    
    for (i, component) in enumerate(evaluator.components)
        println("  Component $i: $(typeof(component))")
        if component isa ContinuousEvaluator
            println("    Column: $(component.column)")
            contains_focal = contains_variable(component, focal_variable)
            println("    Contains focal variable: $contains_focal")
        elseif component isa ConstantEvaluator
            println("    Value: $(component.value)")
        end
    end
    
    # Check for zero components
    has_zero = any(comp -> comp isa ConstantEvaluator && comp.value == 0.0, evaluator.components)
    println("  Has zero component: $has_zero")
    
    return has_zero
end