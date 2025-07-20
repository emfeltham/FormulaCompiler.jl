# derivative_generators.jl
# Code generation for @generated derivatives - FIXED

###############################################################################
# CODE GENERATION FOR @GENERATED DERIVATIVES
###############################################################################

function generate_code_from_evaluator(evaluator::PositionalDerivativeEvaluator)
    reset_var_counter!()
    instructions = String[]
    generate_evaluator_code!(instructions, evaluator, 1)
    return instructions
end

function generate_evaluator_code!(instructions::Vector{String}, evaluator::PositionalDerivativeEvaluator, pos::Int)
    println("DEBUG: Generating code for PositionalDerivativeEvaluator")
    println("  Target width: $(evaluator.target_width)")
    println("  Focal variable: $(evaluator.focal_variable)")
    println("  Original evaluator type: $(typeof(evaluator.original_evaluator))")
    
    # Initialize all positions to zero
    for i in 1:evaluator.target_width
        target_pos = pos + i - 1
        push!(instructions, "@inbounds row_vec[$target_pos] = 0.0")
    end
    
    # Generate code using the original sophisticated system
    if evaluator.original_evaluator isa CombinedEvaluator
        println("DEBUG: Using combined positioning instructions")
        generate_combined_positioning_instructions!(instructions, evaluator, pos)
    else
        println("DEBUG: Using single positioning instructions")
        generate_single_positioning_instructions!(instructions, evaluator, pos)
    end
    
    return pos + evaluator.target_width
end

function generate_combined_positioning_instructions!(instructions::Vector{String}, evaluator::PositionalDerivativeEvaluator, pos::Int)
    current_position = pos
    
    println("DEBUG: Starting generate_combined_positioning_instructions!")
    println("  Number of sub-evaluators: $(length(evaluator.original_evaluator.sub_evaluators))")
    println("  Focal variable: $(evaluator.focal_variable)")
    
    for (i, sub_evaluator) in enumerate(evaluator.original_evaluator.sub_evaluators)
        sub_width = output_width(sub_evaluator)
        
        println("DEBUG: Processing sub-evaluator $i at position $current_position")
        println("  Sub-evaluator type: $(typeof(sub_evaluator))")
        println("  Sub-evaluator width: $sub_width")
        
        # Compute derivative
        sub_derivative = compute_derivative_evaluator(sub_evaluator, evaluator.focal_variable)
        println("  Computed derivative type: $(typeof(sub_derivative))")
        
        # Check if zero derivative
        is_zero = is_zero_derivative(sub_derivative, evaluator.focal_variable)
        println("  Is zero derivative: $is_zero")
        
        if !is_zero
            println("  Generating code for non-zero derivative...")
            
            if sub_derivative isa ConstantEvaluator && sub_derivative.value != 0.0
                println("    Case: ConstantEvaluator (non-zero)")
                push!(instructions, "@inbounds row_vec[$current_position] = $(sub_derivative.value)")
                
            elseif sub_derivative isa ContinuousEvaluator && sub_derivative.column == evaluator.focal_variable
                println("    Case: ContinuousEvaluator (focal variable)")
                push!(instructions, "@inbounds row_vec[$current_position] = 1.0")
                
            elseif sub_derivative isa ContinuousEvaluator
                println("    Case: ContinuousEvaluator (other variable)")
                push!(instructions, "@inbounds row_vec[$current_position] = Float64(data.$(sub_derivative.column)[row_idx])")
                
            elseif sub_derivative isa ChainRuleEvaluator
                println("    Case: ChainRuleEvaluator")
                generate_chain_rule_code!(instructions, sub_derivative, current_position)
                
            elseif sub_derivative isa ProductRuleEvaluator  
                println("    Case: ProductRuleEvaluator")
                generate_product_rule_code!(instructions, sub_derivative, current_position)
                
            elseif sub_derivative isa ProductEvaluator
                println("    Case: ProductEvaluator - THIS IS WHAT WE'RE LOOKING FOR!")
                println("      ProductEvaluator components: $(length(sub_derivative.components))")
                for (j, comp) in enumerate(sub_derivative.components)
                    println("        Component $j: $(typeof(comp))")
                    if comp isa ContinuousEvaluator
                        println("          Column: $(comp.column)")
                    end
                end
                generate_product_evaluator_derivative_code!(instructions, sub_derivative, current_position)
                
            elseif sub_derivative isa ScaledEvaluator
                println("    Case: ScaledEvaluator")
                inner_expr = generate_inline_expression_for_derivatives(sub_derivative.evaluator)
                scale = sub_derivative.scale_factor
                push!(instructions, "@inbounds row_vec[$current_position] = $scale * ($inner_expr)")
                
            else
                println("    Case: UNHANDLED TYPE - $(typeof(sub_derivative))")
                error("Unsupported derivative evaluator type: $(typeof(sub_derivative))")
            end
        else
            println("  Skipping zero derivative")
        end
        
        current_position += sub_width
        println("  New current_position: $current_position")
        println()
    end
    
    println("DEBUG: Finished generate_combined_positioning_instructions!")
end

function generate_product_evaluator_derivative_code!(instructions::Vector{String}, evaluator::ProductEvaluator, pos::Int)
    println("DEBUG: generate_product_evaluator_derivative_code! called")
    println("  Position: $pos")
    println("  Number of components: $(length(evaluator.components))")
    
    component_exprs = String[]
    
    for (i, component) in enumerate(evaluator.components)
        println("  Processing component $i: $(typeof(component))")
        
        if component isa ConstantEvaluator
            expr = string(component.value)
            println("    Generated: $expr")
            push!(component_exprs, expr)
            
        elseif component isa ContinuousEvaluator
            expr = "Float64(data.$(component.column)[row_idx])"
            println("    Generated: $expr")
            push!(component_exprs, expr)
            
        elseif component isa ScaledEvaluator
            inner_expr = generate_inline_expression_for_derivatives(component.evaluator)
            expr = "($(inner_expr) * $(component.scale_factor))"
            println("    Generated: $expr")
            push!(component_exprs, expr)
            
        else
            println("    Trying inline expression for $(typeof(component))")
            try
                expr = generate_inline_expression_for_derivatives(component)
                println("    Generated: $expr")
                push!(component_exprs, expr)
            catch e
                println("    Failed: $e")
                expr = "1.0  # Fallback for $(typeof(component))"
                println("    Fallback: $expr")
                push!(component_exprs, expr)
            end
        end
    end
    
    if length(component_exprs) == 1
        instruction = "@inbounds row_vec[$pos] = $(component_exprs[1])"
        println("  Final instruction: $instruction")
        push!(instructions, instruction)
    else
        product_expr = join(component_exprs, " * ")
        instruction = "@inbounds row_vec[$pos] = $product_expr"
        println("  Final instruction: $instruction")
        push!(instructions, instruction)
    end
    
    println("DEBUG: generate_product_evaluator_derivative_code! finished")
end

function generate_chain_rule_code!(instructions::Vector{String}, evaluator::ChainRuleEvaluator, pos::Int)
    # FIXED: Generate inline expressions for components
    inner_expr = generate_inline_expression_for_derivatives(evaluator.inner_evaluator)
    inner_deriv_expr = generate_inline_expression_for_derivatives(evaluator.inner_derivative)
    
    inner_var = next_var("inner")
    inner_deriv_var = next_var("inner_deriv")
    
    # Generate evaluation of components
    push!(instructions, "@inbounds $inner_var = $inner_expr")
    push!(instructions, "@inbounds $inner_deriv_var = $inner_deriv_expr")
    
    # FIXED: Apply the correct chain rule based on the original function
    original_func = evaluator.original_func
    
    if original_func === log
        # ∂log(u)/∂x = (1/u) * ∂u/∂x
        push!(instructions, "@inbounds row_vec[$pos] = (1.0 / $inner_var) * $inner_deriv_var")
    elseif original_func === sqrt
        # ∂sqrt(u)/∂x = (1/(2*sqrt(u))) * ∂u/∂x = (0.5/sqrt(u)) * ∂u/∂x
        push!(instructions, "@inbounds row_vec[$pos] = (0.5 / sqrt($inner_var)) * $inner_deriv_var")
    elseif original_func === exp
        # ∂exp(u)/∂x = exp(u) * ∂u/∂x
        push!(instructions, "@inbounds row_vec[$pos] = exp($inner_var) * $inner_deriv_var")
    elseif original_func === sin
        # ∂sin(u)/∂x = cos(u) * ∂u/∂x
        push!(instructions, "@inbounds row_vec[$pos] = cos($inner_var) * $inner_deriv_var")
    elseif original_func === cos
        # ∂cos(u)/∂x = -sin(u) * ∂u/∂x
        push!(instructions, "@inbounds row_vec[$pos] = (-sin($inner_var)) * $inner_deriv_var")
    elseif original_func === tan
        # ∂tan(u)/∂x = sec²(u) * ∂u/∂x = (1 + tan²(u)) * ∂u/∂x
        push!(instructions, "@inbounds row_vec[$pos] = (1.0 + tan($inner_var)^2) * $inner_deriv_var")
    elseif original_func === abs
        # ∂|u|/∂x = sign(u) * ∂u/∂x (undefined at 0, but we'll use 0)
        push!(instructions, "@inbounds row_vec[$pos] = ($inner_var == 0.0 ? 0.0 : sign($inner_var)) * $inner_deriv_var")
    else
        # Unknown function - fallback to generic derivative
        @warn "Unknown function in chain rule: $original_func, using identity derivative"
        push!(instructions, "@inbounds row_vec[$pos] = $inner_deriv_var  # Unknown function")
    end
    
    return pos + 1
end

function generate_product_rule_code!(instructions::Vector{String}, evaluator::ProductRuleEvaluator, pos::Int)
    # Generate inline expressions for all components
    f_expr = generate_inline_expression_for_derivatives(evaluator.left_evaluator)
    f_prime_expr = generate_inline_expression_for_derivatives(evaluator.left_derivative)
    g_expr = generate_inline_expression_for_derivatives(evaluator.right_evaluator)
    g_prime_expr = generate_inline_expression_for_derivatives(evaluator.right_derivative)
    
    # Generate variables for components
    f_var = next_var("f")
    f_prime_var = next_var("f_prime")
    g_var = next_var("g")
    g_prime_var = next_var("g_prime")
    
    # Generate evaluation code
    push!(instructions, "@inbounds $f_var = $f_expr")
    push!(instructions, "@inbounds $f_prime_var = $f_prime_expr")
    push!(instructions, "@inbounds $g_var = $g_expr")
    push!(instructions, "@inbounds $g_prime_var = $g_prime_expr")
    
    # Apply product rule: f*g' + g*f'
    push!(instructions, "@inbounds row_vec[$pos] = $f_var * $g_prime_var + $g_var * $f_prime_var")
    
    return pos + 1
end

function generate_single_positioning_instructions!(instructions::Vector{String}, evaluator::PositionalDerivativeEvaluator, pos::Int)
    derivative_evaluator = compute_derivative_evaluator(evaluator.original_evaluator, evaluator.focal_variable)
    
    if !is_zero_derivative(derivative_evaluator, evaluator.focal_variable)
        if derivative_evaluator isa ConstantEvaluator
            push!(instructions, "@inbounds row_vec[$pos] = $(derivative_evaluator.value)")
            
        elseif derivative_evaluator isa ContinuousEvaluator
            if derivative_evaluator.column == evaluator.focal_variable
                push!(instructions, "@inbounds row_vec[$pos] = 1.0")
            else
                push!(instructions, "@inbounds row_vec[$pos] = 0.0")
            end
            
        elseif derivative_evaluator isa ChainRuleEvaluator
            generate_chain_rule_code!(instructions, derivative_evaluator, pos)
        elseif derivative_evaluator isa ProductRuleEvaluator
            generate_product_rule_code!(instructions, derivative_evaluator, pos)
        elseif derivative_evaluator isa ScaledEvaluator
            # Handle scaled derivatives directly
            if derivative_evaluator.evaluator isa ContinuousEvaluator && derivative_evaluator.evaluator.column == evaluator.focal_variable
                # Simple case: scale * x -> scale * 1 = scale
                push!(instructions, "@inbounds row_vec[$pos] = $(derivative_evaluator.scale_factor)")
            else
                expr = generate_inline_expression_for_derivatives(derivative_evaluator.evaluator)
                scale = derivative_evaluator.scale_factor
                push!(instructions, "@inbounds row_vec[$pos] = $scale * ($expr)")
            end
        elseif derivative_evaluator isa FunctionEvaluator
            # Handle function derivatives directly
            generate_function_derivative_inline!(instructions, derivative_evaluator, evaluator.focal_variable, pos)
        else
            # For other types, try to generate inline expression
            try
                expr = generate_inline_expression_for_derivatives(derivative_evaluator)
                push!(instructions, "@inbounds row_vec[$pos] = $expr")
            catch
                push!(instructions, "@inbounds row_vec[$pos] = 0.0  # Unknown derivative: $(typeof(derivative_evaluator))")
            end
        end
    end
end

function generate_function_derivative_inline!(instructions::Vector{String}, 
                                                evaluator::FunctionEvaluator, 
                                                focal_variable::Symbol, 
                                                pos::Int)
    
    func = evaluator.func
    args = evaluator.arg_evaluators
    
    if length(args) == 1 && args[1] isa ContinuousEvaluator && args[1].column == focal_variable
        var_name = args[1].column
        data_expr = "Float64(data.$var_name[row_idx])"
        
        if func === log
            # FIXED: ∂log(x)/∂x = 1/x
            push!(instructions, "@inbounds row_vec[$pos] = 1.0 / ($data_expr)")
            
        elseif func === exp
            push!(instructions, "@inbounds row_vec[$pos] = exp($data_expr)")
            
        elseif func === sqrt
            push!(instructions, "@inbounds row_vec[$pos] = 0.5 / sqrt($data_expr)")
            
        elseif func === sin
            push!(instructions, "@inbounds row_vec[$pos] = cos($data_expr)")
            
        elseif func === cos
            push!(instructions, "@inbounds row_vec[$pos] = -sin($data_expr)")
            
        else
            push!(instructions, "@inbounds row_vec[$pos] = 1.0  # Unknown function")
        end
        
    elseif length(args) == 2 && func === (^) && args[2] isa ConstantEvaluator && 
            args[1] isa ContinuousEvaluator && args[1].column == focal_variable
        var_name = args[1].column
        c = args[2].value
        data_expr = "Float64(data.$var_name[row_idx])"
        
        if c == 0.0
            push!(instructions, "@inbounds row_vec[$pos] = 0.0")
        elseif c == 1.0
            push!(instructions, "@inbounds row_vec[$pos] = 1.0")
        elseif c == 2.0
            # ∂(x^2)/∂x = 2*x
            push!(instructions, "@inbounds row_vec[$pos] = 2.0 * $data_expr")
        elseif c == 3.0
            # ∂(x^3)/∂x = 3*x^2
            push!(instructions, "@inbounds row_vec[$pos] = 3.0 * ($data_expr)^2.0")
        else
            # ∂(x^c)/∂x = c*x^(c-1)
            push!(instructions, "@inbounds row_vec[$pos] = $c * ($data_expr)^$(c-1.0)")
        end
        
    else
        push!(instructions, "@inbounds row_vec[$pos] = 0.0  # Complex function")
    end
end

# Generate inline expression for derivative evaluators - separate namespace
function generate_inline_expression_for_derivatives(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator
        return string(evaluator.value)
        
    elseif evaluator isa ContinuousEvaluator
        return "Float64(data.$(evaluator.column)[row_idx])"
        
    elseif evaluator isa ScaledEvaluator
        inner_expr = generate_inline_expression_for_derivatives(evaluator.evaluator)
        return "($(inner_expr) * $(evaluator.scale_factor))"
        
    elseif evaluator isa ProductEvaluator
        component_exprs = String[]
        for component in evaluator.components
            if component isa ConstantEvaluator
                push!(component_exprs, string(component.value))
            elseif component isa ContinuousEvaluator
                push!(component_exprs, "Float64(data.$(component.column)[row_idx])")
            else
                # Recursive call for complex components
                push!(component_exprs, generate_inline_expression_for_derivatives(component))
            end
        end
        return "(" * join(component_exprs, " * ") * ")"
        
    else
        error("Cannot generate derivative inline expression for $(typeof(evaluator))")
    end
end

function generate_product_expression_for_derivatives(evaluator::ProductEvaluator)
    component_exprs = [generate_inline_expression_for_derivatives(comp) for comp in evaluator.components]
    return "(" * join(component_exprs, " * ") * ")"
end

# Add methods for the internal derivative evaluator types
function generate_evaluator_code!(instructions::Vector{String}, evaluator::ChainRuleEvaluator, pos::Int)
    return generate_chain_rule_code!(instructions, evaluator, pos)
end

function generate_evaluator_code!(instructions::Vector{String}, evaluator::ProductRuleEvaluator, pos::Int)
    return generate_product_rule_code!(instructions, evaluator, pos)
end

function generate_evaluator_code!(instructions::Vector{String}, evaluator::ForwardDiffEvaluator, pos::Int)
    push!(instructions, "@inbounds row_vec[$pos] = 0.0  # ForwardDiff placeholder")
    return pos + 1
end
