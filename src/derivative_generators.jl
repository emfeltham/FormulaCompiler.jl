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
    # println("DEBUG: Generating code for PositionalDerivativeEvaluator")
    # println("  Target width: $(evaluator.target_width)")
    # println("  Focal variable: $(evaluator.focal_variable)")
    # println("  Original evaluator type: $(typeof(evaluator.original_evaluator))")
    
    # Initialize all positions to zero
    for i in 1:evaluator.target_width
        target_pos = pos + i - 1
        push!(instructions, "@inbounds row_vec[$target_pos] = 0.0")
    end
    
    # Generate code using the original sophisticated system
    if evaluator.original_evaluator isa CombinedEvaluator
        # println("DEBUG: Using combined positioning instructions")
        generate_combined_positioning_instructions!(instructions, evaluator, pos)
    else
        # println("DEBUG: Using single positioning instructions")
        generate_single_positioning_instructions!(instructions, evaluator, pos)
    end
    
    return pos + evaluator.target_width
end

function generate_combined_positioning_instructions!(instructions::Vector{String}, evaluator::PositionalDerivativeEvaluator, pos::Int)
    current_position = pos
    
    # println("DEBUG: Starting generate_combined_positioning_instructions!")
    # println("  Number of sub-evaluators: $(length(evaluator.original_evaluator.sub_evaluators))")
    # println("  Focal variable: $(evaluator.focal_variable)")
    
    for (i, sub_evaluator) in enumerate(evaluator.original_evaluator.sub_evaluators)
        sub_width = output_width(sub_evaluator)
        
        # println("DEBUG: Processing sub-evaluator $i at position $current_position")
        # println("  Sub-evaluator type: $(typeof(sub_evaluator))")
        # println("  Sub-evaluator width: $sub_width")
        
        # Compute derivative
        sub_derivative = compute_derivative_evaluator(sub_evaluator, evaluator.focal_variable)
        # println("  Computed derivative type: $(typeof(sub_derivative))")
        
        # Check if zero derivative
        is_zero = is_zero_derivative(sub_derivative, evaluator.focal_variable)
        # println("  Is zero derivative: $is_zero")
        
        if !is_zero
            # println("  Generating code for non-zero derivative...")
            
            if sub_derivative isa ConstantEvaluator && sub_derivative.value != 0.0
                # println("    Case: ConstantEvaluator (non-zero)")
                push!(instructions, "@inbounds row_vec[$current_position] = $(sub_derivative.value)")
                
            elseif sub_derivative isa ContinuousEvaluator && sub_derivative.column == evaluator.focal_variable
                # println("    Case: ContinuousEvaluator (focal variable)")
                push!(instructions, "@inbounds row_vec[$current_position] = 1.0")
                
            elseif sub_derivative isa ContinuousEvaluator
                # println("    Case: ContinuousEvaluator (other variable)")
                push!(instructions, "@inbounds row_vec[$current_position] = Float64(data.$(sub_derivative.column)[row_idx])")
                
            elseif sub_derivative isa ChainRuleEvaluator
                # println("    Case: ChainRuleEvaluator")
                generate_chain_rule_code!(instructions, sub_derivative, current_position)
                
            elseif sub_derivative isa ProductRuleEvaluator  
                # println("    Case: ProductRuleEvaluator")
                generate_product_rule_code!(instructions, sub_derivative, current_position)
                
            elseif sub_derivative isa ProductEvaluator
                # println("    Case: ProductEvaluator - THIS IS WHAT WE'RE LOOKING FOR!")
                # println("      ProductEvaluator components: $(length(sub_derivative.components))")
                # for (j, comp) in enumerate(sub_derivative.components)
                #     println("        Component $j: $(typeof(comp))")
                #     if comp isa ContinuousEvaluator
                #         println("          Column: $(comp.column)")
                #     end
                # end
                generate_product_evaluator_derivative_code!(instructions, sub_derivative, current_position)
                
            elseif sub_derivative isa ScaledEvaluator
                # println("    Case: ScaledEvaluator")
                inner_expr = generate_inline_expression_for_derivatives(sub_derivative.evaluator)
                scale = sub_derivative.scale_factor
                push!(instructions, "@inbounds row_vec[$current_position] = $scale * ($inner_expr)")
                
            else
                println("    Case: UNHANDLED TYPE - $(typeof(sub_derivative))")
                error("Unsupported derivative evaluator type: $(typeof(sub_derivative))")
            end
        else
            # println("  Skipping zero derivative")
        end
        
        current_position += sub_width
        # println("  New current_position: $current_position")
        # println()
    end
    
    # println("DEBUG: Finished generate_combined_positioning_instructions!")
end

# 2. Update generate_product_evaluator_derivative_code! to handle categorical properly
function generate_product_evaluator_derivative_code!(instructions::Vector{String}, evaluator::ProductEvaluator, pos::Int)
    # println("DEBUG: generate_product_evaluator_derivative_code! called")
    # println("  Position: $pos")
    # println("  Number of components: $(length(evaluator.components))")
    
    # Check if we have a categorical component - needs special handling
    has_categorical = any(comp -> comp isa CategoricalEvaluator, evaluator.components)
    
    if has_categorical
        # println("  Special case: ProductEvaluator with CategoricalEvaluator")
        generate_product_with_categorical!(instructions, evaluator, pos)
    else
        # Original logic for non-categorical cases
        component_exprs = String[]
        
        for (i, component) in enumerate(evaluator.components)
            # println("  Processing component $i: $(typeof(component))")
            
            if component isa ConstantEvaluator
                expr = string(component.value)
                # println("    Generated: $expr")
                push!(component_exprs, expr)
                
            elseif component isa ContinuousEvaluator
                expr = "Float64(data.$(component.column)[row_idx])"
                # println("    Generated: $expr")
                push!(component_exprs, expr)
                
            else
                error("fallback for $(typeof(component))")
            end
        end
        
        if length(component_exprs) == 1
            instruction = "@inbounds row_vec[$pos] = $(component_exprs[1])"
            # println("  Final instruction: $instruction")
            push!(instructions, instruction)
        else
            product_expr = join(component_exprs, " * ")
            instruction = "@inbounds row_vec[$pos] = $product_expr"
            # println("  Final instruction: $instruction")
            push!(instructions, instruction)
        end
    end
    
    # println("DEBUG: generate_product_evaluator_derivative_code! finished")
end

# 3. New function to handle ProductEvaluator with CategoricalEvaluator
# Updated generate_product_with_categorical! to handle ChainRuleEvaluator and other complex scalars
function generate_product_with_categorical!(instructions::Vector{String}, evaluator::ProductEvaluator, pos::Int)
    # println("DEBUG: generate_product_with_categorical! called at position $pos")
    
    # Find the categorical component and any scalar factors
    categorical_component = nothing
    scalar_components = AbstractEvaluator[]  # Changed: store evaluators, not just values
    
    for component in evaluator.components
        if component isa CategoricalEvaluator
            categorical_component = component
        else
            # All non-categorical components are scalar factors
            push!(scalar_components, component)
            # println("  Found scalar component: $(typeof(component))")
        end
    end
    
    if categorical_component === nothing
        error("No categorical component found in categorical product evaluator")
    end
    
    # Generate code for scalar factor (if any)
    scalar_var = nothing
    if !isempty(scalar_components)
        scalar_var = next_var("scalar")
        generate_scalar_factor_code!(instructions, scalar_components, scalar_var)
        # println("  Generated scalar factor variable: $scalar_var")
    else
        # println("  No scalar factors")
    end
    
    # Generate categorical contrast lookup code
    col = categorical_component.column
    n_levels = categorical_component.n_levels
    contrast_matrix = categorical_component.contrast_matrix
    width = size(contrast_matrix, 2)
    
    # println("  Categorical: $col with $n_levels levels, $width contrasts")
    
    # Generate lookup variables
    cat_var = next_var("cat")
    level_var = next_var("level")
    
    push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
    push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
    push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
    
    # Generate contrast assignments for each position
    for j in 1:width
        output_pos = pos + j - 1
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        
        if scalar_var === nothing
            # No scaling needed
            if n_levels == 1
                push!(instructions, "@inbounds row_vec[$output_pos] = $(values[1])")
            elseif n_levels == 2
                push!(instructions, "@inbounds row_vec[$output_pos] = $level_var == 1 ? $(values[1]) : $(values[2])")
            else
                # General ternary chain
                ternary_chain = "$level_var == 1 ? $(values[1])"
                for i in 2:(n_levels-1)
                    ternary_chain *= " : $level_var == $i ? $(values[i])"
                end
                ternary_chain *= " : $(values[n_levels])"
                push!(instructions, "@inbounds row_vec[$output_pos] = $ternary_chain")
            end
        else
            # Apply scalar factor
            if n_levels == 1
                push!(instructions, "@inbounds row_vec[$output_pos] = $scalar_var * $(values[1])")
            elseif n_levels == 2
                push!(instructions, "@inbounds row_vec[$output_pos] = $scalar_var * ($level_var == 1 ? $(values[1]) : $(values[2]))")
            else
                # General ternary chain with scaling
                ternary_chain = "$level_var == 1 ? $(values[1])"
                for i in 2:(n_levels-1)
                    ternary_chain *= " : $level_var == $i ? $(values[i])"
                end
                ternary_chain *= " : $(values[n_levels])"
                push!(instructions, "@inbounds row_vec[$output_pos] = $scalar_var * ($ternary_chain)")
            end
        end
    end
    
    # println("DEBUG: generate_product_with_categorical! finished")
end

function generate_scalar_factor_code!(instructions::Vector{String}, scalar_components::Vector{AbstractEvaluator}, scalar_var::String)
    # println("DEBUG: generate_scalar_factor_code! called")
    # println("  Number of scalar components: $(length(scalar_components))")
    
    if length(scalar_components) == 1
        # Single scalar component
        component = scalar_components[1]
        println("  Single scalar: $(typeof(component))")
        
        if component isa ConstantEvaluator
            push!(instructions, "@inbounds $scalar_var = $(component.value)")
            
        elseif component isa ContinuousEvaluator
            push!(instructions, "@inbounds $scalar_var = Float64(data.$(component.column)[row_idx])")
            
        elseif component isa ChainRuleEvaluator
            # This is the key case for ∂(log(z) * group)/∂z
            generate_chain_rule_scalar_code!(instructions, component, scalar_var)
            
        elseif component isa ScaledEvaluator
            generate_scaled_scalar_code!(instructions, component, scalar_var)
            
        else
            println("  Warning: Unsupported scalar component type $(typeof(component)), using 1.0")
            push!(instructions, "@inbounds $scalar_var = 1.0")
        end
        
    else
        # Multiple scalar components - multiply them
        temp_vars = String[]
        
        for (i, component) in enumerate(scalar_components)
            temp_var = next_var("scalar_$i")
            push!(temp_vars, temp_var)
            
            if component isa ConstantEvaluator
                push!(instructions, "@inbounds $temp_var = $(component.value)")
            elseif component isa ContinuousEvaluator
                push!(instructions, "@inbounds $temp_var = Float64(data.$(component.column)[row_idx])")
            elseif component isa ChainRuleEvaluator
                generate_chain_rule_scalar_code!(instructions, component, temp_var)
            else
                println("  Warning: Unsupported scalar component type $(typeof(component)), using 1.0")
                push!(instructions, "@inbounds $temp_var = 1.0")
            end
        end
        
        # Multiply all scalar factors
        product_expr = join(temp_vars, " * ")
        push!(instructions, "@inbounds $scalar_var = $product_expr")
    end
    
    # println("DEBUG: generate_scalar_factor_code! finished")
end

# Generate code for ChainRuleEvaluator as scalar factor
function generate_chain_rule_scalar_code!(instructions::Vector{String}, evaluator::ChainRuleEvaluator, scalar_var::String)
    # println("DEBUG: generate_chain_rule_scalar_code! for $(evaluator.original_func)")
    
    # Generate variables for inner function and its derivative
    inner_var = next_var("inner")
    inner_deriv_var = next_var("inner_deriv")
    
    # Generate code to evaluate inner function
    if evaluator.inner_evaluator isa ContinuousEvaluator
        push!(instructions, "@inbounds $inner_var = Float64(data.$(evaluator.inner_evaluator.column)[row_idx])")
    else
        # For complex inner evaluators, this would need more work
        error("Complex inner evaluator fallback")
    end
    
    # Generate code to evaluate inner derivative
    if evaluator.inner_derivative isa ConstantEvaluator
        push!(instructions, "@inbounds $inner_deriv_var = $(evaluator.inner_derivative.value)")
    else
        # For complex inner derivatives, this would need more work
        error("Complex inner derivative fallback")
    end
    
    # Apply the chain rule based on the original function
    original_func = evaluator.original_func
    
    if original_func === log
        # ∂log(u)/∂x = (1/u) * ∂u/∂x
        push!(instructions, "@inbounds $scalar_var = (1.0 / $inner_var) * $inner_deriv_var")
    elseif original_func === sqrt
        # ∂sqrt(u)/∂x = (0.5/sqrt(u)) * ∂u/∂x
        push!(instructions, "@inbounds $scalar_var = (0.5 / sqrt($inner_var)) * $inner_deriv_var")
    elseif original_func === exp
        # ∂exp(u)/∂x = exp(u) * ∂u/∂x
        push!(instructions, "@inbounds $scalar_var = exp($inner_var) * $inner_deriv_var")
    else
        println("  Warning: Unknown function $(original_func) in chain rule, using 1.0")
        push!(instructions, "@inbounds $scalar_var = 1.0")
    end
end

# Generate code for ScaledEvaluator as scalar factor
function generate_scaled_scalar_code!(instructions::Vector{String}, evaluator::ScaledEvaluator, scalar_var::String)
    inner_var = next_var("inner_scaled")
    
    if evaluator.evaluator isa ConstantEvaluator
        push!(instructions, "@inbounds $inner_var = $(evaluator.evaluator.value)")
    elseif evaluator.evaluator isa ContinuousEvaluator
        push!(instructions, "@inbounds $inner_var = Float64(data.$(evaluator.evaluator.column)[row_idx])")
    else
        error("Complex evaluator fallback")
    end
    
    push!(instructions, "@inbounds $scalar_var = $inner_var * $(evaluator.scale_factor)")
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

function generate_function_derivative_inline!(
    instructions::Vector{String}, 
    evaluator::FunctionEvaluator, 
    focal_variable::Symbol, 
    pos::Int
)
    
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

# 1. Update generate_inline_expression_for_derivatives to handle CategoricalEvaluator
function generate_inline_expression_for_derivatives(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator
        return string(evaluator.value)
        
    elseif evaluator isa ContinuousEvaluator
        return "Float64(data.$(evaluator.column)[row_idx])"
        
    elseif evaluator isa CategoricalEvaluator
        # For categorical in derivatives, we need to generate contrast lookup
        # But since this is inline, we can't generate the full lookup
        # We need to handle this case differently
        error("CategoricalEvaluator requires multi-line code generation, cannot be inlined")
        
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
                # For complex components like CategoricalEvaluator, we can't inline
                error("Complex component $(typeof(component)) cannot be inlined in ProductEvaluator")
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
