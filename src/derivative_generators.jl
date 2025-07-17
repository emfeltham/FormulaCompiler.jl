# derivative_generators.jl
# @generated workflow

###############################################################################
# CODE GENERATION FOR @GENERATED DERIVATIVES
###############################################################################

function generate_evaluator_code!(instructions::Vector{String}, evaluator::PositionalDerivativeEvaluator, pos::Int)
    # Initialize all positions to zero
    for i in 1:evaluator.target_width
        target_pos = pos + i - 1
        push!(instructions, "@inbounds row_vec[$target_pos] = 0.0")
    end
    
    # Generate code using the original sophisticated system
    if evaluator.original_evaluator isa CombinedEvaluator
        generate_combined_positioning_instructions!(instructions, evaluator, pos)
    else
        generate_single_positioning_instructions!(instructions, evaluator, pos)
    end
    
    return pos + evaluator.target_width
end

function generate_combined_positioning_instructions!(instructions::Vector{String}, evaluator::PositionalDerivativeEvaluator, pos::Int)
    current_position = pos
    
    for sub_evaluator in evaluator.original_evaluator.sub_evaluators
        sub_width = output_width(sub_evaluator)
        sub_derivative = compute_derivative_evaluator(sub_evaluator, evaluator.focal_variable)
        
        if !is_zero_derivative(sub_derivative, evaluator.focal_variable)
            if sub_derivative isa ConstantEvaluator && sub_derivative.value != 0.0
                push!(instructions, "@inbounds row_vec[$current_position] = $(sub_derivative.value)")
            elseif sub_derivative isa ContinuousEvaluator
                push!(instructions, "@inbounds row_vec[$current_position] = Float64(data.$(sub_derivative.column)[row_idx])")
            else
                # FIXED: Use proper code generation
                next_pos = generate_evaluator_code!(instructions, sub_derivative, current_position)
                
                # Verify derivative is scalar (should only write to current_position)
                if next_pos != current_position + 1
                    @warn "Sub-derivative $(typeof(sub_derivative)) produced non-scalar output (width=$(next_pos-current_position))"
                end
            end
        end
        
        current_position += sub_width
    end
end

"""
    generate_chain_rule_code!(instructions, evaluator::ChainRuleEvaluator, pos)

Generate code for ChainRuleEvaluator derivatives.
"""
function generate_chain_rule_code!(instructions::Vector{String}, evaluator::ChainRuleEvaluator, pos::Int)
    # Generate variables for inner function value and derivative
    inner_var = next_var("inner")
    inner_deriv_var = next_var("inner_deriv")
    derivative_func_var = next_var("deriv_func")
    
    # Generate code to evaluate inner function
    generate_single_component_code!(instructions, evaluator.inner_evaluator, inner_var)
    
    # Generate code to evaluate inner derivative
    generate_single_component_code!(instructions, evaluator.inner_derivative, inner_deriv_var)
    
    # Apply derivative function (this is tricky - function is stored as a closure)
    # For now, we'll use a runtime evaluation
    push!(instructions, "@inbounds $derivative_func_var = $(evaluator.derivative_func)($inner_var)")
    push!(instructions, "@inbounds row_vec[$pos] = $derivative_func_var * $inner_deriv_var")
    
    return pos + 1
end

"""
    generate_product_rule_code!(instructions, evaluator::ProductRuleEvaluator, pos)

Generate code for ProductRuleEvaluator derivatives.
"""
function generate_product_rule_code!(instructions::Vector{String}, evaluator::ProductRuleEvaluator, pos::Int)
    # Generate variables for all components
    f_var = next_var("f")
    f_prime_var = next_var("f_prime")
    g_var = next_var("g")
    g_prime_var = next_var("g_prime")
    
    # Generate code for each component
    generate_single_component_code!(instructions, evaluator.left_evaluator, f_var)
    generate_single_component_code!(instructions, evaluator.left_derivative, f_prime_var)
    generate_single_component_code!(instructions, evaluator.right_evaluator, g_var)
    generate_single_component_code!(instructions, evaluator.right_derivative, g_prime_var)
    
    # Apply product rule: f*g' + g*f'
    push!(instructions, "@inbounds row_vec[$pos] = $f_var * $g_prime_var + $g_var * $f_prime_var")
    
    return pos + 1
end

"""
Fix for generate_single_positioning_instructions! in derivative_generators.jl

The issue is that generate_single_component_code! from generators.jl writes 
directly to row_vec, but we need the result in a temporary variable.

Solution: Use generate_evaluator_code! instead, which is the proper way to
generate code for any evaluator into row_vec positions.
"""

function generate_single_positioning_instructions!(instructions::Vector{String}, evaluator::PositionalDerivativeEvaluator, pos::Int)
    derivative_evaluator = compute_derivative_evaluator(evaluator.original_evaluator, evaluator.focal_variable)
    
    if !is_zero_derivative(derivative_evaluator, evaluator.focal_variable)
        if derivative_evaluator isa ConstantEvaluator
            # Simple case - just write the constant
            push!(instructions, "@inbounds row_vec[$pos] = $(derivative_evaluator.value)")
            
        elseif derivative_evaluator isa ContinuousEvaluator
            # Simple case - just read the data column
            push!(instructions, "@inbounds row_vec[$pos] = Float64(data.$(derivative_evaluator.column)[row_idx])")
            
        else
            # Complex case - use proper code generation
            # This generates code that writes to row_vec starting at pos
            next_pos = generate_evaluator_code!(instructions, derivative_evaluator, pos)
            
            # Verify we only wrote to one position (derivatives should be scalar)
            if next_pos != pos + 1
                @warn "Derivative evaluator $(typeof(derivative_evaluator)) produced non-scalar output (width=$(next_pos-pos))"
            end
        end
    end
end
