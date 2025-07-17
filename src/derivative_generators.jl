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
                # Use existing generate_single_component_code! from generators.jl
                derivative_var = next_var("deriv")
                generate_single_component_code!(instructions, sub_derivative, derivative_var)
                push!(instructions, "@inbounds row_vec[$current_position] = $derivative_var")
            end
        end
        
        current_position += sub_width
    end
end

function generate_single_positioning_instructions!(instructions::Vector{String}, evaluator::PositionalDerivativeEvaluator, pos::Int)
    derivative_evaluator = compute_derivative_evaluator(evaluator.original_evaluator, evaluator.focal_variable)
    
    if !is_zero_derivative(derivative_evaluator, evaluator.focal_variable)
        if derivative_evaluator isa ConstantEvaluator
            push!(instructions, "@inbounds row_vec[$pos] = $(derivative_evaluator.value)")
        elseif derivative_evaluator isa ContinuousEvaluator
            push!(instructions, "@inbounds row_vec[$pos] = Float64(data.$(derivative_evaluator.column)[row_idx])")
        else
            # Use existing generate_single_component_code! from generators.jl
            derivative_var = next_var("deriv")
            generate_single_component_code!(instructions, derivative_evaluator, derivative_var)
            push!(instructions, "@inbounds row_vec[$pos] = $derivative_var")
        end
    end
end
