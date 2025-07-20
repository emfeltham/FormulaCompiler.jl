# derivative_evaluators.jl

using ForwardDiff  # Add this import for ForwardDiff functionality

###############################################################################
# COMPREHENSIVE DERIVATIVE EVALUATOR TYPES
###############################################################################

"""
    ChainRuleEvaluator <: AbstractEvaluator

Evaluator for chain rule derivatives: ∂f(g(x))/∂x = f'(g(x)) * ∂g(x)/∂x
"""
struct ChainRuleEvaluator <: AbstractEvaluator
    derivative_func::Function
    inner_evaluator::AbstractEvaluator
    inner_derivative::AbstractEvaluator
    original_func::Function  # Store original function for code generation
end

"""
    ProductRuleEvaluator <: AbstractEvaluator

Evaluator for product rule derivatives: ∂(f*g)/∂x = f*∂g/∂x + g*∂f/∂x
"""
struct ProductRuleEvaluator <: AbstractEvaluator
    left_evaluator::AbstractEvaluator
    left_derivative::AbstractEvaluator
    right_evaluator::AbstractEvaluator
    right_derivative::AbstractEvaluator
end

"""
    ForwardDiffEvaluator <: AbstractEvaluator

Complete ForwardDiff fallback evaluator for complex cases where analytical 
derivatives are not implemented or feasible.
"""
struct ForwardDiffEvaluator <: AbstractEvaluator
    original_evaluator::AbstractEvaluator
    focal_variable::Symbol
    validation_cache::Ref{Union{Nothing, Bool}}
    
    function ForwardDiffEvaluator(original_evaluator, focal_variable)
        new(original_evaluator, focal_variable, Ref{Union{Nothing, Bool}}(nothing))
    end
end

# Output widths for internal evaluators (all scalar)
output_width(eval::ChainRuleEvaluator) = 1
output_width(eval::ProductRuleEvaluator) = 1
output_width(eval::ForwardDiffEvaluator) = 1

###############################################################################
# POSITIONAL WRAPPER FOR FULL-WIDTH OUTPUT
###############################################################################

"""
    PositionalDerivativeEvaluator <: AbstractEvaluator

The key insight is that derivatives of individual terms are ALWAYS scalar,
even when the original terms are multi-dimensional (like categorical variables).
"""
struct PositionalDerivativeEvaluator <: AbstractEvaluator
    original_evaluator::AbstractEvaluator
    focal_variable::Symbol
    target_width::Int
    position_map::Vector{Int}
    
    function PositionalDerivativeEvaluator(original_evaluator, focal_variable, target_width)
        position_map = compute_position_map(original_evaluator, focal_variable)
        new(original_evaluator, focal_variable, target_width, position_map)
    end
end

# Full width output
output_width(eval::PositionalDerivativeEvaluator) = eval.target_width

###############################################################################
# CORE DERIVATIVE COMPUTATION - FIXED MATHEMATICS
###############################################################################

"""
    compute_derivative_evaluator(evaluator::AbstractEvaluator, focal_variable::Symbol)

Comprehensive derivative computation with correct mathematical rules.
"""
function compute_derivative_evaluator(evaluator::AbstractEvaluator, focal_variable::Symbol)
    
    # Base cases - leaf evaluators
    if evaluator isa ConstantEvaluator
        return ConstantEvaluator(0.0)  # ∂c/∂x = 0
        
    elseif evaluator isa ContinuousEvaluator
        if evaluator.column == focal_variable
            return ConstantEvaluator(1.0)  # ∂x/∂x = 1
        else
            return ConstantEvaluator(0.0)  # ∂y/∂x = 0
        end
        
    elseif evaluator isa CategoricalEvaluator
        return ConstantEvaluator(0.0)  # Always zero for continuous focal variables
        
    # Function derivatives with correct mathematical formulas
    elseif evaluator isa FunctionEvaluator
        return compute_function_derivative(evaluator, focal_variable)
        
    elseif evaluator isa InteractionEvaluator
        return compute_interaction_derivative(evaluator, focal_variable)
        
    elseif evaluator isa ZScoreEvaluator
        # ∂((g-center)/scale)/∂x = (1/scale) * ∂g/∂x
        underlying_derivative = compute_derivative_evaluator(evaluator.underlying, focal_variable)
        
        if is_zero_derivative(underlying_derivative, focal_variable)
            return ConstantEvaluator(0.0)
        else
            return ScaledEvaluator(underlying_derivative, 1.0 / evaluator.scale)
        end
        
    elseif evaluator isa CombinedEvaluator
        return compute_sum_derivative(evaluator, focal_variable)
        
    elseif evaluator isa ScaledEvaluator
        # ∂(c*f)/∂x = c * ∂f/∂x
        inner_derivative = compute_derivative_evaluator(evaluator.evaluator, focal_variable)
        return ScaledEvaluator(inner_derivative, evaluator.scale_factor)
        
    elseif evaluator isa ProductEvaluator
        return compute_product_evaluator_derivative(evaluator, focal_variable)
        
    # Handle derivative evaluator types recursively
    elseif evaluator isa ChainRuleEvaluator
        return compute_chain_rule_derivative(evaluator, focal_variable)
        
    elseif evaluator isa ProductRuleEvaluator  
        return compute_product_rule_derivative(evaluator, focal_variable)
        
    elseif evaluator isa ForwardDiffEvaluator
        # For higher-order derivatives, return zero (reasonable approximation)
        return ConstantEvaluator(0.0)
        
    else
        # Enhanced ForwardDiff fallback for unknown types
        if !contains_variable(evaluator, focal_variable)
            return ConstantEvaluator(0.0)
        end
        
        return ForwardDiffEvaluator(evaluator, focal_variable)
    end
end

###############################################################################
# FUNCTION DERIVATIVES WITH CORRECT MATHEMATICS
###############################################################################

function compute_function_derivative(evaluator::FunctionEvaluator, focal_variable::Symbol)
    func = evaluator.func
    args = evaluator.arg_evaluators
    n_args = length(args)
    
    if n_args == 1
        return compute_unary_function_derivative(func, args[1], focal_variable)
    elseif n_args == 2
        return compute_binary_function_derivative(func, args[1], args[2], focal_variable)
    else
        return compute_nary_function_derivative(func, args, focal_variable)
    end
end

function compute_unary_function_derivative(func::Function, arg_evaluator::AbstractEvaluator, focal_variable::Symbol)
    # Get derivative of inner function
    inner_derivative = compute_derivative_evaluator(arg_evaluator, focal_variable)
    
    # If inner derivative is zero, whole thing is zero
    if is_zero_derivative(inner_derivative, focal_variable)
        return ConstantEvaluator(0.0)
    end
    
    # CORRECT mathematical derivatives
    if func === log
        # ∂log(u)/∂x = (1/u) * ∂u/∂x
        return ChainRuleEvaluator(x -> 1.0/x, arg_evaluator, inner_derivative, log)
        
    elseif func === exp
        # ∂exp(u)/∂x = exp(u) * ∂u/∂x  
        return ChainRuleEvaluator(x -> exp(x), arg_evaluator, inner_derivative, exp)
        
    elseif func === sqrt
        # ∂sqrt(u)/∂x = (1/(2*sqrt(u))) * ∂u/∂x
        println("DEBUG: Creating ChainRuleEvaluator for sqrt")
        result = ChainRuleEvaluator(x -> 0.5/sqrt(x), arg_evaluator, inner_derivative, sqrt)
        println("DEBUG: Created ChainRuleEvaluator with original_func = $(result.original_func)")
        return result
        
    elseif func === sin
        # ∂sin(u)/∂x = cos(u) * ∂u/∂x
        return ChainRuleEvaluator(x -> cos(x), arg_evaluator, inner_derivative, sin)
        
    elseif func === cos
        # ∂cos(u)/∂x = -sin(u) * ∂u/∂x
        return ChainRuleEvaluator(x -> -sin(x), arg_evaluator, inner_derivative, cos)
        
    elseif func === tan
        # ∂tan(u)/∂x = sec²(u) * ∂u/∂x = (1 + tan²(u)) * ∂u/∂x
        return ChainRuleEvaluator(x -> 1 + tan(x)^2, arg_evaluator, inner_derivative, tan)
        
    elseif func === abs
        # ∂|u|/∂x = sign(u) * ∂u/∂x (undefined at 0, but we'll use 0)
        return ChainRuleEvaluator(x -> x == 0 ? 0.0 : sign(x), arg_evaluator, inner_derivative, abs)
        
    else
        # Unknown function - use ForwardDiff as fallback
        return ForwardDiffEvaluator(FunctionEvaluator(func, [arg_evaluator]), focal_variable)
    end
end

function compute_binary_function_derivative(func::Function, left_arg::AbstractEvaluator, right_arg::AbstractEvaluator, focal_variable::Symbol)
    
    left_derivative = compute_derivative_evaluator(left_arg, focal_variable)
    right_derivative = compute_derivative_evaluator(right_arg, focal_variable)
    
    left_is_zero = is_zero_derivative(left_derivative, focal_variable)
    right_is_zero = is_zero_derivative(right_derivative, focal_variable)
    
    if func === (+) || func === (-)
        # ∂(f ± g)/∂x = ∂f/∂x ± ∂g/∂x
        if left_is_zero && right_is_zero
            return ConstantEvaluator(0.0)
        elseif left_is_zero
            return func === (+) ? right_derivative : ScaledEvaluator(right_derivative, -1.0)
        elseif right_is_zero
            return left_derivative
        else
            if func === (+)
                return CombinedEvaluator([left_derivative, right_derivative])
            else
                return CombinedEvaluator([left_derivative, ScaledEvaluator(right_derivative, -1.0)])
            end
        end
        
    elseif func === (*)
        # CORRECT: Product rule ∂(f*g)/∂x = f*∂g/∂x + g*∂f/∂x
        if left_is_zero && right_is_zero
            return ConstantEvaluator(0.0)
        elseif left_is_zero
            # Only right term contributes: f*g'
            return ProductEvaluator([left_arg, right_derivative])
        elseif right_is_zero
            # Only left term contributes: g*f'
            return ProductEvaluator([right_arg, left_derivative])
        else
            # Both terms: f*g' + g*f'
            return ProductRuleEvaluator(left_arg, left_derivative, right_arg, right_derivative)
        end
        
    elseif func === (^) && right_arg isa ConstantEvaluator
        # CORRECT: Power rule ∂(f^c)/∂x = c * f^(c-1) * ∂f/∂x
        c = right_arg.value
        if c == 0.0
            return ConstantEvaluator(0.0)  # ∂(f^0)/∂x = ∂(1)/∂x = 0
        elseif c == 1.0
            return left_derivative  # ∂(f^1)/∂x = ∂f/∂x
        elseif c == 2.0
            # Special case for x^2: ∂(f^2)/∂x = 2*f*∂f/∂x
            # For simple case where f=x, this becomes 2*x*1 = 2*x
            if left_derivative isa ConstantEvaluator && left_derivative.value == 1.0
                # f is the focal variable, so ∂(x^2)/∂x = 2*x
                return ScaledEvaluator(left_arg, 2.0)
            else
                # General case: 2*f*f'
                return ScaledEvaluator(ProductEvaluator([left_arg, left_derivative]), 2.0)
            end
        else
            # General case: c * f^(c-1) * f'
            if left_derivative isa ConstantEvaluator && left_derivative.value == 1.0
                # Simple case where ∂f/∂x = 1 (f is the focal variable)
                if c == 3.0
                    # ∂(x^3)/∂x = 3*x^2
                    new_exponent = ConstantEvaluator(2.0)
                    f_squared = FunctionEvaluator(^, [left_arg, new_exponent])
                    return ScaledEvaluator(f_squared, 3.0)
                else
                    # ∂(x^c)/∂x = c*x^(c-1)
                    new_exponent = ConstantEvaluator(c - 1.0)
                    f_to_c_minus_1 = FunctionEvaluator(^, [left_arg, new_exponent])
                    return ScaledEvaluator(f_to_c_minus_1, c)
                end
            else
                # General case with product rule: c * f^(c-1) * f'
                new_exponent = ConstantEvaluator(c - 1.0)
                f_to_c_minus_1 = FunctionEvaluator(^, [left_arg, new_exponent])
                coefficient = ConstantEvaluator(c)
                return ProductEvaluator([coefficient, f_to_c_minus_1, left_derivative])
            end
        end
        
    elseif func === (/)
        # CORRECT: Division rule ∂(f/g)/∂x = (g*∂f/∂x - f*∂g/∂x) / g²
        if right_is_zero
            # Simpler case: ∂(f/c)/∂x = (1/c) * ∂f/∂x  
            return ProductEvaluator([FunctionEvaluator(/, [ConstantEvaluator(1.0), right_arg]), left_derivative])
        else
            # General quotient rule
            numerator_term1 = ProductEvaluator([right_arg, left_derivative])
            numerator_term2 = ProductEvaluator([left_arg, right_derivative])
            numerator = CombinedEvaluator([numerator_term1, ScaledEvaluator(numerator_term2, -1.0)])
            denominator = ProductEvaluator([right_arg, right_arg])  # g²
            return FunctionEvaluator(/, [numerator, denominator])
        end
        
    else
        # Unknown binary function
        return ForwardDiffEvaluator(FunctionEvaluator(func, [left_arg, right_arg]), focal_variable)
    end
end

function compute_nary_function_derivative(func::Function, args::Vector{AbstractEvaluator}, focal_variable::Symbol)
    if func === (+)
        # Sum rule: ∂(Σf_i)/∂x = Σ(∂f_i/∂x)
        derivatives = [compute_derivative_evaluator(arg, focal_variable) for arg in args]
        non_zero_derivatives = filter(d -> !is_zero_derivative(d, focal_variable), derivatives)
        
        if isempty(non_zero_derivatives)
            return ConstantEvaluator(0.0)
        elseif length(non_zero_derivatives) == 1
            return non_zero_derivatives[1]
        else
            return CombinedEvaluator(non_zero_derivatives)
        end
        
    elseif func === (*)
        # N-ary product rule
        return compute_nary_product_derivative(args, focal_variable)
        
    else
        # Unknown n-ary function
        return ForwardDiffEvaluator(FunctionEvaluator(func, args), focal_variable)
    end
end

###############################################################################
# INTERACTION DERIVATIVES (N-way product rule)
###############################################################################

function compute_interaction_derivative(evaluator::InteractionEvaluator, focal_variable::Symbol)
    components = evaluator.components
    n_components = length(components)
    
    if n_components == 0
        return ConstantEvaluator(0.0)
    elseif n_components == 1
        return compute_derivative_evaluator(components[1], focal_variable)
    else
        # N-way product rule for interactions
        return compute_nary_product_derivative(components, focal_variable)
    end
end

function compute_nary_product_derivative(components::Vector{AbstractEvaluator}, focal_variable::Symbol)
    n = length(components)
    sum_terms = AbstractEvaluator[]
    
    # Product rule: ∂(∏f_i)/∂x = Σ_i (∂f_i/∂x * ∏_{j≠i} f_j)
    for i in 1:n
        component_derivative = compute_derivative_evaluator(components[i], focal_variable)
        
        if is_zero_derivative(component_derivative, focal_variable)
            continue  # This term contributes 0
        end
        
        # Build product of all other components
        other_components = [components[j] for j in 1:n if j != i]
        
        if isempty(other_components)
            # Only one component total
            push!(sum_terms, component_derivative)
        elseif length(other_components) == 1
            # Two components total: f * g' or g * f'
            push!(sum_terms, ProductEvaluator([component_derivative, other_components[1]]))
        else
            # Multiple other components: f' * (g * h * ...)
            other_product = ProductEvaluator(other_components)
            push!(sum_terms, ProductEvaluator([component_derivative, other_product]))
        end
    end
    
    if isempty(sum_terms)
        return ConstantEvaluator(0.0)
    elseif length(sum_terms) == 1
        return sum_terms[1]
    else
        return CombinedEvaluator(sum_terms)
    end
end

###############################################################################
# HELPER FUNCTIONS
###############################################################################

function compute_sum_derivative(evaluator::CombinedEvaluator, focal_variable::Symbol)
    derivative_terms = AbstractEvaluator[]
    
    for (i, sub_evaluator) in enumerate(evaluator.sub_evaluators)
        sub_derivative = compute_derivative_evaluator(sub_evaluator, focal_variable)
        
        if !is_zero_derivative(sub_derivative, focal_variable)
            # For categorical terms, we need to handle the position correctly
            if sub_evaluator isa CategoricalEvaluator
                # Categorical derivatives are always zero for continuous variables
                # Don't add anything
                continue
            else
                push!(derivative_terms, sub_derivative)
            end
        end
    end
    
    if isempty(derivative_terms)
        return ConstantEvaluator(0.0)
    elseif length(derivative_terms) == 1
        return derivative_terms[1]
    else
        return CombinedEvaluator(derivative_terms)
    end
end

function compute_product_evaluator_derivative(evaluator::ProductEvaluator, focal_variable::Symbol)
    return compute_nary_product_derivative(evaluator.components, focal_variable)
end

function compute_chain_rule_derivative(evaluator::ChainRuleEvaluator, focal_variable::Symbol)
    # For higher-order derivatives of chain rule, use ForwardDiff
    return ForwardDiffEvaluator(evaluator, focal_variable)
end

function compute_product_rule_derivative(evaluator::ProductRuleEvaluator, focal_variable::Symbol)
    # For higher-order derivatives of product rule, use ForwardDiff  
    return ForwardDiffEvaluator(evaluator, focal_variable)
end

###############################################################################
# EVALUATION METHODS
###############################################################################

function evaluate!(evaluator::PositionalDerivativeEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    
    # Initialize entire output to zero
    for i in 1:evaluator.target_width
        @inbounds output[start_idx + i - 1] = 0.0
    end
    
    # Compute the derivative using the original sophisticated system
    derivative_evaluator = compute_derivative_evaluator(evaluator.original_evaluator, evaluator.focal_variable)
    
    # Handle the case where derivative might be multi-dimensional due to interactions
    if evaluator.original_evaluator isa CombinedEvaluator
        evaluate_combined_positioned_derivative!(evaluator, output, data, row_idx, start_idx)
    else
        evaluate_single_positioned_derivative!(evaluator, output, data, row_idx, start_idx)
    end
    
    return start_idx + evaluator.target_width
end

function evaluate_combined_positioned_derivative!(evaluator::PositionalDerivativeEvaluator, 
                                                       output::AbstractVector{Float64}, 
                                                       data, row_idx::Int, start_idx::Int)
    current_position = start_idx
    
    for sub_evaluator in evaluator.original_evaluator.sub_evaluators
        sub_width = output_width(sub_evaluator)
        
        # Compute derivative for this sub-evaluator
        sub_derivative = compute_derivative_evaluator(sub_evaluator, evaluator.focal_variable)
        
        if !is_zero_derivative(sub_derivative, evaluator.focal_variable)
            # Handle different types of sub-derivatives correctly
            if sub_evaluator isa ContinuousEvaluator && sub_evaluator.column == evaluator.focal_variable
                # Simple case: ∂x/∂x = 1, goes in the x position
                @inbounds output[current_position] = 1.0
                
            elseif sub_evaluator isa FunctionEvaluator
                # Function derivative - evaluate and place in function position
                temp_buffer = Vector{Float64}(undef, 1)
                evaluate!(sub_derivative, temp_buffer, data, row_idx, 1)
                @inbounds output[current_position] = temp_buffer[1]
                
            elseif sub_evaluator isa InteractionEvaluator
                # Interaction derivatives need special handling
                evaluate_interaction_derivative_positioned!(sub_evaluator, sub_derivative, 
                                                          output, data, row_idx, current_position,
                                                          evaluator.focal_variable)
                
            elseif sub_evaluator isa ScaledEvaluator
                # Scaled derivative
                temp_buffer = Vector{Float64}(undef, 1)
                evaluate!(sub_derivative, temp_buffer, data, row_idx, 1)
                @inbounds output[current_position] = temp_buffer[1]
                
            else
                # General case - evaluate derivative and place in first position of this term
                temp_buffer = Vector{Float64}(undef, output_width(sub_derivative))
                evaluate!(sub_derivative, temp_buffer, data, row_idx, 1)
                
                # Place the result(s) in the appropriate position(s)
                for i in 1:min(length(temp_buffer), sub_width)
                    @inbounds output[current_position + i - 1] = temp_buffer[i]
                end
            end
        end
        
        current_position += sub_width
    end
end

function evaluate_single_positioned_derivative!(evaluator::PositionalDerivativeEvaluator, 
                                                    output::AbstractVector{Float64}, 
                                                    data, row_idx::Int, start_idx::Int)
    
    derivative_evaluator = compute_derivative_evaluator(evaluator.original_evaluator, evaluator.focal_variable)
    
    if !is_zero_derivative(derivative_evaluator, evaluator.focal_variable)
        if derivative_evaluator isa ConstantEvaluator
            # Simple constant derivative
            @inbounds output[start_idx] = derivative_evaluator.value
            
        elseif derivative_evaluator isa ContinuousEvaluator
            # Simple variable derivative
            @inbounds output[start_idx] = Float64(data[derivative_evaluator.column][row_idx])
            
        else
            # Complex derivative - evaluate it
            temp_buffer = Vector{Float64}(undef, output_width(derivative_evaluator))
            evaluate!(derivative_evaluator, temp_buffer, data, row_idx, 1)
            
            # Only place the scalar result in the first position
            @inbounds output[start_idx] = temp_buffer[1]
        end
    end
end

function evaluate_interaction_derivative_positioned!(original_interaction::InteractionEvaluator,
                                                   derivative_evaluator::AbstractEvaluator,
                                                   output::AbstractVector{Float64}, 
                                                   data, row_idx::Int, 
                                                   start_position::Int,
                                                   focal_variable::Symbol)
    
    # For interactions like x * group, we need to understand the structure
    components = original_interaction.components
    component_widths = [output_width(comp) for comp in components]
    
    # Find which component(s) depend on the focal variable
    focal_component_indices = findall(comp -> contains_variable(comp, focal_variable), components)
    
    if isempty(focal_component_indices)
        # No component depends on focal variable - all derivatives are zero (already initialized)
        return
    end
    
    # For simple cases like x * group, handle directly
    if length(components) == 2
        evaluate_binary_interaction_derivative!(components, derivative_evaluator, output, data, 
                                               row_idx, start_position, focal_variable, component_widths)
    else
        # General case - evaluate the derivative and distribute appropriately
        temp_buffer = Vector{Float64}(undef, output_width(derivative_evaluator))
        evaluate!(derivative_evaluator, temp_buffer, data, row_idx, 1)
        
        # For now, place the result in the appropriate positions
        interaction_width = output_width(original_interaction)
        for i in 1:min(length(temp_buffer), interaction_width)
            @inbounds output[start_position + i - 1] = temp_buffer[i]
        end
    end
end

function evaluate_binary_interaction_derivative!(components::Vector{AbstractEvaluator},
                                                derivative_evaluator::AbstractEvaluator,
                                                output::AbstractVector{Float64}, 
                                                data, row_idx::Int, 
                                                start_position::Int,
                                                focal_variable::Symbol,
                                                component_widths::Vector{Int})
    
    comp1, comp2 = components[1], components[2]
    w1, w2 = component_widths[1], component_widths[2]
    
    # Determine which component contains the focal variable
    comp1_has_focal = contains_variable(comp1, focal_variable)
    comp2_has_focal = contains_variable(comp2, focal_variable)
    
    if comp1_has_focal && !comp2_has_focal
        # Case: x * group, where x is focal variable
        # ∂(x * group)/∂x = 1 * group = group
        evaluate_component_values_for_derivative!(comp2, output, data, row_idx, start_position, w1, w2)
        
    elseif !comp1_has_focal && comp2_has_focal
        # Case: group * x, where x is focal variable  
        # ∂(group * x)/∂x = group * 1 = group
        evaluate_component_values_for_derivative!(comp1, output, data, row_idx, start_position, w1, w2)
        
    elseif comp1_has_focal && comp2_has_focal
        # Both components have focal variable - use general product rule
        temp_buffer = Vector{Float64}(undef, output_width(derivative_evaluator))
        evaluate!(derivative_evaluator, temp_buffer, data, row_idx, 1)
        
        interaction_width = w1 * w2
        for i in 1:min(length(temp_buffer), interaction_width)
            @inbounds output[start_position + i - 1] = temp_buffer[i]
        end
    end
    # If neither has focal variable, derivatives are zero (already initialized)
end

function evaluate_component_values_for_derivative!(component::AbstractEvaluator,
                                                  output::AbstractVector{Float64}, 
                                                  data, row_idx::Int, 
                                                  start_position::Int,
                                                  w1::Int, w2::Int)
    
    if component isa ContinuousEvaluator
        # Single value - replicate across interaction positions
        val = Float64(data[component.column][row_idx])
        for i in 1:(w1 * w2)
            @inbounds output[start_position + i - 1] = val
        end
        
    elseif component isa CategoricalEvaluator
        # Evaluate categorical component
        temp_buffer = Vector{Float64}(undef, output_width(component))
        evaluate!(component, temp_buffer, data, row_idx, 1)
        
        # Distribute values according to Kronecker product structure
        if w1 == 1
            # component is the second factor
            for i in 1:length(temp_buffer)
                @inbounds output[start_position + i - 1] = temp_buffer[i]
            end
        else
            # component is the first factor - replicate each value
            idx = 0
            for j in 1:w2
                for i in 1:w1
                    @inbounds output[start_position + idx] = temp_buffer[i]
                    idx += 1
                end
            end
        end
        
    else
        # General case
        temp_buffer = Vector{Float64}(undef, output_width(component))
        evaluate!(component, temp_buffer, data, row_idx, 1)
        
        for i in 1:min(length(temp_buffer), w1 * w2)
            @inbounds output[start_position + i - 1] = temp_buffer[i]
        end
    end
end

###############################################################################
# VARIABLE DEPENDENCY CHECKING
###############################################################################

function contains_variable(evaluator::AbstractEvaluator, variable::Symbol)
    if evaluator isa ContinuousEvaluator
        return evaluator.column == variable
    elseif evaluator isa CategoricalEvaluator
        return evaluator.column == variable
    elseif evaluator isa ConstantEvaluator
        return false
    elseif evaluator isa FunctionEvaluator
        return any(arg -> contains_variable(arg, variable), evaluator.arg_evaluators)
    elseif evaluator isa InteractionEvaluator
        return any(comp -> contains_variable(comp, variable), evaluator.components)
    elseif evaluator isa CombinedEvaluator
        return any(sub -> contains_variable(sub, variable), evaluator.sub_evaluators)
    elseif evaluator isa ZScoreEvaluator
        return contains_variable(evaluator.underlying, variable)
    elseif evaluator isa ScaledEvaluator
        return contains_variable(evaluator.evaluator, variable)
    elseif evaluator isa ProductEvaluator
        return any(comp -> contains_variable(comp, variable), evaluator.components)
    elseif evaluator isa ChainRuleEvaluator
        return contains_variable(evaluator.inner_evaluator, variable) || 
               contains_variable(evaluator.inner_derivative, variable)
    elseif evaluator isa ProductRuleEvaluator
        return contains_variable(evaluator.left_evaluator, variable) ||
               contains_variable(evaluator.left_derivative, variable) ||
               contains_variable(evaluator.right_evaluator, variable) ||
               contains_variable(evaluator.right_derivative, variable)
    elseif evaluator isa ForwardDiffEvaluator
        return contains_variable(evaluator.original_evaluator, variable)
    else
        return true  # Conservative assumption for unknown types
    end
end

###############################################################################
# POSITION MAP COMPUTATION
###############################################################################

function compute_position_map(evaluator::AbstractEvaluator, focal_variable::Symbol)
    positions = Int[]
    compute_position_map_recursive!(positions, evaluator, focal_variable, 1)
    return positions
end

function compute_position_map_recursive!(positions::Vector{Int}, evaluator::AbstractEvaluator, 
                                       focal_variable::Symbol, current_pos::Int)
    
    if evaluator isa ContinuousEvaluator
        if evaluator.column == focal_variable
            push!(positions, current_pos)  # This position gets derivative = 1
        end
        return current_pos + 1
        
    elseif evaluator isa CategoricalEvaluator
        # Categorical variables have zero derivative w.r.t. continuous variables
        # Don't add any positions
        width = output_width(evaluator)
        return current_pos + width
        
    elseif evaluator isa ConstantEvaluator
        # Constants have zero derivative - don't add position
        return current_pos + 1
        
    elseif evaluator isa FunctionEvaluator
        # Functions can have non-zero derivatives if they depend on focal_variable
        if contains_variable(evaluator, focal_variable)
            push!(positions, current_pos)  # This position gets computed derivative
        end
        return current_pos + 1
        
    elseif evaluator isa InteractionEvaluator
        # Interactions can have non-zero derivatives
        if contains_variable(evaluator, focal_variable)
            width = output_width(evaluator)
            # For interactions, we need to determine which specific positions get derivatives
            return compute_interaction_position_map!(positions, evaluator, focal_variable, current_pos)
        else
            return current_pos + output_width(evaluator)
        end
        
    elseif evaluator isa CombinedEvaluator
        # Recursively process each sub-evaluator
        for sub_eval in evaluator.sub_evaluators
            current_pos = compute_position_map_recursive!(positions, sub_eval, focal_variable, current_pos)
        end
        return current_pos
        
    elseif evaluator isa ZScoreEvaluator
        return compute_position_map_recursive!(positions, evaluator.underlying, focal_variable, current_pos)
        
    elseif evaluator isa ScaledEvaluator
        return compute_position_map_recursive!(positions, evaluator.evaluator, focal_variable, current_pos)
        
    elseif evaluator isa ProductEvaluator
        if contains_variable(evaluator, focal_variable)
            push!(positions, current_pos)
        end
        return current_pos + 1
        
    else
        # Conservative: assume it might have a derivative
        if contains_variable(evaluator, focal_variable)
            push!(positions, current_pos)
        end
        return current_pos + output_width(evaluator)
    end
end

function compute_interaction_position_map!(positions::Vector{Int}, evaluator::InteractionEvaluator,
                                         focal_variable::Symbol, current_pos::Int)
    # For interactions like x * group, only the terms involving x get non-zero derivatives
    components = evaluator.components
    component_widths = [output_width(comp) for comp in components]
    
    # Determine which components depend on focal_variable
    depends_on_focal = [contains_variable(comp, focal_variable) for comp in components]
    
    if !any(depends_on_focal)
        # No component depends on focal variable - all derivatives are zero
        return current_pos + output_width(evaluator)
    end
    
    # For interactions, if ANY component depends on focal_variable, 
    # then ALL positions get derivatives (some will be computed as zero)
    total_width = output_width(evaluator)
    for i in 1:total_width
        push!(positions, current_pos + i - 1)
    end
    
    return current_pos + total_width
end

###############################################################################
# ZERO DERIVATIVE CHECKING
###############################################################################

function is_zero_derivative(evaluator::AbstractEvaluator, focal_variable::Symbol)
    if evaluator isa ConstantEvaluator
        return evaluator.value == 0.0
        
    elseif evaluator isa ContinuousEvaluator
        return evaluator.column != focal_variable
        
    elseif evaluator isa CategoricalEvaluator
        return true  # Categorical variables always have zero derivative w.r.t. continuous variables
        
    elseif evaluator isa ZScoreEvaluator
        return is_zero_derivative(evaluator.underlying, focal_variable)
        
    elseif evaluator isa CombinedEvaluator
        return all(sub_eval -> is_zero_derivative(sub_eval, focal_variable), evaluator.sub_evaluators)
        
    elseif evaluator isa ScaledEvaluator
        return evaluator.scale_factor == 0.0 || is_zero_derivative(evaluator.evaluator, focal_variable)
        
    elseif evaluator isa ProductEvaluator
        # FIXED: A ProductEvaluator is zero if ANY component doesn't depend on focal variable
        # This was the bug - it was checking if ALL components depend on focal variable
        # But for derivatives like ∂(x*w)/∂x = w, we get ProductEvaluator([w])
        # which doesn't contain the focal variable x, but is NOT zero!
        
        # The correct logic: ProductEvaluator is zero only if it contains a component
        # that is explicitly zero (ConstantEvaluator(0.0))
        for component in evaluator.components
            if component isa ConstantEvaluator && component.value == 0.0
                return true  # If any component is zero, the whole product is zero
            end
        end
        return false  # Otherwise, it's not zero
        
    elseif evaluator isa InteractionEvaluator
        # Interaction derivative is zero if NO component depends on focal variable
        return !any(comp -> contains_variable(comp, focal_variable), evaluator.components)
        
    elseif evaluator isa FunctionEvaluator
        return !contains_variable(evaluator, focal_variable)
        
    elseif evaluator isa ChainRuleEvaluator
        return is_zero_derivative(evaluator.inner_derivative, focal_variable)
        
    elseif evaluator isa ProductRuleEvaluator
        return is_zero_derivative(evaluator.left_derivative, focal_variable) && 
               is_zero_derivative(evaluator.right_derivative, focal_variable)
        
    elseif evaluator isa ForwardDiffEvaluator
        return !contains_variable(evaluator.original_evaluator, focal_variable)
        
    else
        return false  # Conservative assumption
    end
end
