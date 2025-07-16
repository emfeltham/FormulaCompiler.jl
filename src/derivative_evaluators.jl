# derivative_evaluators.jl - Add to EfficientModelMatrices/src/evaluators.jl

###############################################################################
# NEW DERIVATIVE EVALUATOR TYPES - Add to existing AbstractEvaluator hierarchy
###############################################################################

"""
    ChainRuleEvaluator <: AbstractEvaluator

Evaluator for chain rule derivatives: ∂f(g(x))/∂x = f'(g(x)) * ∂g(x)/∂x

# Fields
- `derivative_func::Function`: The derivative function f' (e.g., x -> 1/x for log)
- `inner_evaluator::AbstractEvaluator`: The inner function g(x)
- `inner_derivative::AbstractEvaluator`: The derivative ∂g(x)/∂x

# Examples
- ∂log(x)/∂x → ChainRuleEvaluator(x -> 1/x, ContinuousEvaluator(:x), ConstantEvaluator(1.0))
- ∂exp(x²)/∂x → ChainRuleEvaluator(x -> exp(x), FunctionEvaluator(^, [x, 2]), ProductRuleEvaluator(...))
"""
struct ChainRuleEvaluator <: AbstractEvaluator
    derivative_func::Function
    inner_evaluator::AbstractEvaluator
    inner_derivative::AbstractEvaluator
end

"""
    ProductRuleEvaluator <: AbstractEvaluator

Evaluator for product rule derivatives: ∂(f*g)/∂x = f*∂g/∂x + g*∂f/∂x

# Fields
- `left_evaluator::AbstractEvaluator`: The function f
- `left_derivative::AbstractEvaluator`: The derivative ∂f/∂x
- `right_evaluator::AbstractEvaluator`: The function g  
- `right_derivative::AbstractEvaluator`: The derivative ∂g/∂x

# Examples
- ∂(x*y)/∂x → ProductRuleEvaluator(x, 1, y, 0)
- ∂(x*log(z))/∂x → ProductRuleEvaluator(x, 1, log(z), 0)
"""
struct ProductRuleEvaluator <: AbstractEvaluator
    left_evaluator::AbstractEvaluator
    left_derivative::AbstractEvaluator
    right_evaluator::AbstractEvaluator
    right_derivative::AbstractEvaluator
end

"""
    ForwardDiffEvaluator <: AbstractEvaluator

Fallback evaluator that uses ForwardDiff for complex cases that don't have 
analytical derivative implementations.

# Fields
- `original_evaluator::AbstractEvaluator`: The original evaluator to differentiate
- `focal_variable::Symbol`: Variable to differentiate with respect to
- `base_data_sample::NamedTuple`: Sample data for ForwardDiff setup

# Usage
This is a fallback for cases where analytical derivatives are not implemented,
such as custom functions or very complex nested expressions.
"""
struct ForwardDiffEvaluator <: AbstractEvaluator
    original_evaluator::AbstractEvaluator
    focal_variable::Symbol
    base_data_sample::NamedTuple
end

###############################################################################
# OUTPUT WIDTH CALCULATIONS - Add to existing output_width function
###############################################################################

"""
    output_width(eval::ChainRuleEvaluator)

Chain rule evaluators always produce single values (scalar derivatives).
"""
output_width(eval::ChainRuleEvaluator) = 1

"""
    output_width(eval::ProductRuleEvaluator)

Product rule evaluators always produce single values (scalar derivatives).
"""
output_width(eval::ProductRuleEvaluator) = 1

"""
    output_width(eval::ForwardDiffEvaluator)

ForwardDiff evaluators have the same width as their original evaluator.
"""
output_width(eval::ForwardDiffEvaluator) = output_width(eval.original_evaluator)

###############################################################################
# EVALUATION FUNCTIONS - Add to existing evaluate! function
###############################################################################

"""
    evaluate!(evaluator::ChainRuleEvaluator, output, data, row_idx, start_idx)

Evaluate chain rule derivative: f'(g(x)) * g'(x)
"""
function evaluate!(evaluator::ChainRuleEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    
    # Evaluate inner function: g(x)
    inner_value = 0.0
    temp_buffer = Vector{Float64}(undef, 1)
    evaluate!(evaluator.inner_evaluator, temp_buffer, data, row_idx, 1)
    inner_value = temp_buffer[1]
    
    # Evaluate inner derivative: g'(x)
    inner_deriv_value = 0.0
    evaluate!(evaluator.inner_derivative, temp_buffer, data, row_idx, 1)
    inner_deriv_value = temp_buffer[1]
    
    # Apply chain rule: f'(g(x)) * g'(x)
    derivative_at_inner = evaluator.derivative_func(inner_value)
    @inbounds output[start_idx] = derivative_at_inner * inner_deriv_value
    
    return start_idx + 1
end

"""
    evaluate!(evaluator::ProductRuleEvaluator, output, data, row_idx, start_idx)

Evaluate product rule derivative: f*g' + g*f'
"""
function evaluate!(evaluator::ProductRuleEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    
    # Use a small temporary buffer for scalar evaluations
    temp_buffer = Vector{Float64}(undef, 1)
    
    # Evaluate f
    evaluate!(evaluator.left_evaluator, temp_buffer, data, row_idx, 1)
    f_value = temp_buffer[1]
    
    # Evaluate f'
    evaluate!(evaluator.left_derivative, temp_buffer, data, row_idx, 1)
    f_prime_value = temp_buffer[1]
    
    # Evaluate g
    evaluate!(evaluator.right_evaluator, temp_buffer, data, row_idx, 1)
    g_value = temp_buffer[1]
    
    # Evaluate g'
    evaluate!(evaluator.right_derivative, temp_buffer, data, row_idx, 1)
    g_prime_value = temp_buffer[1]
    
    # Apply product rule: f*g' + g*f'
    @inbounds output[start_idx] = f_value * g_prime_value + g_value * f_prime_value
    
    return start_idx + 1
end

"""
    evaluate!(evaluator::ForwardDiffEvaluator, output, data, row_idx, start_idx)

Evaluate derivative using ForwardDiff as fallback for complex cases.
"""
function evaluate!(evaluator::ForwardDiffEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    
    # Get current value of focal variable
    current_value = Float64(data[evaluator.focal_variable][row_idx])
    
    # Define function to differentiate
    function eval_function(var_value::Float64)
        # Create modified data with perturbed variable
        modified_data = merge(data, (evaluator.focal_variable => var_value,))
        
        # Evaluate original evaluator with modified data
        temp_output = Vector{Float64}(undef, output_width(evaluator.original_evaluator))
        evaluate!(evaluator.original_evaluator, temp_output, modified_data, row_idx, 1)
        
        # Return first element (assuming scalar for derivative context)
        return temp_output[1]
    end
    
    # Compute derivative using ForwardDiff
    derivative_value = ForwardDiff.derivative(eval_function, current_value)
    @inbounds output[start_idx] = derivative_value
    
    return start_idx + 1
end
