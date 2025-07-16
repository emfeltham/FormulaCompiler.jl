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

###############################################################################
# UTILITY FUNCTIONS FOR DERIVATIVE COMPUTATION
###############################################################################

"""
    get_standard_derivative_function(func::Function) -> Union{Function, Nothing}

Return the analytical derivative function for standard mathematical functions.
Returns `nothing` if no analytical derivative is available.

# Supported Functions
- `log` → `x -> 1/x`
- `exp` → `x -> exp(x)` 
- `sqrt` → `x -> 1/(2*sqrt(x))`
- `sin` → `x -> cos(x)`
- `cos` → `x -> -sin(x)`
- `abs` → `x -> sign(x)` (except at x=0)
"""
function get_standard_derivative_function(func::Function)
    if func === log
        return x -> 1/x
    elseif func === exp
        return x -> exp(x)
    elseif func === sqrt
        return x -> 1/(2*sqrt(x))
    elseif func === sin
        return x -> cos(x)
    elseif func === cos
        return x -> -sin(x)
    elseif func === tan
        return x -> 1 + tan(x)^2  # sec²(x) = 1 + tan²(x)
    elseif func === atan
        return x -> 1/(1 + x^2)
    elseif func === sinh
        return x -> cosh(x)
    elseif func === cosh
        return x -> sinh(x)
    elseif func === tanh
        return x -> 1 - tanh(x)^2  # sech²(x) = 1 - tanh²(x)
    elseif func === abs
        # Note: derivative undefined at x=0, but we'll use sign function
        return x -> x == 0 ? 0.0 : sign(x)
    else
        return nothing
    end
end

"""
    is_zero_derivative(evaluator::AbstractEvaluator, focal_variable::Symbol) -> Bool

Check if an evaluator's derivative with respect to focal_variable is always zero.
Used for optimization - if derivative is zero, we can use ConstantEvaluator(0.0).

# Examples
- ContinuousEvaluator(:x) w.r.t. :y → true (∂x/∂y = 0)
- ConstantEvaluator(5.0) w.r.t. anything → true (∂c/∂x = 0)
- CategoricalEvaluator(...) w.r.t. continuous var → true
"""
function is_zero_derivative(evaluator::AbstractEvaluator, focal_variable::Symbol)
    if evaluator isa ConstantEvaluator
        return true
    elseif evaluator isa ContinuousEvaluator
        return evaluator.column != focal_variable
    elseif evaluator isa CategoricalEvaluator
        return true  # Categorical variables have zero derivative w.r.t. continuous variables
    elseif evaluator isa ZScoreEvaluator
        return is_zero_derivative(evaluator.underlying, focal_variable)
    elseif evaluator isa CombinedEvaluator
        return all(sub_eval -> is_zero_derivative(sub_eval, focal_variable), evaluator.sub_evaluators)
    else
        return false  # Conservative: assume non-zero for complex cases
    end
end

###############################################################################
# VALIDATION AND TESTING UTILITIES
###############################################################################

"""
    validate_derivative_evaluator(evaluator::AbstractEvaluator, focal_variable::Symbol, 
                                 test_data::NamedTuple, tolerance::Float64 = 1e-8) -> Bool

Validate that a derivative evaluator produces correct results by comparing with 
numerical differentiation using finite differences.

# Arguments
- `evaluator`: The derivative evaluator to test
- `focal_variable`: Variable being differentiated with respect to
- `test_data`: Sample data for testing
- `tolerance`: Tolerance for numerical comparison

# Returns
`true` if analytical and numerical derivatives match within tolerance.
"""
function validate_derivative_evaluator(evaluator::AbstractEvaluator, 
                                     focal_variable::Symbol,
                                     test_data::NamedTuple, 
                                     tolerance::Float64 = 1e-8)
    
    # Test on first few observations
    test_indices = 1:min(5, length(first(test_data)))
    
    for row_idx in test_indices
        try
            # Compute analytical derivative
            analytical_result = Vector{Float64}(undef, output_width(evaluator))
            evaluate!(evaluator, analytical_result, test_data, row_idx, 1)
            analytical_value = analytical_result[1]
            
            # Compute numerical derivative using finite differences
            ε = sqrt(eps(Float64))
            current_value = Float64(test_data[focal_variable][row_idx])
            
            # Evaluate at x + ε
            data_plus = merge(test_data, (focal_variable => current_value + ε,))
            result_plus = Vector{Float64}(undef, 1)
            evaluate!(evaluator, result_plus, data_plus, row_idx, 1)
            
            # Evaluate at x - ε  
            data_minus = merge(test_data, (focal_variable => current_value - ε,))
            result_minus = Vector{Float64}(undef, 1)
            evaluate!(evaluator, result_minus, data_minus, row_idx, 1)
            
            # Numerical derivative
            numerical_value = (result_plus[1] - result_minus[1]) / (2ε)
            
            # Check if they match within tolerance
            if abs(analytical_value - numerical_value) > tolerance
                @warn "Derivative validation failed at row $row_idx: analytical=$analytical_value, numerical=$numerical_value"
                return false
            end
            
        catch e
            @warn "Derivative validation error at row $row_idx: $e"
            return false
        end
    end
    
    return true
end
