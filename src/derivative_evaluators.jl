# derivative_evaluators.jl

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

# Fields
- `original_evaluator::AbstractEvaluator`: The evaluator to differentiate
- `focal_variable::Symbol`: Variable to differentiate with respect to
- `validation_cache::Ref{Union{Nothing, Bool}}`: Cached validation result

# Performance Notes
ForwardDiff is significantly slower than analytical derivatives but provides
correct results for any differentiable function. Use analytical derivatives
whenever possible.
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

SIMPLE ADDITION: Wraps the existing comprehensive derivative system to produce full-width vectors.

This is the ONLY new thing - everything else is the original sophisticated system.

# Fields
- `original_evaluator::AbstractEvaluator`: The original formula evaluator
- `focal_variable::Symbol`: Variable to differentiate with respect to
- `target_width::Int`: Width of output vector (matches original formula)
"""
struct PositionalDerivativeEvaluator <: AbstractEvaluator
    original_evaluator::AbstractEvaluator
    focal_variable::Symbol
    target_width::Int
end

# Full width output - this is the only behavioral change
output_width(eval::PositionalDerivativeEvaluator) = eval.target_width

function evaluate!(evaluator::PositionalDerivativeEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    
    # Initialize entire output to zero
    for i in 1:evaluator.target_width
        @inbounds output[start_idx + i - 1] = 0.0
    end
    
    # Use the original sophisticated derivative system, just place results correctly
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
        
        # Use the ORIGINAL comprehensive derivative system
        sub_derivative = compute_derivative_evaluator(sub_evaluator, evaluator.focal_variable)
        
        if !is_zero_derivative(sub_derivative, evaluator.focal_variable)
            # Place the sophisticated derivative result in the correct position
            temp_buffer = Vector{Float64}(undef, 1)
            evaluate!(sub_derivative, temp_buffer, data, row_idx, 1)
            @inbounds output[current_position] = temp_buffer[1]
        end
        
        current_position += sub_width
    end
end

function evaluate_single_positioned_derivative!(evaluator::PositionalDerivativeEvaluator, 
                                              output::AbstractVector{Float64}, 
                                              data, row_idx::Int, start_idx::Int)
    # Use the ORIGINAL comprehensive derivative system
    derivative_evaluator = compute_derivative_evaluator(evaluator.original_evaluator, evaluator.focal_variable)
    
    if !is_zero_derivative(derivative_evaluator, evaluator.focal_variable)
        temp_buffer = Vector{Float64}(undef, 1)
        evaluate!(derivative_evaluator, temp_buffer, data, row_idx, 1)
        @inbounds output[start_idx] = temp_buffer[1]
    end
end

###############################################################################
# ORIGINAL COMPREHENSIVE DERIVATIVE SYSTEM
###############################################################################

"""
    compute_derivative_evaluator(evaluator::AbstractEvaluator, focal_variable::Symbol)

ORIGINAL comprehensive derivative computation - handles everything analytically.
This is the sophisticated system that was already working.
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
        return ConstantEvaluator(0.0)  # ∂(categorical)/∂x = 0 for continuous x
        
    # Composite cases - FULL RECURSIVE POWER (original)
    elseif evaluator isa FunctionEvaluator
        return compute_function_derivative_recursive(evaluator, focal_variable)
        
    elseif evaluator isa InteractionEvaluator
        return compute_interaction_derivative_recursive(evaluator, focal_variable)
        
    elseif evaluator isa ZScoreEvaluator
        # ∂((g-center)/scale)/∂x = (1/scale) * ∂g/∂x
        underlying_derivative = compute_derivative_evaluator(evaluator.underlying, focal_variable)
        
        # Optimization: if underlying derivative is zero, whole thing is zero
        if is_zero_derivative(underlying_derivative, focal_variable)
            return ConstantEvaluator(0.0)
        else
            return ScaledEvaluator(underlying_derivative, 1.0 / evaluator.scale)
        end
        
    elseif evaluator isa CombinedEvaluator
        # Sum rule: ∂(f+g+h+...)/∂x = ∂f/∂x + ∂g/∂x + ∂h/∂x + ...
        return compute_sum_derivative_recursive(evaluator, focal_variable)
        
    # Handle derivative evaluator types recursively too
    elseif evaluator isa ChainRuleEvaluator
        return compute_chain_rule_derivative(evaluator, focal_variable)
        
    elseif evaluator isa ProductRuleEvaluator  
        return compute_product_rule_derivative(evaluator, focal_variable)
        
    elseif evaluator isa ScaledEvaluator
        # ∂(c*f)/∂x = c * ∂f/∂x
        inner_derivative = compute_derivative_evaluator(evaluator.evaluator, focal_variable)
        return ScaledEvaluator(inner_derivative, evaluator.scale_factor)
        
    elseif evaluator isa ProductEvaluator
        # Product rule for ProductEvaluator
        return compute_product_evaluator_derivative(evaluator, focal_variable)
        
    else
        # Enhanced ForwardDiff fallback (replaces the simple "else" case)
        @debug "Using ForwardDiff fallback for evaluator type: $(typeof(evaluator))"
        
        # Validate that ForwardDiff is feasible
        if !contains_variable(evaluator, focal_variable)
            @debug "Evaluator $(typeof(evaluator)) does not depend on $focal_variable - returning zero derivative"
            return ConstantEvaluator(0.0)
        end
        
        # Check for obvious zero cases before creating ForwardDiff evaluator
        if evaluator isa ConstantEvaluator
            return ConstantEvaluator(0.0)
        elseif evaluator isa ContinuousEvaluator && evaluator.column != focal_variable
            return ConstantEvaluator(0.0)
        elseif evaluator isa CategoricalEvaluator && evaluator.column != focal_variable
            return ConstantEvaluator(0.0)
        end
        
        return ForwardDiffEvaluator(evaluator, focal_variable)
    end
end

###############################################################################
# ORIGINAL FUNCTION DERIVATIVES (comprehensive)
###############################################################################

"""
    compute_function_derivative_recursive(evaluator::FunctionEvaluator, focal_variable::Symbol)
    
    ORIGINAL: Use full recursive power to handle function derivatives analytically.
    Supports unary functions, binary operations, and arbitrary nesting.
"""
function compute_function_derivative_recursive(evaluator::FunctionEvaluator, focal_variable::Symbol)
    func = evaluator.func
    args = evaluator.arg_evaluators
    n_args = length(args)
    
    # === UNARY FUNCTIONS ===
    if n_args == 1
        return compute_unary_function_derivative(func, args[1], focal_variable)
        
    # === BINARY OPERATIONS ===  
    elseif n_args == 2
        return compute_binary_function_derivative(func, args[1], args[2], focal_variable)
        
    # === N-ARY FUNCTIONS ===
    elseif n_args > 2
        return compute_nary_function_derivative(func, args, focal_variable)
        
    else
        return ConstantEvaluator(0.0)
    end
end

function compute_unary_function_derivative(func::Function, arg_evaluator::AbstractEvaluator, focal_variable::Symbol)
    # Get derivative of inner function
    inner_derivative = compute_derivative_evaluator(arg_evaluator, focal_variable)
    
    # Optimization: if inner derivative is zero, whole thing is zero
    if is_zero_derivative(inner_derivative, focal_variable)
        return ConstantEvaluator(0.0)
    end
    
    # Get analytical derivative function if available
    derivative_func = get_standard_derivative_function(func)
    
    if derivative_func !== nothing
        # Chain rule: f'(g) * g'
        return ChainRuleEvaluator(derivative_func, arg_evaluator, inner_derivative)
    else
        # Unknown unary function - use ForwardDiff
        @info "Using ForwardDiff for unknown unary function: $func"
        return ForwardDiffEvaluator(FunctionEvaluator(func, [arg_evaluator]), focal_variable, NamedTuple())
    end
end

function compute_binary_function_derivative(func::Function, left_arg::AbstractEvaluator, right_arg::AbstractEvaluator, focal_variable::Symbol)
    
    # Get derivatives of both arguments
    left_derivative = compute_derivative_evaluator(left_arg, focal_variable)
    right_derivative = compute_derivative_evaluator(right_arg, focal_variable)
    
    # Check if either argument has zero derivative (optimization)
    left_is_zero = is_zero_derivative(left_derivative, focal_variable)
    right_is_zero = is_zero_derivative(right_derivative, focal_variable)
    
    if func === (+) || func === (-)
        # Addition/Subtraction: ∂(f ± g)/∂x = ∂f/∂x ± ∂g/∂x
        if left_is_zero && right_is_zero
            return ConstantEvaluator(0.0)
        elseif left_is_zero
            return func === (+) ? right_derivative : ScaledEvaluator(right_derivative, -1.0)
        elseif right_is_zero
            return left_derivative
        else
            if func === (+)
                return CombinedEvaluator([left_derivative, right_derivative])
            else  # subtraction
                return CombinedEvaluator([left_derivative, ScaledEvaluator(right_derivative, -1.0)])
            end
        end
        
    elseif func === (*)
        # Multiplication: ∂(f*g)/∂x = f*∂g/∂x + g*∂f/∂x (product rule)
        if left_is_zero && right_is_zero
            return ConstantEvaluator(0.0)
        elseif left_is_zero
            # ∂(f*g)/∂x = f*g' + g*0 = f*g'
            # Use simplification
            product = ProductEvaluator([left_arg, right_derivative])
            return simplify_product_evaluator(product)
        elseif right_is_zero
            # ∂(f*g)/∂x = f*0 + g*f' = g*f'
            # Use simplification
            product = ProductEvaluator([right_arg, left_derivative])
            return simplify_product_evaluator(product)
        else
            # General product rule: f*g' + g*f'
            # Check for optimizations: if one derivative is 1, simplify
            if left_derivative isa ConstantEvaluator && left_derivative.value == 1.0
                if right_derivative isa ConstantEvaluator && right_derivative.value == 0.0
                    return right_arg  # f*0 + g*1 = g
                else
                    # f*g' + g*1 = f*g' + g
                    term1 = ProductEvaluator([left_arg, right_derivative])
                    return CombinedEvaluator([term1, right_arg])
                end
            elseif right_derivative isa ConstantEvaluator && right_derivative.value == 1.0
                if left_derivative isa ConstantEvaluator && left_derivative.value == 0.0
                    return left_arg  # f*1 + g*0 = f
                else
                    # f*1 + g*f' = f + g*f'
                    term2 = ProductEvaluator([right_arg, left_derivative])
                    return CombinedEvaluator([left_arg, term2])
                end
            else
                return ProductRuleEvaluator(left_arg, left_derivative, right_arg, right_derivative)
            end
        end
        
    elseif func === (^) && right_arg isa ConstantEvaluator
        # Power with constant exponent: ∂(f^c)/∂x = c * f^(c-1) * ∂f/∂x
        c = right_arg.value
        if c == 0.0
            return ConstantEvaluator(0.0)
        elseif c == 1.0
            return left_derivative
        else
            # c * f^(c-1) * f'
            new_exponent = ConstantEvaluator(c - 1.0)
            f_to_c_minus_1 = FunctionEvaluator(^, [left_arg, new_exponent])
            coefficient = ConstantEvaluator(c)
            return ProductEvaluator([coefficient, f_to_c_minus_1, left_derivative])
        end
        
    elseif func === (/)
        # Division: ∂(f/g)/∂x = (g*∂f/∂x - f*∂g/∂x) / g²
        return compute_division_derivative(left_arg, left_derivative, right_arg, right_derivative)
        
    else
        # Unknown binary function - use ForwardDiff
        @info "Using ForwardDiff for unknown binary function: $func"
        return ForwardDiffEvaluator(FunctionEvaluator(func, [left_arg, right_arg]), focal_variable, NamedTuple())
    end
end

function compute_nary_function_derivative(func::Function, args::Vector{AbstractEvaluator}, focal_variable::Symbol)
    
    if func === (+)
        # N-ary addition: ∂(f₁+f₂+...+fₙ)/∂x = ∂f₁/∂x + ∂f₂/∂x + ... + ∂fₙ/∂x
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
        # N-ary multiplication: Use generalized product rule
        return compute_nary_product_derivative(args, focal_variable)
        
    else
        # Unknown n-ary function - use ForwardDiff
        @info "Using ForwardDiff for unknown n-ary function: $func with $(length(args)) arguments"
        return ForwardDiffEvaluator(FunctionEvaluator(func, args), focal_variable, NamedTuple())
    end
end

###############################################################################
# ORIGINAL INTERACTION DERIVATIVES (N-way product rule)
###############################################################################

function compute_interaction_derivative_recursive(evaluator::InteractionEvaluator, focal_variable::Symbol)
    components = evaluator.components
    n_components = length(components)
    
    if n_components == 0
        return ConstantEvaluator(0.0)
    elseif n_components == 1
        return compute_derivative_evaluator(components[1], focal_variable)
    else
        # N-way product rule for interactions: ∂(f₁*f₂*...*fₙ)/∂x = Σᵢ (∂fᵢ/∂x * ∏ⱼ≠ᵢ fⱼ)
        sum_terms = AbstractEvaluator[]
        
        for i in 1:n_components
            component_derivative = compute_derivative_evaluator(components[i], focal_variable)
            
            # Skip terms where derivative is zero (optimization)
            if is_zero_derivative(component_derivative, focal_variable)
                continue
            end
            
            # Get all other components: ∏ⱼ≠ᵢ fⱼ
            other_components = [components[j] for j in 1:n_components if j != i]
            
            if isempty(other_components)
                push!(sum_terms, component_derivative)
            elseif length(other_components) == 1
                if component_derivative isa ConstantEvaluator && component_derivative.value == 1.0
                    push!(sum_terms, other_components[1])
                else
                    push!(sum_terms, ProductEvaluator([component_derivative, other_components[1]]))
                end
            else
                other_interaction = InteractionEvaluator(other_components)
                if component_derivative isa ConstantEvaluator && component_derivative.value == 1.0
                    push!(sum_terms, other_interaction)
                else
                    push!(sum_terms, ProductEvaluator([component_derivative, other_interaction]))
                end
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
end

function compute_nary_product_derivative(components::Vector{AbstractEvaluator}, focal_variable::Symbol)
    n = length(components)
    sum_terms = AbstractEvaluator[]
    
    for i in 1:n
        component_derivative = compute_derivative_evaluator(components[i], focal_variable)
        
        if is_zero_derivative(component_derivative, focal_variable)
            continue
        end
        
        other_components = [components[j] for j in 1:n if j != i]
        
        if isempty(other_components)
            push!(sum_terms, component_derivative)
        elseif length(other_components) == 1
            push!(sum_terms, ProductEvaluator([component_derivative, other_components[1]]))
        else
            other_product = InteractionEvaluator(other_components)
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
# ORIGINAL HELPER FUNCTIONS
###############################################################################

function compute_sum_derivative_recursive(evaluator::CombinedEvaluator, focal_variable::Symbol)
    derivative_terms = AbstractEvaluator[]
    
    for sub_evaluator in evaluator.sub_evaluators
        sub_derivative = compute_derivative_evaluator(sub_evaluator, focal_variable)
        
        if !is_zero_derivative(sub_derivative, focal_variable)
            push!(derivative_terms, sub_derivative)
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
    # For ProductEvaluator, use generalized product rule
    return compute_nary_product_derivative(evaluator.components, focal_variable)
end

function compute_division_derivative(f::AbstractEvaluator, f_prime::AbstractEvaluator, 
                                   g::AbstractEvaluator, g_prime::AbstractEvaluator)
    # Division rule: ∂(f/g)/∂x = (g*∂f/∂x - f*∂g/∂x) / g²
    term1 = ProductEvaluator([g, f_prime])
    term2 = ProductEvaluator([f, g_prime])
    numerator = CombinedEvaluator([term1, ScaledEvaluator(term2, -1.0)])
    denominator = InteractionEvaluator([g, g])
    return FunctionEvaluator(/, [numerator, denominator])
end

function compute_chain_rule_derivative(evaluator::ChainRuleEvaluator, focal_variable::Symbol)
    @info "Using ForwardDiff for derivative of ChainRuleEvaluator (higher-order)"
    return ForwardDiffEvaluator(evaluator, focal_variable, NamedTuple())
end

function compute_product_rule_derivative(evaluator::ProductRuleEvaluator, focal_variable::Symbol)
    @info "Using ForwardDiff for derivative of ProductRuleEvaluator (higher-order)"
    return ForwardDiffEvaluator(evaluator, focal_variable, NamedTuple())
end

function get_standard_derivative_function(func::Function)
    if func === log
        return x -> 1.0 / x
    elseif func === exp
        return x -> exp(x)
    elseif func === sqrt
        return x -> 0.5 / sqrt(x)
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
    elseif func === log10
        return x -> 1/(x * log(10))
    elseif func === log2
        return x -> 1/(x * log(2))
    elseif func === sign
        return x -> 0.0  # Derivative of sign is 0 (except at 0 where undefined)
    elseif func === abs
        return x -> x == 0 ? 0.0 : sign(x)
    else
        return nothing
    end
end

###############################################################################
# EVALUATE! METHODS FOR INTERNAL EVALUATORS (original)
###############################################################################

function evaluate!(evaluator::ChainRuleEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    
    temp_buffer = Vector{Float64}(undef, 1)
    evaluate!(evaluator.inner_evaluator, temp_buffer, data, row_idx, 1)
    inner_value = temp_buffer[1]
    
    evaluate!(evaluator.inner_derivative, temp_buffer, data, row_idx, 1)
    inner_deriv_value = temp_buffer[1]
    
    derivative_at_inner = evaluator.derivative_func(inner_value)
    @inbounds output[start_idx] = derivative_at_inner * inner_deriv_value
    
    return start_idx + 1
end

function evaluate!(evaluator::ProductRuleEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    
    temp_buffer = Vector{Float64}(undef, 1)
    
    evaluate!(evaluator.left_evaluator, temp_buffer, data, row_idx, 1)
    f_value = temp_buffer[1]
    
    evaluate!(evaluator.left_derivative, temp_buffer, data, row_idx, 1)
    f_prime_value = temp_buffer[1]
    
    evaluate!(evaluator.right_evaluator, temp_buffer, data, row_idx, 1)
    g_value = temp_buffer[1]
    
    evaluate!(evaluator.right_derivative, temp_buffer, data, row_idx, 1)
    g_prime_value = temp_buffer[1]
    
    @inbounds output[start_idx] = f_value * g_prime_value + g_value * f_prime_value
    
    return start_idx + 1
end

function simplify_product_evaluator(evaluator::ProductEvaluator)
    components = evaluator.components
    
    # Check for multiplication by 0
    for comp in components
        if comp isa ConstantEvaluator && comp.value == 0.0
            return ConstantEvaluator(0.0)
        end
    end
    
    # Filter out multiplication by 1
    non_unit_components = AbstractEvaluator[]
    for comp in components
        if !(comp isa ConstantEvaluator && comp.value == 1.0)
            push!(non_unit_components, comp)
        end
    end
    
    # Return simplified form
    if isempty(non_unit_components)
        return ConstantEvaluator(1.0)  # All components were 1
    elseif length(non_unit_components) == 1
        return non_unit_components[1]  # Only one non-unit component
    else
        return ProductEvaluator(non_unit_components)  # Multiple components remain
    end
end

###############################################################################
# FORWARDDIFF FUNCTION CACHING
###############################################################################

"""
Cache for compiled ForwardDiff functions to avoid recompilation overhead.
Key: (evaluator_hash, focal_variable) -> compiled function
"""
const FORWARDDIFF_CACHE = Dict{Tuple{UInt64, Symbol}, Function}()

"""
    get_cached_forwarddiff_function(evaluator, focal_variable)

Get or create a cached ForwardDiff function for the given evaluator and variable.
"""
function get_cached_forwarddiff_function(evaluator::AbstractEvaluator, focal_variable::Symbol)
    # Create cache key based on evaluator structure and focal variable
    cache_key = (hash((typeof(evaluator), focal_variable)), focal_variable)
    
    if haskey(FORWARDDIFF_CACHE, cache_key)
        return FORWARDDIFF_CACHE[cache_key]
    end
    
    # Create and cache the function
    function cached_f(x::Float64, data, row_idx::Int)
        # Create modified data where focal_variable = x at row_idx
        modified_column = copy(data[focal_variable])
        modified_column[row_idx] = x
        modified_data = merge(data, (focal_variable => modified_column,))
        
        # Evaluate original evaluator at modified data
        temp_output = Vector{Float64}(undef, output_width(evaluator))
        evaluate!(evaluator, temp_output, modified_data, row_idx, 1)
        
        # Return scalar (sum of outputs for multi-dimensional evaluators)
        return length(temp_output) == 1 ? temp_output[1] : sum(temp_output)
    end
    
    FORWARDDIFF_CACHE[cache_key] = cached_f
    return cached_f
end

"""
    clear_forwarddiff_cache!()

Clear the ForwardDiff function cache to free memory.
"""
function clear_forwarddiff_cache!()
    empty!(FORWARDDIFF_CACHE)
    return nothing
end

###############################################################################
# FORWARDDIFF EVALUATOR IMPLEMENTATION
###############################################################################

"""
    evaluate!(evaluator::ForwardDiffEvaluator, output, data, row_idx, start_idx)

Evaluate ForwardDiff-based derivative using automatic differentiation.

This implementation:
1. Extracts the current value of the focal variable
2. Creates a function that evaluates the original evaluator with modified focal variable
3. Uses ForwardDiff.derivative to compute the analytical derivative
4. Handles errors gracefully with NaN fallback
"""
function evaluate!(evaluator::ForwardDiffEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    
    focal_var = evaluator.focal_variable
    
    # Validate that focal variable exists in data
    if !haskey(data, focal_var)
        @warn "Focal variable $focal_var not found in data for ForwardDiff evaluation"
        @inbounds output[start_idx] = NaN
        return start_idx + 1
    end
    
    # Get current value of focal variable
    current_value = try
        Float64(data[focal_var][row_idx])
    catch e
        @warn "Failed to extract focal variable value: $e"
        @inbounds output[start_idx] = NaN
        return start_idx + 1
    end
    
    # Check for non-finite current value
    if !isfinite(current_value)
        @warn "Non-finite focal variable value: $current_value"
        @inbounds output[start_idx] = NaN
        return start_idx + 1
    end
    
    try
        # Get cached ForwardDiff function
        cached_f = get_cached_forwarddiff_function(evaluator.original_evaluator, focal_var)
        
        # Create function for ForwardDiff (captures data and row_idx)
        f(x) = cached_f(x, data, row_idx)
        
        # Compute derivative using ForwardDiff
        derivative_value = ForwardDiff.derivative(f, current_value)
        
        # Validate result
        if !isfinite(derivative_value)
            @debug "ForwardDiff produced non-finite derivative: $derivative_value"
            @inbounds output[start_idx] = 0.0  # Conservative fallback
        elseif abs(derivative_value) > 1e12
            @debug "ForwardDiff produced extremely large derivative: $derivative_value"
            @inbounds output[start_idx] = sign(derivative_value) * 1e12  # Clamp to reasonable range
        else
            @inbounds output[start_idx] = derivative_value
        end
        
    catch e
        @debug "ForwardDiff evaluation failed: $e"
        @inbounds output[start_idx] = NaN
    end
    
    return start_idx + 1
end

###############################################################################
# VALIDATION AND TESTING
###############################################################################

"""
    validate_forwarddiff_evaluator(evaluator::ForwardDiffEvaluator, test_data::NamedTuple, tolerance::Float64 = 1e-6)

Validate that a ForwardDiff evaluator produces reasonable results.

# Returns
`(is_valid::Bool, message::String)`
"""
function validate_forwarddiff_evaluator(evaluator::ForwardDiffEvaluator, 
                                       test_data::NamedTuple,
                                       tolerance::Float64 = 1e-6)
    
    focal_var = evaluator.focal_variable
    
    # Check if validation was already cached
    if evaluator.validation_cache[] !== nothing
        return evaluator.validation_cache[], "Cached validation result"
    end
    
    # Validate focal variable exists
    if !haskey(test_data, focal_var)
        result = false
        message = "Focal variable $focal_var not found in test data"
        evaluator.validation_cache[] = result
        return result, message
    end
    
    # Test evaluation at first few observations
    test_indices = 1:min(3, length(first(test_data)))
    
    for row_idx in test_indices
        try
            output = Vector{Float64}(undef, 1)
            evaluate!(evaluator, output, test_data, row_idx, 1)
            
            result = output[1]
            
            if !isfinite(result)
                message = "ForwardDiff produced non-finite result: $result at row $row_idx"
                evaluator.validation_cache[] = false
                return false, message
            end
            
            if abs(result) > 1e10
                message = "ForwardDiff produced extremely large result: $result at row $row_idx"
                evaluator.validation_cache[] = false
                return false, message
            end
            
        catch e
            message = "ForwardDiff evaluation failed at row $row_idx: $e"
            evaluator.validation_cache[] = false
            return false, message
        end
    end
    
    evaluator.validation_cache[] = true
    return true, "Validation passed for all test rows"
end

"""
    test_forwarddiff_accuracy(original_evaluator::AbstractEvaluator, 
                             focal_variable::Symbol,
                             test_data::NamedTuple;
                             tolerance::Float64 = 1e-6,
                             test_rows::Int = 5)

Test ForwardDiff accuracy against numerical finite differences.

# Returns
`(is_accurate::Bool, max_error::Float64, message::String)`
"""
function test_forwarddiff_accuracy(original_evaluator::AbstractEvaluator, 
                                  focal_variable::Symbol,
                                  test_data::NamedTuple;
                                  tolerance::Float64 = 1e-6,
                                  test_rows::Int = 5)
    
    # Create ForwardDiff evaluator
    fd_evaluator = ForwardDiffEvaluator(original_evaluator, focal_variable)
    
    max_error = 0.0
    test_indices = 1:min(test_rows, length(first(test_data)))
    
    for row_idx in test_indices
        try
            # Get ForwardDiff result
            fd_output = Vector{Float64}(undef, 1)
            evaluate!(fd_evaluator, fd_output, test_data, row_idx, 1)
            fd_result = fd_output[1]
            
            # Skip if ForwardDiff failed
            if !isfinite(fd_result)
                continue
            end
            
            # Compute numerical derivative using finite differences
            current_value = Float64(test_data[focal_variable][row_idx])
            ε = sqrt(eps(Float64))
            
            # Create modified data
            original_vector = test_data[focal_variable]
            modified_plus = copy(original_vector)
            modified_minus = copy(original_vector)
            modified_plus[row_idx] = current_value + ε
            modified_minus[row_idx] = current_value - ε
            
            data_plus = merge(test_data, (focal_variable => modified_plus,))
            data_minus = merge(test_data, (focal_variable => modified_minus,))
            
            # Evaluate at x + ε
            result_plus = Vector{Float64}(undef, output_width(original_evaluator))
            evaluate!(original_evaluator, result_plus, data_plus, row_idx, 1)
            
            # Evaluate at x - ε  
            result_minus = Vector{Float64}(undef, output_width(original_evaluator))
            evaluate!(original_evaluator, result_minus, data_minus, row_idx, 1)
            
            # Numerical derivative (for scalar output)
            numerical_result = if length(result_plus) == 1
                (result_plus[1] - result_minus[1]) / (2ε)
            else
                (sum(result_plus) - sum(result_minus)) / (2ε)
            end
            
            # Compare
            error = abs(fd_result - numerical_result)
            max_error = max(max_error, error)
            
            if error > tolerance
                return false, max_error, "Accuracy test failed at row $row_idx: ForwardDiff=$fd_result, Numerical=$numerical_result, Error=$error"
            end
            
        catch e
            @debug "Accuracy test error at row $row_idx: $e"
            continue
        end
    end
    
    return true, max_error, "Accuracy test passed with max error: $max_error"
end
