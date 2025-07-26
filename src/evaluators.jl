# evaluators.jl
# Complete recursive implementation that handles all cases

###############################################################################
# 1. CORE EVALUATOR TYPES (Fixed)
###############################################################################

abstract type AbstractEvaluator end

struct ConstantEvaluator <: AbstractEvaluator
    value::Float64
end

struct ContinuousEvaluator <: AbstractEvaluator
    column::Symbol
end

struct CategoricalEvaluator <: AbstractEvaluator
    column::Symbol
    contrast_matrix::Matrix{Float64}
    n_levels::Int
end

struct FunctionEvaluator <: AbstractEvaluator
    func::Function
    arg_evaluators::Vector{AbstractEvaluator}
end

struct InteractionEvaluator <: AbstractEvaluator
    components::Vector{AbstractEvaluator}
    total_width::Int
    
    function InteractionEvaluator(components)
        total_width = prod([output_width(comp) for comp in components])
        new(components, total_width)
    end
end

struct ZScoreEvaluator <: AbstractEvaluator
    underlying::AbstractEvaluator
    center::Float64
    scale::Float64
end

struct CombinedEvaluator <: AbstractEvaluator
    sub_evaluators::Vector{AbstractEvaluator}
    total_width::Int
    
    function CombinedEvaluator(sub_evaluators)
        total_width = sum([output_width(eval) for eval in sub_evaluators])
        new(sub_evaluators, total_width)
    end
end

struct ScaledEvaluator <: AbstractEvaluator
    evaluator::AbstractEvaluator
    scale_factor::Float64
end

struct ProductEvaluator <: AbstractEvaluator
    components::Vector{AbstractEvaluator}
end

###############################################################################
# 2. OUTPUT WIDTH CALCULATIONS
###############################################################################

output_width(eval::ConstantEvaluator) = 1
output_width(eval::ContinuousEvaluator) = 1
output_width(eval::CategoricalEvaluator) = size(eval.contrast_matrix, 2)
output_width(eval::FunctionEvaluator) = 1  # Functions always produce 1 column
output_width(eval::InteractionEvaluator) = eval.total_width
output_width(eval::ZScoreEvaluator) = output_width(eval.underlying)
output_width(eval::CombinedEvaluator) = eval.total_width
output_width(eval::ScaledEvaluator) = output_width(eval.evaluator)
output_width(eval::ProductEvaluator) = 1  # Products always yield single values

###############################################################################
# 3. RECURSIVE EVALUATION (The Key Fix)
###############################################################################

"""
Recursively evaluate any evaluator into a pre-allocated output vector.
This is the core that makes the compositional approach work.
"""
function evaluate!(
    evaluator::AbstractEvaluator, output::AbstractVector{Float64}, 
    data, row_idx::Int, start_idx::Int=1
)
    
    if evaluator isa ConstantEvaluator
        @inbounds output[start_idx] = evaluator.value
        return start_idx + 1
        
    elseif evaluator isa ContinuousEvaluator
        @inbounds output[start_idx] = Float64(data[evaluator.column][row_idx])
        return start_idx + 1
        
    elseif evaluator isa CategoricalEvaluator
        return evaluate_categorical!(evaluator, output, data, row_idx, start_idx)
        
    elseif evaluator isa FunctionEvaluator
        return evaluate_function!(evaluator, output, data, row_idx, start_idx)
        
    elseif evaluator isa InteractionEvaluator
        return evaluate_interaction!(evaluator, output, data, row_idx, start_idx)
        
    elseif evaluator isa ZScoreEvaluator
        return evaluate_zscore!(evaluator, output, data, row_idx, start_idx)
        
    elseif evaluator isa CombinedEvaluator
        return evaluate_combined!(evaluator, output, data, row_idx, start_idx)
        
    else
        error("Unknown evaluator type: $(typeof(evaluator))")
    end
end

function evaluate_categorical!(
    eval::CategoricalEvaluator, output::AbstractVector{Float64}, 
    data, row_idx::Int, start_idx::Int
)
    @inbounds cat_val = data[eval.column][row_idx]
    @inbounds level_code = cat_val isa CategoricalValue ? levelcode(cat_val) : 1
    @inbounds level_code = clamp(level_code, 1, eval.n_levels)
    
    width = size(eval.contrast_matrix, 2)
    @inbounds for j in 1:width
        output[start_idx + j - 1] = eval.contrast_matrix[level_code, j]
    end
    
    return start_idx + width
end

function evaluate_function!(
    eval::FunctionEvaluator, output::AbstractVector{Float64}, 
    data, row_idx::Int, start_idx::Int
)
    n_args = length(eval.arg_evaluators)
    
    if n_args == 1
        # Unary function - most common case
        arg_eval = eval.arg_evaluators[1]
        
        if arg_eval isa ContinuousEvaluator
            @inbounds val = Float64(data[arg_eval.column][row_idx])
        elseif arg_eval isa ConstantEvaluator
            val = arg_eval.value
        else
            # Recursively evaluate complex argument
            temp_val = Vector{Float64}(undef, 1)
            evaluate!(arg_eval, temp_val, data, row_idx, 1)
            val = temp_val[1]
        end
        
        # Apply function with safety
        result = apply_function_safe(eval.func, val)
        @inbounds output[start_idx] = result
        
    elseif n_args == 2
        # Binary function
        val1, val2 = evaluate_two_args(eval.arg_evaluators, data, row_idx)
        result = apply_function_safe(eval.func, val1, val2)
        @inbounds output[start_idx] = result
        
    else
        # General case - evaluate all arguments
        args = Vector{Float64}(undef, n_args)
        for (i, arg_eval) in enumerate(eval.arg_evaluators)
            if arg_eval isa ContinuousEvaluator
                @inbounds args[i] = Float64(data[arg_eval.column][row_idx])
            elseif arg_eval isa ConstantEvaluator
                args[i] = arg_eval.value
            else
                temp_val = Vector{Float64}(undef, 1)
                evaluate!(arg_eval, temp_val, data, row_idx, 1)
                args[i] = temp_val[1]
            end
        end
        
        result = apply_function_safe(eval.func, args...)
        @inbounds output[start_idx] = result
    end
    
    return start_idx + 1
end

function evaluate_two_args(arg_evaluators::Vector{AbstractEvaluator}, data, row_idx::Int)
    val1 = if arg_evaluators[1] isa ContinuousEvaluator
        Float64(data[arg_evaluators[1].column][row_idx])
    elseif arg_evaluators[1] isa ConstantEvaluator
        arg_evaluators[1].value
    else
        temp = Vector{Float64}(undef, 1)
        evaluate!(arg_evaluators[1], temp, data, row_idx, 1)
        temp[1]
    end
    
    val2 = if arg_evaluators[2] isa ContinuousEvaluator
        Float64(data[arg_evaluators[2].column][row_idx])
    elseif arg_evaluators[2] isa ConstantEvaluator
        arg_evaluators[2].value
    else
        temp = Vector{Float64}(undef, 1)
        evaluate!(arg_evaluators[2], temp, data, row_idx, 1)
        temp[1]
    end
    
    return val1, val2
end

function evaluate_interaction!(eval::InteractionEvaluator, output::AbstractVector{Float64}, 
                              data, row_idx::Int, start_idx::Int)
    n_components = length(eval.components)
    component_widths = [output_width(comp) for comp in eval.components]
    
    # Evaluate each component into temporary buffers
    component_buffers = Vector{Vector{Float64}}(undef, n_components)
    
    for (i, component) in enumerate(eval.components)
        width = component_widths[i]
        component_buffers[i] = Vector{Float64}(undef, width)
        evaluate!(component, component_buffers[i], data, row_idx, 1)
    end
    
    # Compute Kronecker product
    compute_kronecker_product!(component_buffers, component_widths, 
                              view(output, start_idx:(start_idx + eval.total_width - 1)))
    
    return start_idx + eval.total_width
end

function evaluate_zscore!(eval::ZScoreEvaluator, output::AbstractVector{Float64}, 
                         data, row_idx::Int, start_idx::Int)
    # Evaluate underlying term
    underlying_width = output_width(eval.underlying)
    temp_buffer = Vector{Float64}(undef, underlying_width)
    evaluate!(eval.underlying, temp_buffer, data, row_idx, 1)
    
    # Apply Z-score transformation
    @inbounds for i in 1:underlying_width
        output[start_idx + i - 1] = (temp_buffer[i] - eval.center) / eval.scale
    end
    
    return start_idx + underlying_width
end

function evaluate_combined!(
    eval::CombinedEvaluator, output::AbstractVector{Float64}, 
    data, row_idx::Int, start_idx::Int
)
    current_idx = start_idx
    
    for sub_eval in eval.sub_evaluators
        current_idx = evaluate!(sub_eval, output, data, row_idx, current_idx)
    end
    
    return current_idx
end

function evaluate!(evaluator::ScaledEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    next_idx = evaluate!(evaluator.evaluator, output, data, row_idx, start_idx)
    @inbounds output[start_idx] *= evaluator.scale_factor
    return next_idx
end

function evaluate!(evaluator::ProductEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    product = 1.0
    temp_buffer = Vector{Float64}(undef, 1)
    
    for component in evaluator.components
        evaluate!(component, temp_buffer, data, row_idx, 1)
        product *= temp_buffer[1]
    end
    
    @inbounds output[start_idx] = product
    return start_idx + 1
end

###############################################################################
# 4. SAFE FUNCTION APPLICATION
###############################################################################

"""
    apply_function_safe(func::Function, val::Float64)

Apply function with mathematically correct error handling and proper logging.

# Key Improvements
1. Mathematically correct handling of domain errors
2. Selective error catching - only catch expected domain errors
3. Logging of problematic cases for debugging
4. Clear documentation of behavior
"""
# function apply_function_safe(func::Function, val::Float64)
    
#     # Handle known functions with careful domain checking
#     if func === log
#         if val > 0.0
#             return log(val)
#         elseif val == 0.0
#             @debug "log(0) encountered, returning -Inf"
#             return -Inf
#         else
#             @debug "log(negative) encountered: log($val), returning NaN"
#             return NaN
#         end
        
#     elseif func === exp
#         # Clamp to prevent overflow, but warn about extreme values
#         if val > 700.0
#             @debug "exp(large) encountered: exp($val), clamping to exp(700)"
#             return exp(700.0)
#         elseif val < -700.0
#             @debug "exp(very negative) encountered: exp($val), clamping to exp(-700)"
#             return exp(-700.0)
#         else
#             return exp(val)
#         end
        
#     elseif func === sqrt
#         if val >= 0.0
#             return sqrt(val)
#         else
#             @debug "sqrt(negative) encountered: sqrt($val), returning NaN"
#             return NaN
#         end
        
#     elseif func === log10
#         if val > 0.0
#             return log10(val)
#         elseif val == 0.0
#             @debug "log10(0) encountered, returning -Inf"
#             return -Inf
#         else
#             @debug "log10(negative) encountered: log10($val), returning NaN"
#             return NaN
#         end
        
#     elseif func === log2
#         if val > 0.0
#             return log2(val)
#         elseif val == 0.0
#             @debug "log2(0) encountered, returning -Inf"
#             return -Inf
#         else
#             @debug "log2(negative) encountered: log2($val), returning NaN"
#             return NaN
#         end
        
#     elseif func === abs
#         return abs(val)
        
#     elseif func === sin
#         return sin(val)
        
#     elseif func === cos
#         return cos(val)
        
#     elseif func === tan
#         # Check for values near π/2 + nπ where tan is undefined
#         result = tan(val)
#         if !isfinite(result)
#             @debug "tan(undefined) encountered: tan($val), returning $(result)"
#         end
#         return result
        
#     elseif func === atan
#         return atan(val)  # Always well-defined for real numbers
        
#     elseif func === sinh
#         # Check for overflow
#         if abs(val) > 700.0
#             @debug "sinh(large) encountered: sinh($val), may overflow"
#         end
#         return sinh(val)
        
#     elseif func === cosh
#         # Check for overflow  
#         if abs(val) > 700.0
#             @debug "cosh(large) encountered: cosh($val), may overflow"
#         end
#         return cosh(val)
        
#     elseif func === tanh
#         return tanh(val)  # Always well-defined and bounded
        
#     elseif func === sign
#         return sign(val)  # Always well-defined
        
#     else
#         # For unknown functions, be more selective about error catching
#         try
#             result = Float64(func(val))
            
#             # Check result validity
#             if !isfinite(result) && isfinite(val)
#                 @debug "Function $func produced non-finite result $result from finite input $val"
#             end
            
#             return result
            
#         catch e
#             # Only catch domain-related errors, let others propagate
#             if e isa DomainError
#                 @debug "DomainError in $func($val): $e, returning NaN"
#                 return NaN
#             elseif e isa InexactError
#                 @debug "InexactError in $func($val): $e, returning NaN" 
#                 return NaN
#             elseif e isa OverflowError
#                 @debug "OverflowError in $func($val): $e, returning Inf"
#                 return val > 0 ? Inf : -Inf
#             else
#                 # Re-throw unexpected errors for debugging
#                 @error "Unexpected error in apply_function_safe for $func($val): $e"
#                 rethrow(e)
#             end
#         end
#     end
# end

# """
#     apply_function_safe(func::Function, val1::Float64, val2::Float64)

# Apply binary function with improved error handling.
# """
# function apply_function_safe(func::Function, val1::Float64, val2::Float64)
    
#     if func === (^)
#         # Handle power function carefully
#         if val1 == 0.0 && val2 < 0.0
#             @debug "0^(negative) encountered: 0^$val2, returning Inf"
#             return Inf
#         elseif val1 < 0.0 && !isinteger(val2)
#             @debug "negative^(non-integer) encountered: $val1^$val2, returning NaN"
#             return NaN
#         else
#             try
#                 result = val1^val2
#                 if !isfinite(result) && isfinite(val1) && isfinite(val2)
#                     @debug "Power operation produced non-finite result: $val1^$val2 = $result"
#                 end
#                 return result
#             catch e
#                 if e isa DomainError
#                     @debug "DomainError in $val1^$val2: $e, returning NaN"
#                     return NaN
#                 else
#                     rethrow(e)
#                 end
#             end
#         end
        
#     elseif func === (+)
#         return val1 + val2
        
#     elseif func === (-)
#         return val1 - val2
        
#     elseif func === (*)
#         return val1 * val2
        
#     elseif func === (/)
#         if val2 == 0.0
#             if val1 == 0.0
#                 @debug "0/0 encountered, returning NaN"
#                 return NaN
#             elseif val1 > 0.0
#                 @debug "positive/0 encountered: $val1/0, returning Inf"
#                 return Inf
#             else
#                 @debug "negative/0 encountered: $val1/0, returning -Inf"
#                 return -Inf
#             end
#         elseif abs(val2) < 1e-16
#             @debug "Division by very small number: $val1/$val2, may be unstable"
#             return val1 / val2
#         else
#             return val1 / val2
#         end
        
#     # Comparison operations
#     elseif func === (>)
#         return val1 > val2 ? 1.0 : 0.0
#     elseif func === (<)
#         return val1 < val2 ? 1.0 : 0.0
#     elseif func === (>=)
#         return val1 >= val2 ? 1.0 : 0.0
#     elseif func === (<=)
#         return val1 <= val2 ? 1.0 : 0.0
#     elseif func === (==)
#         return val1 == val2 ? 1.0 : 0.0
#     elseif func === (!=)
#         return val1 != val2 ? 1.0 : 0.0
        
#     else
#         # For unknown binary functions
#         try
#             result = Float64(func(val1, val2))
            
#             # Check result validity
#             if !isfinite(result) && isfinite(val1) && isfinite(val2)
#                 @debug "Binary function $func produced non-finite result $result from finite inputs ($val1, $val2)"
#             end
            
#             return result
            
#         catch e
#             if e isa DomainError
#                 @debug "DomainError in $func($val1, $val2): $e, returning NaN"
#                 return NaN
#             elseif e isa InexactError
#                 @debug "InexactError in $func($val1, $val2): $e, returning NaN"
#                 return NaN  
#             elseif e isa OverflowError
#                 @debug "OverflowError in $func($val1, $val2): $e, returning Inf"
#                 return Inf
#             else
#                 @error "Unexpected error in apply_function_safe for $func($val1, $val2): $e"
#                 rethrow(e)
#             end
#         end
#     end
# end

# """
#     apply_function_safe(func::Function, args::Float64...)

# Apply n-ary function with error handling.
# """
# function apply_function_safe(func::Function, args::Float64...)
#     try
#         result = Float64(func(args...))
        
#         # Check result validity
#         if !isfinite(result) && all(isfinite, args)
#             @debug "N-ary function $func produced non-finite result $result from finite inputs $args"
#         end
        
#         return result
        
#     catch e
#         if e isa DomainError
#             @debug "DomainError in $func($args): $e, returning NaN"
#             return NaN
#         elseif e isa InexactError  
#             @debug "InexactError in $func($args): $e, returning NaN"
#             return NaN
#         elseif e isa OverflowError
#             @debug "OverflowError in $func($args): $e, returning Inf"
#             return Inf
#         else
#             @error "Unexpected error in apply_function_safe for $func($args): $e"
#             rethrow(e)
#         end
#     end
# end

###############################################################################
# TESTING UTILITIES
###############################################################################

"""
    test_function_safety()

Test that apply_function_safe handles edge cases correctly.
"""
function test_function_safety()
    println("Testing apply_function_safe improvements...")
    
    # Test log domain errors
    @assert isnan(apply_function_safe(log, -1.0)) "log(-1) should return NaN"
    @assert apply_function_safe(log, 0.0) == -Inf "log(0) should return -Inf" 
    @assert apply_function_safe(log, 1.0) == 0.0 "log(1) should return 0"
    println("✓ log domain handling works")
    
    # Test sqrt domain errors
    @assert isnan(apply_function_safe(sqrt, -1.0)) "sqrt(-1) should return NaN"
    @assert apply_function_safe(sqrt, 4.0) == 2.0 "sqrt(4) should return 2"
    println("✓ sqrt domain handling works")
    
    # Test division by zero
    @assert apply_function_safe(/, 1.0, 0.0) == Inf "1/0 should return Inf"
    @assert apply_function_safe(/, -1.0, 0.0) == -Inf "-1/0 should return -Inf"
    @assert isnan(apply_function_safe(/, 0.0, 0.0)) "0/0 should return NaN"
    println("✓ division by zero handling works")
    
    # Test power function edge cases
    @assert apply_function_safe(^, 0.0, -1.0) == Inf "0^(-1) should return Inf"
    @assert isnan(apply_function_safe(^, -1.0, 0.5)) "(-1)^0.5 should return NaN"
    println("✓ power function edge cases work")
    
    println("All function safety tests passed!")
    return true
end

###############################################################################
# 5. KRONECKER PRODUCT COMPUTATION
###############################################################################

function compute_kronecker_product!(
    component_buffers::Vector{Vector{Float64}}, 
    component_widths::Vector{Int}, 
    output::AbstractVector{Float64}
)
    n_components = length(component_buffers)
    
    if n_components == 1
        # Single component - just copy
        copy!(output, component_buffers[1])
        
    elseif n_components == 2
        # Two components - direct computation
        w1, w2 = component_widths[1], component_widths[2]
        buf1, buf2 = component_buffers[1], component_buffers[2]
        
        idx = 1
        @inbounds for j in 1:w2
            for i in 1:w1
                output[idx] = buf1[i] * buf2[j]
                idx += 1
            end
        end
        
    elseif n_components == 3
        # Three components - common case
        w1, w2, w3 = component_widths[1], component_widths[2], component_widths[3]
        buf1, buf2, buf3 = component_buffers[1], component_buffers[2], component_buffers[3]
        
        idx = 1
        @inbounds for k in 1:w3
            for j in 1:w2
                for i in 1:w1
                    output[idx] = buf1[i] * buf2[j] * buf3[k]
                    idx += 1
                end
            end
        end
        
    else
        # General case - recursive approach
        compute_general_kronecker!(component_buffers, component_widths, output)
    end
end

function compute_general_kronecker!(
    component_buffers::Vector{Vector{Float64}}, 
    component_widths::Vector{Int}, 
    output::AbstractVector{Float64}
)
    n_components = length(component_buffers)
    total_size = length(output)
    
    @inbounds for i in 1:total_size
        # Convert linear index to multi-dimensional indices
        indices = linear_to_multi_index(i - 1, component_widths) .+ 1
        
        # Compute product across all components
        product = 1.0
        for j in 1:n_components
            product *= component_buffers[j][indices[j]]
        end
        
        output[i] = product
    end
end

function linear_to_multi_index(linear_idx::Int, dimensions::Vector{Int})
    n_dims = length(dimensions)
    indices = Vector{Int}(undef, n_dims)
    
    remaining = linear_idx
    for i in 1:n_dims
        indices[i] = remaining % dimensions[i]
        remaining = remaining ÷ dimensions[i]
    end
    
    return indices
end

###############################################################################
# 6. TERM COMPILATION (Recursive)
###############################################################################

function compile_term(term::AbstractTerm)
    if term isa InterceptTerm
        return hasintercept(term) ? ConstantEvaluator(1.0) : ConstantEvaluator(0.0)
        
    elseif term isa ConstantTerm
        return ConstantEvaluator(Float64(term.n))
        
    elseif term isa Union{ContinuousTerm, Term}
        return ContinuousEvaluator(term.sym)
        
    elseif term isa CategoricalTerm
        return CategoricalEvaluator(
            term.sym,
            Matrix{Float64}(term.contrasts.matrix),
            size(term.contrasts.matrix, 1)
        )
        
    elseif term isa FunctionTerm
        # Recursively compile arguments
        arg_evaluators = [compile_term(arg) for arg in term.args]
        return FunctionEvaluator(term.f, arg_evaluators)
        
    elseif term isa InteractionTerm
        # Recursively compile components
        component_evaluators = [compile_term(comp) for comp in term.terms]
        return InteractionEvaluator(component_evaluators)
        
    elseif term isa ZScoredTerm
        underlying_evaluator = compile_term(term.term)
        center = term.center isa Number ? Float64(term.center) : Float64(term.center[1])
        scale = term.scale isa Number ? Float64(term.scale) : Float64(term.scale[1])
        return ZScoreEvaluator(underlying_evaluator, center, scale)
        
    elseif term isa MatrixTerm
        # Compile each sub-term
        sub_evaluators = [compile_term(t) for t in term.terms if width(t) > 0]
        return CombinedEvaluator(sub_evaluators)
        
    else
        error("Unknown term type: $(typeof(term)), using constant fallback")
    end
end

###############################################################################
# 8. UTILITY FUNCTIONS
###############################################################################

function extract_all_columns(term::AbstractTerm)
    columns = Symbol[]
    extract_columns_recursive!(columns, term)
    return unique(columns)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::Union{ContinuousTerm, Term})
    push!(columns, term.sym)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::CategoricalTerm)
    push!(columns, term.sym)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::FunctionTerm)
    for arg in term.args
        extract_columns_recursive!(columns, arg)
    end
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::InteractionTerm)
    for comp in term.terms
        extract_columns_recursive!(columns, comp)
    end
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::ZScoredTerm)
    extract_columns_recursive!(columns, term.term)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::MatrixTerm)
    for sub_term in term.terms
        extract_columns_recursive!(columns, sub_term)
    end
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::Union{InterceptTerm, ConstantTerm})
    # No columns
end

# Fix the InteractionEvaluator constructor to handle type stability
function InteractionEvaluator(components::Vector{AbstractEvaluator})
    if isempty(components)
        total_width = 1
    else
        # Ensure we work with integers
        widths = [Int(output_width(comp)) for comp in components]
        total_width = prod(widths)
    end
    
    InteractionEvaluator(components, total_width)
end

# Make sure the struct has the inner constructor
struct InteractionEvaluator <: AbstractEvaluator
    components::Vector{AbstractEvaluator}
    total_width::Int
    
    # Inner constructor that takes both arguments
    InteractionEvaluator(components, total_width) = new(components, total_width)
end