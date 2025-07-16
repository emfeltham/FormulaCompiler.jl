# derivatives.jl
# All derivatives return full-width vectors matching original formula

###############################################################################
# CORE POSITIONAL DERIVATIVE EVALUATOR
###############################################################################

"""
    PositionalDerivativeEvaluator <: AbstractEvaluator

BREAKING CHANGE: All derivatives now produce full-width output vectors.

This evaluator ensures derivative output matches the width of the original formula,
with derivatives placed in correct positions and zeros elsewhere.

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

# Full width output
output_width(eval::PositionalDerivativeEvaluator) = eval.target_width

function evaluate!(evaluator::PositionalDerivativeEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    
    # Initialize entire output to zero
    for i in 1:evaluator.target_width
        @inbounds output[start_idx + i - 1] = 0.0
    end
    
    # Handle different types of original evaluators
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
        
        # Compute derivative of this sub-evaluator
        sub_derivative = compute_scalar_derivative(sub_evaluator, evaluator.focal_variable)
        
        if !is_zero_derivative(sub_derivative, evaluator.focal_variable)
            # Evaluate and place in correct position
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
    derivative_evaluator = compute_scalar_derivative(evaluator.original_evaluator, evaluator.focal_variable)
    
    if !is_zero_derivative(derivative_evaluator, evaluator.focal_variable)
        temp_buffer = Vector{Float64}(undef, 1)
        evaluate!(derivative_evaluator, temp_buffer, data, row_idx, 1)
        @inbounds output[start_idx] = temp_buffer[1]
    end
end

###############################################################################
# CODE GENERATION FOR @GENERATED DERIVATIVES
###############################################################################

"""
    generate_evaluator_code!(instructions, evaluator::PositionalDerivativeEvaluator, pos)

Generate code for PositionalDerivativeEvaluator that integrates with existing @generated workflow.
"""
function generate_evaluator_code!(instructions::Vector{String}, evaluator::PositionalDerivativeEvaluator, pos::Int)
    # Initialize all positions to zero
    for i in 1:evaluator.target_width
        target_pos = pos + i - 1
        push!(instructions, "@inbounds row_vec[$target_pos] = 0.0")
    end
    
    # Generate code to compute derivatives and place them in correct positions
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
        sub_derivative = compute_scalar_derivative(sub_evaluator, evaluator.focal_variable)
        
        if !is_zero_derivative(sub_derivative, evaluator.focal_variable)
            if sub_derivative isa ConstantEvaluator && sub_derivative.value != 0.0
                push!(instructions, "@inbounds row_vec[$current_position] = $(sub_derivative.value)")
            elseif sub_derivative isa ContinuousEvaluator
                push!(instructions, "@inbounds row_vec[$current_position] = Float64(data.$(sub_derivative.column)[row_idx])")
            else
                # For complex derivatives, generate a variable and compute
                derivative_var = next_var("deriv")
                generate_single_component_code!(instructions, sub_derivative, derivative_var)
                push!(instructions, "@inbounds row_vec[$current_position] = $derivative_var")
            end
        end
        
        current_position += sub_width
    end
end

function generate_single_positioning_instructions!(instructions::Vector{String}, evaluator::PositionalDerivativeEvaluator, pos::Int)
    derivative_evaluator = compute_scalar_derivative(evaluator.original_evaluator, evaluator.focal_variable)
    
    if !is_zero_derivative(derivative_evaluator, evaluator.focal_variable)
        if derivative_evaluator isa ConstantEvaluator
            push!(instructions, "@inbounds row_vec[$pos] = $(derivative_evaluator.value)")
        elseif derivative_evaluator isa ContinuousEvaluator
            push!(instructions, "@inbounds row_vec[$pos] = Float64(data.$(derivative_evaluator.column)[row_idx])")
        else
            # For complex derivatives, generate a variable
            derivative_var = next_var("deriv")
            generate_single_component_code!(instructions, derivative_evaluator, derivative_var)
            push!(instructions, "@inbounds row_vec[$pos] = $derivative_var")
        end
    end
end

"""
    generate_single_component_code!(instructions, evaluator, var_name)

Generate code to compute a single component derivative and store in var_name.
"""
function generate_single_component_code!(instructions::Vector{String}, evaluator::AbstractEvaluator, var_name::String)
    if evaluator isa ConstantEvaluator
        push!(instructions, "$var_name = $(evaluator.value)")
    elseif evaluator isa ContinuousEvaluator
        push!(instructions, "$var_name = Float64(data.$(evaluator.column)[row_idx])")
    elseif evaluator isa FunctionEvaluator
        # Generate code for function derivative
        inner_var = next_var("inner")
        generate_single_component_code!(instructions, evaluator.arg_evaluator, inner_var)
        
        func_name = string(evaluator.func)
        if func_name == "log"
            push!(instructions, "$var_name = 1.0 / $inner_var")
        elseif func_name == "exp"
            push!(instructions, "$var_name = exp($inner_var)")
        elseif func_name == "sqrt"
            push!(instructions, "$var_name = 0.5 / sqrt($inner_var)")
        else
            # Fallback for unknown functions
            push!(instructions, "$var_name = 1.0  # TODO: implement derivative for $func_name")
        end
    elseif evaluator isa ChainRuleEvaluator
        # For chain rule: f'(g(x)) * g'(x)
        inner_var = next_var("inner")
        inner_deriv_var = next_var("inner_deriv")
        
        generate_single_component_code!(instructions, evaluator.inner_evaluator, inner_var)
        generate_single_component_code!(instructions, evaluator.inner_derivative, inner_deriv_var)
        
        # Apply derivative function
        func_deriv_var = next_var("func_deriv")
        push!(instructions, "$func_deriv_var = $(evaluator.derivative_func)($inner_var)")
        push!(instructions, "$var_name = $func_deriv_var * $inner_deriv_var")
        
    elseif evaluator isa ProductRuleEvaluator
        # For product rule: f*g' + g*f'
        f_var = next_var("f")
        f_prime_var = next_var("f_prime")
        g_var = next_var("g")
        g_prime_var = next_var("g_prime")
        
        generate_single_component_code!(instructions, evaluator.left_evaluator, f_var)
        generate_single_component_code!(instructions, evaluator.left_derivative, f_prime_var)
        generate_single_component_code!(instructions, evaluator.right_evaluator, g_var)
        generate_single_component_code!(instructions, evaluator.right_derivative, g_prime_var)
        
        push!(instructions, "$var_name = $f_var * $g_prime_var + $g_var * $f_prime_var")
    else
        # Fallback for other evaluator types
        push!(instructions, "$var_name = 0.0  # TODO: implement derivative for $(typeof(evaluator))")
    end
end

###############################################################################
# DERIVATIVE CACHE AND COMPILATION
###############################################################################

const DERIVATIVE_CACHE = Dict{UInt64, Tuple{Vector{String}, Vector{Symbol}, Int}}()

"""
    CompiledDerivativeFormula{H}

BREAKING CHANGE: Now always has output_width == original_formula.output_width

# Fields
- `formula_val::Val{H}`: Hash-based identifier
- `output_width::Int`: SAME as original formula width
- `focal_variable::Symbol`: Variable this derivative is with respect to
- `root_derivative_evaluator::PositionalDerivativeEvaluator`: Always positional
"""
struct CompiledDerivativeFormula{H}
    formula_val::Val{H}
    output_width::Int  # BREAKING CHANGE: Now same as original
    focal_variable::Symbol
    root_derivative_evaluator::PositionalDerivativeEvaluator
end

Base.length(cdf::CompiledDerivativeFormula) = cdf.output_width

function (cdf::CompiledDerivativeFormula)(row_vec::AbstractVector{Float64}, data, row_idx::Int)
    _derivative_modelrow_generated!(row_vec, cdf.formula_val, data, row_idx)
    return row_vec
end

"""
    compile_derivative_formula(compiled_formula::CompiledFormula, focal_variable::Symbol; verbose=false)

BREAKING CHANGE: Now returns full-width derivatives.

# Returns
`CompiledDerivativeFormula` where `length(derivative) == length(compiled_formula)`
"""
function compile_derivative_formula(compiled_formula::CompiledFormula, focal_variable::Symbol; verbose=false)
    target_width = compiled_formula.output_width
    root_evaluator = compiled_formula.root_evaluator
    
    if verbose
        println("=== Compiling Full-Width Derivative Formula ===")
        println("Original formula width: $target_width")
        println("Focal variable: $focal_variable")
    end
    
    # Create PositionalDerivativeEvaluator - this is the key integration point
    positional_derivative_evaluator = PositionalDerivativeEvaluator(
        root_evaluator, 
        focal_variable, 
        target_width
    )
    
    # Generate code instructions using existing infrastructure
    instructions = generate_code_from_evaluator(positional_derivative_evaluator)
    
    if verbose
        println("Generated $(length(instructions)) full-width derivative instructions:")
        for (i, instr) in enumerate(instructions[1:min(3, length(instructions))])
            println("  $i: $instr")
        end
        if length(instructions) > 3
            println("  ... ($(length(instructions) - 3) more)")
        end
    end
    
    # Cache for @generated function
    derivative_hash = hash((compiled_formula.formula_val, focal_variable, :full_width))
    DERIVATIVE_CACHE[derivative_hash] = (instructions, [focal_variable], target_width)
    
    if verbose
        println("Cached derivative with hash: $derivative_hash")
    end
    
    return CompiledDerivativeFormula(
        Val(derivative_hash), 
        target_width,  # BREAKING CHANGE: Same width as original
        focal_variable, 
        positional_derivative_evaluator
    )
end

@generated function _derivative_modelrow_generated!(
    row_vec::AbstractVector{Float64}, 
    ::Val{derivative_hash}, 
    data, 
    row_idx::Int
) where derivative_hash
    
    if !haskey(DERIVATIVE_CACHE, derivative_hash)
        return quote
            error("Derivative hash $($derivative_hash) not found in cache")
        end
    end
    
    instructions, _, _ = DERIVATIVE_CACHE[derivative_hash]
    
    try
        code_exprs = [Meta.parse(line) for line in instructions]
        return quote
            @inbounds begin
                $(code_exprs...)
            end
            return row_vec
        end
    catch e
        return quote
            error("Failed to parse derivative instructions: $($e)")
        end
    end
end

###############################################################################
# SCALAR DERIVATIVE COMPUTATION (internal only)
###############################################################################

"""
    compute_scalar_derivative(evaluator::AbstractEvaluator, focal_variable::Symbol)

Internal function for computing scalar derivatives. Used by PositionalDerivativeEvaluator
to compute derivatives of individual terms.

BREAKING CHANGE: This is now internal-only. External API is compile_derivative_formula.
"""
function compute_scalar_derivative(evaluator::AbstractEvaluator, focal_variable::Symbol)
    if evaluator isa ConstantEvaluator
        return ConstantEvaluator(0.0)
    elseif evaluator isa ContinuousEvaluator
        return evaluator.column == focal_variable ? ConstantEvaluator(1.0) : ConstantEvaluator(0.0)
    elseif evaluator isa CategoricalEvaluator
        return ConstantEvaluator(0.0)
    elseif evaluator isa FunctionEvaluator
        return compute_function_derivative(evaluator, focal_variable)
    elseif evaluator isa CombinedEvaluator
        return compute_sum_derivative(evaluator, focal_variable)
    elseif evaluator isa ProductEvaluator
        return compute_product_derivative(evaluator, focal_variable)
    elseif evaluator isa ScaledEvaluator
        inner_derivative = compute_scalar_derivative(evaluator.evaluator, focal_variable)
        return ScaledEvaluator(inner_derivative, evaluator.scale_factor)
    else
        @warn "Using ForwardDiff fallback for $(typeof(evaluator))"
        return ForwardDiffEvaluator(evaluator, focal_variable, NamedTuple())
    end
end

# All the derivative computation helpers remain the same:
function compute_function_derivative(evaluator::FunctionEvaluator, focal_variable::Symbol)
    arg_derivative = compute_scalar_derivative(evaluator.arg_evaluator, focal_variable)
    
    if is_zero_derivative(arg_derivative, focal_variable)
        return ConstantEvaluator(0.0)
    end
    
    derivative_func = get_standard_derivative_function(evaluator.func)
    
    return ChainRuleEvaluator(
        derivative_func,
        evaluator.arg_evaluator,
        arg_derivative
    )
end

function compute_sum_derivative(evaluator::CombinedEvaluator, focal_variable::Symbol)
    derivative_terms = AbstractEvaluator[]
    
    for sub_evaluator in evaluator.sub_evaluators
        sub_derivative = compute_scalar_derivative(sub_evaluator, focal_variable)
        
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

function compute_product_derivative(evaluator::ProductEvaluator, focal_variable::Symbol)
    left_derivative = compute_scalar_derivative(evaluator.components[1], focal_variable)
    right_derivative = compute_scalar_derivative(evaluator.components[2], focal_variable)
    
    left_is_zero = is_zero_derivative(left_derivative, focal_variable)
    right_is_zero = is_zero_derivative(right_derivative, focal_variable)
    
    if left_is_zero && right_is_zero
        return ConstantEvaluator(0.0)
    elseif left_is_zero
        return ProductEvaluator([evaluator.components[1], right_derivative])
    elseif right_is_zero
        return ProductEvaluator([left_derivative, evaluator.components[2]])
    else
        return ProductRuleEvaluator(
            evaluator.components[1], left_derivative,
            evaluator.components[2], right_derivative
        )
    end
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
    else
        error("Derivative not implemented for function: $func")
    end
end

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

function clear_derivative_cache!()
    empty!(DERIVATIVE_CACHE)
    return nothing
end

function is_zero_derivative(evaluator::AbstractEvaluator, focal_variable::Symbol)
    if evaluator isa ConstantEvaluator
        return evaluator.value == 0.0
    elseif evaluator isa ContinuousEvaluator
        return evaluator.column != focal_variable
    elseif evaluator isa CategoricalEvaluator
        return true
    else
        return false
    end
end

# Helper for variable naming - should integrate with existing next_var
function next_var(prefix::String="v")
    # This should integrate with the existing VAR_COUNTER from generators.jl
    VAR_COUNTER[] += 1
    return "$(prefix)_$(VAR_COUNTER[])"
end
