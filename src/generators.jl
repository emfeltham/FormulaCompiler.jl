# generators.jl
# Add code generation to your existing recursive evaluators

###############################################################################
# CODE GENERATION FROM EVALUATOR TREE
###############################################################################

# Global variable counter for unique names
const VAR_COUNTER = Ref(0)

"""
Generate globally unique variable names to prevent conflicts.
"""
function next_var(prefix::String="v")
    VAR_COUNTER[] += 1
    return "$(prefix)_$(VAR_COUNTER[])"
end

"""
Reset counter for each new formula compilation.
"""
function reset_var_counter!()
    VAR_COUNTER[] = 0
end

###############################################################################
# MAIN CODE GENERATION INTERFACE
###############################################################################

"""
Generate instruction strings from evaluator tree for @generated function.
"""
function generate_code_from_evaluator(evaluator::AbstractEvaluator)
    reset_var_counter!()
    instructions = String[]
    generate_evaluator_code!(instructions, evaluator, 1)
    return instructions
end

"""
Recursively generate code for any evaluator, writing to row_vec starting at pos.
Returns the next position to write to.
"""
function generate_evaluator_code!(instructions::Vector{String}, evaluator::AbstractEvaluator, pos::Int)
    
    if evaluator isa ConstantEvaluator
        push!(instructions, "@inbounds row_vec[$pos] = $(evaluator.value)")
        return pos + 1
        
    elseif evaluator isa ContinuousEvaluator
        push!(instructions, "@inbounds row_vec[$pos] = Float64(data.$(evaluator.column)[row_idx])")
        return pos + 1
        
    elseif evaluator isa CategoricalEvaluator
        return generate_categorical_code!(instructions, evaluator, pos)
        
    elseif evaluator isa FunctionEvaluator
        if is_simple_for_inline(evaluator)
            return generate_function_code!(instructions, evaluator, pos)  # New simple version
        else
            return generate_complex_function_code!(instructions, evaluator, pos)  # New complex version
        end
        
    elseif evaluator isa InteractionEvaluator
        return generate_interaction_code!(instructions, evaluator, pos)
        
    elseif evaluator isa ZScoreEvaluator
        return generate_zscore_code!(instructions, evaluator, pos)
        
    elseif evaluator isa CombinedEvaluator
        return generate_combined_code!(instructions, evaluator, pos)
    elseif evaluator isa ScaledEvaluator
    return generate_scaled_code!(instructions, evaluator, pos)    
    elseif evaluator isa ProductEvaluator  
        return generate_product_code!(instructions, evaluator, pos)
    elseif evaluator isa ChainRuleEvaluator
        return generate_chain_rule_code!(instructions, evaluator, pos)
    elseif evaluator isa ProductRuleEvaluator
        return generate_product_rule_code!(instructions, evaluator, pos)
    elseif evaluator isa ChainRuleEvaluator
        return generate_chain_rule_code!(instructions, evaluator, pos)
    elseif evaluator isa ProductRuleEvaluator
        return generate_product_rule_code!(instructions, evaluator, pos)
    elseif evaluator isa ForwardDiffEvaluator
        # For now, generate a placeholder - ForwardDiff would need special handling
        push!(instructions, "@inbounds row_vec[$pos] = 0.0  # ForwardDiff placeholder")
        return pos + 1
    else
        error("Unknown evaluator type: $(typeof(evaluator))")
    end
end

###############################################################################
# CATEGORICAL CODE GENERATION
###############################################################################

function generate_categorical_code!(instructions::Vector{String}, eval::CategoricalEvaluator, pos::Int)
    col = eval.column
    n_levels = eval.n_levels
    contrast_matrix = eval.contrast_matrix
    width = size(contrast_matrix, 2)
    
    # Generate unique variable names for this categorical
    cat_var = next_var("cat")
    level_var = next_var("level")
    
    # Extract categorical value and level code
    push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
    push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
    push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
    
    # Generate code for each contrast column
    for j in 1:width
        output_pos = pos + j - 1
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        
        # Generate efficient lookup
        if n_levels == 1
            push!(instructions, "@inbounds row_vec[$output_pos] = $(values[1])")
        elseif n_levels == 2
            push!(instructions, "@inbounds row_vec[$output_pos] = $level_var == 1 ? $(values[1]) : $(values[2])")
        elseif n_levels == 3
            push!(instructions, "@inbounds row_vec[$output_pos] = $level_var == 1 ? $(values[1]) : $level_var == 2 ? $(values[2]) : $(values[3])")
        else
            # Chain of ternaries for more levels
            ternary_chain = "$level_var == 1 ? $(values[1])"
            for i in 2:(n_levels-1)
                ternary_chain *= " : $level_var == $i ? $(values[i])"
            end
            ternary_chain *= " : $(values[n_levels])"
            push!(instructions, "@inbounds row_vec[$output_pos] = $ternary_chain")
        end
    end
    
    return pos + width
end

###############################################################################
# FUNCTION CODE GENERATION
###############################################################################

"""
    generate_function_code!(instructions, eval::FunctionEvaluator, pos)

Generate allocation-free function evaluation code using inline expressions.
"""
function generate_function_code!(instructions::Vector{String}, eval::FunctionEvaluator, pos::Int)
    func = eval.func
    n_args = length(eval.arg_evaluators)
    
    if n_args == 0
        # Zero-argument function
        result_expr = "$func()"
        push!(instructions, "@inbounds row_vec[$pos] = $result_expr")
        
    elseif n_args == 1
        # Unary function - most common case
        arg_expr = generate_inline_expression(eval.arg_evaluators[1])
        result_expr = generate_function_call(func, [arg_expr])
        push!(instructions, "@inbounds row_vec[$pos] = $result_expr")
        
    elseif n_args == 2
        # Binary function
        arg1_expr = generate_inline_expression(eval.arg_evaluators[1])
        arg2_expr = generate_inline_expression(eval.arg_evaluators[2])
        result_expr = generate_function_call(func, [arg1_expr, arg2_expr])
        push!(instructions, "@inbounds row_vec[$pos] = $result_expr")
        
    else
        # General case - evaluate all arguments as expressions
        arg_exprs = [generate_inline_expression(arg_eval) for arg_eval in eval.arg_evaluators]
        result_expr = generate_function_call(func, arg_exprs)
        push!(instructions, "@inbounds row_vec[$pos] = $result_expr")
    end
    
    return pos + 1
end

###############################################################################
# HELPER: GENERATE INLINE EXPRESSION (needed for fixes above)
###############################################################################

"""
    generate_inline_expression(evaluator::AbstractEvaluator) -> String

Generate an inline expression for an evaluator that can be used directly
in larger expressions without requiring temporary variable assignment.

This is a simplified version of the function from the interaction fixes.
"""
function generate_inline_expression(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator
        return string(evaluator.value)
        
    elseif evaluator isa ContinuousEvaluator
        return "Float64(data.$(evaluator.column)[row_idx])"
        
    elseif evaluator isa CategoricalEvaluator
        # FIXED: No more multi-line generation in expressions
        error("CategoricalEvaluator cannot be used in inline expressions. Use statement-level generation instead.")
        
    elseif evaluator isa FunctionEvaluator
        # FIXED: Check for simple vs complex
        if all(arg -> arg isa Union{ConstantEvaluator, ContinuousEvaluator}, evaluator.arg_evaluators)
            # Simple case - generate inline
            arg_exprs = [arg isa ConstantEvaluator ? string(arg.value) : "Float64(data.$(arg.column)[row_idx])" for arg in evaluator.arg_evaluators]
            return generate_function_call(evaluator.func, arg_exprs)
        else
            error("Complex function arguments not supported in inline expressions: $(evaluator.func)")
        end
        
    elseif evaluator isa ScaledEvaluator
        inner_expr = generate_inline_expression(evaluator.evaluator)
        return "($(inner_expr) * $(evaluator.scale_factor))"
        
    elseif evaluator isa ProductEvaluator
        # FIXED: Check if all components are simple
        if all(comp -> comp isa Union{ConstantEvaluator, ContinuousEvaluator}, evaluator.components)
            component_exprs = [comp isa ConstantEvaluator ? string(comp.value) : "Float64(data.$(comp.column)[row_idx])" for comp in evaluator.components]
            return "(" * join(component_exprs, " * ") * ")"
        else
            error("Complex ProductEvaluator components not supported in inline expressions")
        end
        
    else
        # FIXED: No placeholder - clear error
        error("Inline expression generation not implemented for $(typeof(evaluator))")
    end
end


"""
Generate inline categorical expression using ternary operators.
"""
function generate_categorical_inline_expression(evaluator::CategoricalEvaluator)
    col = evaluator.column
    n_levels = evaluator.n_levels
    values = [evaluator.contrast_matrix[i, 1] for i in 1:n_levels]  # Use first contrast column
    
    if n_levels == 1
        return string(values[1])
    elseif n_levels == 2
        return "(data.$col[row_idx] isa CategoricalValue ? (clamp(levelcode(data.$col[row_idx]), 1, 2) == 1 ? $(values[1]) : $(values[2])) : $(values[1]))"
    else
        # Build nested ternary expression
        level_expr = "clamp(data.$col[row_idx] isa CategoricalValue ? levelcode(data.$col[row_idx]) : 1, 1, $n_levels)"
        
        ternary_chain = "$level_expr == 1 ? $(values[1])"
        for i in 2:(n_levels-1)
            ternary_chain *= " : $level_expr == $i ? $(values[i])"
        end
        ternary_chain *= " : $(values[n_levels])"
        
        return "($ternary_chain)"
    end
end

"""
Generate nested function expression recursively.
"""
function generate_nested_function_expression(evaluator::FunctionEvaluator)
    func = evaluator.func
    arg_exprs = [generate_inline_expression(arg) for arg in evaluator.arg_evaluators]
    return generate_function_call(func, arg_exprs)
end


function generate_argument_code!(instructions::Vector{String}, arg_eval::AbstractEvaluator, var_name::String)
    if arg_eval isa ContinuousEvaluator
        push!(instructions, "@inbounds $var_name = Float64(data.$(arg_eval.column)[row_idx])")
        
    elseif arg_eval isa ConstantEvaluator
        push!(instructions, "@inbounds $var_name = $(arg_eval.value)")
        
    elseif arg_eval isa CategoricalEvaluator
        # For categorical in function argument, use first contrast column
        col = arg_eval.column
        n_levels = arg_eval.n_levels
        values = [arg_eval.contrast_matrix[i, 1] for i in 1:n_levels]
        
        cat_var = next_var("cat")
        level_var = next_var("level")
        
        push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
        push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
        push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
        
        if n_levels == 2
            push!(instructions, "@inbounds $var_name = $level_var == 1 ? $(values[1]) : $(values[2])")
        else
            ternary_chain = "$level_var == 1 ? $(values[1])"
            for i in 2:(n_levels-1)
                ternary_chain *= " : $level_var == $i ? $(values[i])"
            end
            ternary_chain *= " : $(values[n_levels])"
            push!(instructions, "@inbounds $var_name = $ternary_chain")
        end
        
    elseif arg_eval isa FunctionEvaluator
        # Recursively handle nested functions
        nested_args = [next_var("nested_arg") for _ in arg_eval.arg_evaluators]
        for (i, nested_arg_eval) in enumerate(arg_eval.arg_evaluators)
            generate_argument_code!(instructions, nested_arg_eval, nested_args[i])
        end
        nested_expr = generate_function_call(arg_eval.func, nested_args)
        push!(instructions, "@inbounds $var_name = $nested_expr")
        
    else
        error("Variable generation not implemented for $(typeof(evaluator))")
    end
end

"""
Enhanced generate_function_call that handles more function types and 
works with inline expressions.
"""
function generate_function_call(func::Function, arg_exprs::Vector{String})
    if func === log
        @assert length(arg_exprs) == 1 "log expects 1 argument"
        arg = arg_exprs[1]
        return "($arg > 0.0 ? log($arg) : ($arg == 0.0 ? -Inf : NaN))"
        
    elseif func === exp
        @assert length(arg_exprs) == 1 "exp expects 1 argument"
        arg = arg_exprs[1]
        return "exp(clamp($arg, -700.0, 700.0))"
        
    elseif func === sqrt
        @assert length(arg_exprs) == 1 "sqrt expects 1 argument"
        arg = arg_exprs[1]
        return "($arg >= 0.0 ? sqrt($arg) : NaN)"
        
    elseif func === abs
        @assert length(arg_exprs) == 1 "abs expects 1 argument"
        return "abs($(arg_exprs[1]))"
        
    elseif func === sin
        @assert length(arg_exprs) == 1 "sin expects 1 argument"
        return "sin($(arg_exprs[1]))"
        
    elseif func === cos
        @assert length(arg_exprs) == 1 "cos expects 1 argument"
        return "cos($(arg_exprs[1]))"
        
    elseif func === tan
        @assert length(arg_exprs) == 1 "tan expects 1 argument"
        return "tan($(arg_exprs[1]))"
        
    elseif func === (^) && length(arg_exprs) == 2
        arg1, arg2 = arg_exprs[1], arg_exprs[2]
        return "($arg1^$arg2)"
        
    elseif func === (+) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) + $(arg_exprs[2]))"
        
    elseif func === (-) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) - $(arg_exprs[2]))"
        
    elseif func === (*) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) * $(arg_exprs[2]))"
        
    elseif func === (/) && length(arg_exprs) == 2
        arg1, arg2 = arg_exprs[1], arg_exprs[2]
        return "(abs($arg2) > 1e-16 ? $arg1 / $arg2 : ($arg1 == 0.0 ? NaN : ($arg1 > 0.0 ? Inf : -Inf)))"
        
    elseif func === (>) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) > $(arg_exprs[2]) ? 1.0 : 0.0)"
    elseif func === (<) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) < $(arg_exprs[2]) ? 1.0 : 0.0)"
    elseif func === (>=) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) >= $(arg_exprs[2]) ? 1.0 : 0.0)"
    elseif func === (<=) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) <= $(arg_exprs[2]) ? 1.0 : 0.0)"
    elseif func === (==) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) == $(arg_exprs[2]) ? 1.0 : 0.0)"
    elseif func === (!=) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) != $(arg_exprs[2]) ? 1.0 : 0.0)"
        
    else
        error("Function $func with $(length(arg_exprs)) arguments not yet supported. " *
              "Supported: log, exp, sqrt, abs, sin, cos, tan, ^, +, -, *, /, comparison ops")
    end
end

###############################################################################
# COMPLEX EXPRESSION HANDLING
###############################################################################

"""
For very complex nested expressions, we might need to break them into
multiple statements to avoid extremely long lines. This function detects
when an expression is getting too complex and breaks it down.
"""
function generate_complex_function_code!(instructions::Vector{String}, eval::FunctionEvaluator, pos::Int)
    # Check if any argument is too complex for inline generation
    complex_args = AbstractEvaluator[]
    simple_exprs = String[]
    
    for (i, arg_eval) in enumerate(eval.arg_evaluators)
        if is_simple_for_inline(arg_eval)
            push!(simple_exprs, generate_inline_expression(arg_eval))
        else
            push!(complex_args, arg_eval)
            # Generate temporary variable for complex argument
            temp_var = next_var("complex_arg")
            generate_evaluator_to_variable!(instructions, arg_eval, temp_var)
            push!(simple_exprs, temp_var)
        end
    end
    
    # Now generate the function call with mix of expressions and variables
    result_expr = generate_function_call(eval.func, simple_exprs)
    push!(instructions, "@inbounds row_vec[$pos] = $result_expr")
    
    return pos + 1
end

"""
Check if an evaluator is simple enough for inline expression generation.
"""
function is_simple_for_inline(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator || evaluator isa ContinuousEvaluator
        return true
    elseif evaluator isa CategoricalEvaluator
        return evaluator.n_levels <= 3  # Avoid very long ternary chains
    elseif evaluator isa FunctionEvaluator
        return length(evaluator.arg_evaluators) <= 2 && all(is_simple_for_inline, evaluator.arg_evaluators)
    else
        return false
    end
end

"""
Generate code to evaluate an evaluator into a single variable.
This is used as a fallback for complex expressions.
"""
function generate_evaluator_to_variable!(instructions::Vector{String}, evaluator::AbstractEvaluator, var_name::String)
    if evaluator isa ConstantEvaluator
        push!(instructions, "@inbounds $var_name = $(evaluator.value)")
    elseif evaluator isa ContinuousEvaluator
        push!(instructions, "@inbounds $var_name = Float64(data.$(evaluator.column)[row_idx])")
    elseif evaluator isa FunctionEvaluator
        # Use the new inline approach
        expr = generate_inline_expression(evaluator)
        push!(instructions, "@inbounds $var_name = $expr")
    else
        error("Variable generation not implemented for $(typeof(evaluator))")
    end
end

###############################################################################
# INTERACTION CODE GENERATION
###############################################################################

"""
    generate_interaction_code!(instructions, eval::InteractionEvaluator, pos)

Generate fully unrolled, allocation-free code for interaction terms.

# Strategy
Instead of computing Kronecker products dynamically, we pre-analyze the 
interaction structure and generate explicit code for each output position.

# Key Insight
An interaction x₁ × x₂ × ... × xₙ produces a Kronecker product where each
output element is the product of specific elements from each component.
We can pre-compute which elements multiply together and generate direct code.

# Example
For x * group (where group has 3 levels):
- Output[1] = x * group_contrast[1] 
- Output[2] = x * group_contrast[2]
- Output[3] = x * group_contrast[3]

Instead of computing this as a Kronecker product, we generate 3 separate
multiplication instructions.
"""
function generate_interaction_code!(instructions::Vector{String}, eval::InteractionEvaluator, pos::Int)
    n_components = length(eval.components)
    component_widths = [output_width(comp) for comp in eval.components]
    total_width = eval.total_width
    
    if n_components == 0
        # Empty interaction - shouldn't happen but handle gracefully
        return pos
        
    elseif n_components == 1
        # Single component - just delegate to regular generation
        return generate_evaluator_code!(instructions, eval.components[1], pos)
        
    elseif n_components == 2
        # Two-way interaction - optimize common case
        return generate_binary_interaction_unrolled!(instructions, eval.components, component_widths, pos)
        
    elseif n_components == 3
        # Three-way interaction - still manageable to unroll
        return generate_ternary_interaction_unrolled!(instructions, eval.components, component_widths, pos)
        
    else
        # High-order interactions - use general unrolling
        return generate_general_interaction_unrolled!(instructions, eval.components, component_widths, pos)
    end
end

###############################################################################
# BINARY INTERACTION UNROLLING (Most Common Case)
###############################################################################

"""
Generate unrolled code for two-way interactions like x * group.
This handles the most common case efficiently.
"""
function generate_binary_interaction_unrolled!(instructions::Vector{String}, 
                                              components::Vector{AbstractEvaluator}, 
                                              widths::Vector{Int}, 
                                              pos::Int)
    
    comp1, comp2 = components[1], components[2]
    w1, w2 = widths[1], widths[2]
    
    # Generate variable names for first component values
    comp1_vars = if w1 == 1
        [next_var("c1")]
    else
        [next_var("c1_$i") for i in 1:w1]
    end
    
    # Generate variable names for second component values  
    comp2_vars = if w2 == 1
        [next_var("c2")]
    else
        [next_var("c2_$i") for i in 1:w2]
    end
    
    # Generate code to evaluate first component
    generate_component_values!(instructions, comp1, comp1_vars)
    
    # Generate code to evaluate second component
    generate_component_values!(instructions, comp2, comp2_vars)
    
    # Generate unrolled Kronecker product: comp1[i] * comp2[j] for all i,j
    output_idx = pos
    for j in 1:w2
        for i in 1:w1
            push!(instructions, "@inbounds row_vec[$output_idx] = $(comp1_vars[i]) * $(comp2_vars[j])")
            output_idx += 1
        end
    end
    
    return output_idx
end

###############################################################################
# TERNARY INTERACTION UNROLLING
###############################################################################

"""
Generate unrolled code for three-way interactions like x * y * group.
Still manageable for reasonable component widths.
"""
function generate_ternary_interaction_unrolled!(instructions::Vector{String}, 
                                               components::Vector{AbstractEvaluator}, 
                                               widths::Vector{Int}, 
                                               pos::Int)
    
    comp1, comp2, comp3 = components[1], components[2], components[3]
    w1, w2, w3 = widths[1], widths[2], widths[3]
    
    # Check if unrolling is reasonable (avoid code explosion)
    total_terms = w1 * w2 * w3
    if total_terms > 100
        @warn "Three-way interaction has $total_terms terms, code may be large"
    end
    
    # Generate variable names for each component
    comp1_vars = [next_var("c1_$i") for i in 1:w1]
    comp2_vars = [next_var("c2_$i") for i in 1:w2] 
    comp3_vars = [next_var("c3_$i") for i in 1:w3]
    
    # Generate code to evaluate each component
    generate_component_values!(instructions, comp1, comp1_vars)
    generate_component_values!(instructions, comp2, comp2_vars)
    generate_component_values!(instructions, comp3, comp3_vars)
    
    # Generate triple-nested unrolled product
    output_idx = pos
    for k in 1:w3
        for j in 1:w2
            for i in 1:w1
                push!(instructions, "@inbounds row_vec[$output_idx] = $(comp1_vars[i]) * $(comp2_vars[j]) * $(comp3_vars[k])")
                output_idx += 1
            end
        end
    end
    
    return output_idx
end

###############################################################################
# GENERAL INTERACTION UNROLLING
###############################################################################

"""
Generate unrolled code for n-way interactions.
Uses index arithmetic to avoid exponential code generation.
"""
function generate_general_interaction_unrolled!(instructions::Vector{String}, 
                                               components::Vector{AbstractEvaluator}, 
                                               widths::Vector{Int}, 
                                               pos::Int)
    
    n_components = length(components)
    total_width = prod(widths)
    
    # Warn about potential code explosion
    if total_width > 200
        @warn "High-order interaction has $total_width terms, generating compact code"
        return generate_compact_interaction_code!(instructions, components, widths, pos)
    end
    
    # Generate variable names for each component
    all_component_vars = Vector{Vector{String}}(undef, n_components)
    for i in 1:n_components
        all_component_vars[i] = [next_var("c$(i)_$j") for j in 1:widths[i]]
        generate_component_values!(instructions, components[i], all_component_vars[i])
    end
    
    # Generate unrolled products using index arithmetic
    for linear_idx in 0:(total_width-1)
        # Convert linear index to multi-dimensional indices
        indices = linear_to_multi_index(linear_idx, widths)
        
        # Generate product expression
        product_terms = String[]
        for comp_idx in 1:n_components
            element_idx = indices[comp_idx] + 1  # Convert to 1-based
            push!(product_terms, all_component_vars[comp_idx][element_idx])
        end
        
        product_expr = join(product_terms, " * ")
        output_pos = pos + linear_idx
        push!(instructions, "@inbounds row_vec[$output_pos] = $product_expr")
    end
    
    return pos + total_width
end

###############################################################################
# COMPONENT VALUE GENERATION
###############################################################################

"""
    generate_component_values!(instructions, component, var_names)

Generate code to evaluate a single interaction component into named variables.
This is the allocation-free replacement for the old buffer-based approach.
"""
function generate_component_values!(instructions::Vector{String}, 
                                   component::AbstractEvaluator, 
                                   var_names::Vector{String})
    
    if component isa ConstantEvaluator
        # All variables get the same constant value
        for var_name in var_names
            push!(instructions, "@inbounds $var_name = $(component.value)")
        end
        
    elseif component isa ContinuousEvaluator
        # Single variable gets the data value (should only be one var_name)
        @assert length(var_names) == 1 "Continuous component should have width 1"
        push!(instructions, "@inbounds $(var_names[1]) = Float64(data.$(component.column)[row_idx])")
        
    elseif component isa CategoricalEvaluator
        # Generate categorical contrast lookup code
        generate_categorical_component_values!(instructions, component, var_names)
        
    elseif component isa FunctionEvaluator
        # Single variable gets the function result (should only be one var_name)
        @assert length(var_names) == 1 "Function component should have width 1"
        func_result = generate_function_expression(component)
        push!(instructions, "@inbounds $(var_names[1]) = $func_result")
        
    else
        @error "Unsupported component type in interaction: $(typeof(component))"
        error("Unsupported component type in interaction: $(typeof(component))")
    end
end

"""
Generate categorical contrast values efficiently.
"""
function generate_categorical_component_values!(instructions::Vector{String}, 
                                               component::CategoricalEvaluator, 
                                               var_names::Vector{String})
    
    col = component.column
    n_levels = component.n_levels
    contrast_matrix = component.contrast_matrix
    width = length(var_names)
    
    @assert width == size(contrast_matrix, 2) "Variable count must match contrast width"
    
    # Generate categorical lookup variables
    cat_var = next_var("cat")
    level_var = next_var("level")
    
    push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
    push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
    push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
    
    # Generate efficient contrast lookup for each variable
    for j in 1:width
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        var_name = var_names[j]
        
        if n_levels == 1
            push!(instructions, "@inbounds $var_name = $(values[1])")
        elseif n_levels == 2
            push!(instructions, "@inbounds $var_name = $level_var == 1 ? $(values[1]) : $(values[2])")
        elseif n_levels == 3
            push!(instructions, "@inbounds $var_name = $level_var == 1 ? $(values[1]) : $level_var == 2 ? $(values[2]) : $(values[3])")
        else
            # Chain of ternaries for more levels
            ternary_chain = "$level_var == 1 ? $(values[1])"
            for i in 2:(n_levels-1)
                ternary_chain *= " : $level_var == $i ? $(values[i])"
            end
            ternary_chain *= " : $(values[n_levels])"
            push!(instructions, "@inbounds $var_name = $ternary_chain")
        end
    end
end

"""
Generate inline function evaluation expression.
"""
function generate_function_expression(component::FunctionEvaluator)
    func = component.func
    n_args = length(component.arg_evaluators)
    
    if n_args == 1
        arg_expr = generate_argument_expression(component.arg_evaluators[1])
        return generate_function_call(func, [arg_expr])
    elseif n_args == 2
        arg1_expr = generate_argument_expression(component.arg_evaluators[1])
        arg2_expr = generate_argument_expression(component.arg_evaluators[2])
        return generate_function_call(func, [arg1_expr, arg2_expr])
    else
        # General case
        arg_exprs = [generate_argument_expression(arg) for arg in component.arg_evaluators]
        return generate_function_call(func, arg_exprs)
    end
end

"""
Generate expression for function argument.
"""
function generate_argument_expression(arg_evaluator::AbstractEvaluator)
    if arg_evaluator isa ConstantEvaluator
        return string(arg_evaluator.value)
    elseif arg_evaluator isa ContinuousEvaluator
        return "Float64(data.$(arg_evaluator.column)[row_idx])"
    else
        error("Complex argument type $(typeof(arg_evaluator)) not supported in function arguments")
    end
end

###############################################################################
# COMPACT CODE GENERATION FOR VERY LARGE INTERACTIONS
###############################################################################

"""
For interactions with hundreds of terms, generate compact loop-based code
that's still allocation-free but uses runtime loops instead of full unrolling.
"""
function generate_compact_interaction_code!(instructions::Vector{String}, 
                                           components::Vector{AbstractEvaluator}, 
                                           widths::Vector{Int}, 
                                           pos::Int)
    
    n_components = length(components)
    total_width = prod(widths)
    
    # Generate component evaluation code
    all_component_vars = Vector{Vector{String}}(undef, n_components)
    for i in 1:n_components
        all_component_vars[i] = [next_var("c$(i)_$j") for j in 1:widths[i]]
        generate_component_values!(instructions, components[i], all_component_vars[i])
    end
    
    # Generate compact nested loops (still no allocations)
    if n_components == 2
        w1, w2 = widths[1], widths[2]
        output_idx = pos
        push!(instructions, "@inbounds for j in 1:$w2")
        push!(instructions, "@inbounds   for i in 1:$w1")
        push!(instructions, "@inbounds     row_vec[$output_idx] = $(all_component_vars[1])[i] * $(all_component_vars[2])[j]")
        push!(instructions, "@inbounds     $output_idx += 1")
        push!(instructions, "@inbounds   end")
        push!(instructions, "@inbounds end")
    else
        # For higher dimensions, fall back to index arithmetic
        errors("higher dimension error")
        # push!(instructions, "@inbounds for linear_idx in 0:$(total_width-1)")
        # push!(instructions, "@inbounds   # Multi-index computation and product would go here")
        # push!(instructions, "@inbounds   row_vec[$(pos) + linear_idx] = 1.0  # Placeholder")
        # push!(instructions, "@inbounds end")
    end
    
    return pos + total_width
end

###############################################################################
# ZSCORE AND COMBINED CODE GENERATION
###############################################################################

function generate_zscore_code!(instructions::Vector{String}, eval::ZScoreEvaluator, pos::Int)
    center = eval.center
    scale = eval.scale
    
    # Generate code for underlying evaluator
    underlying_width = output_width(eval.underlying)
    
    if underlying_width == 1
        # Single-column case - generate inline
        temp_var = next_var("zscore_temp")
        
        if eval.underlying isa ContinuousEvaluator
            col = eval.underlying.column
            push!(instructions, "@inbounds $temp_var = Float64(data.$col[row_idx])")
        elseif eval.underlying isa FunctionEvaluator
            # Generate function evaluation
            arg_vars = [next_var("zarg") for _ in eval.underlying.arg_evaluators]
            for (i, arg_eval) in enumerate(eval.underlying.arg_evaluators)
                generate_argument_code!(instructions, arg_eval, arg_vars[i])
            end
            result_expr = generate_function_call(eval.underlying.func, arg_vars)
            push!(instructions, "@inbounds $temp_var = $result_expr")
        else
            error("ZScore fallback")
        end
        
        # Apply Z-score transformation
        push!(instructions, "@inbounds row_vec[$pos] = ($temp_var - $center) / $scale")
        
        return pos + 1
    else
        # Multi-column case - recursively generate underlying code then transform
        # For now, simplified handling
        next_pos = generate_evaluator_code!(instructions, eval.underlying, pos)
        
        # Apply Z-score to each generated column
        for i in 0:(underlying_width-1)
            col_pos = pos + i
            temp_var = next_var("zscore_col")
            push!(instructions, "@inbounds $temp_var = row_vec[$col_pos]")
            push!(instructions, "@inbounds row_vec[$col_pos] = ($temp_var - $center) / $scale")
        end
        
        return next_pos
    end
end

function generate_combined_code!(instructions::Vector{String}, eval::CombinedEvaluator, pos::Int)
    current_pos = pos
    
    for sub_eval in eval.sub_evaluators
        current_pos = generate_evaluator_code!(instructions, sub_eval, current_pos)
    end
    
    return current_pos
end

###############################################################################
# SCALED AND PRODUCT CODE GENERATION
###############################################################################

"""
    generate_scaled_code!(instructions, eval::ScaledEvaluator, pos)

Generate code for ScaledEvaluator that properly scales ALL output positions.

# Problem with Original
The original implementation only scaled row_vec[pos], but the inner evaluator
might write to multiple positions (e.g., categorical with 3 levels writes to
pos, pos+1, pos+2). This left positions pos+1, pos+2, ... unscaled.

# Fix
Scale all positions from pos to next_pos-1 that the inner evaluator wrote to.

# Example
For categorical with 3 levels and scale factor 2.5:
- Inner evaluator writes to row_vec[pos], row_vec[pos+1], row_vec[pos+2]  
- We need to scale ALL three positions, not just row_vec[pos]
"""
function generate_scaled_code!(instructions::Vector{String}, eval::ScaledEvaluator, pos::Int)
    # Generate code for inner evaluator first
    next_pos = generate_evaluator_code!(instructions, eval.evaluator, pos)
    
    # Scale ALL positions that the inner evaluator wrote to
    width = next_pos - pos  # Number of positions written
    
    if width == 1
        # Single position - direct scaling (most common case)
        push!(instructions, "@inbounds row_vec[$pos] *= $(eval.scale_factor)")
    else
        # Multiple positions - scale each one
        for i in 0:(width-1)
            current_pos = pos + i
            push!(instructions, "@inbounds row_vec[$current_pos] *= $(eval.scale_factor)")
        end
    end
    
    return next_pos
end

"""
Test case that demonstrates the bug and verifies the fix:

```julia
# Create a categorical with 3 levels scaled by 2.0
cat_eval = CategoricalEvaluator(:group, contrast_matrix, 3)
scaled_eval = ScaledEvaluator(cat_eval, 2.0)

# The inner categorical writes to positions [pos, pos+1, pos+2]
# The scale should apply to ALL three positions
```

Before fix: Only row_vec[pos] gets scaled
After fix: All row_vec[pos], row_vec[pos+1], row_vec[pos+2] get scaled
"""

function generate_product_code!(instructions::Vector{String}, eval::ProductEvaluator, pos::Int)
    component_vars = [next_var("prod_comp") for _ in eval.components]
    
    for (i, component) in enumerate(eval.components)
        generate_argument_code!(instructions, component, component_vars[i])
    end
    
    product_expr = join(component_vars, " * ")
    push!(instructions, "@inbounds row_vec[$pos] = $product_expr")
    return pos + 1
end
