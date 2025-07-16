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
        return generate_function_code!(instructions, evaluator, pos)
        
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

function generate_function_code!(instructions::Vector{String}, eval::FunctionEvaluator, pos::Int)
    func = eval.func
    n_args = length(eval.arg_evaluators)
    
    # Generate unique variable names for function arguments
    arg_vars = [next_var("arg") for _ in 1:n_args]
    
    # Generate code to evaluate each argument
    for (i, arg_eval) in enumerate(eval.arg_evaluators)
        generate_argument_code!(instructions, arg_eval, arg_vars[i])
    end
    
    # Generate function application with safety checks
    result_expr = generate_function_call(func, arg_vars)
    push!(instructions, "@inbounds row_vec[$pos] = $result_expr")
    
    return pos + 1
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
        # Fallback for complex arguments
        push!(instructions, "@inbounds $var_name = 1.0  # Complex argument fallback")
    end
end

function generate_function_call(func::Function, arg_vars::Vector{String})
    if func === log
        return "$(arg_vars[1]) > 0.0 ? log($(arg_vars[1])) : log(abs($(arg_vars[1])) + 1e-16)"
    elseif func === exp
        return "exp(clamp($(arg_vars[1]), -700.0, 700.0))"
    elseif func === sqrt
        return "sqrt(abs($(arg_vars[1])))"
    elseif func === abs
        return "abs($(arg_vars[1]))"
    elseif func === (^) && length(arg_vars) == 2
        return "$(arg_vars[1])^$(arg_vars[2])"
    elseif func === (+) && length(arg_vars) == 2
        return "$(arg_vars[1]) + $(arg_vars[2])"
    elseif func === (-) && length(arg_vars) == 2
        return "$(arg_vars[1]) - $(arg_vars[2])"
    elseif func === (*) && length(arg_vars) == 2
        return "$(arg_vars[1]) * $(arg_vars[2])"
    elseif func === (/) && length(arg_vars) == 2
        return "abs($(arg_vars[2])) > 1e-16 ? $(arg_vars[1]) / $(arg_vars[2]) : $(arg_vars[1])"
    elseif func === (>) && length(arg_vars) == 2
        return "$(arg_vars[1]) > $(arg_vars[2]) ? 1.0 : 0.0"
    elseif func === (<) && length(arg_vars) == 2
        return "$(arg_vars[1]) < $(arg_vars[2]) ? 1.0 : 0.0"
    elseif func === (>=) && length(arg_vars) == 2
        return "$(arg_vars[1]) >= $(arg_vars[2]) ? 1.0 : 0.0"
    elseif func === (<=) && length(arg_vars) == 2
        return "$(arg_vars[1]) <= $(arg_vars[2]) ? 1.0 : 0.0"
    elseif func === (==) && length(arg_vars) == 2
        return "$(arg_vars[1]) == $(arg_vars[2]) ? 1.0 : 0.0"
    elseif func === (!=) && length(arg_vars) == 2
        return "$(arg_vars[1]) != $(arg_vars[2]) ? 1.0 : 0.0"
    else
        # General function - handles ANY user-defined function
        args_str = join(arg_vars, ", ")
        return "$func($args_str)"
    end
end

###############################################################################
# INTERACTION CODE GENERATION
###############################################################################

function generate_interaction_code!(instructions::Vector{String}, eval::InteractionEvaluator, pos::Int)
    n_components = length(eval.components)
    component_widths = [output_width(comp) for comp in eval.components]
    total_width = eval.total_width
    
    # Generate unique identifiers for this interaction
    interaction_id = next_var("int")
    
    # Evaluate each component into variables
    component_vars = Vector{Vector{String}}(undef, n_components)
    
    for (i, component) in enumerate(eval.components)
        width = component_widths[i]
        component_vars[i] = Vector{String}(undef, width)
        
        if width == 1
            # Single-column component
            var_name = next_var("comp")
            component_vars[i][1] = var_name
            generate_single_component_code!(instructions, component, var_name)
        else
            # Multi-column component
            generate_multi_component_code!(instructions, component, component_vars[i], interaction_id)
        end
    end
    
    # Generate Kronecker product code
    generate_kronecker_code!(instructions, component_vars, component_widths, pos)
    
    return pos + total_width
end

function generate_single_component_code!(instructions::Vector{String}, component::AbstractEvaluator, var_name::String)
    if component isa ContinuousEvaluator
        push!(instructions, "@inbounds $var_name = Float64(data.$(component.column)[row_idx])")
        
    elseif component isa ConstantEvaluator
        push!(instructions, "@inbounds $var_name = $(component.value)")
        
    elseif component isa CategoricalEvaluator
        # Use first contrast column for single-width categorical
        col = component.column
        n_levels = component.n_levels
        values = [component.contrast_matrix[i, 1] for i in 1:n_levels]
        
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
        
    elseif component isa FunctionEvaluator
        # Generate function evaluation code
        arg_vars = [next_var("farg") for _ in component.arg_evaluators]
        for (i, arg_eval) in enumerate(component.arg_evaluators)
            generate_argument_code!(instructions, arg_eval, arg_vars[i])
        end
        result_expr = generate_function_call(component.func, arg_vars)
        push!(instructions, "@inbounds $var_name = $result_expr")
        
    else
        push!(instructions, "@inbounds $var_name = 1.0  # Component fallback")
    end
end

function generate_multi_component_code!(instructions::Vector{String}, component::AbstractEvaluator, var_names::Vector{String}, interaction_id::String)
    if component isa CategoricalEvaluator
        # Generate code for multi-column categorical
        col = component.column
        n_levels = component.n_levels
        contrast_matrix = component.contrast_matrix
        width = length(var_names)
        
        cat_var = next_var("cat")
        level_var = next_var("level")
        
        push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
        push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
        push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
        
        for j in 1:width
            var_names[j] = next_var("multicomp")
            values = [contrast_matrix[i, j] for i in 1:n_levels]
            
            if n_levels == 2
                push!(instructions, "@inbounds $(var_names[j]) = $level_var == 1 ? $(values[1]) : $(values[2])")
            else
                ternary_chain = "$level_var == 1 ? $(values[1])"
                for i in 2:(n_levels-1)
                    ternary_chain *= " : $level_var == $i ? $(values[i])"
                end
                ternary_chain *= " : $(values[n_levels])"
                push!(instructions, "@inbounds $(var_names[j]) = $ternary_chain")
            end
        end
    else
        # Fallback for other multi-column components
        for j in 1:length(var_names)
            var_names[j] = next_var("multicomp")
            push!(instructions, "@inbounds $(var_names[j]) = 1.0  # Multi-component fallback")
        end
    end
end

function generate_kronecker_code!(instructions::Vector{String}, component_vars::Vector{Vector{String}}, component_widths::Vector{Int}, pos::Int)
    n_components = length(component_vars)
    
    if n_components == 1
        # Single component - direct assignment
        for (i, var) in enumerate(component_vars[1])
            push!(instructions, "@inbounds row_vec[$(pos + i - 1)] = $var")
        end
        
    elseif all(length(vars) == 1 for vars in component_vars)
        # All single-column components - simple product
        product_expr = join([vars[1] for vars in component_vars], " * ")
        push!(instructions, "@inbounds row_vec[$pos] = $product_expr")
        
    elseif n_components == 2
        # Two components - direct Kronecker
        w1, w2 = component_widths[1], component_widths[2]
        vars1, vars2 = component_vars[1], component_vars[2]
        
        output_idx = pos
        for j in 1:w2
            for i in 1:w1
                push!(instructions, "@inbounds row_vec[$output_idx] = $(vars1[i]) * $(vars2[j])")
                output_idx += 1
            end
        end
        
    elseif n_components == 3
        # Three components - triple loop
        w1, w2, w3 = component_widths[1], component_widths[2], component_widths[3]
        vars1, vars2, vars3 = component_vars[1], component_vars[2], component_vars[3]
        
        output_idx = pos
        for k in 1:w3
            for j in 1:w2
                for i in 1:w1
                    push!(instructions, "@inbounds row_vec[$output_idx] = $(vars1[i]) * $(vars2[j]) * $(vars3[k])")
                    output_idx += 1
                end
            end
        end
        
    else
        # General case - generate index computation code
        generate_general_kronecker_code!(instructions, component_vars, component_widths, pos)
    end
end

function generate_general_kronecker_code!(instructions::Vector{String}, component_vars::Vector{Vector{String}}, component_widths::Vector{Int}, pos::Int)
    # For general N-way interactions, generate nested loops
    n_components = length(component_vars)
    total_width = prod(component_widths)
    
    # Generate code that computes the Kronecker product element by element
    for linear_idx in 0:(total_width-1)
        indices = linear_to_multi_index(linear_idx, component_widths) .+ 1
        
        product_terms = String[]
        for (comp_idx, var_idx) in enumerate(indices)
            push!(product_terms, component_vars[comp_idx][var_idx])
        end
        
        product_expr = join(product_terms, " * ")
        output_pos = pos + linear_idx
        push!(instructions, "@inbounds row_vec[$output_pos] = $product_expr")
    end
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
            push!(instructions, "@inbounds $temp_var = 1.0  # ZScore fallback")
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

function generate_scaled_code!(instructions::Vector{String}, eval::ScaledEvaluator, pos::Int)
    next_pos = generate_evaluator_code!(instructions, eval.evaluator, pos)
    push!(instructions, "@inbounds row_vec[$pos] *= $(eval.scale_factor)")
    return next_pos
end

function generate_product_code!(instructions::Vector{String}, eval::ProductEvaluator, pos::Int)
    component_vars = [next_var("prod_comp") for _ in eval.components]
    
    for (i, component) in enumerate(eval.components)
        generate_argument_code!(instructions, component, component_vars[i])
    end
    
    product_expr = join(component_vars, " * ")
    push!(instructions, "@inbounds row_vec[$pos] = $product_expr")
    return pos + 1
end
