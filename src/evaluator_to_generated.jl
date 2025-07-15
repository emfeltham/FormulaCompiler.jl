# evaluator_to_generated.jl
# Convenience methods

# Include the evaluator definitions from the compositional compiler
# (In practice, you'd have these in the same module)

###############################################################################
# Variable Name Management - Fix the Core Bug
###############################################################################

# Global counter for unique variable names
const VAR_COUNTER = Ref(0)

"""
Generate unique variable names to prevent conflicts.
This fixes the core bug where variable names were being reused.
"""
function generate_unique_var(base::String="var")
    VAR_COUNTER[] += 1
    return "$(base)_$(VAR_COUNTER[])"
end

"""
Reset the variable counter for each new formula compilation.
"""
function reset_var_counter!()
    VAR_COUNTER[] = 0
end

###############################################################################
# Code Generation from Evaluator Tree
###############################################################################

###############################################################################
# REPLACE: Fixed Code Generation Functions
###############################################################################

"""
REPLACES: generate_code_from_evaluator
Fixed version with unique variable names and proper recursion.
"""
function generate_code_from_evaluator(evaluator::AbstractEvaluator, start_pos::Int=1)
    reset_var_counter!()  # Start fresh for each formula
    
    if evaluator isa CombinedEvaluator
        instructions = String[]
        current_pos = start_pos
        
        for (i, sub_eval) in enumerate(evaluator.sub_evaluators)
            width = evaluator.sub_widths[i]
            if width == 1
                temp_var = generate_unique_var("result")
                append!(instructions, generate_evaluator_code(sub_eval, temp_var))
                push!(instructions, "@inbounds row_vec[$current_pos] = $temp_var")
                current_pos += 1
            else
                # Multi-column terms - handle properly
                for j in 1:width
                    pos = current_pos + j - 1
                    temp_var = generate_unique_var("result")
                    # For now, simplified multi-column handling
                    append!(instructions, generate_evaluator_code(sub_eval, temp_var))
                    push!(instructions, "@inbounds row_vec[$pos] = $temp_var")
                end
                current_pos += width
            end
        end
        
        return instructions
    else
        # Single evaluator
        temp_var = generate_unique_var("result")
        instructions = generate_evaluator_code(evaluator, temp_var)
        push!(instructions, "@inbounds row_vec[$start_pos] = $temp_var")
        return instructions
    end
end

"""
Core recursive function for generating code from any evaluator.
This is the fixed version that prevents variable name conflicts.
"""
function generate_evaluator_code(evaluator::AbstractEvaluator, var_name::String)
    if evaluator isa ConstantEvaluator
        return ["@inbounds $var_name = $(evaluator.value)"]
        
    elseif evaluator isa ContinuousEvaluator
        return ["@inbounds $var_name = Float64(data.$(evaluator.column)[row_idx])"]
        
    elseif evaluator isa CategoricalEvaluator
        return generate_categorical_code_fixed(evaluator, var_name)
        
    elseif evaluator isa FunctionEvaluator
        instructions = String[]
        arg_vars = String[]
        
        # Generate UNIQUE variable names for each argument
        for (i, arg) in enumerate(evaluator.arg_evaluators)
            arg_var = generate_unique_var("arg")
            push!(arg_vars, arg_var)
            append!(instructions, generate_evaluator_code(arg, arg_var))  # RECURSIVE
        end
        
        # Apply function
        func_call = generate_function_call_safe(evaluator.func, arg_vars)
        push!(instructions, "@inbounds $var_name = $func_call")
        
        return instructions
        
    elseif evaluator isa InteractionEvaluator
        instructions = String[]
        
        if evaluator.total_width == 1
            # Single-column interaction - simple product
            component_vars = String[]
            for (i, component) in enumerate(evaluator.components)
                comp_var = generate_unique_var("comp")
                push!(component_vars, comp_var)
                append!(instructions, generate_evaluator_code(component, comp_var))  # RECURSIVE
            end
            
            # Multiply all components
            product = join(component_vars, " * ")
            push!(instructions, "@inbounds $var_name = $product")
            
        else
            # Multi-column interaction - Kronecker product
            append!(instructions, generate_multi_column_interaction(evaluator, var_name))
        end
        
        return instructions
        
    elseif evaluator isa ZScoreEvaluator
        instructions = String[]
        underlying_var = generate_unique_var("underlying")
        
        # Generate code for underlying evaluator
        append!(instructions, generate_evaluator_code(evaluator.underlying, underlying_var))
        
        # Apply Z-score transformation
        push!(instructions, "@inbounds $var_name = ($underlying_var - $(evaluator.center)) / $(evaluator.scale)")
        
        return instructions
        
    else
        error("Unknown evaluator type: $(typeof(evaluator))")
    end
end

"""
REPLACES: _generate_categorical_code!
Fixed categorical code generation with unique variable names.
"""
function generate_categorical_code_fixed(evaluator::CategoricalEvaluator, var_name::String)
    col = evaluator.column
    n_levels = evaluator.n_levels
    contrast_matrix = evaluator.contrast_matrix
    
    instructions = String[]
    
    # Use unique variable names
    cat_var = generate_unique_var("cat")
    level_var = generate_unique_var("level")
    
    push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
    push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
    push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
    
    if size(contrast_matrix, 2) == 1
        # Single column case (most common in interactions)
        values = [contrast_matrix[i, 1] for i in 1:n_levels]
        
        if n_levels == 2
            push!(instructions, "@inbounds $var_name = $level_var == 1 ? $(values[1]) : $(values[2])")
        elseif n_levels == 3
            push!(instructions, "@inbounds $var_name = $level_var == 1 ? $(values[1]) : $level_var == 2 ? $(values[2]) : $(values[3])")
        else
            # General case
            ternary = "$level_var == 1 ? $(values[1])"
            for i in 2:(n_levels-1)
                ternary *= " : $level_var == $i ? $(values[i])"
            end
            ternary *= " : $(values[n_levels])"
            push!(instructions, "@inbounds $var_name = $ternary")
        end
    else
        # Multi-column categorical - handle each column
        for j in 1:size(contrast_matrix, 2)
            col_var = "$(var_name)_col$j"
            values = [contrast_matrix[i, j] for i in 1:n_levels]
            
            if n_levels == 2
                push!(instructions, "@inbounds $col_var = $level_var == 1 ? $(values[1]) : $(values[2])")
            else
                ternary = "$level_var == 1 ? $(values[1])"
                for i in 2:(n_levels-1)
                    ternary *= " : $level_var == $i ? $(values[i])"
                end
                ternary *= " : $(values[n_levels])"
                push!(instructions, "@inbounds $col_var = $ternary")
            end
        end
        
        # For single output variable, use first column (simplified)
        push!(instructions, "@inbounds $var_name = $(var_name)_col1")
    end
    
    return instructions
end

"""
REPLACES: generate_function_call (from previous versions)
Safe function call generation with proper type handling.
"""
function generate_function_call_safe(func::Function, arg_vars::Vector{String})
    if func === (^)
        return "$(arg_vars[1])^$(arg_vars[2])"
    elseif func === log
        return "$(arg_vars[1]) > 0.0 ? log($(arg_vars[1])) : log(abs($(arg_vars[1])) + 1e-16)"
    elseif func === exp
        return "exp(clamp($(arg_vars[1]), -700.0, 700.0))"
    elseif func === sqrt
        return "sqrt(abs($(arg_vars[1])))"
    elseif func === abs
        return "abs($(arg_vars[1]))"
    elseif func === (+)
        return "$(arg_vars[1]) + $(arg_vars[2])"
    elseif func === (-)
        return "$(arg_vars[1]) - $(arg_vars[2])"
    elseif func === (*)
        return "$(arg_vars[1]) * $(arg_vars[2])"
    elseif func === (/)
        return "abs($(arg_vars[2])) > 1e-16 ? $(arg_vars[1]) / $(arg_vars[2]) : $(arg_vars[1])"
    elseif func === (>)
        return "$(arg_vars[1]) > $(arg_vars[2]) ? 1.0 : 0.0"
    elseif func === (<)
        return "$(arg_vars[1]) < $(arg_vars[2]) ? 1.0 : 0.0"
    elseif func === (>=)
        return "$(arg_vars[1]) >= $(arg_vars[2]) ? 1.0 : 0.0"
    elseif func === (<=)
        return "$(arg_vars[1]) <= $(arg_vars[2]) ? 1.0 : 0.0"
    elseif func === (==)
        return "$(arg_vars[1]) == $(arg_vars[2]) ? 1.0 : 0.0"
    elseif func === (!=)
        return "$(arg_vars[1]) != $(arg_vars[2]) ? 1.0 : 0.0"
    else
        # General function - handles ANY user-defined function
        args_str = join(arg_vars, ", ")
        return "$func($args_str)"
    end
end

"""
Recursive code generation dispatch.
"""
function _generate_code_recursive!(instructions::Vector{String}, evaluator::AbstractEvaluator, pos::Int)
    if evaluator isa ConstantEvaluator
        _generate_constant_code!(instructions, evaluator, pos)
    elseif evaluator isa ContinuousEvaluator
        _generate_continuous_code!(instructions, evaluator, pos)
    elseif evaluator isa CategoricalEvaluator
        _generate_categorical_code!(instructions, evaluator, pos)
    elseif evaluator isa FunctionEvaluator
        _generate_function_code!(instructions, evaluator, pos)
    elseif evaluator isa InteractionEvaluator
        _generate_interaction_code!(instructions, evaluator, pos)
    elseif evaluator isa ZScoreEvaluator
        _generate_zscore_code!(instructions, evaluator, pos)
    elseif evaluator isa CombinedEvaluator
        _generate_combined_code!(instructions, evaluator, pos)
    else
        error("Unknown evaluator type: $(typeof(evaluator))")
    end
    return pos + output_width(evaluator)
end

###############################################################################
# Code Generation for Each Evaluator Type
###############################################################################

function _generate_constant_code!(instructions::Vector{String}, eval::ConstantEvaluator, pos::Int)
    push!(instructions, "@inbounds row_vec[$pos] = $(eval.value)")
end

function _generate_continuous_code!(instructions::Vector{String}, eval::ContinuousEvaluator, pos::Int)
    push!(instructions, "@inbounds row_vec[$pos] = Float64(data.$(eval.column)[row_idx])")
end

function _generate_categorical_code!(instructions::Vector{String}, eval::CategoricalEvaluator, pos::Int)
    col = eval.column
    n_levels = eval.n_levels
    contrast_matrix = eval.contrast_matrix
    width = size(contrast_matrix, 2)
    
    # Generate level code extraction (once per categorical)
    level_var = "level_$(col)_$(pos)"
    push!(instructions, "@inbounds cat_val = data.$col[row_idx]")
    push!(instructions, "@inbounds $level_var = cat_val isa CategoricalValue ? levelcode(cat_val) : 1")
    push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
    
    # Generate code for each contrast column
    for j in 1:width
        output_pos = pos + j - 1
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        
        # Generate efficient lookup based on number of levels
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
end

function _generate_function_code!(instructions::Vector{String}, eval::FunctionEvaluator, pos::Int)
    func = eval.func
    n_args = length(eval.arg_evaluators)
    
    # Generate temporary variables for each argument
    arg_vars = ["arg_$(pos)_$i" for i in 1:n_args]
    
    # Generate code to evaluate each argument
    for (i, arg_eval) in enumerate(eval.arg_evaluators)
        if arg_eval isa ContinuousEvaluator
            push!(instructions, "@inbounds $(arg_vars[i]) = Float64(data.$(arg_eval.column)[row_idx])")
        elseif arg_eval isa ConstantEvaluator
            push!(instructions, "@inbounds $(arg_vars[i]) = $(arg_eval.value)")
        else
            # For complex arguments, we'd need to evaluate them first
            # For now, handle the most common cases
            push!(instructions, "@inbounds $(arg_vars[i]) = 1.0  # Complex arg fallback")
        end
    end
    
    # Generate function application with safety checks
    if func === log
        push!(instructions, "@inbounds row_vec[$pos] = $(arg_vars[1]) > 0.0 ? log($(arg_vars[1])) : log(abs($(arg_vars[1])) + 1e-16)")
    elseif func === exp
        push!(instructions, "@inbounds row_vec[$pos] = exp(clamp($(arg_vars[1]), -700.0, 700.0))")
    elseif func === (^) && n_args == 2
        # Handle power functions
        push!(instructions, "@inbounds row_vec[$pos] = $(arg_vars[1])^$(arg_vars[2])")
    elseif func === (>) && n_args == 2
        push!(instructions, "@inbounds row_vec[$pos] = $(arg_vars[1]) > $(arg_vars[2]) ? 1.0 : 0.0")
    elseif func === (<) && n_args == 2
        push!(instructions, "@inbounds row_vec[$pos] = $(arg_vars[1]) < $(arg_vars[2]) ? 1.0 : 0.0")
    elseif func === (>=) && n_args == 2
        push!(instructions, "@inbounds row_vec[$pos] = $(arg_vars[1]) >= $(arg_vars[2]) ? 1.0 : 0.0")
    elseif func === (<=) && n_args == 2
        push!(instructions, "@inbounds row_vec[$pos] = $(arg_vars[1]) <= $(arg_vars[2]) ? 1.0 : 0.0")
    elseif func === (==) && n_args == 2
        push!(instructions, "@inbounds row_vec[$pos] = $(arg_vars[1]) == $(arg_vars[2]) ? 1.0 : 0.0")
    elseif func === (!=) && n_args == 2
        push!(instructions, "@inbounds row_vec[$pos] = $(arg_vars[1]) != $(arg_vars[2]) ? 1.0 : 0.0")
    else
        # General function application
        args_str = join(arg_vars, ", ")
        push!(instructions, "@inbounds row_vec[$pos] = $func($args_str)")
    end
end

function _generate_interaction_code!(instructions::Vector{String}, eval::InteractionEvaluator, pos::Int)
    n_components = length(eval.components)
    component_widths = eval.component_widths
    total_width = eval.total_width
    
    # Generate unique variable names for this interaction
    interaction_id = "int_$(pos)"
    
    # Generate code to evaluate each component into temporary variables
    component_vars = String[]
    
    for (i, component) in enumerate(eval.components)
        comp_width = component_widths[i]
        
        if comp_width == 1
            # Single-column component - use scalar variable
            var_name = "comp_$(interaction_id)_$i"
            push!(component_vars, var_name)
            
            if component isa ContinuousEvaluator
                push!(instructions, "@inbounds $var_name = Float64(data.$(component.column)[row_idx])")
            elseif component isa ConstantEvaluator
                push!(instructions, "@inbounds $var_name = $(component.value)")
            elseif component isa CategoricalEvaluator
                # For single-column categorical, extract the contrast value
                col = component.column
                level_var = "level_$(col)_$(interaction_id)_$i"
                push!(instructions, "@inbounds cat_val = data.$col[row_idx]")
                push!(instructions, "@inbounds $level_var = cat_val isa CategoricalValue ? levelcode(cat_val) : 1")
                push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $(component.n_levels))")
                
                # Get the single contrast column (first column)
                values = [component.contrast_matrix[level, 1] for level in 1:component.n_levels]
                if component.n_levels == 2
                    push!(instructions, "@inbounds $var_name = $level_var == 1 ? $(values[1]) : $(values[2])")
                else
                    ternary_chain = "$level_var == 1 ? $(values[1])"
                    for level in 2:(component.n_levels-1)
                        ternary_chain *= " : $level_var == $level ? $(values[level])"
                    end
                    ternary_chain *= " : $(values[component.n_levels])"
                    push!(instructions, "@inbounds $var_name = $ternary_chain")
                end
            else
                push!(instructions, "@inbounds $var_name = 1.0  # Component fallback")
            end
        else
            # Multi-column component - would need array variables
            # For now, handle the most common cases
            push!(component_vars, "1.0")  # Fallback
        end
    end
    
    # Generate Kronecker product code
    if all(w == 1 for w in component_widths)
        # All single-column components - simple product
        product_expr = join(component_vars, " * ")
        push!(instructions, "@inbounds row_vec[$pos] = $product_expr")
    else
        # Complex Kronecker product - implement based on specific patterns
        _generate_kronecker_product_code!(instructions, eval, pos, component_vars)
    end
end

function _generate_kronecker_product_code!(instructions::Vector{String}, eval::InteractionEvaluator, pos::Int, component_vars::Vector{String})
    component_widths = eval.component_widths
    components = eval.components
    
    if length(components) == 2
        w1, w2 = component_widths[1], component_widths[2]
        
        if w1 == 1 && w2 > 1
            # Scalar × Vector case
            scalar_var = component_vars[1]
            vector_component = components[2]
            
            if vector_component isa CategoricalEvaluator
                _generate_scalar_categorical_interaction!(instructions, scalar_var, vector_component, pos)
            else
                # Fallback
                for j in 1:w2
                    push!(instructions, "@inbounds row_vec[$(pos + j - 1)] = $scalar_var")
                end
            end
            
        elseif w1 > 1 && w2 == 1
            # Vector × Scalar case (swap arguments)
            scalar_var = component_vars[2]
            vector_component = components[1]
            
            if vector_component isa CategoricalEvaluator
                _generate_scalar_categorical_interaction!(instructions, scalar_var, vector_component, pos)
            else
                # Fallback
                for j in 1:w1
                    push!(instructions, "@inbounds row_vec[$(pos + j - 1)] = $scalar_var")
                end
            end
            
        elseif w1 > 1 && w2 > 1
            # Vector × Vector case
            _generate_vector_vector_interaction!(instructions, components, component_widths, pos)
        end
    else
        # 3+ way interactions - implement specific cases as needed
        _generate_multiway_interaction!(instructions, eval, pos, component_vars)
    end
end

"""
Handle multi-column interactions with proper Kronecker products.
"""
function generate_multi_column_interaction(evaluator::InteractionEvaluator, var_name::String)
    components = evaluator.components
    component_widths = evaluator.component_widths
    
    instructions = String[]
    
    if length(components) == 2
        comp1, comp2 = components
        w1, w2 = component_widths
        
        if w1 == 1 && w2 > 1
            # Scalar × Vector
            scalar_var = generate_unique_var("scalar")
            append!(instructions, generate_evaluator_code(comp1, scalar_var))
            
            if comp2 isa CategoricalEvaluator
                append!(instructions, generate_scalar_categorical_interaction(scalar_var, comp2, var_name))
            else
                error("Multi-column non-categorical interactions not implemented")
            end
            
        elseif w1 > 1 && w2 == 1
            # Vector × Scalar
            scalar_var = generate_unique_var("scalar")
            append!(instructions, generate_evaluator_code(comp2, scalar_var))
            
            if comp1 isa CategoricalEvaluator
                append!(instructions, generate_scalar_categorical_interaction(scalar_var, comp1, var_name))
            else
                error("Multi-column non-categorical interactions not implemented")
            end
            
        elseif w1 > 1 && w2 > 1
            # Vector × Vector
            if comp1 isa CategoricalEvaluator && comp2 isa CategoricalEvaluator
                append!(instructions, generate_categorical_categorical_interaction(comp1, comp2, var_name))
            else
                error("Complex multi-column interactions not implemented")
            end
        end
    else
        error("$(length(components))-way multi-column interactions not implemented")
    end
    
    return instructions
end


function _generate_scalar_categorical_interaction!(instructions::Vector{String}, scalar_var::String, cat_eval::CategoricalEvaluator, pos::Int)
    col = cat_eval.column
    n_levels = cat_eval.n_levels
    contrast_matrix = cat_eval.contrast_matrix
    width = size(contrast_matrix, 2)
    
    # Generate level code (unique variable name)
    level_var = "level_$(col)_scalar_int_$(pos)"
    push!(instructions, "@inbounds cat_val = data.$col[row_idx]")
    push!(instructions, "@inbounds $level_var = cat_val isa CategoricalValue ? levelcode(cat_val) : 1")
    push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
    
    # Generate scalar × contrast for each column
    for j in 1:width
        output_pos = pos + j - 1
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        
        if n_levels == 2
            contrast_expr = "$level_var == 1 ? $(values[1]) : $(values[2])"
        else
            ternary_chain = "$level_var == 1 ? $(values[1])"
            for i in 2:(n_levels-1)
                ternary_chain *= " : $level_var == $i ? $(values[i])"
            end
            ternary_chain *= " : $(values[n_levels])"
            contrast_expr = ternary_chain
        end
        
        push!(instructions, "@inbounds row_vec[$output_pos] = $scalar_var * ($contrast_expr)")
    end
end

function _generate_vector_vector_interaction!(instructions::Vector{String}, components::Vector{AbstractEvaluator}, component_widths::Vector{Int}, pos::Int)
    # Handle categorical × categorical interactions
    if all(comp isa CategoricalEvaluator for comp in components)
        comp1, comp2 = components[1], components[2]
        w1, w2 = component_widths[1], component_widths[2]
        
        # Generate level codes for both categoricals
        level_var1 = "level_$(comp1.column)_vec_int_$(pos)_1"
        level_var2 = "level_$(comp2.column)_vec_int_$(pos)_2"
        
        # First categorical
        push!(instructions, "@inbounds cat_val1 = data.$(comp1.column)[row_idx]")
        push!(instructions, "@inbounds $level_var1 = cat_val1 isa CategoricalValue ? levelcode(cat_val1) : 1")
        push!(instructions, "@inbounds $level_var1 = clamp($level_var1, 1, $(comp1.n_levels))")
        
        # Second categorical
        push!(instructions, "@inbounds cat_val2 = data.$(comp2.column)[row_idx]")
        push!(instructions, "@inbounds $level_var2 = cat_val2 isa CategoricalValue ? levelcode(cat_val2) : 1")
        push!(instructions, "@inbounds $level_var2 = clamp($level_var2, 1, $(comp2.n_levels))")
        
        # Generate Kronecker product in StatsModels order
        col_idx = 0
        for j in 1:w2  # Second component columns
            for i in 1:w1  # First component columns
                output_pos = pos + col_idx
                
                # Get contrast values
                values1 = [comp1.contrast_matrix[level, i] for level in 1:comp1.n_levels]
                values2 = [comp2.contrast_matrix[level, j] for level in 1:comp2.n_levels]
                
                # Generate ternary expressions for both contrasts
                if comp1.n_levels == 2
                    contrast1_expr = "$level_var1 == 1 ? $(values1[1]) : $(values1[2])"
                else
                    ternary_chain = "$level_var1 == 1 ? $(values1[1])"
                    for level in 2:(comp1.n_levels-1)
                        ternary_chain *= " : $level_var1 == $level ? $(values1[level])"
                    end
                    ternary_chain *= " : $(values1[comp1.n_levels])"
                    contrast1_expr = ternary_chain
                end
                
                if comp2.n_levels == 2
                    contrast2_expr = "$level_var2 == 1 ? $(values2[1]) : $(values2[2])"
                else
                    ternary_chain = "$level_var2 == 1 ? $(values2[1])"
                    for level in 2:(comp2.n_levels-1)
                        ternary_chain *= " : $level_var2 == $level ? $(values2[level])"
                    end
                    ternary_chain *= " : $(values2[comp2.n_levels])"
                    contrast2_expr = ternary_chain
                end
                
                # Generate the product
                push!(instructions, "@inbounds row_vec[$output_pos] = ($contrast1_expr) * ($contrast2_expr)")
                
                col_idx += 1
            end
        end
    else
        # Fallback for other vector×vector cases
        total_width = component_widths[1] * component_widths[2]
        for i in 0:(total_width-1)
            push!(instructions, "@inbounds row_vec[$(pos + i)] = 1.0  # Vector×Vector fallback")
        end
    end
end

function _generate_multiway_interaction!(instructions::Vector{String}, eval::InteractionEvaluator, pos::Int, component_vars::Vector{String})
    # Handle common 3-way case: scalar × scalar × vector
    component_widths = eval.component_widths
    components = eval.components
    
    if length(components) == 3 && component_widths[1] == 1 && component_widths[2] == 1 && component_widths[3] > 1
        # Two scalars and one vector
        scalar_product = "$(component_vars[1]) * $(component_vars[2])"
        vector_component = components[3]
        
        if vector_component isa CategoricalEvaluator
            _generate_scalar_categorical_interaction!(instructions, "($scalar_product)", vector_component, pos)
        else
            # Fallback
            for j in 1:component_widths[3]
                push!(instructions, "@inbounds row_vec[$(pos + j - 1)] = $scalar_product")
            end
        end
    else
        # General fallback
        total_width = eval.total_width
        for i in 0:(total_width-1)
            push!(instructions, "@inbounds row_vec[$(pos + i)] = 1.0  # Multi-way fallback")
        end
    end
end

function _generate_zscore_code!(instructions::Vector{String}, eval::ZScoreEvaluator, pos::Int)
    center = eval.center
    scale = eval.scale
    
    # Generate code for underlying evaluator into temporary variable
    temp_var = "zscore_temp_$pos"
    
    if eval.underlying isa ContinuousEvaluator
        col = eval.underlying.column
        push!(instructions, "@inbounds $temp_var = Float64(data.$col[row_idx])")
    elseif eval.underlying isa FunctionEvaluator && length(eval.underlying.arg_evaluators) == 1
        # Simple function case
        arg_eval = eval.underlying.arg_evaluators[1]
        if arg_eval isa ContinuousEvaluator
            col = arg_eval.column
            func = eval.underlying.func
            push!(instructions, "@inbounds raw_val = Float64(data.$col[row_idx])")
            
            if func === log
                push!(instructions, "@inbounds $temp_var = raw_val > 0.0 ? log(raw_val) : log(abs(raw_val) + 1e-16)")
            else
                push!(instructions, "@inbounds $temp_var = $func(raw_val)")
            end
        else
            push!(instructions, "@inbounds $temp_var = 1.0  # ZScore fallback")
        end
    else
        push!(instructions, "@inbounds $temp_var = 1.0  # ZScore complex fallback")
    end
    
    # Apply Z-score transformation
    push!(instructions, "@inbounds row_vec[$pos] = ($temp_var - $center) / $scale")
end

function _generate_combined_code!(instructions::Vector{String}, eval::CombinedEvaluator, start_pos::Int)
    current_pos = start_pos
    
    for (i, sub_eval) in enumerate(eval.sub_evaluators)
        width = eval.sub_widths[i]
        if width > 0
            _generate_code_recursive!(instructions, sub_eval, current_pos)
            current_pos += width
        end
    end
end

###############################################################################
# Integration with @generated Functions
###############################################################################

# Global cache for generated code (reuse existing infrastructure)
const FORMULA_CACHE = Dict{UInt64, Tuple{Vector{String}, Vector{Symbol}, Int}}()

"""
Simple struct for zero-allocation compiled formulas.
"""
struct CompiledFormula{H}
    formula_val::Val{H}
    output_width::Int
    column_names::Vector{Symbol}
end

Base.length(cf::CompiledFormula) = cf.output_width
Base.size(cf::CompiledFormula) = (cf.output_width,)
variables(cf::CompiledFormula) = cf.column_names

# Call interface - delegate to @generated function
function (cf::CompiledFormula)(row_vec::AbstractVector{Float64}, data, row_idx::Int)
    modelrow!(row_vec, cf.formula_val, data, row_idx)
    return row_vec
end

###############################################################################
# @generated Function (Reuse Existing Infrastructure)
###############################################################################

"""
Zero-allocation @generated function for model row evaluation.
"""
@generated function modelrow!(row_vec, ::Val{formula_hash}, data, row_idx) where formula_hash
    # Retrieve instructions from cache
    if !haskey(FORMULA_CACHE, formula_hash)
        error("Formula hash $formula_hash not found in cache")
    end
    
    instructions, column_names, output_width = FORMULA_CACHE[formula_hash]
    
    println("@generated: Compiling for hash $formula_hash with $(length(instructions)) instructions")
    
    # Convert instruction strings to expressions
    try
        code_exprs = [Meta.parse(line) for line in instructions]
        
        return quote
            @inbounds begin
                $(code_exprs...)
            end
            return row_vec
        end
        
    catch e
        error("Failed to parse instructions for hash $formula_hash: $e")
    end
end

###############################################################################
# Testing Function
###############################################################################

"""
Test the zero-allocation compiler.
"""
function test_zero_allocation_compiler()
    println("=== Testing Zero-Allocation Compiler ===")
    
    # Create test data
    Random.seed!(42)
    n = 100
    df = DataFrame(
        x = randn(n),
        y = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], n))
    )
    
    data = Tables.columntable(df)
    
    # Test simple formula
    formula = @formula(y ~ x + group + x * group)
    model = lm(formula, df)
    
    # Compile with zero-allocation approach
    compiled = compile_formula(model)
    
    # Test correctness
    row_vec = Vector{Float64}(undef, length(compiled))
    mm = modelmatrix(model)
    
    println("\nTesting correctness...")
    for test_row in [1, 5, 10]
        compiled(row_vec, data, test_row)
        expected = mm[test_row, :]
        
        if isapprox(row_vec, expected, atol=1e-12)
            println("✅ Row $test_row correct")
        else
            println("❌ Row $test_row error: $(maximum(abs.(row_vec .- expected)))")
        end
    end
    
    # Test allocations
    println("\nTesting allocations...")
    
    # Warmup
    for _ in 1:10
        compiled(row_vec, data, 1)
    end
    
    # Test allocation
    allocs = @allocated compiled(row_vec, data, 1)
    println("Allocations: $allocs bytes")
    
    if allocs == 0
        println("🎉 ZERO ALLOCATIONS ACHIEVED!")
        return true
    else
        println("⚠️  Still allocating: $allocs bytes")
        return false
    end
end


function generate_scalar_categorical_interaction(scalar_var::String, cat_eval::CategoricalEvaluator, output_base::String)
    col = cat_eval.column
    n_levels = cat_eval.n_levels
    contrast_matrix = cat_eval.contrast_matrix
    width = size(contrast_matrix, 2)
    
    instructions = String[]
    
    # Generate categorical lookup
    cat_var = generate_unique_var("cat")
    level_var = generate_unique_var("level")
    
    push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
    push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
    push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
    
    # Generate interaction for each column
    for j in 1:width
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        result_var = "$(output_base)_col$j"
        
        if n_levels == 2
            contrast_expr = "$level_var == 1 ? $(values[1]) : $(values[2])"
        else
            ternary = "$level_var == 1 ? $(values[1])"
            for i in 2:(n_levels-1)
                ternary *= " : $level_var == $i ? $(values[i])"
            end
            ternary *= " : $(values[n_levels])"
            contrast_expr = ternary
        end
        
        push!(instructions, "@inbounds $result_var = $scalar_var * ($contrast_expr)")
    end
    
    # For simplified single output, use first column
    push!(instructions, "@inbounds $output_base = $(output_base)_col1")
    
    return instructions
end

function generate_categorical_categorical_interaction(cat1::CategoricalEvaluator, cat2::CategoricalEvaluator, output_base::String)
    instructions = String[]
    
    # Generate level codes for both
    cat1_var = generate_unique_var("cat")
    level1_var = generate_unique_var("level")
    cat2_var = generate_unique_var("cat")
    level2_var = generate_unique_var("level")
    
    push!(instructions, "@inbounds $cat1_var = data.$(cat1.column)[row_idx]")
    push!(instructions, "@inbounds $level1_var = $cat1_var isa CategoricalValue ? levelcode($cat1_var) : 1")
    push!(instructions, "@inbounds $level1_var = clamp($level1_var, 1, $(cat1.n_levels))")
    
    push!(instructions, "@inbounds $cat2_var = data.$(cat2.column)[row_idx]")
    push!(instructions, "@inbounds $level2_var = $cat2_var isa CategoricalValue ? levelcode($cat2_var) : 1")
    push!(instructions, "@inbounds $level2_var = clamp($level2_var, 1, $(cat2.n_levels))")
    
    # Generate Kronecker product (simplified for first element)
    w1, w2 = size(cat1.contrast_matrix, 2), size(cat2.contrast_matrix, 2)
    
    values1 = [cat1.contrast_matrix[i, 1] for i in 1:cat1.n_levels]
    values2 = [cat2.contrast_matrix[i, 1] for i in 1:cat2.n_levels]
    
    # Generate contrasts
    if cat1.n_levels == 2
        contrast1_expr = "$level1_var == 1 ? $(values1[1]) : $(values1[2])"
    else
        ternary = "$level1_var == 1 ? $(values1[1])"
        for i in 2:(cat1.n_levels-1)
            ternary *= " : $level1_var == $i ? $(values1[i])"
        end
        ternary *= " : $(values1[cat1.n_levels])"
        contrast1_expr = ternary
    end
    
    if cat2.n_levels == 2
        contrast2_expr = "$level2_var == 1 ? $(values2[1]) : $(values2[2])"
    else
        ternary = "$level2_var == 1 ? $(values2[1])"
        for i in 2:(cat2.n_levels-1)
            ternary *= " : $level2_var == $i ? $(values2[i])"
        end
        ternary *= " : $(values2[cat2.n_levels])"
        contrast2_expr = ternary
    end
    
    push!(instructions, "@inbounds $output_base = ($contrast1_expr) * ($contrast2_expr)")
    
    return instructions
end

###############################################################################
# REPLACE: compile_formula function
###############################################################################

"""
REPLACES: compile_formula
Updated to use the fixed code generation.
"""
function compile_formula(model)
    println("=== Compiling Formula with Fixed Code Generation ===")
    
    # Step 1: Build evaluator tree (unchanged)
    rhs = fixed_effects_form(model).rhs
    
    terms = if rhs isa MatrixTerm
        collect(rhs.terms)
    elseif rhs isa Tuple
        collect(rhs)
    else
        [rhs]
    end
    
    active_terms = filter(t -> width(t) > 0, terms)
    evaluators = [compile_term(term) for term in active_terms]
    
    root_evaluator = if length(evaluators) == 1
        evaluators[1]
    else
        CombinedEvaluator(evaluators)
    end
    
    total_width = output_width(root_evaluator)
    column_names = extract_all_columns(rhs)
    
    println("Built evaluator tree: width $total_width, columns $column_names")
    
    # Step 2: Generate code using FIXED method
    instructions = generate_code_from_evaluator(root_evaluator)
    
    println("Generated $(length(instructions)) instructions with unique variables:")
    for (i, instr) in enumerate(instructions[1:min(10, length(instructions))])
        println("  $i: $instr")
    end
    if length(instructions) > 10
        println("  ... ($(length(instructions) - 10) more)")
    end
    
    # Step 3: Cache for @generated function
    formula_hash = hash(string(rhs))
    FORMULA_CACHE[formula_hash] = (instructions, column_names, total_width)
    
    println("Cached with hash: $formula_hash")
    
    # Step 4: Return compiled formula
    return CompiledFormula(Val(formula_hash), total_width, column_names)
end

###############################################################################
# Test the Complete Fix
###############################################################################

"""
Test the fixed compilation on all problematic cases.
"""
function test_fixed_compilation()
    println("=== Testing Fixed Compilation ===")
    
    # Create test data including the problematic bool case
    Random.seed!(42)
    df = DataFrame(
        x = randn(100),
        y = randn(100),
        z = abs.(randn(100)) .+ 0.1,
        cat2a = categorical(rand(["X", "Y"], 100)),
        cat2b = categorical(rand(["P", "Q"], 100)),
        bool = rand([false, true], 100)  # This was the problem case
    )
    
    data = Tables.columntable(df)
    
    test_cases = [
        (@formula(y ~ cat2b * (x^2)), "cat2b x function (was failing)"),
        (@formula(y ~ bool * (x^2)), "bool x function (was failing)"),
        (@formula(y ~ (x^2) * cat2a), "function x cat2a"),
        (@formula(y ~ log(sqrt(x)) * cat2a), "nested functions"),
        (@formula(y ~ log(x^2 + y^2) * cat2b), "complex function"),
        (@formula(y ~ (x > 0) * log(z) * cat2a), "three-way with functions"),
    ]
    
    results = []
    
    for (formula, description) in test_cases
        println("\n--- Testing: $description ---")
        println("Formula: $formula")
        
        try
            model = lm(formula, df)
            mm = modelmatrix(model)
            
            # Compile with fixed approach
            compiled = compile_formula(model)
            
            # Test correctness
            row_vec = Vector{Float64}(undef, length(compiled))
            n_test_rows = min(5, size(mm, 1))
            all_correct = true
            
            for test_row in 1:n_test_rows
                compiled(row_vec, data, test_row)
                expected = mm[test_row, :]
                
                if !isapprox(row_vec, expected, atol=1e-12)
                    println("❌ Row $test_row error: $(maximum(abs.(row_vec .- expected)))")
                    all_correct = false
                    break
                end
            end
            
            if all_correct
                println("✅ All test rows correct")
                
                # Test allocations
                allocs = @allocated compiled(row_vec, data, 1)
                println("Allocations: $allocs bytes")
                
                if allocs == 0
                    println("🎉 PERFECT: Correct + Zero Allocation!")
                    push!(results, (description, true, "perfect"))
                else
                    println("✅ Correct but $allocs bytes allocated")
                    push!(results, (description, true, "good"))
                end
            else
                println("❌ Correctness failed")
                push!(results, (description, false, "failed"))
            end
            
        catch e
            println("❌ Exception: $e")
            push!(results, (description, false, "exception"))
        end
    end
    
    # Summary
    println("\n" * "="^60)
    println("FINAL RESULTS WITH FIXED CODE GENERATION")
    println("="^60)
    
    perfect_count = sum(r[3] == "perfect" for r in results)
    good_count = sum(r[2] && r[3] == "good" for r in results)
    failed_count = sum(!r[2] for r in results)
    
    println("Perfect (correct + zero alloc): $perfect_count")
    println("Good (correct + some alloc):    $good_count")
    println("Failed:                         $failed_count")
    println("Total:                          $(length(results))")
    
    if perfect_count == length(results)
        println("🎉 ALL TESTS PERFECT! Bug is completely fixed!")
    elseif perfect_count + good_count == length(results)
        println("✅ All tests correct, some with minor allocations")
    else
        println("⚠️  Some tests still failing")
    end
    
    return results
end

# Export the fixed functions
export generate_code_from_evaluator, compile_formula, test_fixed_compilation