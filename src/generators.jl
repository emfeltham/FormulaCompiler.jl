# Export main functions
"""
Generate statements for ZScore evaluators with multi-output underlying terms.
"""
function generate_zscore_statements_recursive(evaluator::ZScoreEvaluator, start_pos::Int)
    underlying = evaluator.underlying
    center = evaluator.center
    scale = evaluator.scale
    width = output_width(underlying)
    
    instructions = String[]
    
    # Generate statements for underlying evaluator
    underlying_instructions, next_pos = generate_statements_recursive(underlying, start_pos)
    append!(instructions, underlying_instructions)
    
    # Apply Z-score transformation to each position
    for i in 0:(width-1)
        pos = start_pos + i
        temp_var = next_var("zscore_temp")
        push!(instructions, "@inbounds $temp_var = row_vec[$pos]")
        push!(instructions, "@inbounds row_vec[$pos] = ($temp_var - $center) / $scale")
    end
    
    return instructions, next_pos
end

###############################################################################
# PLACEHOLDER IMPLEMENTATIONS FOR PHASE 2D
###############################################################################

"""
Phase 2D: Advanced Evaluator Types
"""
function generate_interaction_expression_recursive(evaluator::InteractionEvaluator)
    error("InteractionEvaluator expression generation will be implemented in Phase 2D. " *
          "Components: $(length(evaluator.components))")
end

function generate_zscore_expression_recursive(evaluator::ZScoreEvaluator)
    error("ZScoreEvaluator expression generation will be implemented in Phase 2D. " *
          "Underlying: $(typeof(evaluator.underlying))")
end# generators.jl
# Phase 2A: Core Recursive Expression Generation Architecture
# This replaces the previous generators.jl with a recursive architecture that mirrors compile_term()

###############################################################################
# VARIABLE GENERATION SYSTEM (Preserved from original)
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
# CORE RECURSIVE EXPRESSION GENERATION
###############################################################################

"""
    generate_expression_recursive(evaluator::AbstractEvaluator) -> String

The inverse of compile_term() - converts any evaluator back to a Julia expression string.
This function MUST handle every AbstractEvaluator type that compile_term() can create.

This mirrors the exact structure of your compile_term() function:
- ConstantTerm → ConstantEvaluator → back to constant expression
- ContinuousTerm → ContinuousEvaluator → back to data access expression  
- FunctionTerm → FunctionEvaluator → back to function call expression
- etc.
"""
function generate_expression_recursive(evaluator::AbstractEvaluator)
    
    if evaluator isa ConstantEvaluator
        # Mirror: ConstantTerm → ConstantEvaluator → back to constant
        return string(evaluator.value)
        
    elseif evaluator isa ContinuousEvaluator
        # Mirror: ContinuousTerm/Term → ContinuousEvaluator → back to data access
        return "Float64(data.$(evaluator.column)[row_idx])"
        
    elseif evaluator isa CategoricalEvaluator
        # Mirror: CategoricalTerm → CategoricalEvaluator → back to categorical expression
        return generate_categorical_expression_recursive(evaluator)
        
    elseif evaluator isa FunctionEvaluator
        # Mirror: FunctionTerm → FunctionEvaluator → back to function call
        # This is the CRITICAL recursive case
        return generate_function_expression_recursive(evaluator)
        
    elseif evaluator isa InteractionEvaluator
        # Mirror: InteractionTerm → InteractionEvaluator → back to product expression
        return generate_interaction_expression_recursive(evaluator)
        
    elseif evaluator isa ZScoreEvaluator
        # Mirror: ZScoredTerm → ZScoreEvaluator → back to standardization expression
        return generate_zscore_expression_recursive(evaluator)
        
    elseif evaluator isa ScaledEvaluator
        # Handle scaled expressions recursively
        inner_expr = generate_expression_recursive(evaluator.evaluator)
        return "($(inner_expr) * $(evaluator.scale_factor))"
        
    elseif evaluator isa ProductEvaluator
        # Handle product expressions recursively
        return generate_product_expression_recursive(evaluator)
        
    else
        error("Expression generation not implemented for $(typeof(evaluator)). " *
              "This must be implemented to maintain completeness with compile_term().")
    end
end

###############################################################################
# MULTI-OUTPUT STATEMENT GENERATION
###############################################################################

"""
    generate_statements_recursive(evaluator::AbstractEvaluator, start_pos::Int) -> (Vector{String}, Int)

Handle evaluators that produce multiple outputs (like CombinedEvaluator).
Returns (instructions, next_position).

This mirrors how compile_term() handles MatrixTerm → CombinedEvaluator.
"""
function generate_statements_recursive(evaluator::AbstractEvaluator, start_pos::Int)
    
    if evaluator isa CombinedEvaluator
        # Mirror: MatrixTerm → CombinedEvaluator → back to multiple statements
        return generate_combined_statements_recursive(evaluator, start_pos)
        
    elseif is_multi_output_evaluator(evaluator)
        # Handle other multi-output cases (complex categoricals, interactions)
        return generate_multi_output_statements_recursive(evaluator, start_pos)
        
    else
        # Single output - use expression generator
        expr = generate_expression_recursive(evaluator)
        instructions = ["@inbounds row_vec[$start_pos] = $expr"]
        return instructions, start_pos + 1
    end
end

"""
Check if an evaluator produces multiple outputs.
"""
function is_multi_output_evaluator(evaluator::AbstractEvaluator)
    return evaluator isa CombinedEvaluator ||
           (evaluator isa CategoricalEvaluator && size(evaluator.contrast_matrix, 2) > 1) ||
           (evaluator isa InteractionEvaluator && output_width(evaluator) > 1)
end

###############################################################################
# COMBINED EVALUATOR HANDLING (EXACT INVERSE OF MATRIXTERM)
###############################################################################

"""
    generate_combined_statements_recursive(evaluator::CombinedEvaluator, start_pos::Int)

EXACT INVERSE of MatrixTerm → CombinedEvaluator parsing.
Mirrors this code from compile_term():

    if term isa MatrixTerm
        sub_evaluators = [compile_term(t) for t in term.terms if width(t) > 0]
        return CombinedEvaluator(sub_evaluators)
"""
function generate_combined_statements_recursive(evaluator::CombinedEvaluator, start_pos::Int)
    
    instructions = String[]
    current_pos = start_pos
    
    # Mirror the recursive compilation in compile_term()
    for sub_evaluator in evaluator.sub_evaluators
        # Recursively generate each sub-evaluator (mirrors compile_term recursion)
        sub_instructions, new_pos = generate_statements_recursive(sub_evaluator, current_pos)
        append!(instructions, sub_instructions)
        current_pos = new_pos
    end
    
    return instructions, current_pos
end

###############################################################################
# FUNCTION EXPRESSION GENERATION (Phase 2B - Full Recursive Implementation)
###############################################################################

"""
    generate_function_expression_recursive(evaluator::FunctionEvaluator) -> String

Phase 2B: Full recursive function expression generation.
Handles arbitrarily nested functions like log(x^2 + sin(y) * z).

This is the key recursive case that mirrors how compile_term() handles FunctionTerm:
- FunctionTerm(log, [FunctionTerm(^, [x, 2])]) → FunctionEvaluator(log, [FunctionEvaluator(^, [x, 2])])
- Now we reverse it: FunctionEvaluator(log, [FunctionEvaluator(^, [x, 2])]) → "log(x^2)"
"""
function generate_function_expression_recursive(evaluator::FunctionEvaluator)
    func = evaluator.func
    args = evaluator.arg_evaluators
    
    # Phase 2B: Handle ALL cases recursively
    # Generate argument expressions recursively - this is the key insight!
    arg_exprs = [generate_expression_recursive(arg) for arg in args]
    
    # Generate the function call with domain safety
    return generate_function_call_safe(func, arg_exprs)
end

"""
Check if an evaluator can be expressed as a simple inline expression.
Phase 2B: Expanded to handle more cases recursively.
"""
function is_simple_expression(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator || evaluator isa ContinuousEvaluator
        return true
    elseif evaluator isa CategoricalEvaluator
        return size(evaluator.contrast_matrix, 2) == 1 && evaluator.n_levels <= 3
    elseif evaluator isa FunctionEvaluator
        # Phase 2B: Functions are simple if their arguments are simple
        return all(arg -> is_simple_expression(arg), evaluator.arg_evaluators)
    elseif evaluator isa ScaledEvaluator
        return is_simple_expression(evaluator.evaluator)
    elseif evaluator isa ProductEvaluator
        # Simple if all components are simple
        return all(comp -> is_simple_expression(comp), evaluator.components)
    else
        return false
    end
end

"""
Generate function calls with domain safety and enhanced mathematical function support.
Phase 2B: Expanded with more functions and better error handling.
"""
function generate_function_call_safe(func::Function, arg_exprs::Vector{String})
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
        
    # Trigonometric functions
    elseif func === sin
        @assert length(arg_exprs) == 1 "sin expects 1 argument"
        return "sin($(arg_exprs[1]))"
        
    elseif func === cos
        @assert length(arg_exprs) == 1 "cos expects 1 argument"
        return "cos($(arg_exprs[1]))"
        
    elseif func === tan
        @assert length(arg_exprs) == 1 "tan expects 1 argument"
        arg = arg_exprs[1]
        return "tan($arg)"  # Let Julia handle undefined values
        
    elseif func === asin
        @assert length(arg_exprs) == 1 "asin expects 1 argument"
        arg = arg_exprs[1]
        return "(abs($arg) <= 1.0 ? asin($arg) : NaN)"
        
    elseif func === acos
        @assert length(arg_exprs) == 1 "acos expects 1 argument"
        arg = arg_exprs[1]
        return "(abs($arg) <= 1.0 ? acos($arg) : NaN)"
        
    elseif func === atan
        @assert length(arg_exprs) == 1 "atan expects 1 argument"
        return "atan($(arg_exprs[1]))"
        
    # Hyperbolic functions
    elseif func === sinh
        @assert length(arg_exprs) == 1 "sinh expects 1 argument"
        arg = arg_exprs[1]
        return "sinh(clamp($arg, -700.0, 700.0))"
        
    elseif func === cosh
        @assert length(arg_exprs) == 1 "cosh expects 1 argument"
        arg = arg_exprs[1]
        return "cosh(clamp($arg, -700.0, 700.0))"
        
    elseif func === tanh
        @assert length(arg_exprs) == 1 "tanh expects 1 argument"
        return "tanh($(arg_exprs[1]))"
        
    # Logarithmic functions
    elseif func === log10
        @assert length(arg_exprs) == 1 "log10 expects 1 argument"
        arg = arg_exprs[1]
        return "($arg > 0.0 ? log10($arg) : ($arg == 0.0 ? -Inf : NaN))"
        
    elseif func === log2
        @assert length(arg_exprs) == 1 "log2 expects 1 argument"
        arg = arg_exprs[1]
        return "($arg > 0.0 ? log2($arg) : ($arg == 0.0 ? -Inf : NaN))"
        
    elseif func === log1p
        @assert length(arg_exprs) == 1 "log1p expects 1 argument"
        arg = arg_exprs[1]
        return "($arg > -1.0 ? log1p($arg) : NaN)"
        
    # Power and exponential functions
    elseif func === exp2
        @assert length(arg_exprs) == 1 "exp2 expects 1 argument"
        arg = arg_exprs[1]
        return "exp2(clamp($arg, -1000.0, 1000.0))"
        
    elseif func === exp10
        @assert length(arg_exprs) == 1 "exp10 expects 1 argument"
        arg = arg_exprs[1]
        return "exp10(clamp($arg, -300.0, 300.0))"
        
    elseif func === expm1
        @assert length(arg_exprs) == 1 "expm1 expects 1 argument"
        arg = arg_exprs[1]
        return "expm1(clamp($arg, -700.0, 700.0))"
        
    # Rounding and sign functions
    elseif func === round
        @assert length(arg_exprs) == 1 "round expects 1 argument"
        return "round($(arg_exprs[1]))"
        
    elseif func === floor
        @assert length(arg_exprs) == 1 "floor expects 1 argument"
        return "floor($(arg_exprs[1]))"
        
    elseif func === ceil
        @assert length(arg_exprs) == 1 "ceil expects 1 argument"
        return "ceil($(arg_exprs[1]))"
        
    elseif func === sign
        @assert length(arg_exprs) == 1 "sign expects 1 argument"
        return "sign($(arg_exprs[1]))"
        
    # Binary arithmetic operations
    elseif func === (+) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) + $(arg_exprs[2]))"
        
    elseif func === (-) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) - $(arg_exprs[2]))"
        
    elseif func === (*) && length(arg_exprs) == 2
        return "($(arg_exprs[1]) * $(arg_exprs[2]))"
        
    elseif func === (/) && length(arg_exprs) == 2
        arg1, arg2 = arg_exprs[1], arg_exprs[2]
        return "(abs($arg2) > 1e-16 ? $arg1 / $arg2 : ($arg1 == 0.0 ? NaN : ($arg1 > 0.0 ? Inf : -Inf)))"
        
    elseif func === (^) && length(arg_exprs) == 2
        arg1, arg2 = arg_exprs[1], arg_exprs[2]
        return "($arg1 == 0.0 && $arg2 < 0.0 ? Inf : ($arg1 < 0.0 && !isinteger($arg2) ? NaN : $arg1^$arg2))"
        
    elseif func === mod && length(arg_exprs) == 2
        arg1, arg2 = arg_exprs[1], arg_exprs[2]
        return "($arg2 == 0.0 ? NaN : mod($arg1, $arg2))"
        
    elseif func === rem && length(arg_exprs) == 2
        arg1, arg2 = arg_exprs[1], arg_exprs[2]
        return "($arg2 == 0.0 ? NaN : rem($arg1, $arg2))"
        
    # Comparison operations
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
        
    # Min/max functions
    elseif func === min && length(arg_exprs) == 2
        return "min($(arg_exprs[1]), $(arg_exprs[2]))"
    elseif func === max && length(arg_exprs) == 2
        return "max($(arg_exprs[1]), $(arg_exprs[2]))"
        
    # N-ary operations
    elseif func === (+) && length(arg_exprs) > 2
        return "(" * join(arg_exprs, " + ") * ")"
    elseif func === (*) && length(arg_exprs) > 2
        return "(" * join(arg_exprs, " * ") * ")"
    elseif func === min && length(arg_exprs) > 2
        return "min(" * join(arg_exprs, ", ") * ")"
    elseif func === max && length(arg_exprs) > 2
        return "max(" * join(arg_exprs, ", ") * ")"
        
    else
        error("Function $func with $(length(arg_exprs)) arguments not supported in Phase 2B. " *
              "Supported functions: log, exp, sqrt, abs, sin, cos, tan, asin, acos, atan, " *
              "sinh, cosh, tanh, log10, log2, log1p, exp2, exp10, expm1, round, floor, ceil, sign, " *
              "+, -, *, /, ^, mod, rem, >, <, >=, <=, ==, !=, min, max")
    end
end

###############################################################################
# CATEGORICAL EXPRESSION GENERATION (Phase 2A - Simple Cases Only)
###############################################################################

###############################################################################
# CATEGORICAL EXPRESSION GENERATION (Phase 2C - Full Implementation)
###############################################################################

"""
    generate_categorical_expression_recursive(evaluator::CategoricalEvaluator) -> String

Phase 2C: Full categorical expression generation.
Handles both single-contrast (inline expressions) and multi-contrast (requires statements).
"""
function generate_categorical_expression_recursive(evaluator::CategoricalEvaluator)
    col = evaluator.column
    n_levels = evaluator.n_levels
    contrast_matrix = evaluator.contrast_matrix
    
    if size(contrast_matrix, 2) == 1
        # Single contrast - can generate as inline expression
        return generate_single_contrast_categorical_expression(evaluator)
    else
        # Multi-contrast - requires statement generation
        error("Multi-contrast categorical expressions require statement generation. " *
              "Use generate_statements_recursive() instead of generate_expression_recursive(). " *
              "Contrasts: $(size(contrast_matrix, 2))")
    end
end

"""
Generate single-contrast categorical expressions (Phase 2C enhanced).
"""
function generate_single_contrast_categorical_expression(evaluator::CategoricalEvaluator)
    col = evaluator.column
    n_levels = evaluator.n_levels
    contrast_matrix = evaluator.contrast_matrix
    
    values = [contrast_matrix[i, 1] for i in 1:n_levels]
    level_expr = "clamp(data.$col[row_idx] isa CategoricalValue ? levelcode(data.$col[row_idx]) : 1, 1, $n_levels)"
    
    if n_levels == 1
        return string(values[1])
    elseif n_levels == 2
        return "($level_expr == 1 ? $(values[1]) : $(values[2]))"
    else
        # Phase 2C: Handle larger categoricals more efficiently
        return generate_categorical_lookup_expression(level_expr, values, n_levels)
    end
end

"""
Generate efficient categorical lookup expressions for larger categoricals.
"""
function generate_categorical_lookup_expression(level_expr::String, values::Vector{Float64}, n_levels::Int)
    if n_levels <= 8
        # Use ternary chain for small categoricals
        ternary = "$level_expr == 1 ? $(values[1])"
        for i in 2:(n_levels-1)
            ternary *= " : $level_expr == $i ? $(values[i])"
        end
        ternary *= " : $(values[n_levels])"
        return "($ternary)"
    else
        # For large categoricals, this becomes unwieldy - should use statement generation
        error("Categorical with $n_levels levels is too large for inline expression. " *
              "Use generate_statements_recursive() for multi-statement generation.")
    end
end

###############################################################################
# PLACEHOLDER IMPLEMENTATIONS FOR PHASE 2B-2D
###############################################################################

"""
Phase 2B: Complex Function Expression Generation
Will handle arbitrarily nested functions like log(x^2 + sin(y)).
"""
function generate_complex_function_expression_recursive(evaluator::FunctionEvaluator)
    error("Complex nested function expressions will be implemented in Phase 2B. " *
          "Function: $(evaluator.func)")
end

###############################################################################
# MULTI-OUTPUT STATEMENT GENERATION (Phase 2C - Full Implementation)
###############################################################################

"""
    generate_multi_output_statements_recursive(evaluator::AbstractEvaluator, start_pos::Int)

Phase 2C: Full multi-output statement generation.
Handles complex categoricals, interactions, and other multi-output evaluators.
"""
function generate_multi_output_statements_recursive(evaluator::AbstractEvaluator, start_pos::Int)
    
    if evaluator isa CategoricalEvaluator
        return generate_categorical_statements_recursive(evaluator, start_pos)
        
    elseif evaluator isa InteractionEvaluator
        return generate_interaction_statements_recursive(evaluator, start_pos)
        
    elseif evaluator isa ZScoreEvaluator && output_width(evaluator.underlying) > 1
        return generate_zscore_statements_recursive(evaluator, start_pos)
        
    else
        error("Multi-output statement generation for $(typeof(evaluator)) not implemented. " *
              "Output width: $(output_width(evaluator))")
    end
end

"""
Generate statements for multi-contrast categorical evaluators.
"""
function generate_categorical_statements_recursive(evaluator::CategoricalEvaluator, start_pos::Int)
    col = evaluator.column
    n_levels = evaluator.n_levels
    contrast_matrix = evaluator.contrast_matrix
    width = size(contrast_matrix, 2)
    
    if width == 1
        # Single contrast - can use expression generation
        expr = generate_single_contrast_categorical_expression(evaluator)
        instructions = ["@inbounds row_vec[$start_pos] = $expr"]
        return instructions, start_pos + 1
    end
    
    instructions = String[]
    
    # Generate unique variable names for this categorical
    cat_var = next_var("cat")
    level_var = next_var("level")
    
    # Extract categorical value and level code
    push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
    push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
    push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
    
    # Generate efficient contrast assignments
    if n_levels <= 4 && width <= 3
        # Small categorical - use direct ternary assignments
        generate_small_categorical_statements!(instructions, contrast_matrix, level_var, n_levels, width, start_pos)
    else
        # Large categorical - use lookup table approach
        generate_large_categorical_statements!(instructions, contrast_matrix, level_var, n_levels, width, start_pos)
    end
    
    return instructions, start_pos + width
end

"""
Generate statements for small categoricals using direct ternary operations.
"""
function generate_small_categorical_statements!(instructions::Vector{String}, 
                                               contrast_matrix::Matrix{Float64},
                                               level_var::String, 
                                               n_levels::Int, 
                                               width::Int, 
                                               start_pos::Int)
    
    for j in 1:width
        output_pos = start_pos + j - 1
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        
        if n_levels == 1
            push!(instructions, "@inbounds row_vec[$output_pos] = $(values[1])")
        elseif n_levels == 2
            push!(instructions, "@inbounds row_vec[$output_pos] = $level_var == 1 ? $(values[1]) : $(values[2])")
        elseif n_levels == 3
            push!(instructions, "@inbounds row_vec[$output_pos] = $level_var == 1 ? $(values[1]) : $level_var == 2 ? $(values[2]) : $(values[3])")
        elseif n_levels == 4
            push!(instructions, "@inbounds row_vec[$output_pos] = $level_var == 1 ? $(values[1]) : $level_var == 2 ? $(values[2]) : $level_var == 3 ? $(values[3]) : $(values[4])")
        else
            # Build general ternary chain
            ternary_chain = "$level_var == 1 ? $(values[1])"
            for i in 2:(n_levels-1)
                ternary_chain *= " : $level_var == $i ? $(values[i])"
            end
            ternary_chain *= " : $(values[n_levels])"
            push!(instructions, "@inbounds row_vec[$output_pos] = $ternary_chain")
        end
    end
end

"""
Generate statements for large categoricals using lookup table approach.
"""
function generate_large_categorical_statements!(instructions::Vector{String}, 
                                               contrast_matrix::Matrix{Float64},
                                               level_var::String, 
                                               n_levels::Int, 
                                               width::Int, 
                                               start_pos::Int)
    
    # For very large categoricals, we can use a more compact approach
    # Generate lookup arrays as local variables
    for j in 1:width
        output_pos = start_pos + j - 1
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        
        # Create a lookup array variable
        lookup_var = next_var("lookup")
        values_str = "[" * join(string.(values), ", ") * "]"
        push!(instructions, "@inbounds $lookup_var = $values_str")
        push!(instructions, "@inbounds row_vec[$output_pos] = $lookup_var[$level_var]")
    end
end

"""
Generate statements for simple interactions (Phase 2C).
More complex interactions will be handled in Phase 2D.
"""
function generate_interaction_statements_recursive(evaluator::InteractionEvaluator, start_pos::Int)
    components = evaluator.components
    n_components = length(components)
    
    if n_components == 0
        return String[], start_pos
    elseif n_components == 1
        # Single component - delegate to regular statement generation
        return generate_statements_recursive(components[1], start_pos)
    elseif n_components == 2
        # Binary interaction - handle in Phase 2C
        return generate_binary_interaction_statements(components, start_pos)
    else
        # Complex interactions - defer to Phase 2D
        error("Interactions with $(n_components) components will be implemented in Phase 2D. " *
              "Components: $(typeof.(components))")
    end
end

"""
Generate statements for binary interactions (Phase 2C).
"""
function generate_binary_interaction_statements(components::Vector{AbstractEvaluator}, start_pos::Int)
    comp1, comp2 = components[1], components[2]
    w1, w2 = output_width(comp1), output_width(comp2)
    total_width = w1 * w2
    
    instructions = String[]
    
    # Check if both components can be generated as simple expressions
    if is_simple_expression(comp1) && is_simple_expression(comp2)
        return generate_simple_binary_interaction_statements(comp1, comp2, start_pos)
    else
        # One or both components are complex - use statement-based approach
        return generate_complex_binary_interaction_statements(comp1, comp2, w1, w2, start_pos)
    end
end

"""
Generate statements for simple binary interactions (both components expressible inline).
"""
function generate_simple_binary_interaction_statements(comp1::AbstractEvaluator, comp2::AbstractEvaluator, start_pos::Int)
    w1, w2 = output_width(comp1), output_width(comp2)
    instructions = String[]
    
    # Generate expressions for both components
    if w1 == 1 && w2 == 1
        # Both scalar - simple product
        expr1 = generate_expression_recursive(comp1)
        expr2 = generate_expression_recursive(comp2)
        push!(instructions, "@inbounds row_vec[$start_pos] = ($expr1) * ($expr2)")
        return instructions, start_pos + 1
        
    elseif w1 == 1 && w2 > 1
        # Scalar × Vector interaction (e.g., x * group)
        expr1 = generate_expression_recursive(comp1)
        return generate_scalar_vector_interaction(expr1, comp2, start_pos, w2)
        
    elseif w1 > 1 && w2 == 1
        # Vector × Scalar interaction (e.g., group * x) 
        expr2 = generate_expression_recursive(comp2)
        return generate_vector_scalar_interaction(comp1, expr2, start_pos, w1)
        
    else
        # Vector × Vector - defer to complex case
        return generate_complex_binary_interaction_statements(comp1, comp2, w1, w2, start_pos)
    end
end

"""
Generate statements for scalar × vector interactions.
"""
function generate_scalar_vector_interaction(scalar_expr::String, vector_comp::AbstractEvaluator, start_pos::Int, width::Int)
    instructions = String[]
    
    if vector_comp isa CategoricalEvaluator
        # Scalar × Categorical: multiply scalar by each contrast
        col = vector_comp.column
        n_levels = vector_comp.n_levels
        contrast_matrix = vector_comp.contrast_matrix
        
        # Generate categorical lookup
        cat_var = next_var("cat")
        level_var = next_var("level")
        push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
        push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
        push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
        
        # Generate scaled contrasts
        for j in 1:width
            output_pos = start_pos + j - 1
            values = [contrast_matrix[i, j] for i in 1:n_levels]
            
            if n_levels <= 4
                # Small categorical - direct ternary with scaling
                if n_levels == 2
                    contrast_expr = "$level_var == 1 ? $(values[1]) : $(values[2])"
                else
                    contrast_expr = "$level_var == 1 ? $(values[1])"
                    for i in 2:(n_levels-1)
                        contrast_expr *= " : $level_var == $i ? $(values[i])"
                    end
                    contrast_expr *= " : $(values[n_levels])"
                end
                
                push!(instructions, "@inbounds row_vec[$output_pos] = ($scalar_expr) * ($contrast_expr)")
            else
                # Large categorical - use lookup
                lookup_var = next_var("lookup")
                values_str = "[" * join(string.(values), ", ") * "]"
                push!(instructions, "@inbounds $lookup_var = $values_str")
                push!(instructions, "@inbounds row_vec[$output_pos] = ($scalar_expr) * $lookup_var[$level_var]")
            end
        end
        
        return instructions, start_pos + width
    else
        error("Scalar × Vector interaction with $(typeof(vector_comp)) not implemented in Phase 2C")
    end
end

"""
Generate statements for vector × scalar interactions.
"""
function generate_vector_scalar_interaction(vector_comp::AbstractEvaluator, scalar_expr::String, start_pos::Int, width::Int)
    # Vector × Scalar is the same as Scalar × Vector
    return generate_scalar_vector_interaction(scalar_expr, vector_comp, start_pos, width)
end

"""
Generate statements for complex binary interactions (fallback).
"""
function generate_complex_binary_interaction_statements(comp1::AbstractEvaluator, comp2::AbstractEvaluator, w1::Int, w2::Int, start_pos::Int)
    error("Complex binary interactions between $(typeof(comp1)) (width $w1) and $(typeof(comp2)) (width $w2) " *
          "will be implemented in Phase 2D. Use simpler interaction patterns for Phase 2C.")
end

###############################################################################
# PRODUCT EXPRESSION GENERATION (Phase 2B Implementation)
###############################################################################

"""
    generate_product_expression_recursive(evaluator::ProductEvaluator) -> String

Phase 2B: Generate product expressions recursively.
Handles products like (x + 1) * (y + 2) by recursively generating each component.
"""
function generate_product_expression_recursive(evaluator::ProductEvaluator)
    # Recursively generate each component expression
    component_exprs = [generate_expression_recursive(comp) for comp in evaluator.components]
    
    if length(component_exprs) == 1
        return component_exprs[1]
    else
        return "(" * join(component_exprs, " * ") * ")"
    end
end

###############################################################################
# MAIN CODE GENERATION INTERFACE (Replaces previous generate_code_from_evaluator)
###############################################################################

"""
    generate_code_from_evaluator(evaluator::AbstractEvaluator) -> Vector{String}

Main interface for generating code from evaluator trees.
This replaces the previous non-recursive generate_code_from_evaluator function.
"""
function generate_code_from_evaluator(evaluator::AbstractEvaluator)
    reset_var_counter!()
    instructions, _ = generate_statements_recursive(evaluator, 1)
    return instructions
end

"""
    generate_evaluator_code!(instructions::Vector{String}, evaluator::AbstractEvaluator, pos::Int) -> Int

Backward compatibility interface that uses the new recursive system.
This maintains compatibility with existing code while using the new architecture.
"""
function generate_evaluator_code!(instructions::Vector{String}, evaluator::AbstractEvaluator, pos::Int)
    # Use the new recursive statement generation
    new_instructions, next_pos = generate_statements_recursive(evaluator, pos)
    append!(instructions, new_instructions)
    return next_pos
end

# Export main functions
export generate_code_from_evaluator, generate_evaluator_code!
export generate_expression_recursive, generate_statements_recursive
