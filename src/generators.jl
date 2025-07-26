# # generators.jl - Complete New Framework
# # Pure expression-based approach with minimal necessary functions

# ###############################################################################
# # COMPLEXITY LIMITS AND THRESHOLDS
# ###############################################################################

# # Configurable limits for different types of complexity
# const CATEGORICAL_INLINE_LIMIT = 8           # Max levels for inline categorical expressions
# const CATEGORICAL_LOOKUP_LIMIT = 50          # Max levels for lookup-based categoricals
# const INTERACTION_TERM_LIMIT = 1000          # Max terms in any interaction
# const THREEWAY_INTERACTION_LIMIT = 200       # Max terms in three-way interactions
# const NWAY_INTERACTION_LIMIT = 1000          # Max terms in n-way interactions
# const EXPRESSION_DEPTH_LIMIT = 10            # Max nesting depth for expressions
# const TOTAL_FORMULA_WIDTH_LIMIT = 10000      # Max total terms in entire formula

# ###############################################################################
# # COMPLEXITY CHECKING FUNCTIONS
# ###############################################################################

# """
# Check if an evaluator is too complex for efficient code generation.
# """
# function is_too_complex_for_generation(evaluator::AbstractEvaluator)
    
#     # Check total output width
#     width = output_width(evaluator)
#     if width > TOTAL_FORMULA_WIDTH_LIMIT
#         return true, "Formula produces $width terms (limit: $TOTAL_FORMULA_WIDTH_LIMIT)"
#     end
    
#     # Check specific evaluator types
#     if evaluator isa CategoricalEvaluator
#         return check_categorical_complexity(evaluator)
        
#     elseif evaluator isa InteractionEvaluator
#         return check_interaction_complexity(evaluator)
        
#     elseif evaluator isa FunctionEvaluator
#         return check_function_complexity(evaluator)
        
#     elseif evaluator isa CombinedEvaluator
#         return check_combined_complexity(evaluator)
        
#     else
#         return false, ""
#     end
# end

# """
# Check categorical evaluator complexity.
# """
# function check_categorical_complexity(evaluator::CategoricalEvaluator)
#     n_levels = evaluator.n_levels
#     n_contrasts = size(evaluator.contrast_matrix, 2)
    
#     if n_levels > CATEGORICAL_LOOKUP_LIMIT
#         return true, "Categorical variable has $n_levels levels (limit: $CATEGORICAL_LOOKUP_LIMIT)"
#     end
    
#     if n_contrasts > 20  # Reasonable limit for contrast matrix width
#         return true, "Categorical variable has $n_contrasts contrasts (limit: 20)"
#     end
    
#     return false, ""
# end

# """
# Check interaction evaluator complexity.
# """
# function check_interaction_complexity(evaluator::InteractionEvaluator)
#     components = evaluator.components
#     n_components = length(components)
#     total_width = output_width(evaluator)
    
#     # Check total interaction size
#     if total_width > INTERACTION_TERM_LIMIT
#         return true, "Interaction produces $total_width terms (limit: $INTERACTION_TERM_LIMIT)"
#     end
    
#     # Check specific interaction patterns
#     if n_components == 3 && total_width > THREEWAY_INTERACTION_LIMIT
#         return true, "Three-way interaction produces $total_width terms (limit: $THREEWAY_INTERACTION_LIMIT)"
#     end
    
#     if n_components >= 4 && total_width > NWAY_INTERACTION_LIMIT
#         return true, "$n_components-way interaction produces $total_width terms (limit: $NWAY_INTERACTION_LIMIT)"
#     end
    
#     # Check for pathological cases
#     component_widths = [output_width(comp) for comp in components]
#     max_width = maximum(component_widths)
    
#     if max_width > 50
#         return true, "Interaction component has $max_width terms (individual component limit: 50)"
#     end
    
#     # Check for categorical × categorical with many levels
#     cat_components = [comp for comp in components if comp isa CategoricalEvaluator]
#     if length(cat_components) >= 2
#         cat_levels = [comp.n_levels for comp in cat_components]
#         if prod(cat_levels) > 500
#             return true, "Multiple categorical interaction: $(cat_levels) levels = $(prod(cat_levels)) terms (limit: 500)"
#         end
#     end
    
#     return false, ""
# end

# """
# Check function evaluator complexity (recursive depth).
# """
# function check_function_complexity(evaluator::FunctionEvaluator, depth::Int = 1)
#     if depth > EXPRESSION_DEPTH_LIMIT
#         return true, "Function nesting depth $depth exceeds limit ($EXPRESSION_DEPTH_LIMIT)"
#     end
    
#     # Check argument complexity recursively
#     for arg in evaluator.arg_evaluators
#         if arg isa FunctionEvaluator
#             is_complex, message = check_function_complexity(arg, depth + 1)
#             if is_complex
#                 return true, message
#             end
#         end
#     end
    
#     return false, ""
# end

# """
# Check combined evaluator complexity.
# """
# function check_combined_complexity(evaluator::CombinedEvaluator)
#     total_width = output_width(evaluator)
#     n_terms = length(evaluator.sub_evaluators)
    
#     if total_width > TOTAL_FORMULA_WIDTH_LIMIT
#         return true, "Combined formula produces $total_width terms (limit: $TOTAL_FORMULA_WIDTH_LIMIT)"
#     end
    
#     if n_terms > 1000  # Reasonable limit on number of terms
#         return true, "Formula has $n_terms terms (limit: 1000)"
#     end
    
#     # Check each sub-evaluator
#     for (i, sub_eval) in enumerate(evaluator.sub_evaluators)
#         is_complex, message = is_too_complex_for_generation(sub_eval)
#         if is_complex
#             return true, "Term $i: $message"
#         end
#     end
    
#     return false, ""
# end

# ###############################################################################
# # GLOBAL VARIABLE COUNTER
# ###############################################################################

# const VAR_COUNTER = Ref(0)

# function next_var(prefix::String="v")
#     VAR_COUNTER[] += 1
#     return "$(prefix)_$(VAR_COUNTER[])"
# end

# function reset_var_counter!()
#     VAR_COUNTER[] = 0
# end

# ###############################################################################
# # MAIN INTERFACE FUNCTIONS
# ###############################################################################

# """
#     generate_code_from_evaluator(evaluator::AbstractEvaluator) -> Vector{String}

# Main interface for generating code from evaluator trees.
# This is the entry point called by CompiledFormula.

# Enhanced with with complexity checking.
# """
# function generate_code_from_evaluator(evaluator::AbstractEvaluator)
#     # Check complexity before attempting generation
#     is_complex, message = is_too_complex_for_generation(evaluator)
    
#     if is_complex
#         throw(ComplexityError("Formula too complex for efficient compilation: $message\n" *
#                             "Consider:\n" *
#                             "1. Simplifying the formula\n" *
#                             "2. Using fewer categorical levels\n" *
#                             "3. Reducing interaction complexity\n" *
#                             "4. Splitting into multiple simpler formulas"))
#     end
    
#     reset_var_counter!()
#     instructions, _ = generate_statements_recursive(evaluator, 1)
#     return instructions
# end

# """
# Custom error type for complexity issues.
# """
# struct ComplexityError <: Exception
#     message::String
# end

# Base.show(io::IO, e::ComplexityError) = print(io, "ComplexityError: ", e.message)

# """
#     generate_evaluator_code!(instructions::Vector{String}, evaluator::AbstractEvaluator, pos::Int) -> Int

# Backward compatibility interface for existing code.
# """
# function generate_evaluator_code!(instructions::Vector{String}, evaluator::AbstractEvaluator, pos::Int)
#     new_instructions, next_pos = generate_statements_recursive(evaluator, pos)
#     append!(instructions, new_instructions)
#     return next_pos
# end

# ###############################################################################
# # CORE RECURSIVE FUNCTIONS
# ###############################################################################

# # """
# #     generate_expression_recursive(evaluator::AbstractEvaluator) -> String

# # Generate a single expression for simple evaluators.
# # This is the core recursive function that handles all basic evaluator types.
# # """
# # function generate_expression_recursive(evaluator::AbstractEvaluator)
    
# #     if evaluator isa ConstantEvaluator
# #         return string(evaluator.value)
        
# #     elseif evaluator isa ContinuousEvaluator
# #         return "Float64(data.$(evaluator.column)[row_idx])"
        
# #     elseif evaluator isa CategoricalEvaluator
# #         return generate_categorical_expression(evaluator)
        
# #     elseif evaluator isa FunctionEvaluator
# #         return generate_function_expression(evaluator)
        
# #     elseif evaluator isa ScaledEvaluator
# #         inner_expr = generate_expression_recursive(evaluator.evaluator)
# #         return "($(inner_expr) * $(evaluator.scale_factor))"
        
# #     elseif evaluator isa ProductEvaluator
# #         component_exprs = [generate_expression_recursive(comp) for comp in evaluator.components]
# #         return "(" * join(component_exprs, " * ") * ")"
        
# #     elseif evaluator isa ZScoreEvaluator
# #         underlying_expr = generate_expression_recursive(evaluator.underlying)
# #         return "(($(underlying_expr) - $(evaluator.center)) / $(evaluator.scale))"
        
# #     elseif evaluator isa InteractionEvaluator
# #         return generate_interaction_expression(evaluator)
        
# #     else
# #         error("Expression generation not implemented for $(typeof(evaluator))")
# #     end
# # end

# # """
# #     generate_statements_recursive(evaluator::AbstractEvaluator, start_pos::Int) -> (Vector{String}, Int)

# # Generate multiple statements for complex evaluators.
# # Returns (instructions, next_position).
# # """
# # function generate_statements_recursive(evaluator::AbstractEvaluator, start_pos::Int)
    
# #     if evaluator isa CombinedEvaluator
# #         return generate_combined_statements(evaluator, start_pos)
        
# #     elseif is_multi_output_evaluator(evaluator)
# #         return generate_multi_output_statements(evaluator, start_pos)
        
# #     else
# #         # Single output - use expression generator
# #         if is_simple_expression(evaluator)
# #             expr = generate_expression_recursive(evaluator)
# #             instructions = ["@inbounds row_vec[$start_pos] = $expr"]
# #             return instructions, start_pos + 1
# #         else
# #             # Complex single output - try to handle with helper functions
# #             return generate_complex_single_output(evaluator, start_pos)
# #         end
# #     end
# # end

# ###############################################################################
# # EVALUATOR TYPE HANDLERS
# ###############################################################################

# """
# Generate expressions for categorical evaluators.

# Enhanced categorical expression generation with size limits.
# """
# function generate_categorical_expression(evaluator::CategoricalEvaluator)
#     col = evaluator.column
#     n_levels = evaluator.n_levels
#     contrast_matrix = evaluator.contrast_matrix
    
#     # Check if this categorical is too large for inline expression
#     if n_levels > CATEGORICAL_INLINE_LIMIT
#         error("Categorical variable '$col' has $n_levels levels. " *
#               "For inline expressions, limit is $CATEGORICAL_INLINE_LIMIT levels. " *
#               "Use generate_categorical_statements() instead.")
#     end
    
#     if size(contrast_matrix, 2) == 1
#         # Single contrast - can generate as expression
#         values = [contrast_matrix[i, 1] for i in 1:n_levels]
#         level_expr = "clamp(data.$col[row_idx] isa CategoricalValue ? levelcode(data.$col[row_idx]) : 1, 1, $n_levels)"
        
#         if n_levels == 1
#             return string(values[1])
#         elseif n_levels == 2
#             return "($level_expr == 1 ? $(values[1]) : $(values[2]))"
#         else
#             # Build ternary chain for reasonable sizes
#             ternary = "$level_expr == 1 ? $(values[1])"
#             for i in 2:(n_levels-1)
#                 ternary *= " : $level_expr == $i ? $(values[i])"
#             end
#             ternary *= " : $(values[n_levels])"
#             return "($ternary)"
#         end
#     else
#         error("Multi-contrast categorical requires statement generation")
#     end
# end

# """
# Generate expressions for function evaluators.
# """
# function generate_function_expression(evaluator::FunctionEvaluator)
#     func = evaluator.func
#     args = evaluator.arg_evaluators
    
#     # Generate argument expressions recursively
#     arg_exprs = [generate_expression_recursive(arg) for arg in args]
    
#     # Generate the function call with domain safety
#     return generate_function_call_safe(func, arg_exprs)
# end

# """
# Generate expressions for interaction evaluators.
# """
# function generate_interaction_expression(evaluator::InteractionEvaluator)
#     components = evaluator.components
#     n_components = length(components)
    
#     if n_components == 0
#         return "1.0"
#     elseif n_components == 1
#         return generate_expression_recursive(components[1])
#     elseif all(comp -> is_simple_expression(comp) && output_width(comp) == 1, components)
#         # All components are simple scalar expressions
#         component_exprs = [generate_expression_recursive(comp) for comp in components]
#         return "(" * join(component_exprs, " * ") * ")"
#     else
#         error("Complex interaction requires statement generation")
#     end
# end

# """
# Generate function calls with domain safety.
# """
# function generate_function_call_safe(func::Function, arg_exprs::Vector{String})
#     if func === log
#         @assert length(arg_exprs) == 1 "log expects 1 argument"
#         arg = arg_exprs[1]
#         return "($arg > 0.0 ? log($arg) : ($arg == 0.0 ? -Inf : NaN))"
        
#     elseif func === exp
#         @assert length(arg_exprs) == 1 "exp expects 1 argument"
#         arg = arg_exprs[1]
#         return "exp(clamp($arg, -700.0, 700.0))"
        
#     elseif func === sqrt
#         @assert length(arg_exprs) == 1 "sqrt expects 1 argument"
#         arg = arg_exprs[1]
#         return "($arg >= 0.0 ? sqrt($arg) : NaN)"
        
#     elseif func === abs
#         @assert length(arg_exprs) == 1 "abs expects 1 argument"
#         return "abs($(arg_exprs[1]))"
        
#     elseif func === sin
#         @assert length(arg_exprs) == 1 "sin expects 1 argument"
#         return "sin($(arg_exprs[1]))"
        
#     elseif func === cos
#         @assert length(arg_exprs) == 1 "cos expects 1 argument"
#         return "cos($(arg_exprs[1]))"
        
#     elseif func === tan
#         @assert length(arg_exprs) == 1 "tan expects 1 argument"
#         return "tan($(arg_exprs[1]))"
        
#     elseif func === (+) && length(arg_exprs) == 2
#         return "($(arg_exprs[1]) + $(arg_exprs[2]))"
        
#     elseif func === (-) && length(arg_exprs) == 2
#         return "($(arg_exprs[1]) - $(arg_exprs[2]))"
        
#     elseif func === (*) && length(arg_exprs) == 2
#         return "($(arg_exprs[1]) * $(arg_exprs[2]))"
        
#     elseif func === (/) && length(arg_exprs) == 2
#         arg1, arg2 = arg_exprs[1], arg_exprs[2]
#         return "(abs($arg2) > 1e-16 ? $arg1 / $arg2 : ($arg1 == 0.0 ? NaN : ($arg1 > 0.0 ? Inf : -Inf)))"
        
#     elseif func === (^) && length(arg_exprs) == 2
#         arg1, arg2 = arg_exprs[1], arg_exprs[2]
#         return "($arg1 == 0.0 && $arg2 < 0.0 ? Inf : ($arg1 < 0.0 && !isinteger($arg2) ? NaN : $arg1^$arg2))"
        
#     elseif func === (+) && length(arg_exprs) > 2
#         return "(" * join(arg_exprs, " + ") * ")"
        
#     elseif func === (*) && length(arg_exprs) > 2
#         return "(" * join(arg_exprs, " * ") * ")"
        
#     # Comparison operations (return 1.0 or 0.0)
#     elseif func === (>) && length(arg_exprs) == 2
#         return "($(arg_exprs[1]) > $(arg_exprs[2]) ? 1.0 : 0.0)"
#     elseif func === (<) && length(arg_exprs) == 2
#         return "($(arg_exprs[1]) < $(arg_exprs[2]) ? 1.0 : 0.0)"
#     elseif func === (>=) && length(arg_exprs) == 2
#         return "($(arg_exprs[1]) >= $(arg_exprs[2]) ? 1.0 : 0.0)"
#     elseif func === (<=) && length(arg_exprs) == 2
#         return "($(arg_exprs[1]) <= $(arg_exprs[2]) ? 1.0 : 0.0)"
#     elseif func === (==) && length(arg_exprs) == 2
#         return "($(arg_exprs[1]) == $(arg_exprs[2]) ? 1.0 : 0.0)"
#     elseif func === (!=) && length(arg_exprs) == 2
#         return "($(arg_exprs[1]) != $(arg_exprs[2]) ? 1.0 : 0.0)"
        
#     # Additional mathematical functions
#     elseif func === min && length(arg_exprs) == 2
#         return "min($(arg_exprs[1]), $(arg_exprs[2]))"
#     elseif func === max && length(arg_exprs) == 2
#         return "max($(arg_exprs[1]), $(arg_exprs[2]))"
#     elseif func === min && length(arg_exprs) > 2
#         return "min(" * join(arg_exprs, ", ") * ")"
#     elseif func === max && length(arg_exprs) > 2
#         return "max(" * join(arg_exprs, ", ") * ")"
        
#     # Rounding functions
#     elseif func === round && length(arg_exprs) == 1
#         return "round($(arg_exprs[1]))"
#     elseif func === floor && length(arg_exprs) == 1
#         return "floor($(arg_exprs[1]))"
#     elseif func === ceil && length(arg_exprs) == 1
#         return "ceil($(arg_exprs[1]))"
#     elseif func === sign && length(arg_exprs) == 1
#         return "sign($(arg_exprs[1]))"
        
#     else
#         error("Function $func with $(length(arg_exprs)) arguments not supported")
#     end
# end

# ###############################################################################
# # MULTI-OUTPUT HANDLERS
# ###############################################################################

# """
# Generate statements for CombinedEvaluator (mirrors MatrixTerm).
# """
# function generate_combined_statements(evaluator::CombinedEvaluator, start_pos::Int)
#     instructions = String[]
#     current_pos = start_pos
    
#     for sub_evaluator in evaluator.sub_evaluators
#         sub_instructions, new_pos = generate_statements_recursive(sub_evaluator, current_pos)
#         append!(instructions, sub_instructions)
#         current_pos = new_pos
#     end
    
#     return instructions, current_pos
# end

# """
# Generate statements for multi-output evaluators.
# """
# function generate_multi_output_statements(evaluator::AbstractEvaluator, start_pos::Int)
    
#     if evaluator isa CategoricalEvaluator
#         return generate_categorical_statements(evaluator, start_pos)
        
#     elseif evaluator isa InteractionEvaluator
#         return generate_interaction_statements(evaluator, start_pos)
        
#     elseif evaluator isa ZScoreEvaluator && output_width(evaluator) > 1
#         return generate_zscore_statements(evaluator, start_pos)
        
#     else
#         error("Multi-output statement generation for $(typeof(evaluator)) not implemented")
#     end
# end

# """
# Generate statements for multi-contrast categorical evaluators.

# Enhanced categorical statements with size limits.
# """
# function generate_categorical_statements(evaluator::CategoricalEvaluator, start_pos::Int)
#     col = evaluator.column
#     n_levels = evaluator.n_levels
#     contrast_matrix = evaluator.contrast_matrix
#     width = size(contrast_matrix, 2)
    
#     # Check size limits
#     if n_levels > CATEGORICAL_LOOKUP_LIMIT
#         error("Categorical variable '$col' has $n_levels levels (limit: $CATEGORICAL_LOOKUP_LIMIT). " *
#               "Consider:\n" *
#               "1. Reducing the number of levels\n" *
#               "2. Grouping similar categories\n" *
#               "3. Using continuous variables instead")
#     end
    
#     instructions = String[]
    
#     # Generate categorical lookup variables
#     cat_var = next_var("cat")
#     level_var = next_var("level")
    
#     push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
#     push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
#     push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
    
#     # Choose generation strategy based on size
#     if n_levels <= CATEGORICAL_INLINE_LIMIT
#         # Use ternary expressions for smaller categoricals
#         generate_ternary_categorical_assignments!(instructions, contrast_matrix, level_var, n_levels, width, start_pos)
#     else
#         # Use lookup arrays for larger categoricals
#         generate_lookup_categorical_assignments!(instructions, contrast_matrix, level_var, n_levels, width, start_pos)
#     end
    
#     return instructions, start_pos + width
# end

# """
# Generate ternary-based categorical assignments.
# """
# function generate_ternary_categorical_assignments!(instructions, contrast_matrix, level_var, n_levels, width, start_pos)
#     for j in 1:width
#         output_pos = start_pos + j - 1
#         values = [contrast_matrix[i, j] for i in 1:n_levels]
        
#         if n_levels == 2
#             contrast_expr = "$level_var == 1 ? $(values[1]) : $(values[2])"
#         else
#             contrast_expr = "$level_var == 1 ? $(values[1])"
#             for i in 2:(n_levels-1)
#                 contrast_expr *= " : $level_var == $i ? $(values[i])"
#             end
#             contrast_expr *= " : $(values[n_levels])"
#         end
#         push!(instructions, "@inbounds row_vec[$output_pos] = $contrast_expr")
#     end
# end

# """
# Generate lookup-based categorical assignments.
# """
# function generate_lookup_categorical_assignments!(instructions, contrast_matrix, level_var, n_levels, width, start_pos)
#     for j in 1:width
#         output_pos = start_pos + j - 1
#         values = [contrast_matrix[i, j] for i in 1:n_levels]
        
#         # Create lookup array as local variable
#         lookup_var = next_var("lookup")
#         values_str = "[" * join(string.(values), ", ") * "]"
#         push!(instructions, "@inbounds $lookup_var = $values_str")
#         push!(instructions, "@inbounds row_vec[$output_pos] = $lookup_var[$level_var]")
#     end
# end

# """
# Enhanced interaction statements with complexity checking.
# """
# function generate_interaction_statements(evaluator::InteractionEvaluator, start_pos::Int)
#     # Check complexity first
#     is_complex, message = check_interaction_complexity(evaluator)
#     if is_complex
#         error("Interaction too complex: $message")
#     end
    
#     components = evaluator.components
#     n_components = length(components)
    
#     if n_components == 0
#         instructions = ["@inbounds row_vec[$start_pos] = 1.0"]
#         return instructions, start_pos + 1
        
#     elseif n_components == 1
#         return generate_statements_recursive(components[1], start_pos)
        
#     elseif n_components == 2
#         return generate_binary_interaction_statements(components, start_pos)
        
#     elseif n_components == 3
#         return generate_ternary_interaction_statements_phase2d(components, start_pos)
        
#     else
#         return generate_nary_interaction_statements_phase2d(components, start_pos)
#     end
# end

# """
# Generate statements for ZScore evaluators with multi-output.
# """
# function generate_zscore_statements(evaluator::ZScoreEvaluator, start_pos::Int)
#     underlying = evaluator.underlying
#     center = evaluator.center
#     scale = evaluator.scale
#     width = output_width(underlying)
    
#     instructions = String[]
    
#     # Generate statements for underlying evaluator
#     underlying_instructions, next_pos = generate_statements_recursive(underlying, start_pos)
#     append!(instructions, underlying_instructions)
    
#     # Apply Z-score transformation to each position
#     for i in 0:(width-1)
#         pos = start_pos + i
#         temp_var = next_var("zscore_temp")
#         push!(instructions, "@inbounds $temp_var = row_vec[$pos]")
#         push!(instructions, "@inbounds row_vec[$pos] = ($temp_var - $center) / $scale")
#     end
    
#     return instructions, next_pos
# end

# """
# Generate statements for complex single-output evaluators.
# """
# function generate_complex_single_output(evaluator::AbstractEvaluator, start_pos::Int)
#     # Try to convert to expression using helper functions
#     if can_generate_as_expression(evaluator)
#         expr = generate_complex_component_as_expression(evaluator)
#         instructions = ["@inbounds row_vec[$start_pos] = $expr"]
#         return instructions, start_pos + 1
#     else
#         error("Complex single output for $(typeof(evaluator)) not implemented")
#     end
# end

# ###############################################################################
# # BINARY INTERACTION HANDLERS (FIXED - NO SCRATCH POSITIONS)
# ###############################################################################

# """
# Generate statements for binary interactions.
# """
# function generate_binary_interaction_statements(components::Vector{AbstractEvaluator}, start_pos::Int)
#     comp1, comp2 = components[1], components[2]
#     w1, w2 = output_width(comp1), output_width(comp2)
    
#     if w1 == 1 && w2 == 1
#         # Scalar × Scalar
#         return generate_scalar_scalar_interaction(comp1, comp2, start_pos)
#     elseif w1 == 1 && w2 > 1
#         # Scalar × Vector
#         return generate_enhanced_scalar_vector_interaction(comp1, comp2, start_pos, w2)
#     elseif w1 > 1 && w2 == 1
#         # Vector × Scalar
#         return generate_enhanced_vector_scalar_interaction(comp1, comp2, start_pos, w1)
#     else
#         # Vector × Vector
#         return generate_vector_vector_interaction(comp1, comp2, w1, w2, start_pos)
#     end
# end

# ###############################################################################
# # HELPER FUNCTIONS (NEW - SOLVE SCRATCH POSITION PROBLEM)
# ###############################################################################

# """
# Check if an evaluator can be expressed as a simple inline expression.
# """
# function is_simple_expression(evaluator::AbstractEvaluator)
#     if evaluator isa ConstantEvaluator || evaluator isa ContinuousEvaluator
#         return true
        
#     elseif evaluator isa CategoricalEvaluator
#         return size(evaluator.contrast_matrix, 2) == 1 && evaluator.n_levels <= 6
        
#     elseif evaluator isa FunctionEvaluator
#         return all(arg -> is_simple_expression(arg), evaluator.arg_evaluators)
        
#     elseif evaluator isa ScaledEvaluator
#         return is_simple_expression(evaluator.evaluator)
        
#     elseif evaluator isa ProductEvaluator
#         return output_width(evaluator) == 1 && all(comp -> is_simple_expression(comp), evaluator.components)
        
#     elseif evaluator isa ZScoreEvaluator
#         return output_width(evaluator) == 1 && is_simple_expression(evaluator.underlying)
        
#     elseif evaluator isa InteractionEvaluator
#         return output_width(evaluator) == 1 && all(comp -> is_simple_expression(comp), evaluator.components)
        
#     else
#         return false
#     end
# end

# """
# Check if an evaluator produces multiple outputs.
# """
# function is_multi_output_evaluator(evaluator::AbstractEvaluator)
#     return evaluator isa CombinedEvaluator ||
#            (evaluator isa CategoricalEvaluator && size(evaluator.contrast_matrix, 2) > 1) ||
#            (evaluator isa InteractionEvaluator && output_width(evaluator) > 1) ||
#            (evaluator isa ZScoreEvaluator && output_width(evaluator) > 1)
# end

# """
# Generate a complex component directly as an expression without using scratch space.
# """
# function generate_complex_component_as_expression(evaluator::AbstractEvaluator)
#     if evaluator isa ConstantEvaluator
#         return string(evaluator.value)
        
#     elseif evaluator isa ContinuousEvaluator
#         return "Float64(data.$(evaluator.column)[row_idx])"
        
#     elseif evaluator isa FunctionEvaluator
#         return generate_expression_recursive(evaluator)
        
#     elseif evaluator isa CategoricalEvaluator && size(evaluator.contrast_matrix, 2) == 1
#         return generate_categorical_expression(evaluator)
        
#     elseif evaluator isa ScaledEvaluator && is_simple_expression(evaluator.evaluator)
#         inner_expr = generate_expression_recursive(evaluator.evaluator)
#         return "($(inner_expr) * $(evaluator.scale_factor))"
        
#     elseif evaluator isa ProductEvaluator && all(is_simple_expression, evaluator.components)
#         component_exprs = [generate_expression_recursive(comp) for comp in evaluator.components]
#         return "(" * join(component_exprs, " * ") * ")"
        
#     else
#         error("Complex component $(typeof(evaluator)) cannot be expressed as single expression")
#     end
# end

# """
# Check if a component can be generated as a single expression.
# """
# function can_generate_as_expression(evaluator::AbstractEvaluator)
#     try
#         generate_complex_component_as_expression(evaluator)
#         return true
#     catch
#         return false
#     end
# end

# """
# Generate a component value into a variable using safe methods.
# """
# function generate_component_to_variable(evaluator::AbstractEvaluator, variable_name::String)
#     instructions = String[]
    
#     if can_generate_as_expression(evaluator)
#         expr = generate_complex_component_as_expression(evaluator)
#         push!(instructions, "@inbounds $variable_name = $expr")
#     else
#         error("Component $(typeof(evaluator)) is too complex for current expression generation")
#     end
    
#     return instructions
# end

# """
# Generate values for a vector component into a list of variables.
# """
# function generate_vector_component_to_variables(evaluator::AbstractEvaluator, var_names::Vector{String})
#     instructions = String[]
    
#     if evaluator isa CategoricalEvaluator
#         col = evaluator.column
#         n_levels = evaluator.n_levels
#         contrast_matrix = evaluator.contrast_matrix
#         width = length(var_names)
        
#         # Generate categorical lookup
#         cat_var = next_var("cat")
#         level_var = next_var("level")
#         push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
#         push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
#         push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
        
#         # Generate each contrast value
#         for j in 1:width
#             values = [contrast_matrix[i, j] for i in 1:n_levels]
#             var_name = var_names[j]
            
#             if n_levels <= 6
#                 if n_levels == 2
#                     contrast_expr = "$level_var == 1 ? $(values[1]) : $(values[2])"
#                 else
#                     contrast_expr = "$level_var == 1 ? $(values[1])"
#                     for i in 2:(n_levels-1)
#                         contrast_expr *= " : $level_var == $i ? $(values[i])"
#                     end
#                     contrast_expr *= " : $(values[n_levels])"
#                 end
#                 push!(instructions, "@inbounds $var_name = $contrast_expr")
#             else
#                 lookup_var = next_var("lookup")
#                 values_str = "[" * join(string.(values), ", ") * "]"
#                 push!(instructions, "@inbounds $lookup_var = $values_str")
#                 push!(instructions, "@inbounds $var_name = $lookup_var[$level_var]")
#             end
#         end
        
#     elseif evaluator isa ContinuousEvaluator && length(var_names) == 1
#         push!(instructions, "@inbounds $(var_names[1]) = Float64(data.$(evaluator.column)[row_idx])")
        
#     else
#         error("Vector component generation for $(typeof(evaluator)) with $(length(var_names)) variables not implemented")
#     end
    
#     return instructions
# end

# ###############################################################################
# # INTERACTION FUNCTIONS (FIXED - FROM PREVIOUS ARTIFACT)
# ###############################################################################

# """
# FIXED: Generate scalar × scalar interactions using pure expressions.
# """
# function generate_scalar_scalar_interaction(comp1::AbstractEvaluator, comp2::AbstractEvaluator, start_pos::Int)
#     instructions = String[]
    
#     if is_simple_expression(comp1) && is_simple_expression(comp2)
#         expr1 = generate_expression_recursive(comp1)
#         expr2 = generate_expression_recursive(comp2)
#         push!(instructions, "@inbounds row_vec[$start_pos] = ($expr1) * ($expr2)")
        
#     elseif is_simple_expression(comp1) && can_generate_as_expression(comp2)
#         expr1 = generate_expression_recursive(comp1)
#         comp2_expr = generate_complex_component_as_expression(comp2)
#         push!(instructions, "@inbounds row_vec[$start_pos] = ($expr1) * ($comp2_expr)")
        
#     elseif can_generate_as_expression(comp1) && is_simple_expression(comp2)
#         comp1_expr = generate_complex_component_as_expression(comp1)
#         expr2 = generate_expression_recursive(comp2)
#         push!(instructions, "@inbounds row_vec[$start_pos] = ($comp1_expr) * ($expr2)")
        
#     else
#         # Both complex - use variables
#         temp1 = next_var("comp1_val")
#         temp2 = next_var("comp2_val")
        
#         comp1_instructions = generate_component_to_variable(comp1, temp1)
#         comp2_instructions = generate_component_to_variable(comp2, temp2)
#         append!(instructions, comp1_instructions)
#         append!(instructions, comp2_instructions)
        
#         push!(instructions, "@inbounds row_vec[$start_pos] = $temp1 * $temp2")
#     end
    
#     return instructions, start_pos + 1
# end

# """
# FIXED: Enhanced scalar × vector interaction.
# """
# function generate_enhanced_scalar_vector_interaction(scalar_comp::AbstractEvaluator, vector_comp::AbstractEvaluator, start_pos::Int, width::Int)
#     instructions = String[]
    
#     # Get scalar value
#     if is_simple_expression(scalar_comp)
#         scalar_expr = generate_expression_recursive(scalar_comp)
#     else
#         scalar_var = next_var("scalar")
#         scalar_instructions = generate_component_to_variable(scalar_comp, scalar_var)
#         append!(instructions, scalar_instructions)
#         scalar_expr = scalar_var
#     end
    
#     # Handle vector component
#     if vector_comp isa CategoricalEvaluator
#         return generate_scalar_categorical_interaction(scalar_expr, vector_comp, instructions, start_pos, width)
#     elseif vector_comp isa ContinuousEvaluator
#         vector_expr = "Float64(data.$(vector_comp.column)[row_idx])"
#         push!(instructions, "@inbounds row_vec[$start_pos] = ($scalar_expr) * ($vector_expr)")
#         return instructions, start_pos + 1
#     else
#         error("Vector component $(typeof(vector_comp)) not supported in scalar × vector interaction")
#     end
# end

# """
# Enhanced vector × scalar interaction.
# """
# function generate_enhanced_vector_scalar_interaction(vector_comp::AbstractEvaluator, scalar_comp::AbstractEvaluator, start_pos::Int, width::Int)
#     return generate_enhanced_scalar_vector_interaction(scalar_comp, vector_comp, start_pos, width)
# end

# """
# Scalar × categorical interaction.
# """
# function generate_scalar_categorical_interaction(scalar_expr::String, cat_comp::CategoricalEvaluator, instructions::Vector{String}, start_pos::Int, width::Int)
#     col = cat_comp.column
#     n_levels = cat_comp.n_levels
#     contrast_matrix = cat_comp.contrast_matrix
    
#     # Generate categorical lookup
#     cat_var = next_var("cat")
#     level_var = next_var("level")
#     push!(instructions, "@inbounds $cat_var = data.$col[row_idx]")
#     push!(instructions, "@inbounds $level_var = $cat_var isa CategoricalValue ? levelcode($cat_var) : 1")
#     push!(instructions, "@inbounds $level_var = clamp($level_var, 1, $n_levels)")
    
#     # Generate scaled contrasts
#     for j in 1:width
#         output_pos = start_pos + j - 1
#         values = [contrast_matrix[i, j] for i in 1:n_levels]
        
#         if n_levels <= 6
#             if n_levels == 2
#                 contrast_expr = "$level_var == 1 ? $(values[1]) : $(values[2])"
#             else
#                 contrast_expr = "$level_var == 1 ? $(values[1])"
#                 for i in 2:(n_levels-1)
#                     contrast_expr *= " : $level_var == $i ? $(values[i])"
#                 end
#                 contrast_expr *= " : $(values[n_levels])"
#             end
#             push!(instructions, "@inbounds row_vec[$output_pos] = ($scalar_expr) * ($contrast_expr)")
#         else
#             lookup_var = next_var("lookup")
#             values_str = "[" * join(string.(values), ", ") * "]"
#             push!(instructions, "@inbounds $lookup_var = $values_str")
#             push!(instructions, "@inbounds row_vec[$output_pos] = ($scalar_expr) * $lookup_var[$level_var]")
#         end
#     end
    
#     return instructions, start_pos + width
# end

# """
# FIXED: Vector × Vector interaction.
# """
# function generate_vector_vector_interaction(comp1::AbstractEvaluator, comp2::AbstractEvaluator, w1::Int, w2::Int, start_pos::Int)
#     instructions = String[]
#     total_width = w1 * w2
    
#     # Generate variables for each component
#     comp1_vars = [next_var("c1_$i") for i in 1:w1]
#     comp2_vars = [next_var("c2_$i") for i in 1:w2]
    
#     # Generate component values
#     comp1_instructions = generate_vector_component_to_variables(comp1, comp1_vars)
#     comp2_instructions = generate_vector_component_to_variables(comp2, comp2_vars)
#     append!(instructions, comp1_instructions)
#     append!(instructions, comp2_instructions)
    
#     # Generate Kronecker product
#     output_idx = start_pos
#     for j in 1:w2
#         for i in 1:w1
#             push!(instructions, "@inbounds row_vec[$output_idx] = $(comp1_vars[i]) * $(comp2_vars[j])")
#             output_idx += 1
#         end
#     end
    
#     return instructions, start_pos + total_width
# end

# """
# Enhanced three-way interaction with size checking.
# """
# function generate_ternary_interaction_statements_phase2d(components::Vector{AbstractEvaluator}, start_pos::Int)
#     w1, w2, w3 = output_width(components[1]), output_width(components[2]), output_width(components[3])
#     total_width = w1 * w2 * w3
    
#     # Enhanced size checking
#     if total_width > THREEWAY_INTERACTION_LIMIT
#         component_info = ["$(typeof(comp).name.name)(width=$w)" for (comp, w) in zip(components, [w1, w2, w3])]
#         error("Three-way interaction too large: $(join(component_info, " × ")) = $total_width terms (limit: $THREEWAY_INTERACTION_LIMIT).\n" *
#               "Consider:\n" *
#               "1. Using fewer categorical levels\n" *
#               "2. Simplifying to two-way interactions\n" *
#               "3. Using continuous variables instead of categoricals")
#     end
    
#     instructions = String[]
#     comp_vars = Vector{Vector{String}}(undef, 3)
    
#     # Generate variables for each component
#     for (i, comp) in enumerate(components)
#         width = output_width(comp)
#         comp_vars[i] = [next_var("c$(i)_$j") for j in 1:width]
#         comp_instructions = generate_vector_component_to_variables(comp, comp_vars[i])
#         append!(instructions, comp_instructions)
#     end
    
#     # Generate triple product
#     output_idx = start_pos
#     for k in 1:w3
#         for j in 1:w2
#             for i in 1:w1
#                 push!(instructions, "@inbounds row_vec[$output_idx] = $(comp_vars[1][i]) * $(comp_vars[2][j]) * $(comp_vars[3][k])")
#                 output_idx += 1
#             end
#         end
#     end
    
#     return instructions, start_pos + total_width
# end

# """
# FIXED: N-ary interaction generation.
# """
# function generate_nary_interaction_statements_phase2d(components::Vector{AbstractEvaluator}, start_pos::Int)
#     n_components = length(components)
#     widths = [output_width(comp) for comp in components]
#     total_width = prod(widths)
    
#     if total_width > 500
#         error("High-order interaction with $n_components components and $total_width terms is too complex")
#     end
    
#     instructions = String[]
#     all_comp_vars = Vector{Vector{String}}(undef, n_components)
    
#     # Generate variables for each component
#     for (i, comp) in enumerate(components)
#         width = widths[i]
#         all_comp_vars[i] = [next_var("comp$(i)_$j") for j in 1:width]
#         comp_instructions = generate_vector_component_to_variables(comp, all_comp_vars[i])
#         append!(instructions, comp_instructions)
#     end
    
#     # Generate n-way product
#     for linear_idx in 0:(total_width-1)
#         indices = linear_to_multi_index_custom(linear_idx, widths)
        
#         product_terms = String[]
#         for comp_idx in 1:n_components
#             element_idx = indices[comp_idx] + 1  # Convert to 1-based
#             push!(product_terms, all_comp_vars[comp_idx][element_idx])
#         end
        
#         product_expr = join(product_terms, " * ")
#         output_pos = start_pos + linear_idx
#         push!(instructions, "@inbounds row_vec[$output_pos] = $product_expr")
#     end
    
#     return instructions, start_pos + total_width
# end

# ###############################################################################
# # UTILITY FUNCTIONS
# ###############################################################################

# """
# Convert linear index to multi-dimensional indices for n-way interactions.
# """
# function linear_to_multi_index_custom(linear_idx::Int, dimensions::Vector{Int})
#     n_dims = length(dimensions)
#     indices = Vector{Int}(undef, n_dims)
    
#     remaining = linear_idx
#     for i in 1:n_dims
#         indices[i] = remaining % dimensions[i]
#         remaining = remaining ÷ dimensions[i]
#     end
    
#     return indices
# end

# ###############################################################################
# # USER-FRIENDLY ERROR MESSAGES
# ###############################################################################

# """
# Provide helpful suggestions for common complexity issues.
# """
# function suggest_formula_simplification(evaluator::AbstractEvaluator)
#     suggestions = String[]
    
#     if evaluator isa CombinedEvaluator
#         # Count different types of terms
#         categorical_terms = count(e -> e isa CategoricalEvaluator, evaluator.sub_evaluators)
#         interaction_terms = count(e -> e isa InteractionEvaluator, evaluator.sub_evaluators)
        
#         if categorical_terms > 10
#             push!(suggestions, "Consider combining some categorical variables")
#         end
        
#         if interaction_terms > 5
#             push!(suggestions, "Consider reducing the number of interaction terms")
#         end
#     end
    
#     if evaluator isa InteractionEvaluator && length(evaluator.components) > 3
#         push!(suggestions, "Consider using lower-order interactions (2-way or 3-way maximum)")
#     end
    
#     return suggestions
# end

# ###############################################################################
# # CONFIGURATION FUNCTIONS
# ###############################################################################

# """
# Allow users to adjust complexity limits if needed.
# """
# function set_complexity_limits(;
#     categorical_inline_limit = CATEGORICAL_INLINE_LIMIT,
#     categorical_lookup_limit = CATEGORICAL_LOOKUP_LIMIT,
#     interaction_term_limit = INTERACTION_TERM_LIMIT,
#     threeway_interaction_limit = THREEWAY_INTERACTION_LIMIT,
#     total_formula_width_limit = TOTAL_FORMULA_WIDTH_LIMIT
# )
#     # This would update global constants (in a real implementation)
#     @warn "Complexity limit adjustment not implemented in this version"
# end

# """
# Get current complexity limits for user reference.
# """
# function get_complexity_limits()
#     return (
#         categorical_inline_limit = CATEGORICAL_INLINE_LIMIT,
#         categorical_lookup_limit = CATEGORICAL_LOOKUP_LIMIT,
#         interaction_term_limit = INTERACTION_TERM_LIMIT,
#         threeway_interaction_limit = THREEWAY_INTERACTION_LIMIT,
#         total_formula_width_limit = TOTAL_FORMULA_WIDTH_LIMIT
#     )
# end

# # Export the new functions
# export ComplexityError, set_complexity_limits, get_complexity_limits
