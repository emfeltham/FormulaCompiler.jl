# generated_function_interactions.jl
# Metaprogramming-generated specialized types for function×interaction zero allocations

using CategoricalArrays: levelcode

###############################################################################
# COMMON PATTERNS THAT CAUSE ALLOCATIONS
###############################################################################

# From test_cases.jl results:
# - "Function × Categorical": 137ns, 1 allocation: 96 bytes  
# - "Continuous × Function": 76ns, 1 allocation: 96 bytes
# - "Multiple functions × categorical": 455ns, 6 allocations: 576 bytes

# Target the most common function×component patterns
const COMMON_FUNCTIONS = [:log, :exp, :sqrt, :abs]
const INTERACTION_PATTERNS = [
    (:function, :continuous),    # log(z) * x
    (:continuous, :function),    # x * log(z)  
    (:function, :categorical),   # log(z) * group
    (:categorical, :function),   # group * log(z)
]

###############################################################################
# HELPER FUNCTIONS FOR CODE GENERATION (DEFINED FIRST)
###############################################################################

"""
Generate domain-safe function application expression for each supported function.
"""
function generate_function_application_expr(func::Symbol)
    if func == :log
        :(func_val > 0.0 ? log(func_val) : (func_val == 0.0 ? -Inf : NaN))
    elseif func == :exp
        :(exp(clamp(func_val, -700.0, 700.0)))  # Prevent overflow
    elseif func == :sqrt
        :(func_val >= 0.0 ? sqrt(func_val) : NaN)
    elseif func == :abs
        :(abs(func_val))
    else
        # Fallback for any other functions
        :($func(func_val))
    end
end

"""
Generate interaction computation expression based on the component pattern.
"""
function generate_component_interaction_expr(pos1, pos2)
    if pos1 == :function && pos2 == :continuous
        # log(z) * x pattern - function first, component second
        # The component is continuous (width=1), so always use index 1
        # Use direct data access instead of get_component_interaction_value to avoid allocations
        quote
            component_val = get_data_value_specialized(data, eval.component.column, row_idx)
            interaction_result = func_result * component_val
        end
    elseif pos1 == :continuous && pos2 == :function  
        # x * log(z) pattern - component first, function second
        # The component is continuous (width=1), so always use index 1
        quote
            component_val = get_data_value_specialized(data, eval.component.column, row_idx)
            interaction_result = component_val * func_result
        end
    elseif pos1 == :function && pos2 == :categorical
        # log(z) * group pattern - function first, component second
        # The component is categorical, use j to select the right contrast column
        quote
            # For categorical, get the level code and use contrast matrix exactly like the system
            column_data = getproperty(data, eval.component.column)
            level_code = Int(levelcode(column_data[row_idx]))
            component_val = eval.component.contrast_matrix[level_code, j]
            interaction_result = func_result * component_val
        end
    elseif pos1 == :categorical && pos2 == :function
        # group * log(z) pattern - component first, function second
        # The component is categorical, use i to select the right contrast column
        quote
            # For categorical, get the level code and use contrast matrix exactly like the system
            column_data = getproperty(data, eval.component.column)
            level_code = Int(levelcode(column_data[row_idx]))
            component_val = eval.component.contrast_matrix[level_code, i]
            interaction_result = component_val * func_result
        end
    else
        error("Unsupported interaction pattern: $pos1 × $pos2")
    end
end

###############################################################################
# GENERATED SPECIALIZED INTERACTION TYPES
###############################################################################

"""
Generate specialized interaction types for common function×component patterns.
These replace InteractionScratchReference + FunctionPreEvalOperation coordination
with direct, zero-allocation execution.
"""

# Generate all combinations of functions and patterns
for func in COMMON_FUNCTIONS
    for (pos1, pos2) in INTERACTION_PATTERNS
        
        # Create the type name (e.g., LogFunctionContinuousInteraction)
        type_name = Symbol(
            titlecase(string(func)),
            titlecase(string(pos1)), 
            titlecase(string(pos2)),
            "Interaction"
        )
        
        @eval begin            
            # Specialized zero-allocation interaction type
            struct $type_name{ComponentType} <: AbstractEvaluator
                func_variable::Symbol                                    # Function input (e.g., :z for log(z))
                component::ComponentType                                 # Other component (continuous/categorical)  
                output_positions::Vector{Int}                           # Final output positions
                interaction_pattern::NTuple{N, Tuple{Int,Int}} where N  # Pre-computed Kronecker pattern
            end
            
            # Interface methods for the specialized type
            output_width(eval::$type_name) = length(eval.output_positions)
            get_positions(eval::$type_name) = eval.output_positions
            get_scratch_positions(eval::$type_name) = Int[]  # No scratch space needed!
            max_scratch_needed(eval::$type_name) = 0         # Zero scratch space required
            
            # Zero-allocation execution method
            @inline function execute_interaction!(
                eval::$type_name{CT},
                output::AbstractVector{Float64},
                data::NamedTuple,
                row_idx::Int
            ) where CT
                
                # Get function input value directly (no scratch space)
                func_val = get_data_value_specialized(data, eval.func_variable, row_idx)
                
                # Apply function with domain checking (specialized per function)
                func_result = $(generate_function_application_expr(func))
                
                # Execute interaction pattern directly  
                @inbounds for (pattern_idx, (i, j)) in enumerate(eval.interaction_pattern)
                    
                    # Get component values based on pattern position
                    $(generate_component_interaction_expr(pos1, pos2))
                    
                    # Write result to output
                    output_pos = eval.output_positions[pattern_idx]
                    output[output_pos] = interaction_result
                end
                
                return nothing
            end
        end
    end
end


###############################################################################
# PATTERN DETECTION FOR COMPILATION INTEGRATION
###############################################################################

"""
    is_specialized_function_interaction(func_eval, other_component) -> Bool

Check if a function×component combination matches a generated specialized pattern.
"""
function is_specialized_function_interaction(func_eval::FunctionEvaluator, other_component::AbstractEvaluator)
    # Check if function is in our supported list
    func_name = extract_function_name(func_eval)
    if !(func_name in COMMON_FUNCTIONS)
        return false
    end
    
    # Check if it's a simple unary function (not nested)
    if !is_simple_unary_function(func_eval)
        return false
    end
    
    # Check if other component is supported
    if !(other_component isa ContinuousEvaluator || other_component isa CategoricalEvaluator)
        return false
    end
    
    return true
end

"""
    create_specialized_function_interaction(func_eval, other_component, positions) -> AbstractEvaluator

Create a specialized interaction type for the given function×component pattern.
"""
function create_specialized_function_interaction(func_eval::FunctionEvaluator, other_component::AbstractEvaluator, positions::Vector{Int})
    func_name = extract_function_name(func_eval)
    func_variable = extract_function_input_variable(func_eval)
    
    # Determine pattern and create appropriate type
    if other_component isa ContinuousEvaluator
        type_name = Symbol(titlecase(string(func_name)), "FunctionContinuousInteraction")
    elseif other_component isa CategoricalEvaluator  
        type_name = Symbol(titlecase(string(func_name)), "FunctionCategoricalInteraction")
    else
        error("Unsupported component type: $(typeof(other_component))")
    end
    
    # Get the generated type
    SpecializedType = @eval $type_name
    
    # Compute interaction pattern (reuse existing logic)
    interaction_pattern = compute_interaction_pattern_tuple(1, output_width(other_component))
    
    return SpecializedType(func_variable, other_component, positions, interaction_pattern)
end

###############################################################################
# HELPER FUNCTIONS FOR FUNCTION ANALYSIS
###############################################################################

"""Extract the main function name from a FunctionEvaluator."""
function extract_function_name(func_eval::FunctionEvaluator)
    # The function is stored directly in func_eval.func
    func = func_eval.func
    
    # Convert function to symbol for pattern matching
    if func === log
        return :log
    elseif func === exp
        return :exp
    elseif func === sqrt
        return :sqrt
    elseif func === abs
        return :abs
    else
        # Return the function itself for other cases
        return Symbol(func)
    end
end

"""Check if this is a simple unary function (not nested)."""
function is_simple_unary_function(func_eval::FunctionEvaluator)
    # Check if it has exactly one argument evaluator
    if length(func_eval.arg_evaluators) != 1
        return false
    end
    
    # Check if the argument is a simple continuous variable (not nested function)
    arg_eval = func_eval.arg_evaluators[1]
    return arg_eval isa ContinuousEvaluator
end

"""Extract the input variable for a simple unary function."""
function extract_function_input_variable(func_eval::FunctionEvaluator)
    # For simple unary functions, get the variable from the first argument
    if length(func_eval.arg_evaluators) == 1
        arg_eval = func_eval.arg_evaluators[1]
        if arg_eval isa ContinuousEvaluator
            return arg_eval.column
        end
    end
    
    error("Cannot extract input variable from complex function: $(func_eval.func)")
end


###############################################################################
# TYPE-BASED DISPATCH FOR ZERO-ALLOCATION EXECUTION
###############################################################################

"""
    is_specialized_interaction_type(evaluator) -> Bool

Check if an evaluator is a specialized interaction type without method lookups.
This avoids allocations from hasmethod() calls.
"""
is_specialized_interaction_type(::Any) = false

# Generate type checkers for all specialized interaction types
for func in COMMON_FUNCTIONS
    for (pos1, pos2) in INTERACTION_PATTERNS
        type_name = Symbol(
            titlecase(string(func)),
            titlecase(string(pos1)), 
            titlecase(string(pos2)),
            "Interaction"
        )
        
        @eval is_specialized_interaction_type(::$type_name) = true
    end
end

# Export the main integration functions
export is_specialized_function_interaction, create_specialized_function_interaction, is_specialized_interaction_type