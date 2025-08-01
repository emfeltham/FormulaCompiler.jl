###############################################################################
# FUNCTION DATA TYPES
###############################################################################

"""
    ArgumentData

Data for a single function argument (can be constant, continuous, or nested function).
Uses Any for value to avoid circular reference.
"""
struct ArgumentData
    arg_type::Symbol                      # :constant, :continuous, :function
    value::Any                            # Float64, Symbol, or FunctionData
end

"""
    FunctionData

Pre-computed data for function evaluation.
"""
struct FunctionData
    func::Function                        # The function to call (log, exp, etc.)
    arg_data::Vector{ArgumentData}        # Data for each argument
    position::Int                         # Output position
end

"""
    FunctionOp

Compile-time encoding of function operations.
"""
struct FunctionOp end

###############################################################################
# COMPREHENSIVE FORMULA DATA TYPES
###############################################################################

"""
    ComprehensiveFormulaOp{ConstOp, ContOp, CatOp, FuncOp}

Combined operation encoding for comprehensive formulas.
"""
struct ComprehensiveFormulaOp{ConstOp, ContOp, CatOp, FuncOp}
    constants::ConstOp
    continuous::ContOp
    categorical::CatOp
    functions::FuncOp                     # FunctionOp
end

###############################################################################
# FUNCTION ANALYSIS FUNCTIONS
###############################################################################

"""
    analyze_function_operations(evaluator::CombinedEvaluator) -> (Vector{FunctionData}, FunctionOp)

Extract function data from a CombinedEvaluator's function evaluators.
"""
function analyze_function_operations(evaluator::CombinedEvaluator)
    function_evaluators = evaluator.function_evaluators
    n_funcs = length(function_evaluators)
    
    if n_funcs == 0
        # No function operations - return empty vector
        return FunctionData[], FunctionOp()
    end
    
    # Create vector of FunctionData
    function_data = Vector{FunctionData}(undef, n_funcs)
    
    for i in 1:n_funcs
        func_eval = function_evaluators[i]
        
        # Analyze arguments
        arg_data = Vector{ArgumentData}(undef, length(func_eval.arg_evaluators))
        for j in 1:length(func_eval.arg_evaluators)
            arg_eval = func_eval.arg_evaluators[j]
            arg_data[j] = analyze_function_argument(arg_eval)
        end
        
        function_data[i] = FunctionData(
            func_eval.func,
            arg_data,
            func_eval.position
        )
    end
    
    return function_data, FunctionOp()
end

"""
    analyze_function_argument(evaluator::AbstractEvaluator) -> ArgumentData

Analyze a function argument and create ArgumentData.
"""
function analyze_function_argument(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator
        return ArgumentData(:constant, evaluator.value)
    elseif evaluator isa ContinuousEvaluator
        return ArgumentData(:continuous, evaluator.column)
    elseif evaluator isa FunctionEvaluator
        # Nested function - recursively analyze
        nested_arg_data = Vector{ArgumentData}(undef, length(evaluator.arg_evaluators))
        for i in 1:length(evaluator.arg_evaluators)
            nested_arg_data[i] = analyze_function_argument(evaluator.arg_evaluators[i])
        end
        
        nested_function_data = FunctionData(
            evaluator.func,
            nested_arg_data,
            evaluator.position  # This will be overridden in context
        )
        
        return ArgumentData(:function, nested_function_data)
    else
        error("Unsupported function argument type: $(typeof(evaluator))")
    end
end

###############################################################################
# FUNCTION EXECUTION FUNCTIONS
###############################################################################

"""
    execute_function_operations!(function_data::Vector{FunctionData}, output, input_data, row_idx)

Execute multiple function operations.
"""
function execute_function_operations!(function_data::Vector{FunctionData}, output, input_data, row_idx)
    # Handle empty case
    if isempty(function_data)
        return nothing
    end
    
    # Process all function operations
    @inbounds for func_data in function_data
        result = evaluate_function_direct(func_data, input_data, row_idx)
        output[func_data.position] = result
    end
    return nothing
end

"""
    evaluate_function_direct(func_data::FunctionData, input_data, row_idx) -> Float64

Evaluate a function directly without apply_function_safe overhead.
"""
function evaluate_function_direct(func_data::FunctionData, input_data, row_idx)
    # Evaluate arguments
    n_args = length(func_data.arg_data)
    
    if n_args == 1
        # Single argument - most common case
        arg_val = evaluate_argument(func_data.arg_data[1], input_data, row_idx)
        return apply_function_direct_single(func_data.func, arg_val)
    elseif n_args == 2
        # Two arguments
        arg1_val = evaluate_argument(func_data.arg_data[1], input_data, row_idx)
        arg2_val = evaluate_argument(func_data.arg_data[2], input_data, row_idx)
        return apply_function_direct_binary(func_data.func, arg1_val, arg2_val)
    elseif n_args == 3
        # Three arguments (SEEMS TO BE SAME AS N>#)
        arg1_val = evaluate_argument(func_data.arg_data[1], input_data, row_idx)
        arg2_val = evaluate_argument(func_data.arg_data[2], input_data, row_idx)
        arg3_val = evaluate_argument(func_data.arg_data[3], input_data, row_idx)
        return apply_function_direct_varargs(func_data.func, arg1_val, arg2_val, arg3_val)
    else
        # General case for more arguments
        arg_values = Vector{Float64}(undef, n_args)
        for i in 1:n_args
            arg_values[i] = evaluate_argument(func_data.arg_data[i], input_data, row_idx)
        end
        return apply_function_direct_varargs(func_data.func, arg_values...)
    end
end

"""
    evaluate_argument(arg_data::ArgumentData, input_data, row_idx) -> Float64

Evaluate a single function argument.
"""
function evaluate_argument(arg_data::ArgumentData, input_data, row_idx)
    if arg_data.arg_type === :constant
        return Float64(arg_data.value)
    elseif arg_data.arg_type === :continuous
        col = arg_data.value::Symbol
        return Float64(get_data_value_specialized(input_data, col, row_idx))
    elseif arg_data.arg_type === :function
        nested_func_data = arg_data.value::FunctionData
        return evaluate_function_direct(nested_func_data, input_data, row_idx)
    else
        error("Unknown argument type: $(arg_data.arg_type)")
    end
end
