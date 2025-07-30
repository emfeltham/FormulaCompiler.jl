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
    ComprehensiveFormulaData{ConstData, ContData, CatData, FuncData}

Combined data for formulas with constants, continuous, categorical, and function variables.
"""
struct ComprehensiveFormulaData{ConstData, ContData, CatData, FuncData}
    constants::ConstData
    continuous::ContData
    categorical::CatData
    functions::FuncData                   # Vector{FunctionData}
end

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

"""
    analyze_evaluator_comprehensive(evaluator::AbstractEvaluator) -> (DataTuple, OpTuple)

Comprehensive analysis for constants, continuous, categorical, and function variables.
"""
function analyze_evaluator_comprehensive(evaluator::AbstractEvaluator)
    if evaluator isa CombinedEvaluator
        # Check that this only has simple operation types (no interactions yet)
        has_interactions = !isempty(evaluator.interaction_evaluators)
        
        if has_interactions
            error("Step 3 only supports constants, continuous, categorical, and functions. Found interactions.")
        end
        
        # Analyze all four operation types
        constant_data, constant_op = analyze_constant_operations(evaluator)
        continuous_data, continuous_op = analyze_continuous_operations(evaluator)
        categorical_data, categorical_op = analyze_categorical_operations(evaluator)
        function_data, function_op = analyze_function_operations(evaluator)
        
        # Combine into comprehensive formula data
        formula_data = ComprehensiveFormulaData(constant_data, continuous_data, categorical_data, function_data)
        formula_op = ComprehensiveFormulaOp(constant_op, continuous_op, categorical_op, function_op)
        
        return formula_data, formula_op
        
    else
        error("Step 3 only supports CombinedEvaluator with constants, continuous, categorical, and function operations")
    end
end

###############################################################################
# FUNCTION EXECUTION FUNCTIONS
###############################################################################

"""
    execute_function_operations!(function_data::Vector{FunctionData}, output, input_data, row_idx)

Execute multiple function operations with zero allocations.
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
        return apply_function_direct(func_data.func, arg_val)
    elseif n_args == 2
        # Two arguments
        arg1_val = evaluate_argument(func_data.arg_data[1], input_data, row_idx)
        arg2_val = evaluate_argument(func_data.arg_data[2], input_data, row_idx)
        return apply_function_direct(func_data.func, arg1_val, arg2_val)
    elseif n_args == 3
        # Three arguments
        arg1_val = evaluate_argument(func_data.arg_data[1], input_data, row_idx)
        arg2_val = evaluate_argument(func_data.arg_data[2], input_data, row_idx)
        arg3_val = evaluate_argument(func_data.arg_data[3], input_data, row_idx)
        return apply_function_direct(func_data.func, arg1_val, arg2_val, arg3_val)
    else
        # General case for more arguments
        arg_values = Vector{Float64}(undef, n_args)
        for i in 1:n_args
            arg_values[i] = evaluate_argument(func_data.arg_data[i], input_data, row_idx)
        end
        return apply_function_direct(func_data.func, arg_values...)
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

"""
    apply_function_direct(func::Function, args...) -> Float64

Apply function directly with domain checking, replacing apply_function_safe.
"""
function apply_function_direct(func::Function, args...)
    if length(args) == 1
        val = args[1]
        if func === log
            return val > 0.0 ? log(val) : (val == 0.0 ? -Inf : NaN)
        elseif func === exp
            return exp(clamp(val, -700.0, 700.0))  # Prevent overflow
        elseif func === sqrt
            return val ≥ 0.0 ? sqrt(val) : NaN
        elseif func === abs
            return abs(val)
        elseif func === sin
            return sin(val)
        elseif func === cos
            return cos(val)
        elseif func === tan
            return tan(val)
        else
            # Direct function call for other functions
            return Float64(func(val))
        end
    elseif length(args) == 2
        val1, val2 = args[1], args[2]
        if func === (+)
            return val1 + val2
        elseif func === (-)
            return val1 - val2
        elseif func === (*)
            return val1 * val2
        elseif func === (/)
            return val2 == 0.0 ? (val1 == 0.0 ? NaN : (val1 > 0.0 ? Inf : -Inf)) : val1 / val2
        elseif func === (^)
            if val1 == 0.0 && val2 < 0.0
                return Inf
            elseif val1 < 0.0 && !isinteger(val2)
                return NaN
            else
                return val1^val2
            end
        else
            return Float64(func(val1, val2))
        end
    else
        # General case
        return Float64(func(args...))
    end
end

"""
    execute_operation!(data::ComprehensiveFormulaData{ConstData, ContData, CatData, FuncData}, 
                      op::ComprehensiveFormulaOp{ConstOp, ContOp, CatOp, FuncOp}, 
                      output, input_data, row_idx) where {ConstData, ContData, CatData, FuncData, ConstOp, ContOp, CatOp, FuncOp}

Execute comprehensive formulas with constants, continuous, categorical, and function variables.
"""
function execute_operation!(data::ComprehensiveFormulaData{ConstData, ContData, CatData, FuncData}, 
                           op::ComprehensiveFormulaOp{ConstOp, ContOp, CatOp, FuncOp}, 
                           output, input_data, row_idx) where {ConstData, ContData, CatData, FuncData, ConstOp, ContOp, CatOp, FuncOp}
    
    # Execute constants
    execute_operation!(data.constants, op.constants, output, input_data, row_idx)
    
    # Execute continuous variables
    execute_operation!(data.continuous, op.continuous, output, input_data, row_idx)
    
    # Execute categorical variables
    execute_categorical_operations!(data.categorical, output, input_data, row_idx)
    
    # Execute functions
    execute_function_operations!(data.functions, output, input_data, row_idx)
    
    return nothing
end

###############################################################################
# COMPREHENSIVE COMPILATION FUNCTIONS
###############################################################################

"""
    create_specialized_formula_comprehensive(compiled_formula::CompiledFormula) -> SpecializedFormula

Convert a CompiledFormula to a SpecializedFormula (Step 3: includes functions).
"""
function create_specialized_formula_comprehensive(compiled_formula::CompiledFormula)
    # Analyze the evaluator tree with comprehensive support
    data_tuple, op_tuple = analyze_evaluator_comprehensive(compiled_formula.root_evaluator)
    
    # Create specialized formula
    return SpecializedFormula{typeof(data_tuple), typeof(op_tuple)}(
        data_tuple,
        op_tuple,
        compiled_formula.output_width
    )
end

"""
    compile_formula_specialized_comprehensive(model, data::NamedTuple) -> SpecializedFormula

Direct compilation to specialized formula (Step 3: includes functions).
"""
function compile_formula_specialized_comprehensive(model, data::NamedTuple)
    # Use existing compilation logic to build evaluator tree
    compiled = compile_formula(model, data)
    
    # Convert to comprehensive specialized form
    return create_specialized_formula_comprehensive(compiled)
end

###############################################################################
# COMPREHENSIVE UTILITY FUNCTIONS
###############################################################################

"""
    show_comprehensive_specialized_info(sf::SpecializedFormula)

Display detailed information about a comprehensive specialized formula.
"""
function show_comprehensive_specialized_info(sf::SpecializedFormula{D, O}) where {D, O}
    println("Comprehensive SpecializedFormula Information:")
    println("  Data type: $D")
    println("  Operation type: $O") 
    println("  Output width: $(sf.output_width)")
    
    if sf.data isa ComprehensiveFormulaData
        println("  Constants: $(sf.data.constants.values)")
        println("  Continuous variables: $(sf.data.continuous.columns)")
        
        if !isempty(sf.data.categorical)
            println("  Categorical variables:")
            for (i, cat_data) in enumerate(sf.data.categorical)
                n_levels = cat_data.n_levels
                n_contrasts = cat_data.n_contrasts
                println("    Categorical $i: $n_levels levels, $n_contrasts contrasts")
            end
        else
            println("  Categorical variables: none")
        end
        
        if !isempty(sf.data.functions)
            println("  Functions:")
            for (i, func_data) in enumerate(sf.data.functions)
                func_name = func_data.func
                n_args = length(func_data.arg_data)
                println("    Function $i: $func_name with $n_args arguments")
            end
        else
            println("  Functions: none")
        end
    end
end
