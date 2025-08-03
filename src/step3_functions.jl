# step3_functions.jl
# Universal function execution with compile-time specialized sequences

###############################################################################
# UNIVERSAL FUNCTION SEQUENCE TYPE
###############################################################################

"""
    UniversalFunctionSequence{N, Operations, Functions, Columns}

Universal pre-extracted execution sequence for any function structure.
All complexity pre-extracted during compilation, simple indexing during execution.
"""
struct UniversalFunctionSequence{N, Operations, Functions, Columns}
    operations::Operations      # NTuple{N, Int} - operation type codes
    functions::Functions        # NTuple{M, Function} - all functions used
    columns::Columns           # NTuple{P, Symbol} - all columns used
    constants::NTuple{16, Float64}  # Fixed-size constant storage (expand as needed)
    n_constants::Int            # Number of constants actually used
    output_position::Int
    scratch_size::Int
    
    function UniversalFunctionSequence(operations::NTuple{N, Int}, functions::NTuple{M, Function}, columns::NTuple{P, Symbol}, constants::Vector{Float64}, output_position::Int, scratch_size::Int) where {N, M, P}
        # Pad constants to fixed size
        padded_constants = ntuple(16) do i
            i <= length(constants) ? constants[i] : 0.0
        end
        new{N, typeof(operations), typeof(functions), typeof(columns)}(operations, functions, columns, padded_constants, length(constants), output_position, scratch_size)
    end
end

###############################################################################
# OPERATION TYPE CODES
###############################################################################

const LOAD_CONTINUOUS = 1
const LOAD_CONSTANT = 2
const CALL_UNARY = 3
const CALL_BINARY = 4

###############################################################################
# UNIVERSAL FUNCTION EXTRACTION
###############################################################################

function extract_universal_function_sequence(func_eval::FunctionEvaluator, output_position::Int)
    # Collect all components
    operations = Int[]
    functions = Function[]
    columns = Symbol[]
    constants = Float64[]
    
    # Extract the complete tree
    extract_evaluator_recursive!(operations, functions, columns, constants, func_eval)
    
    # Convert to compile-time tuples
    operations_tuple = ntuple(i -> operations[i], length(operations))
    functions_tuple = ntuple(i -> functions[i], length(functions))
    columns_tuple = ntuple(i -> columns[i], length(columns))
    
    return UniversalFunctionSequence(operations_tuple, functions_tuple, columns_tuple, constants, output_position, 0)
end

"""
    extract_evaluator_recursive!(operations, functions, columns, constants, evaluator)

Recursively extract ALL components from evaluator tree into flat sequences.
"""
function extract_evaluator_recursive!(operations::Vector{Int}, functions::Vector{Function}, columns::Vector{Symbol}, constants::Vector{Float64}, evaluator::AbstractEvaluator)
    
    if evaluator isa ContinuousEvaluator
        push!(operations, LOAD_CONTINUOUS)
        # FIX: Always add column (don't check duplicates yet)
        push!(columns, evaluator.column)
        
    elseif evaluator isa ConstantEvaluator
        push!(operations, LOAD_CONSTANT)
        # FIX: Always add constant (don't check duplicates yet)
        push!(constants, evaluator.value)
        
    elseif evaluator isa FunctionEvaluator
        # First, process all arguments recursively
        for arg_eval in evaluator.arg_evaluators
            extract_evaluator_recursive!(operations, functions, columns, constants, arg_eval)
        end
        
        # Then add the function call operation
        n_args = length(evaluator.arg_evaluators)
        if n_args == 1
            push!(operations, CALL_UNARY)
        elseif n_args == 2
            push!(operations, CALL_BINARY)
        else
            error("Functions with $(n_args) arguments not supported yet")
        end
        
        # FIX: Always add function (don't check duplicates yet)
        push!(functions, evaluator.func)
        
    else
        error("Unsupported evaluator type: $(typeof(evaluator))")
    end
end

"""
    execute_double_nested_with_constant_pattern!(sequence, output, data, row_idx)

Specialized execution for f(g(h(x)), c) pattern like log(abs(z))^2 - zero allocations.
"""
function execute_double_nested_with_constant_pattern!(
    sequence::UniversalFunctionSequence,
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Pattern: LOAD_CONTINUOUS, CALL_UNARY, CALL_UNARY, LOAD_CONSTANT, CALL_BINARY
    # This means: x → h(x) → g(h(x)) → f(g(h(x)), c)
    
    if length(sequence.columns) < 1 || length(sequence.functions) < 3 || sequence.n_constants < 1
        error("Invalid double nested with constant pattern: insufficient data")
    end
    
    column = sequence.columns[1]        # z
    func1 = sequence.functions[1]       # abs (h)
    func2 = sequence.functions[2]       # log (g)  
    func3 = sequence.functions[3]       # ^ (f)
    constant = sequence.constants[1]    # 2.0 (c)
    
    # Execute: z → abs(z) → log(abs(z)) → log(abs(z))^2
    val = get_data_value_specialized(data, column, row_idx)
    result1 = apply_function_direct_single(func1, Float64(val))  # abs(z)
    result2 = apply_function_direct_single(func2, result1)        # log(abs(z))
    final_result = apply_function_direct_binary(func3, result2, constant)  # log(abs(z))^2
    output[sequence.output_position] = final_result
end

###############################################################################
# UNIVERSAL SEQUENCE EXECUTION
###############################################################################

"""
    execute_universal_sequence!(
        sequence::UniversalFunctionSequence{N, Operations, Functions, Columns},
        scratch, output, data, row_idx
    ) where {N, Operations, Functions, Columns}

Execute universal sequence using compile-time specialized dispatch.
"""
function execute_universal_sequence!(
    sequence::UniversalFunctionSequence{N, Operations, Functions, Columns},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
) where {N, Operations, Functions, Columns}
    
    # Execute based on compile-time known pattern
    # THIS IS NOT GENERAL REALLY, NEED TO CONSIDER
    if N == 2 && sequence.operations == (LOAD_CONTINUOUS, CALL_UNARY)
        # Pattern: f(x) - unary continuous
        execute_unary_continuous_pattern!(sequence, output, data, row_idx)
        
    elseif N == 3 && sequence.operations == (LOAD_CONTINUOUS, LOAD_CONTINUOUS, CALL_BINARY)
        # Pattern: f(x, y) - binary continuous
        execute_binary_continuous_pattern!(sequence, output, data, row_idx)
        
    elseif N == 2 && sequence.operations == (LOAD_CONSTANT, CALL_UNARY)
        # Pattern: f(c) - unary constant
        execute_unary_constant_pattern!(sequence, output, data, row_idx)
    elseif N == 4 && sequence.operations == (LOAD_CONTINUOUS, CALL_UNARY, CALL_UNARY, CALL_UNARY)
        # Pattern: f(g(h(x))) - triple nested
        execute_triple_nested_pattern!(sequence, output, data, row_idx)
        
    elseif N == 3 && sequence.operations == (LOAD_CONTINUOUS, CALL_UNARY, CALL_UNARY)
        # Pattern: f(g(x)) - double nested
        execute_double_nested_pattern!(sequence, output, data, row_idx)
    elseif N == 4 && sequence.operations == (LOAD_CONTINUOUS, CALL_UNARY, LOAD_CONSTANT, CALL_BINARY)
        # Pattern: f(g(x), c) - nested function with constant
        execute_nested_with_constant_pattern!(sequence, output, data, row_idx)
    elseif N == 5 && sequence.operations == (LOAD_CONTINUOUS, CALL_UNARY, CALL_UNARY, LOAD_CONSTANT, CALL_BINARY)
        # Pattern: f(g(h(x)), c) - double nested with constant like log(abs(z))^2
        execute_double_nested_with_constant_pattern!(sequence, output, data, row_idx)
    elseif N == 5 && sequence.operations == (LOAD_CONTINUOUS, LOAD_CONTINUOUS, CALL_BINARY, LOAD_CONSTANT, CALL_BINARY)
        # Pattern: f(g(x, y), c) - binary function with constant like (z + y)^2
        execute_binary_with_constant_pattern!(sequence, output, data, row_idx)
    else
        # General execution for any pattern
        execute_general_pattern!(sequence, scratch, output, data, row_idx)
    end
    
    return nothing
end

###############################################################################
# SPECIALIZED PATTERN EXECUTION METHODS
###############################################################################

"""
    execute_unary_continuous_pattern!(sequence, output, data, row_idx)

Specialized execution for f(x) pattern - zero allocations.
"""
function execute_unary_continuous_pattern!(
    sequence::UniversalFunctionSequence,
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Add bounds checking
    if length(sequence.columns) < 1 || length(sequence.functions) < 1
        error("Invalid unary continuous pattern: insufficient columns or functions")
    end
    
    column = sequence.columns[1]
    func = sequence.functions[1]
    
    val = get_data_value_specialized(data, column, row_idx)
    result = apply_function_direct_single(func, Float64(val))
    output[sequence.output_position] = result
end

"""
    execute_binary_continuous_pattern!(sequence, output, data, row_idx)

Specialized execution for f(x, y) pattern - zero allocations.
"""
function execute_binary_continuous_pattern!(
    sequence::UniversalFunctionSequence,
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    col1 = sequence.columns[1]
    col2 = sequence.columns[2]
    func = sequence.functions[1]
    
    val1 = get_data_value_specialized(data, col1, row_idx)
    val2 = get_data_value_specialized(data, col2, row_idx)
    result = apply_function_direct_binary(func, Float64(val1), Float64(val2))
    output[sequence.output_position] = result
end

"""
    execute_unary_constant_pattern!(sequence, output, data, row_idx)

Specialized execution for f(c) pattern - zero allocations.
"""
function execute_unary_constant_pattern!(
    sequence::UniversalFunctionSequence,
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Add bounds checking
    if sequence.n_constants < 1 || length(sequence.functions) < 1
        error("Invalid unary constant pattern: insufficient constants or functions")
    end
    
    constant = sequence.constants[1]
    func = sequence.functions[1]
    
    result = apply_function_direct_single(func, constant)
    output[sequence.output_position] = result
end

"""
    execute_double_nested_pattern!(sequence, output, data, row_idx)

Specialized execution for f(g(x)) pattern - zero allocations.
"""
function execute_double_nested_pattern!(
    sequence::UniversalFunctionSequence,
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    column = sequence.columns[1]
    inner_func = sequence.functions[1]  # g
    outer_func = sequence.functions[2]  # f
    
    val = get_data_value_specialized(data, column, row_idx)
    inner_result = apply_function_direct_single(inner_func, Float64(val))
    final_result = apply_function_direct_single(outer_func, inner_result)
    output[sequence.output_position] = final_result
end

"""
    execute_triple_nested_pattern!(sequence, output, data, row_idx)

Specialized execution for f(g(h(x))) pattern - zero allocations.
"""
function execute_triple_nested_pattern!(
    sequence::UniversalFunctionSequence,
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    column = sequence.columns[1]
    func1 = sequence.functions[1]  # h
    func2 = sequence.functions[2]  # g  
    func3 = sequence.functions[3]  # f
    
    val = get_data_value_specialized(data, column, row_idx)
    result1 = apply_function_direct_single(func1, Float64(val))
    result2 = apply_function_direct_single(func2, result1)
    final_result = apply_function_direct_single(func3, result2)
    output[sequence.output_position] = final_result
end

"""
    execute_nested_with_constant_pattern!(sequence, output, data, row_idx)

Specialized execution for f(g(x), c) pattern like log(abs(z))^2 - zero allocations.
"""
function execute_nested_with_constant_pattern!(
    sequence::UniversalFunctionSequence,
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Pattern: LOAD_CONTINUOUS, CALL_UNARY, LOAD_CONSTANT, CALL_BINARY
    # This means: g(x), then f(g(x), c)
    
    if length(sequence.columns) < 1 || length(sequence.functions) < 2 || sequence.n_constants < 1
        error("Invalid nested with constant pattern: insufficient data")
    end
    
    column = sequence.columns[1]        # x
    inner_func = sequence.functions[1]  # g (like abs)  
    outer_func = sequence.functions[2]  # f (like ^)
    constant = sequence.constants[1]    # c (like 2)
    
    # Execute: x → g(x) → f(g(x), c)
    val = get_data_value_specialized(data, column, row_idx)
    inner_result = apply_function_direct_single(inner_func, Float64(val))
    final_result = apply_function_direct_binary(outer_func, inner_result, constant)
    output[sequence.output_position] = final_result
end

"""
    execute_binary_with_constant_pattern!(sequence, output, data, row_idx)

Specialized execution for f(g(x, y), c) pattern like (z + y)^2 - zero allocations.
"""
function execute_binary_with_constant_pattern!(
    sequence::UniversalFunctionSequence,
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Pattern: LOAD_CONTINUOUS, LOAD_CONTINUOUS, CALL_BINARY, LOAD_CONSTANT, CALL_BINARY
    # This means: x, y → g(x, y) → f(g(x, y), c)
    
    if length(sequence.columns) < 2 || length(sequence.functions) < 2 || sequence.n_constants < 1
        error("Invalid binary with constant pattern: insufficient data")
    end
    
    col1 = sequence.columns[1]          # z
    col2 = sequence.columns[2]          # y
    inner_func = sequence.functions[1]  # + (g)
    outer_func = sequence.functions[2]  # ^ (f)
    constant = sequence.constants[1]    # 2.0 (c)
    
    # Execute: z, y → z + y → (z + y)^2
    val1 = get_data_value_specialized(data, col1, row_idx)
    val2 = get_data_value_specialized(data, col2, row_idx)
    inner_result = apply_function_direct_binary(inner_func, Float64(val1), Float64(val2))  # z + y
    final_result = apply_function_direct_binary(outer_func, inner_result, constant)        # (z + y)^2
    output[sequence.output_position] = final_result
end

"""
    execute_general_pattern!(sequence, scratch, output, data, row_idx)

General execution for complex patterns using scratch space.
"""
function execute_general_pattern!(
    sequence::UniversalFunctionSequence{N},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
) where N
    
    scratch_pos = 1
    col_idx = 1
    const_idx = 1
    func_idx = 1
    
    @inbounds for i in 1:N
        op = sequence.operations[i]
        
        if op == LOAD_CONTINUOUS
            column = sequence.columns[col_idx]
            val = get_data_value_specialized(data, column, row_idx)
            scratch[scratch_pos] = Float64(val)
            scratch_pos += 1
            col_idx += 1
            
        elseif op == LOAD_CONSTANT
            constant = sequence.constants[const_idx]
            scratch[scratch_pos] = constant
            scratch_pos += 1
            const_idx += 1
            
        elseif op == CALL_UNARY
            func = sequence.functions[func_idx]
            input_val = scratch[scratch_pos - 1]
            result = apply_function_direct_single(func, input_val)
            scratch[scratch_pos - 1] = result  # Overwrite input
            func_idx += 1
            
        elseif op == CALL_BINARY
            func = sequence.functions[func_idx]
            val1 = scratch[scratch_pos - 2]
            val2 = scratch[scratch_pos - 1]
            result = apply_function_direct_binary(func, val1, val2)
            scratch[scratch_pos - 2] = result  # Overwrite first input
            scratch_pos -= 1  # One less value on stack
            func_idx += 1
        end
    end
    
    # Final result is at top of scratch stack
    output[sequence.output_position] = scratch[scratch_pos - 1]
end

###############################################################################
# MAIN ANALYSIS AND EXECUTION FUNCTIONS
###############################################################################

"""
    analyze_function_operations_linear(evaluator::CombinedEvaluator)

Main analysis function - creates universal sequences for ALL functions.
"""
function analyze_function_operations_linear(evaluator::CombinedEvaluator)
    function_evaluators = evaluator.function_evaluators
    n_funcs = length(function_evaluators)
    
    if n_funcs == 0
        return (), LinearFunctionOp(0)
    end
    
    # Create tuple of universal function sequences
    function_data = ntuple(n_funcs) do i
        func_eval = function_evaluators[i]
        extract_universal_function_sequence(func_eval, func_eval.position)
    end
    
    return function_data, LinearFunctionOp(n_funcs)
end

"""
    execute_linear_function_operations!(function_data::Tuple, scratch, output, data, row_idx)

Execute ALL function sequences using universal execution.
"""
function execute_linear_function_operations!(
    function_data::Tuple,
    scratch::Vector{Float64},
    output::Vector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Execute each function sequence
    for func_data in function_data
        execute_universal_sequence!(func_data, scratch, output, data, row_idx)
    end
    return nothing
end

# Operation encoding (unchanged interface)
struct LinearFunctionOp{N}
    function LinearFunctionOp(n::Int)
        new{n}()
    end
end

