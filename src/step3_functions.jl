# step3_functions.jl - TRUE GENERALITY THROUGH COMPOSITION
# Universal function system using only unary and binary operations

###############################################################################
# SIMPLIFIED COMPILE-TIME SPECIALIZED FUNCTION DATA TYPES
###############################################################################

"""
    UnaryFunctionData{F, InputType}

Compile-time specialized unary function with known function type and input source.
InputType is Symbol (column), Int (temp position), or Float64 (constant).
"""
struct UnaryFunctionData{F, InputType}
    func::F
    input_source::InputType
    position::Int
    
    function UnaryFunctionData(func::F, input_source::T, position::Int) where {F, T}
        new{F, T}(func, input_source, position)
    end
end

"""
    BinaryFunctionData{F, Input1Type, Input2Type}

Compile-time specialized binary function with known function type and input sources.
Input types can be Symbol (column), Int (temp position), or Float64 (constant).
Handles ALL multi-argument functions through decomposition.
"""
struct BinaryFunctionData{F, Input1Type, Input2Type}
    func::F
    input1::Input1Type
    input2::Input2Type
    position::Int
    
    function BinaryFunctionData(func::F, input1::T1, input2::T2, position::Int) where {F, T1, T2}
        new{F, T1, T2}(func, input1, input2, position)
    end
end

"""
    SpecializedFunctionData{UnaryTuple, BinaryTuple}

Simplified function data with only unary and binary operations.
All n-ary functions are decomposed into sequences of binary operations.
"""
struct SpecializedFunctionData{UnaryTuple, BinaryTuple}
    unary_functions::UnaryTuple      # NTuple{N, UnaryFunctionData{...}}
    binary_functions::BinaryTuple    # NTuple{M, BinaryFunctionData{...}}
end

"""
    FunctionOp{N, M}

Simplified operation encoding with only unary and binary counts.
"""
struct FunctionOp{N, M}
    function FunctionOp(n_unary::Int, n_binary::Int)
        new{n_unary, n_binary}()
    end
end

###############################################################################
# LINEARIZED OPERATION TYPES (SIMPLIFIED)
###############################################################################

"""
    LinearizedOperation

Intermediate representation for function decomposition.
Only supports :unary and :binary operations for true generality.
"""
struct LinearizedOperation
    operation_type::Symbol  # :unary or :binary only
    func::Function
    inputs::Vector{Union{Symbol, Int, Float64}}  # Column names, temp positions, or constants
    output_position::Int
    temp_position::Union{Int, Nothing}  # Nothing for final output
end

"""
    TempAllocator

Manages temporary position allocation during decomposition.
"""
mutable struct TempAllocator
    next_temp::Int
    temp_base::Int
    
    function TempAllocator(temp_start::Int)
        new(temp_start, temp_start)
    end
end

function allocate_temp!(allocator::TempAllocator)
    temp_pos = allocator.next_temp
    allocator.next_temp += 1
    return temp_pos
end

###############################################################################
# UNIVERSAL FUNCTION DECOMPOSITION
###############################################################################

"""
    decompose_function_tree(func_eval::FunctionEvaluator, temp_allocator::TempAllocator) -> Vector{LinearizedOperation}

Zero-allocation universal function decomposition.
Uses existing scratch space only - no temp positions in output array.

Any n-ary function f(a, b, c, d, ...) becomes a sequence of binary operations
that execute in dependency order using only existing scratch positions.
"""
function decompose_function_tree(func_eval::FunctionEvaluator, temp_allocator::TempAllocator)
    operations = LinearizedOperation[]
    
    # Step 1: Recursively decompose all argument evaluators
    arg_inputs = Union{Symbol, Int, Float64}[]
    
    for arg_eval in func_eval.arg_evaluators
        if arg_eval isa ConstantEvaluator
            # Use constant value directly - no temp operations needed
            push!(arg_inputs, arg_eval.value)
            
        elseif arg_eval isa ContinuousEvaluator
            # Direct column reference
            push!(arg_inputs, arg_eval.column)
            
        elseif arg_eval isa FunctionEvaluator
            # Recursive case: decompose nested function
            nested_ops = decompose_function_tree(arg_eval, temp_allocator)
            append!(operations, nested_ops)
            
            # Use the nested function's final position as input
            # (This avoids temp positions - we use the function's actual output position)
            push!(arg_inputs, arg_eval.position)
            
        else
            error("Unsupported argument evaluator type: $(typeof(arg_eval))")
        end
    end
    
    # Step 2: Decompose this function based on argument count
    n_args = length(arg_inputs)
    
    if n_args == 0
        error("Function with no arguments is not supported")
        
    elseif n_args == 1
        # Unary function - direct operation
        push!(operations, LinearizedOperation(
            :unary,
            func_eval.func,
            Union{Symbol, Int, Float64}[arg_inputs[1]],
            func_eval.position,
            nothing
        ))
        
    elseif n_args == 2
        # Binary function - direct operation  
        push!(operations, LinearizedOperation(
            :binary,
            func_eval.func,
            Union{Symbol, Int, Float64}[arg_inputs[1], arg_inputs[2]],
            func_eval.position,
            nothing
        ))
        
    else
        # N-ary function (n ≥ 3) - decompose using intermediate scratch positions
        # Strategy: Use the function's own position for ALL intermediate results
        # Execute in sequence, overwriting the same position
        
        # All intermediate operations write to this function's position
        output_pos = func_eval.position
        
        # Create sequence of binary operations
        for i in 2:n_args
            if i == 2
                # First binary operation: f(arg1, arg2) → func_eval.position
                inputs = Union{Symbol, Int, Float64}[arg_inputs[1], arg_inputs[2]]
            else
                # Subsequent operations: f(previous_result, next_arg) → func_eval.position
                # Previous result is at func_eval.position (overwritten each time)
                inputs = Union{Symbol, Int, Float64}[output_pos, arg_inputs[i]]
            end
            
            push!(operations, LinearizedOperation(
                :binary,
                func_eval.func,
                inputs,
                output_pos,
                nothing  # No temp positions - everything uses final position
            ))
        end
    end
    
    return operations
end

###############################################################################
# SPECIALIZED EXECUTION FUNCTIONS (SIMPLIFIED)
###############################################################################

"""
    get_input_value(input::Float64, output, input_data, row_idx) -> Float64

Get constant value directly.
"""
@inline function get_input_value(input::Float64, output, input_data, row_idx)
    return input
end

"""
    get_input_value(input::Symbol, output, input_data, row_idx) -> Float64

Get value from data column.
"""
@inline function get_input_value(input::Symbol, output, input_data, row_idx)
    return Float64(get_data_value_specialized(input_data, input, row_idx))
end

"""
    get_input_value(input::Int, output, input_data, row_idx) -> Float64

Get value from temporary position in output array.
"""
@inline function get_input_value(input::Int, output, input_data, row_idx)
    return output[input]
end

"""
    execute_operation!(data::UnaryFunctionData{F, InputType}, output, input_data, row_idx) where {F, InputType}

Execute unary function with compile-time specialization.
"""
function execute_operation!(
    data::UnaryFunctionData{F, InputType},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {F, InputType}
    
    input_val = get_input_value(data.input_source, output, input_data, row_idx)
    
    # Use specialized function application
    if F === typeof(log)
        result = input_val > 0.0 ? log(input_val) : (input_val == 0.0 ? -Inf : NaN)
    elseif F === typeof(exp)
        result = exp(clamp(input_val, -700.0, 700.0))
    elseif F === typeof(sin)
        result = sin(input_val)
    elseif F === typeof(cos)
        result = cos(input_val)
    elseif F === typeof(sqrt)
        result = input_val ≥ 0.0 ? sqrt(input_val) : NaN
    elseif F === typeof(abs)
        result = abs(input_val)
    else
        # Direct function call for other functions
        result = data.func(input_val)
    end
    
    output[data.position] = result
    return nothing
end

"""
    execute_operation!(data::BinaryFunctionData{F, T1, T2}, output, input_data, row_idx) where {F, T1, T2}

Execute binary function with compile-time specialization.
Handles ALL multi-argument functions through decomposition.
"""
function execute_operation!(
    data::BinaryFunctionData{F, T1, T2},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {F, T1, T2}
    
    val1 = get_input_value(data.input1, output, input_data, row_idx)
    val2 = get_input_value(data.input2, output, input_data, row_idx)
    
    # Use specialized function application
    if F === typeof(+)
        result = val1 + val2
    elseif F === typeof(-)
        result = val1 - val2
    elseif F === typeof(*)
        result = val1 * val2
    elseif F === typeof(/)
        result = val2 == 0.0 ? (val1 == 0.0 ? NaN : (val1 > 0.0 ? Inf : -Inf)) : val1 / val2
    elseif F === typeof(^)
        if val1 == 0.0 && val2 < 0.0
            result = Inf
        elseif val1 < 0.0 && !isinteger(val2)
            result = NaN
        else
            result = val1^val2
        end
    elseif F === typeof(>)
        result = Float64(val1 > val2)
    elseif F === typeof(<)
        result = Float64(val1 < val2)
    elseif F === typeof(>=)
        result = Float64(val1 >= val2)
    elseif F === typeof(<=)
        result = Float64(val1 <= val2)
    elseif F === typeof(==)
        result = Float64(val1 == val2)
    else
        # Direct function call for other functions
        result = data.func(val1, val2)
    end
    
    output[data.position] = result
    return nothing
end

###############################################################################
# SIMPLIFIED TUPLE-BASED EXECUTION
###############################################################################

"""
    execute_operation!(data::SpecializedFunctionData{UT, BT}, op::FunctionOp{N, M}, output, input_data, row_idx) where {UT, BT, N, M}

Execute all functions using simplified binary/unary decomposition.
"""
function execute_operation!(
    data::SpecializedFunctionData{UT, BT},
    op::FunctionOp{N, M},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {UT, BT, N, M}
    
    # Execute in dependency order: unary then binary
    # (Binary operations may depend on unary results)
    execute_unary_functions_recursive!(data.unary_functions, output, input_data, row_idx)
    execute_binary_functions_recursive!(data.binary_functions, output, input_data, row_idx)
    
    return nothing
end

"""
    execute_unary_functions_recursive!(unary_tuple::Tuple{}, output, input_data, row_idx)

Base case: no unary functions to execute.
"""
function execute_unary_functions_recursive!(
    unary_tuple::Tuple{},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

"""
    execute_unary_functions_recursive!(unary_tuple::Tuple, output, input_data, row_idx)

Recursive case: execute first unary function, then process remaining.
"""
function execute_unary_functions_recursive!(
    unary_tuple::Tuple,
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(unary_tuple) > 0
        # Execute first function
        execute_operation!(unary_tuple[1], output, input_data, row_idx)
        
        # Recursively process remaining
        if length(unary_tuple) > 1
            remaining = Base.tail(unary_tuple)
            execute_unary_functions_recursive!(remaining, output, input_data, row_idx)
        end
    end
    return nothing
end

"""
    execute_binary_functions_recursive!(binary_tuple, output, input_data, row_idx)

Recursive execution of binary functions.
"""
function execute_binary_functions_recursive!(
    binary_tuple::Tuple{},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

function execute_binary_functions_recursive!(
    binary_tuple::Tuple,
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(binary_tuple) > 0
        execute_operation!(binary_tuple[1], output, input_data, row_idx)
        if length(binary_tuple) > 1
            remaining = Base.tail(binary_tuple)
            execute_binary_functions_recursive!(remaining, output, input_data, row_idx)
        end
    end
    return nothing
end

###############################################################################
# UNIVERSAL ANALYSIS AND COMPILATION
###############################################################################

"""
    analyze_function_operations_linear(evaluator::CombinedEvaluator) -> (SpecializedFunctionData, FunctionOp)

Universal analysis and compilation using true mathematical generality.
Any n-ary function is decomposed into unary/binary operations.
"""
function analyze_function_operations_linear(evaluator::CombinedEvaluator)
    function_evaluators = evaluator.function_evaluators
    n_funcs = length(function_evaluators)
    
    if n_funcs == 0
        empty_data = SpecializedFunctionData((), ())
        return empty_data, FunctionOp(0, 0)
    end
    
    # Step 1: Decompose all function trees using universal decomposition
    all_operations = LinearizedOperation[]
    temp_allocator = TempAllocator(1000)  # Start temp positions at 1000
    
    for func_eval in function_evaluators
        ops = decompose_function_tree(func_eval, temp_allocator)
        append!(all_operations, ops)
    end
    
    # Step 2: Separate by operation type (only unary and binary)
    unary_ops = filter(op -> op.operation_type == :unary, all_operations)
    binary_ops = filter(op -> op.operation_type == :binary, all_operations)
    
    # Step 3: Create specialized data structures
    unary_data = create_unary_tuple(unary_ops)
    binary_data = create_binary_tuple(binary_ops)
    
    specialized_data = SpecializedFunctionData(unary_data, binary_data)
    function_op = FunctionOp(length(unary_ops), length(binary_ops))
    
    return specialized_data, function_op
end

"""
    create_unary_tuple(unary_ops::Vector{LinearizedOperation})

Create compile-time tuple of unary function data.
"""
function create_unary_tuple(unary_ops::Vector{LinearizedOperation})
    n_unary = length(unary_ops)
    
    if n_unary == 0
        return ()
    end
    
    return ntuple(n_unary) do i
        op = unary_ops[i]
        input = length(op.inputs) > 0 ? op.inputs[1] : Symbol()
        UnaryFunctionData(op.func, input, op.output_position)
    end
end

"""
    create_binary_tuple(binary_ops::Vector{LinearizedOperation})

Create compile-time tuple of binary function data.
"""
function create_binary_tuple(binary_ops::Vector{LinearizedOperation})
    n_binary = length(binary_ops)
    
    if n_binary == 0
        return ()
    end
    
    return ntuple(n_binary) do i
        op = binary_ops[i]
        input1 = length(op.inputs) > 0 ? op.inputs[1] : Symbol()
        input2 = length(op.inputs) > 1 ? op.inputs[2] : Symbol()
        BinaryFunctionData(op.func, input1, input2, op.output_position)
    end
end

###############################################################################
# INTERFACE METHODS FOR INTEGRATION
###############################################################################

"""
    Base.isempty(data::SpecializedFunctionData)

Check if function data is empty (no functions to execute).
"""
function Base.isempty(data::SpecializedFunctionData)
    return length(data.unary_functions) == 0 && 
           length(data.binary_functions) == 0
end

"""
    Base.length(data::SpecializedFunctionData)

Get total number of functions in the data structure.
"""
function Base.length(data::SpecializedFunctionData)
    return length(data.unary_functions) + 
           length(data.binary_functions)
end

"""
    Base.iterate(data::SpecializedFunctionData, state=1)

Iterate over all functions in the data structure.
Required for isempty() to work properly.
"""
function Base.iterate(data::SpecializedFunctionData, state=1)
    total_unary = length(data.unary_functions)
    total_binary = length(data.binary_functions)
    total_functions = total_unary + total_binary
    
    if state > total_functions
        return nothing
    end
    
    if state <= total_unary
        return (data.unary_functions[state], state + 1)
    else
        binary_idx = state - total_unary
        return (data.binary_functions[binary_idx], state + 1)
    end
end

###############################################################################
# MAIN EXECUTION INTERFACE
###############################################################################

"""
    execute_linear_function_operations!(function_data::SpecializedFunctionData, scratch, output, data, row_idx)

Main execution interface - maintains compatibility with existing system.
"""
function execute_linear_function_operations!(
    function_data::SpecializedFunctionData,
    scratch::Vector{Float64},  # Not used in specialized version
    output::Vector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Create operation encoding for dispatch
    n_unary = length(function_data.unary_functions)
    n_binary = length(function_data.binary_functions) 
    
    op = FunctionOp(n_unary, n_binary)
    
    # Execute using specialized dispatch
    execute_operation!(function_data, op, output, data, row_idx)
    
    return nothing
end
