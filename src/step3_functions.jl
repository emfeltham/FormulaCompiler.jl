# step3_functions.jl - COMPILE-TIME SPECIALIZED FUNCTION SYSTEM
# Zero-allocation execution with full type specialization

###############################################################################
# COMPILE-TIME SPECIALIZED FUNCTION DATA TYPES
###############################################################################

"""
    UnaryFunctionData{F, InputType}

Compile-time specialized unary function with known function type and input source.
InputType is either Symbol (column) or Int (temp position).
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
    TernaryFunctionData{F, Input1Type, Input2Type, Input3Type}

Compile-time specialized ternary function.
Input types can be Symbol (column), Int (temp position), or Float64 (constant).
"""
struct TernaryFunctionData{F, Input1Type, Input2Type, Input3Type}
    func::F
    input1::Input1Type
    input2::Input2Type
    input3::Input3Type
    position::Int
    
    function TernaryFunctionData(func::F, input1::T1, input2::T2, input3::T3, position::Int) where {F, T1, T2, T3}
        new{F, T1, T2, T3}(func, input1, input2, input3, position)
    end
end

"""
    SpecializedFunctionData{UnaryTuple, BinaryTuple, TernaryTuple}

Complete function data with compile-time tuples following categorical pattern.
"""
struct SpecializedFunctionData{UnaryTuple, BinaryTuple, TernaryTuple}
    unary_functions::UnaryTuple      # NTuple{N, UnaryFunctionData{...}}
    binary_functions::BinaryTuple    # NTuple{M, BinaryFunctionData{...}}
    ternary_functions::TernaryTuple  # NTuple{P, TernaryFunctionData{...}}
end

"""
    FunctionOp{N, M, P}

Compile-time operation encoding for function execution.
"""
struct FunctionOp{N, M, P}
    function FunctionOp(n_unary::Int, n_binary::Int, n_ternary::Int)
        new{n_unary, n_binary, n_ternary}()
    end
end

###############################################################################
# LINEARIZED OPERATION TYPES
###############################################################################

"""
    LinearizedOperation

Intermediate representation for function tree linearization.
"""
struct LinearizedOperation
    operation_type::Symbol  # :unary, :binary, :ternary
    func::Function
    inputs::Vector{Union{Symbol, Int, Float64}}  # Column names, temp positions, or constants
    output_position::Int
    temp_position::Union{Int, Nothing}  # Nothing for final output
end

"""
    TempAllocator

Manages temporary position allocation during linearization.
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
# FUNCTION TREE LINEARIZATION
###############################################################################

"""
    linearize_function_tree(func_eval::FunctionEvaluator, temp_allocator::TempAllocator) -> Vector{LinearizedOperation}

Convert nested function tree to linear execution sequence.
Post-order traversal ensures dependencies are computed before use.
"""
function linearize_function_tree(func_eval::FunctionEvaluator, temp_allocator::TempAllocator)
    operations = LinearizedOperation[]
    
    # Step 1: Recursively linearize all argument evaluators
    arg_inputs = Union{Symbol, Int, Float64}[]
    
    for arg_eval in func_eval.arg_evaluators
        if arg_eval isa ConstantEvaluator
            # Use constant value directly - no temp operations needed
            push!(arg_inputs, arg_eval.value)  # Direct Float64 value
            
        elseif arg_eval isa ContinuousEvaluator
            # Direct column reference
            push!(arg_inputs, arg_eval.column)
            
        elseif arg_eval isa FunctionEvaluator
            # Recursive case: linearize nested function
            nested_ops = linearize_function_tree(arg_eval, temp_allocator)
            append!(operations, nested_ops)
            
            # The last operation's output becomes our input
            last_op = nested_ops[end]
            input_pos = last_op.temp_position !== nothing ? last_op.temp_position : last_op.output_position
            push!(arg_inputs, input_pos)
            
        else
            error("Unsupported argument evaluator type: $(typeof(arg_eval))")
        end
    end
    
    # Step 2: Create operation for this function
    n_args = length(arg_inputs)
    operation_type = if n_args == 1
        :unary
    elseif n_args == 2
        :binary
    elseif n_args == 3
        :ternary
    else
        error("Functions with $n_args arguments not yet supported")
    end
    
    # Determine if we need a temp position or use final position
    temp_position = nothing  # Will be set if this is an intermediate result
    
    push!(operations, LinearizedOperation(
        operation_type,
        func_eval.func,
        Union{Symbol, Int, Float64}[arg_inputs...],  # Explicit type conversion
        func_eval.position,
        temp_position
    ))
    
    return operations
end

"""
    assign_temp_positions!(operations::Vector{LinearizedOperation}, func_eval::FunctionEvaluator)

Assign temporary positions to intermediate results.
"""
function assign_temp_positions!(operations::Vector{LinearizedOperation}, func_eval::FunctionEvaluator)
    # The final operation uses the function evaluator's position
    # All others need temporary positions
    for i in 1:(length(operations) - 1)
        if operations[i].temp_position === nothing
            # This is an intermediate result, assign a temp position
            operations[i] = LinearizedOperation(
                operations[i].operation_type,
                operations[i].func,
                operations[i].inputs,
                operations[i].output_position,
                operations[i].output_position  # Use output position as temp
            )
        end
    end
    
    return operations
end

###############################################################################
# SPECIALIZED EXECUTION FUNCTIONS
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
    else
        # Direct function call for other functions
        result = data.func(val1, val2)
    end
    
    output[data.position] = result
    return nothing
end

"""
    execute_operation!(data::TernaryFunctionData{F, T1, T2, T3}, output, input_data, row_idx) where {F, T1, T2, T3}

Execute ternary function with compile-time specialization.
"""
function execute_operation!(
    data::TernaryFunctionData{F, T1, T2, T3},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {F, T1, T2, T3}
    
    val1 = get_input_value(data.input1, output, input_data, row_idx)
    val2 = get_input_value(data.input2, output, input_data, row_idx)
    val3 = get_input_value(data.input3, output, input_data, row_idx)
    
    # Direct function call
    result = data.func(val1, val2, val3)
    output[data.position] = result
    return nothing
end

###############################################################################
# TUPLE-BASED EXECUTION (Following Categorical Pattern)
###############################################################################

"""
    execute_operation!(data::SpecializedFunctionData{UT, BT, TT}, op::FunctionOp{N, M, P}, output, input_data, row_idx) where {UT, BT, TT, N, M, P}

Execute all functions using tuple-based recursive pattern like categorical system.
"""
function execute_operation!(
    data::SpecializedFunctionData{UT, BT, TT},
    op::FunctionOp{N, M, P},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {UT, BT, TT, N, M, P}
    
    # Execute in dependency order: unary, then binary, then ternary
    execute_unary_functions_recursive!(data.unary_functions, output, input_data, row_idx)
    execute_binary_functions_recursive!(data.binary_functions, output, input_data, row_idx)
    execute_ternary_functions_recursive!(data.ternary_functions, output, input_data, row_idx)
    
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

Recursive execution of binary functions following categorical pattern.
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

"""
    execute_ternary_functions_recursive!(ternary_tuple, output, input_data, row_idx)

Recursive execution of ternary functions following categorical pattern.
"""
function execute_ternary_functions_recursive!(
    ternary_tuple::Tuple{},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

function execute_ternary_functions_recursive!(
    ternary_tuple::Tuple,
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(ternary_tuple) > 0
        execute_operation!(ternary_tuple[1], output, input_data, row_idx)
        if length(ternary_tuple) > 1
            remaining = Base.tail(ternary_tuple)
            execute_ternary_functions_recursive!(remaining, output, input_data, row_idx)
        end
    end
    return nothing
end

###############################################################################
# ANALYSIS AND COMPILATION
###############################################################################

"""
    analyze_function_operations_linear(evaluator::CombinedEvaluator) -> (SpecializedFunctionData, FunctionOp)

Complete analysis and compilation to specialized function data.
REPLACES previous implementation with compile-time specialization.
"""
function analyze_function_operations_linear(evaluator::CombinedEvaluator)
    function_evaluators = evaluator.function_evaluators
    n_funcs = length(function_evaluators)
    
    if n_funcs == 0
        empty_data = SpecializedFunctionData((), (), ())
        return empty_data, FunctionOp(0, 0, 0)
    end
    
    # Step 1: Linearize all function trees
    all_operations = LinearizedOperation[]
    temp_allocator = TempAllocator(1000)  # Start temp positions at 1000
    
    for func_eval in function_evaluators
        ops = linearize_function_tree(func_eval, temp_allocator)
        assign_temp_positions!(ops, func_eval)
        append!(all_operations, ops)
    end
    
    # Step 2: Separate by operation type (exclude constant operations)
    unary_ops = filter(op -> op.operation_type == :unary, all_operations)
    binary_ops = filter(op -> op.operation_type == :binary, all_operations)
    ternary_ops = filter(op -> op.operation_type == :ternary, all_operations)
    
    # Step 3: Create specialized data structures (no separate constant handling)
    unary_data = create_unary_tuple(unary_ops)
    binary_data = create_binary_tuple(binary_ops)
    ternary_data = create_ternary_tuple(ternary_ops)
    
    specialized_data = SpecializedFunctionData(unary_data, binary_data, ternary_data)
    function_op = FunctionOp(length(unary_ops), length(binary_ops), length(ternary_ops))
    
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

"""
    create_ternary_tuple(ternary_ops::Vector{LinearizedOperation})

Create compile-time tuple of ternary function data.
"""
function create_ternary_tuple(ternary_ops::Vector{LinearizedOperation})
    n_ternary = length(ternary_ops)
    
    if n_ternary == 0
        return ()
    end
    
    return ntuple(n_ternary) do i
        op = ternary_ops[i]
        input1 = length(op.inputs) > 0 ? op.inputs[1] : Symbol()
        input2 = length(op.inputs) > 1 ? op.inputs[2] : Symbol()
        input3 = length(op.inputs) > 2 ? op.inputs[3] : Symbol()
        TernaryFunctionData(op.func, input1, input2, input3, op.output_position)
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
           length(data.binary_functions) == 0 && 
           length(data.ternary_functions) == 0
end

"""
    Base.length(data::SpecializedFunctionData)

Get total number of functions in the data structure.
"""
function Base.length(data::SpecializedFunctionData)
    return length(data.unary_functions) + 
           length(data.binary_functions) + 
           length(data.ternary_functions)
end

"""
    Base.iterate(data::SpecializedFunctionData, state=1)

Iterate over all functions in the data structure.
Required for isempty() to work properly.
"""
function Base.iterate(data::SpecializedFunctionData, state=1)
    total_unary = length(data.unary_functions)
    total_binary = length(data.binary_functions)
    total_ternary = length(data.ternary_functions)
    total_functions = total_unary + total_binary + total_ternary
    
    if state > total_functions
        return nothing
    end
    
    if state <= total_unary
        return (data.unary_functions[state], state + 1)
    elseif state <= total_unary + total_binary
        binary_idx = state - total_unary
        return (data.binary_functions[binary_idx], state + 1)
    else
        ternary_idx = state - total_unary - total_binary
        return (data.ternary_functions[ternary_idx], state + 1)
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
    # Create dummy operation for dispatch
    n_unary = length(function_data.unary_functions)
    n_binary = length(function_data.binary_functions) 
    n_ternary = length(function_data.ternary_functions)
    
    op = FunctionOp(n_unary, n_binary, n_ternary)
    
    # Execute using specialized dispatch
    execute_operation!(function_data, op, output, data, row_idx)
    
    return nothing
end
