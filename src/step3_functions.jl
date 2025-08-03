# step3_type_stable.jl
# Type-stable function execution mirroring categorical success pattern

###############################################################################
# SIMPLE TYPE-STABLE FUNCTION DATA - MIRROR CATEGORICAL PATTERN
###############################################################################

"""
    SimpleFunctionOperation

Simple operation data with compile-time known structure.
Mirrors SpecializedCategoricalData pattern - no Union types, no complexity.
"""
struct SimpleFunctionOperation
    operation_type::Int                 # 1=load_constant, 2=load_continuous, 3=call_unary, 4=call_binary
    input_pos1::Int                     # First input position (0 if not used)
    input_pos2::Int                     # Second input position (0 if not used)  
    output_pos::Int                     # Output scratch position
    func::Union{Function, Nothing}      # Function to call (nothing for loads)
    constant_value::Float64             # Constant value (NaN if not used)
    column_symbol::Symbol               # Column symbol (:none if not used)
    
    # Constructors for different operation types
    function SimpleFunctionOperation(::Val{:load_constant}, output_pos::Int, value::Float64)
        new(1, 0, 0, output_pos, nothing, value, :none)
    end
    
    function SimpleFunctionOperation(::Val{:load_continuous}, output_pos::Int, col::Symbol)
        new(2, 0, 0, output_pos, nothing, NaN, col)
    end
    
    function SimpleFunctionOperation(::Val{:call_unary}, input_pos::Int, output_pos::Int, func::Function)
        new(3, input_pos, 0, output_pos, func, NaN, :none)
    end
    
    function SimpleFunctionOperation(::Val{:call_binary}, input_pos1::Int, input_pos2::Int, output_pos::Int, func::Function)
        new(4, input_pos1, input_pos2, output_pos, func, NaN, :none)
    end
end

"""
    SimpleFunctionData{N, Operations}

Simple function data mirroring categorical success pattern.
N = number of operations, Operations = NTuple{N, SimpleFunctionOperation}
"""
struct SimpleFunctionData{N, Operations}
    operations::Operations              # NTuple{N, SimpleFunctionOperation}
    output_position::Int                # Final output position
    scratch_size::Int                   # Scratch space needed
    
    function SimpleFunctionData(operations::NTuple{N, SimpleFunctionOperation}, output_position::Int, scratch_size::Int) where N
        new{N, typeof(operations)}(operations, output_position, scratch_size)
    end
end

"""
    LinearFunctionOp{N}

Simple operation encoding.
"""
struct LinearFunctionOp{N}
    function LinearFunctionOp(n::Int)
        new{n}()
    end
end

"""
    FunctionScratchAllocator

Simple scratch allocator.
"""
mutable struct FunctionScratchAllocator
    next_position::Int
    FunctionScratchAllocator() = new(1)
end

function allocate_scratch_position!(allocator::FunctionScratchAllocator)
    pos = allocator.next_position
    allocator.next_position += 1
    return pos
end

###############################################################################
# SIMPLE ANALYSIS - MIRROR CATEGORICAL ANALYSIS PATTERN
###############################################################################

"""
    analyze_function_to_simple_data(func_eval, output_position) -> SimpleFunctionData

Simple analysis that creates type-stable data directly.
Mirrors the categorical analysis pattern exactly.
"""
function analyze_function_to_simple_data(func_eval::FunctionEvaluator, output_position::Int)
    allocator = FunctionScratchAllocator()
    
    # Collect operations in a simple Vector first (like categoricals do)
    operations_vec = SimpleFunctionOperation[]
    
    function analyze_recursive(evaluator)
        if evaluator isa ConstantEvaluator
            scratch_pos = allocate_scratch_position!(allocator)
            push!(operations_vec, SimpleFunctionOperation(Val(:load_constant), scratch_pos, evaluator.value))
            return scratch_pos
            
        elseif evaluator isa ContinuousEvaluator
            scratch_pos = allocate_scratch_position!(allocator)
            push!(operations_vec, SimpleFunctionOperation(Val(:load_continuous), scratch_pos, evaluator.column))
            return scratch_pos
            
        elseif evaluator isa FunctionEvaluator
            arg_evaluators = evaluator.arg_evaluators
            n_args = length(arg_evaluators)
            
            if n_args == 1
                arg_pos = analyze_recursive(arg_evaluators[1])
                result_pos = allocate_scratch_position!(allocator)
                push!(operations_vec, SimpleFunctionOperation(Val(:call_unary), arg_pos, result_pos, evaluator.func))
                return result_pos
                
            elseif n_args == 2
                arg1_pos = analyze_recursive(arg_evaluators[1])
                arg2_pos = analyze_recursive(arg_evaluators[2])
                result_pos = allocate_scratch_position!(allocator)
                push!(operations_vec, SimpleFunctionOperation(Val(:call_binary), arg1_pos, arg2_pos, result_pos, evaluator.func))
                return result_pos
                
            else
                error("Functions with $(n_args) arguments not supported")
            end
            
        else
            error("Unsupported evaluator type: $(typeof(evaluator))")
        end
    end
    
    # Analyze the function
    final_pos = analyze_recursive(func_eval)
    
    # Convert to compile-time tuple (mirrors categorical pattern)
    n_ops = length(operations_vec)
    operations_tuple = ntuple(i -> operations_vec[i], n_ops)
    
    scratch_size = allocator.next_position - 1
    
    return SimpleFunctionData(operations_tuple, output_position, scratch_size)
end

###############################################################################
# OVERWRITE: Main Analysis Function
###############################################################################

"""
    analyze_function_operations_linear(evaluator::CombinedEvaluator)

OVERWRITE: Simple analysis mirroring categorical success pattern.
"""
function analyze_function_operations_linear(evaluator::CombinedEvaluator)
    function_evaluators = evaluator.function_evaluators
    n_funcs = length(function_evaluators)
    
    if n_funcs == 0
        return (), LinearFunctionOp(0)
    end
    
    # Create tuple of simple function data (mirrors categorical pattern)
    function_data = ntuple(n_funcs) do i
        func_eval = function_evaluators[i]
        analyze_function_to_simple_data(func_eval, func_eval.position)
    end
    
    return function_data, LinearFunctionOp(n_funcs)
end

###############################################################################
# SIMPLE TYPE-STABLE EXECUTION - MIRROR CATEGORICAL EXECUTION
###############################################################################

"""
    execute_simple_operation!(op::SimpleFunctionOperation, scratch, output, data, row_idx, scratch_offset)

Execute single operation - type-stable like categorical execution.
"""
function execute_simple_operation!(
    op::SimpleFunctionOperation,
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int,
    scratch_offset::Int = 0
)
    if op.operation_type == 1  # load_constant
        scratch_pos = scratch_offset + op.output_pos
        scratch[scratch_pos] = op.constant_value
        
    elseif op.operation_type == 2  # load_continuous
        scratch_pos = scratch_offset + op.output_pos
        val = get_data_value_specialized(data, op.column_symbol, row_idx)
        scratch[scratch_pos] = Float64(val)
        
    elseif op.operation_type == 3  # call_unary
        input_pos = scratch_offset + op.input_pos1
        output_pos = scratch_offset + op.output_pos
        input_val = scratch[input_pos]
        result = apply_function_direct_single(op.func, input_val)
        scratch[output_pos] = result
        
    elseif op.operation_type == 4  # call_binary
        input_pos1 = scratch_offset + op.input_pos1
        input_pos2 = scratch_offset + op.input_pos2
        output_pos = scratch_offset + op.output_pos
        input_val1 = scratch[input_pos1]
        input_val2 = scratch[input_pos2]
        result = apply_function_direct_binary(op.func, input_val1, input_val2)
        scratch[output_pos] = result
        
    else
        error("Unknown operation type: $(op.operation_type)")
    end
    
    return nothing
end

"""
    execute_simple_function!(func_data::SimpleFunctionData{N}, scratch, output, data, row_idx, scratch_offset) where N

Execute simple function - mirrors categorical execution pattern exactly.
"""
function execute_simple_function!(
    func_data::SimpleFunctionData{N},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int,
    scratch_offset::Int = 0
) where N
    
    # Execute all operations (mirrors categorical loop pattern)
    @inbounds for i in 1:N  # N is compile-time constant!
        op = func_data.operations[i]  # Direct tuple access, known type
        execute_simple_operation!(op, scratch, output, data, row_idx, scratch_offset)
    end
    
    # Write final result to output
    if N > 0
        final_op = func_data.operations[N]
        final_scratch_pos = scratch_offset + final_op.output_pos
        output[func_data.output_position] = scratch[final_scratch_pos]
    end
    
    return nothing
end

###############################################################################
# RECURSIVE EXECUTION - MIRROR CATEGORICAL RECURSIVE PATTERN
###############################################################################

"""
    execute_simple_functions_recursive!(function_data::Tuple{}, ...) 

Base case: empty tuple.
"""
function execute_simple_functions_recursive!(
    function_data::Tuple{},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int,
    scratch_offset::Int = 0
)
    return nothing
end

"""
    execute_simple_functions_recursive!(function_data::Tuple, ...)

Recursive case - mirrors categorical recursive pattern exactly.
"""
function execute_simple_functions_recursive!(
    function_data::Tuple,
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int,
    scratch_offset::Int = 0
)
    if length(function_data) == 0
        return nothing
    end
    
    # Process first function
    func_data = function_data[1]
    execute_simple_function!(func_data, scratch, output, data, row_idx, scratch_offset)
    
    # Calculate next scratch offset
    next_scratch_offset = scratch_offset + func_data.scratch_size
    
    # Recurse on remaining functions
    if length(function_data) > 1
        remaining_data = Base.tail(function_data)
        execute_simple_functions_recursive!(
            remaining_data, scratch, output, data, row_idx, next_scratch_offset
        )
    end
    
    return nothing
end

###############################################################################
# OVERWRITE: Main Execution Function
###############################################################################

"""
    execute_linear_function_operations!(function_data::Tuple, scratch, output, data, row_idx)

OVERWRITE: Simple execution mirroring categorical success.
"""
function execute_linear_function_operations!(
    function_data::Tuple,
    scratch::Vector{Float64},
    output::Vector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Use simple recursive execution (mirrors categorical pattern)
    execute_simple_functions_recursive!(function_data, scratch, output, data, row_idx, 0)
    return nothing
end
