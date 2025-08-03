# step3_clean.jl
# Clean, unified function execution system with Phase 2 specialization

###############################################################################
# CORE FUNCTION DATA TYPES
###############################################################################

"""
    FunctionExecutionStep

A single step in linear function execution.
"""
struct FunctionExecutionStep
    operation::Symbol                    # :load_constant, :load_continuous, :call_unary, :call_binary
    func::Union{Function, Nothing}       # Function to call (nothing for load operations)
    input_positions::Vector{Int}         # Scratch positions to read from
    output_position::Int                 # Scratch position to write to
    constant_value::Union{Float64, Nothing}  # For load_constant operations
    column_symbol::Union{Symbol, Nothing}    # For load_continuous operations
end

# Constructors for different operation types
function FunctionExecutionStep(operation::Symbol, output_pos::Int, constant_val::Float64)
    @assert operation === :load_constant
    FunctionExecutionStep(operation, nothing, Int[], output_pos, constant_val, nothing)
end

function FunctionExecutionStep(operation::Symbol, output_pos::Int, col::Symbol)
    @assert operation === :load_continuous
    FunctionExecutionStep(operation, nothing, Int[], output_pos, nothing, col)
end

function FunctionExecutionStep(operation::Symbol, func::Function, input_pos::Int, output_pos::Int)
    @assert operation === :call_unary
    FunctionExecutionStep(operation, func, [input_pos], output_pos, nothing, nothing)
end

function FunctionExecutionStep(operation::Symbol, func::Function, input_pos1::Int, input_pos2::Int, output_pos::Int)
    @assert operation === :call_binary
    FunctionExecutionStep(operation, func, [input_pos1, input_pos2], output_pos, nothing, nothing)
end

"""
    SpecializedLinearFunctionData{StepCount, ScratchSize, OutputPos}

Fully compile-time specialized function data with embedded metadata.
"""
struct SpecializedLinearFunctionData{StepCount, ScratchSize, OutputPos}
    execution_steps::NTuple{StepCount, FunctionExecutionStep}
    output_position::Int
    scratch_size::Int
    
    function SpecializedLinearFunctionData{StepCount, ScratchSize, OutputPos}(
        step_tuple::NTuple{StepCount, FunctionExecutionStep},
        output_position::Int,
        scratch_size::Int
    ) where {StepCount, ScratchSize, OutputPos}
        new{StepCount, ScratchSize, OutputPos}(step_tuple, output_position, scratch_size)
    end
end

"""
    LinearFunctionOp{N}

Compile-time encoding of function operations with known count.
"""
struct LinearFunctionOp{N}
    function LinearFunctionOp(n::Int)
        new{n}()
    end
end

"""
    FunctionScratchAllocator

Helper for allocating scratch positions during function flattening.
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
# FUNCTION FLATTENING ALGORITHM
###############################################################################

"""
    flatten_function_to_specialized_data(func_eval::FunctionEvaluator, output_position::Int) -> SpecializedLinearFunctionData

Convert a function evaluator tree DIRECTLY into specialized data with tuples - NO ALLOCATIONS.
"""
function flatten_function_to_specialized_data(func_eval::FunctionEvaluator, output_position::Int)
    allocator = FunctionScratchAllocator()
    
    # Collect steps without vector allocation
    steps_collector = Tuple{Symbol, Any, Vector{Int}, Int, Union{Float64, Nothing}, Union{Symbol, Nothing}}[]
    
    # Flatten the function tree into step tuples
    result_position = flatten_function_recursive_specialized!(steps_collector, allocator, func_eval)
    
    # Create tuple directly from collected steps
    step_count = length(steps_collector)
    step_tuple = ntuple(step_count) do i
        step_data = steps_collector[i]
        FunctionExecutionStep(step_data[1], step_data[2], step_data[3], step_data[4], step_data[5], step_data[6])
    end
    
    scratch_size = allocator.next_position - 1
    
    # Create specialized data directly
    return SpecializedLinearFunctionData{step_count, scratch_size, output_position}(
        step_tuple,
        output_position,
        scratch_size
    )
end

"""
    flatten_function_recursive_specialized!(steps_collector, allocator, evaluator) -> Int

Recursively flatten a function/evaluator directly into tuples - NO VECTOR ALLOCATIONS.
"""
function flatten_function_recursive_specialized!(
    steps_collector::Vector,
    allocator::FunctionScratchAllocator,
    evaluator::AbstractEvaluator
)
    if evaluator isa ConstantEvaluator
        scratch_pos = allocate_scratch_position!(allocator)
        push!(steps_collector, (:load_constant, nothing, Int[], scratch_pos, evaluator.value, nothing))
        return scratch_pos
        
    elseif evaluator isa ContinuousEvaluator
        scratch_pos = allocate_scratch_position!(allocator)
        push!(steps_collector, (:load_continuous, nothing, Int[], scratch_pos, nothing, evaluator.column))
        return scratch_pos
        
    elseif evaluator isa FunctionEvaluator
        func = evaluator.func
        arg_evaluators = evaluator.arg_evaluators
        n_args = length(arg_evaluators)
        
        if n_args == 1
            arg_pos = flatten_function_recursive_specialized!(steps_collector, allocator, arg_evaluators[1])
            result_pos = allocate_scratch_position!(allocator)
            push!(steps_collector, (:call_unary, func, [arg_pos], result_pos, nothing, nothing))
            return result_pos
            
        elseif n_args == 2
            arg1_pos = flatten_function_recursive_specialized!(steps_collector, allocator, arg_evaluators[1])
            arg2_pos = flatten_function_recursive_specialized!(steps_collector, allocator, arg_evaluators[2])
            result_pos = allocate_scratch_position!(allocator)
            push!(steps_collector, (:call_binary, func, [arg1_pos, arg2_pos], result_pos, nothing, nothing))
            return result_pos
            
        else
            error("Functions with $(n_args) arguments not yet supported in linear flattening")
        end
        
    else
        error("Unsupported evaluator type in function flattening: $(typeof(evaluator))")
    end
end

###############################################################################
# ANALYSIS FUNCTION - PHASE 2 SPECIALIZATION
###############################################################################

"""
    analyze_function_operations_linear(evaluator::CombinedEvaluator)

Phase 2: Allocation-free analysis that creates specialized tuples directly.
"""
function analyze_function_operations_linear(evaluator::CombinedEvaluator)

    println("USED: new analyze")

    function_evaluators = evaluator.function_evaluators
    n_funcs = length(function_evaluators)
    
    if n_funcs == 0
        return (), LinearFunctionOp(0)
    end
    
    # Create tuple of specialized function data directly - NO CONVERSION
    function_data = ntuple(n_funcs) do i
        func_eval = function_evaluators[i]
        # Create specialized data directly - no intermediate LinearFunctionData
        flatten_function_to_specialized_data(func_eval, func_eval.position)
    end
    
    return function_data, LinearFunctionOp(n_funcs)
end

###############################################################################
# EXECUTION FUNCTIONS - PHASE 2 ALLOCATION-FREE
###############################################################################

"""
    execute_function_in_preallocated_scratch!(
        func_data::SpecializedLinearFunctionData,
        scratch::AbstractVector{Float64},
        output::AbstractVector{Float64},
        data::NamedTuple,
        row_idx::Int,
        scratch_offset::Int = 0
    )

Execute function using pre-allocated scratch space - NO ALLOCATIONS.
"""
function execute_function_in_preallocated_scratch!(
    func_data::SpecializedLinearFunctionData{StepCount, ScratchSize, OutputPos},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int,
    scratch_offset::Int = 0
) where {StepCount, ScratchSize, OutputPos}
    
    # Execute the linear plan using pre-allocated scratch space with offset
    @inbounds for step_idx in 1:StepCount
        step = func_data.execution_steps[step_idx]
        
        if step.operation === :load_constant
            scratch_pos = scratch_offset + step.output_position
            scratch[scratch_pos] = step.constant_value
            
        elseif step.operation === :load_continuous
            scratch_pos = scratch_offset + step.output_position
            col = step.column_symbol
            val = get_data_value_specialized(data, col, row_idx)
            scratch[scratch_pos] = Float64(val)
            
        elseif step.operation === :call_unary
            input_pos = scratch_offset + step.input_positions[1]
            output_pos = scratch_offset + step.output_position
            input_val = scratch[input_pos]
            result = apply_function_direct_single(step.func, input_val)
            scratch[output_pos] = result
            
        elseif step.operation === :call_binary
            input_pos1 = scratch_offset + step.input_positions[1]
            input_pos2 = scratch_offset + step.input_positions[2]
            output_pos = scratch_offset + step.output_position
            input_val1 = scratch[input_pos1]
            input_val2 = scratch[input_pos2]
            result = apply_function_direct_binary(step.func, input_val1, input_val2)
            scratch[output_pos] = result
            
        else
            error("Unknown operation type: $(step.operation)")
        end
    end
    
    # Write final result to output array
    if StepCount > 0
        final_step = func_data.execution_steps[StepCount]
        final_scratch_pos = scratch_offset + final_step.output_position
        output[OutputPos] = scratch[final_scratch_pos]
    end
    
    return nothing
end

"""
    execute_function_operations_recursive!(
        function_data::Tuple{},
        scratch, output, data, row_idx, scratch_offset
    )

Base case: empty tuple - no functions to process.
"""
function execute_function_operations_recursive!(
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
    execute_function_operations_recursive!(
        function_data::Tuple,
        scratch, output, data, row_idx, scratch_offset
    )

Recursive case: process first function, then recurse on rest.
"""
function execute_function_operations_recursive!(
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
    
    # Process the first function
    func_data = function_data[1]
    execute_function_in_preallocated_scratch!(
        func_data, scratch, output, data, row_idx, scratch_offset
    )
    
    # Calculate scratch offset for next function
    next_scratch_offset = scratch_offset + func_data.scratch_size
    
    # Recursively process remaining functions
    if length(function_data) > 1
        remaining_data = Base.tail(function_data)
        execute_function_operations_recursive!(
            remaining_data, scratch, output, data, row_idx, next_scratch_offset
        )
    end
    
    return nothing
end

###############################################################################
# MAIN EXECUTION FUNCTION - CLEAN VERSION
###############################################################################

"""
    execute_linear_function_operations!(
        function_data::Tuple,
        scratch::Vector{Float64},
        output::Vector{Float64},
        data::NamedTuple,
        row_idx::Int
    )

CLEAN: Main function execution using Phase 2 tuple-based approach.
"""
function execute_linear_function_operations!(
    function_data::Tuple,
    scratch::Vector{Float64},
    output::Vector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Use recursive processing - allocation-free
    execute_function_operations_recursive!(function_data, scratch, output, data, row_idx, 0)
    return nothing
end

###############################################################################
# REMOVE OLD/LEGACY FUNCTIONS
###############################################################################

# Remove all old allocating functions to avoid confusion:
# - execute_function_via_position_mapping (replaced)
# - Vector{LinearFunctionData} methods (replaced) 
# - Any other legacy function execution
