# step3/main.jl
# Implementation of the function system (Step 3 of compilation pipeline)
# Universal function system using only unary and binary operations
# Phase A Fix: Zero-allocation function position management

# Types are defined in step3/types.jl and included by step3_functions.jl

###############################################################################
# ZERO-ALLOCATION FUNCTION DECOMPOSITION (PHASE A FIX)
###############################################################################

"""
    decompose_function_tree_zero_alloc(func_eval::FunctionEvaluator, temp_allocator::TempAllocator)

SIMPLIFIED FIX: Handle nested functions by treating them as intermediate operations.
"""
function decompose_function_tree_zero_alloc(func_eval::FunctionEvaluator, temp_allocator::TempAllocator)
    operations = LinearizedOperation[]
    
    # # println("DEBUG: Decomposing function with $(length(func_eval.arg_evaluators)) args, final position $(func_eval.position)")
    
    # Step 1: Process all argument evaluators
    arg_inputs = Union{Symbol, Int, Float64, ScratchPosition}[]
    
    for arg_eval in func_eval.arg_evaluators
        if arg_eval isa ConstantEvaluator
            push!(arg_inputs, arg_eval.value)
            # # println("DEBUG:   Arg: constant $(arg_eval.value)")
            
        elseif arg_eval isa ContinuousEvaluator
            push!(arg_inputs, arg_eval.column)
            # # println("DEBUG:   Arg: column $(arg_eval.column)")
            
        elseif arg_eval isa FunctionEvaluator
            # FIXED: Nested function - create intermediate operations and use scratch
            # # println("DEBUG:   Arg: nested function with $(length(arg_eval.arg_evaluators)) args")
            
            # Create a scratch position for the nested function result
            nested_scratch_pos = allocate_temp!(temp_allocator)
            
            # Recursively decompose the nested function, but force it to write to scratch
            nested_ops = decompose_function_tree_as_intermediate(arg_eval, nested_scratch_pos, temp_allocator)
            append!(operations, nested_ops)
            
            # Use scratch position as input to outer function
            push!(arg_inputs, ScratchPosition(nested_scratch_pos))
            # # println("DEBUG:   Nested function writes to scratch[$nested_scratch_pos]")
            
        else
            error("Unsupported argument evaluator type: $(typeof(arg_eval))")
        end
    end
    
    # Step 2: Create the main function operation (always final)
    n_args = length(arg_inputs)
    
    if n_args == 1
        # Unary function
        # # println("DEBUG: Main unary operation: $(func_eval.func)($(arg_inputs[1])) → position $(func_eval.position)")
        push!(operations, LinearizedOperation(
            :unary,
            func_eval.func,
            Union{Symbol, Int, Float64, ScratchPosition}[arg_inputs[1]],
            func_eval.position,
            nothing
        ))
        
    elseif n_args == 2
        # Binary function
        # # println("DEBUG: Main binary operation: $(func_eval.func)($(arg_inputs[1]), $(arg_inputs[2])) → position $(func_eval.position)")
        push!(operations, LinearizedOperation(
            :final_binary,
            func_eval.func,
            Union{Symbol, Int, Float64, ScratchPosition}[arg_inputs[1], arg_inputs[2]],
            func_eval.position,
            nothing
        ))
        
    else
        # N-ary function - decompose into binary sequence
        # # println("DEBUG: Main n-ary operation with $(n_args) args")
        
        for i in 2:n_args
            if i == 2
                inputs = Union{Symbol, Int, Float64, ScratchPosition}[arg_inputs[1], arg_inputs[2]]
            else
                prev_scratch = ScratchPosition(temp_allocator.next_temp - 1)
                inputs = Union{Symbol, Int, Float64, ScratchPosition}[prev_scratch, arg_inputs[i]]
            end
            
            if i == n_args
                # Final operation
                # # println("DEBUG:   Final n-ary step: $(func_eval.func)($(inputs[1]), $(inputs[2])) → position $(func_eval.position)")
                push!(operations, LinearizedOperation(
                    :final_binary,
                    func_eval.func,
                    inputs,
                    func_eval.position,
                    nothing
                ))
            else
                # Intermediate operation
                scratch_pos = allocate_temp!(temp_allocator)
                # # println("DEBUG:   Intermediate n-ary step: $(func_eval.func)($(inputs[1]), $(inputs[2])) → scratch[$scratch_pos]")
                push!(operations, LinearizedOperation(
                    :intermediate_binary,
                    func_eval.func,
                    inputs,
                    scratch_pos,
                    scratch_pos
                ))
            end
        end
    end
    
    return operations
end

"""
    decompose_function_tree_as_intermediate(func_eval::FunctionEvaluator, target_scratch::Int, temp_allocator::TempAllocator)

FIXED: Better handling of intermediate unary operations.
Instead of creating :intermediate_unary and converting to binary, create proper intermediate binary operations.
"""
function decompose_function_tree_as_intermediate(func_eval::FunctionEvaluator, target_scratch::Int, temp_allocator::TempAllocator)
    operations = LinearizedOperation[]
    
    # # println("DEBUG: FIXED decompose_as_intermediate with $(length(func_eval.arg_evaluators)) args → scratch[$target_scratch]")
    
    # Process arguments (same logic as main function)
    arg_inputs = Union{Symbol, Int, Float64, ScratchPosition}[]
    
    for arg_eval in func_eval.arg_evaluators
        if arg_eval isa ConstantEvaluator
            push!(arg_inputs, arg_eval.value)
        elseif arg_eval isa ContinuousEvaluator
            push!(arg_inputs, arg_eval.column)
        elseif arg_eval isa FunctionEvaluator
            # Nested-nested function
            nested_scratch_pos = allocate_temp!(temp_allocator)
            nested_ops = decompose_function_tree_as_intermediate(arg_eval, nested_scratch_pos, temp_allocator)
            append!(operations, nested_ops)
            push!(arg_inputs, ScratchPosition(nested_scratch_pos))
        else
            error("Unsupported argument evaluator type: $(typeof(arg_eval))")
        end
    end
    
    # FIXED: Create operations that write to the target scratch position
    n_args = length(arg_inputs)
    
    if n_args == 1
        # FIXED: For unary functions, convert to intermediate binary with dummy second input
        # This ensures consistent handling in the execution logic
        # # println("DEBUG:   Fixed intermediate unary: $(func_eval.func)($(arg_inputs[1]), 1.0) → scratch[$target_scratch]")
        push!(operations, LinearizedOperation(
            :intermediate_binary,  # Use binary type with dummy input
            func_eval.func,
            Union{Symbol, Int, Float64, ScratchPosition}[arg_inputs[1], 1.0],  # Add dummy 1.0
            target_scratch,
            target_scratch
        ))
        
    elseif n_args == 2
        # Binary intermediate
        # # println("DEBUG:   Intermediate binary: $(func_eval.func)($(arg_inputs[1]), $(arg_inputs[2])) → scratch[$target_scratch]")
        push!(operations, LinearizedOperation(
            :intermediate_binary,
            func_eval.func,
            Union{Symbol, Int, Float64, ScratchPosition}[arg_inputs[1], arg_inputs[2]],
            target_scratch,
            target_scratch
        ))
        
    else
        # N-ary intermediate - all steps are intermediate, last writes to target_scratch
        for i in 2:n_args
            if i == 2
                inputs = Union{Symbol, Int, Float64, ScratchPosition}[arg_inputs[1], arg_inputs[2]]
            else
                prev_scratch = ScratchPosition(temp_allocator.next_temp - 1)
                inputs = Union{Symbol, Int, Float64, ScratchPosition}[prev_scratch, arg_inputs[i]]
            end
            
            if i == n_args
                # Final step of n-ary intermediate writes to target_scratch
                # # println("DEBUG:   Final intermediate n-ary: $(func_eval.func)($(inputs[1]), $(inputs[2])) → scratch[$target_scratch]")
                push!(operations, LinearizedOperation(
                    :intermediate_binary,
                    func_eval.func,
                    inputs,
                    target_scratch,
                    target_scratch
                ))
            else
                # Regular intermediate step
                scratch_pos = allocate_temp!(temp_allocator)
                # # println("DEBUG:   Intermediate n-ary step: $(func_eval.func)($(inputs[1]), $(inputs[2])) → scratch[$scratch_pos]")
                push!(operations, LinearizedOperation(
                    :intermediate_binary,
                    func_eval.func,
                    inputs,
                    scratch_pos,
                    scratch_pos
                ))
            end
        end
    end
    
    return operations
end

# Keep the old function for backward compatibility, but mark it as deprecated
function decompose_function_tree(func_eval::FunctionEvaluator, temp_allocator::TempAllocator)
    @warn "decompose_function_tree is deprecated, use decompose_function_tree_zero_alloc"
    return decompose_function_tree_zero_alloc(func_eval, temp_allocator)
end

###############################################################################
# ZERO-ALLOCATION INPUT VALUE ACCESS (PHASE A FIX)
###############################################################################

"""
    get_input_value_zero_alloc(input, output, scratch, input_data, row_idx) -> Float64

Zero-allocation input value access with compile-time type dispatch.
No runtime branching - uses method dispatch instead.
"""

# Constant values - compile-time dispatch
@inline function get_input_value_zero_alloc(input::Float64, output, scratch, input_data, row_idx)
    return input
end

# Column references - compile-time dispatch  
@inline function get_input_value_zero_alloc(input::Symbol, output, scratch, input_data, row_idx)
    return Float64(get_data_value_specialized(input_data, input, row_idx))
end

# Output positions - compile-time dispatch
@inline function get_input_value_zero_alloc(input::Int, output, scratch, input_data, row_idx)
    return output[input]
end

# Scratch positions - compile-time dispatch
@inline function get_input_value_zero_alloc(input::ScratchPosition{P}, output, scratch, input_data, row_idx) where P
    return scratch[input.position]
end

# Keep old function for backward compatibility
@inline function get_input_value(input::Float64, output, input_data, row_idx)
    return input
end

@inline function get_input_value(input::Symbol, output, input_data, row_idx)
    return Float64(get_data_value_specialized(input_data, input, row_idx))
end

@inline function get_input_value(input::Int, output, input_data, row_idx)
    return output[input]
end

###############################################################################
# ZERO-ALLOCATION EXECUTION FUNCTIONS (PHASE A FIX)
###############################################################################

"""
    execute_operation!(data::UnaryFunctionData{F, InputType}, scratch, output, input_data, row_idx)

FIXED: Better debugging and scratch position handling for unary functions.
"""
function execute_operation!(
    data::UnaryFunctionData{F, InputType},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {F, InputType}
    
    input_val = get_input_value_zero_alloc(data.input_source, output, scratch, input_data, row_idx)
    
    # # println("DEBUG: Unary operation $(F) on input_val=$(input_val) from $(data.input_source)")
    
    # Use specialized function application
    result = if F === typeof(log)
        input_val > 0.0 ? log(input_val) : (input_val == 0.0 ? -Inf : NaN)
    elseif F === typeof(exp)
        exp(clamp(input_val, -700.0, 700.0))
    elseif F === typeof(sin)
        sin(input_val)
    elseif F === typeof(cos)
        cos(input_val)
    elseif F === typeof(sqrt)
        input_val ≥ 0.0 ? sqrt(input_val) : NaN
    elseif F === typeof(abs)
        abs(input_val)
    else
        Float64(data.func(input_val))
    end
    
    # Unary functions always write to output (never intermediate)
    output[data.position] = result
    
    # # println("DEBUG: Unary operation $(F) wrote result=$(result) to output[$(data.position)]")
    
    return nothing
end

function execute_operation!(
    data::IntermediateBinaryFunctionData{F, T1, T2},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {F, T1, T2}
    
    val1 = get_input_value_zero_alloc(data.input1, output, scratch, input_data, row_idx)
    val2 = get_input_value_zero_alloc(data.input2, output, scratch, input_data, row_idx)
    
    # FIXED: Better handling of converted unary functions
    # Check if this is a converted unary function (second input is 1.0)
    result = if F === typeof(+)
        val1 + val2
    elseif F === typeof(-)
        val1 - val2
    elseif F === typeof(*)
        val1 * val2
    elseif F === typeof(/)
        val2 == 0.0 ? (val1 == 0.0 ? NaN : (val1 > 0.0 ? Inf : -Inf)) : val1 / val2
    elseif F === typeof(^)
        if val1 == 0.0 && val2 < 0.0
            Inf
        elseif val1 < 0.0 && !isinteger(val2)
            NaN
        else
            val1^val2
        end
    elseif F === typeof(>)
        Float64(val1 > val2)
    elseif F === typeof(<)
        Float64(val1 < val2)
    elseif F === typeof(>=)
        Float64(val1 >= val2)
    elseif F === typeof(<=)
        Float64(val1 <= val2)
    elseif F === typeof(==)
        Float64(val1 == val2)
    # FIXED: Better handling of converted unary functions
    elseif F === typeof(abs)
        # For abs function, if val2 is 1.0, this was a converted unary operation
        if val2 == 1.0
            abs(val1)  # Ignore second input
        else
            # This shouldn't happen, but handle gracefully
            Float64(data.func(val1, val2))
        end
    elseif F === typeof(log)
        # For log function, if val2 is 1.0, this was a converted unary operation
        if val2 == 1.0
            val1 > 0.0 ? log(val1) : (val1 == 0.0 ? -Inf : NaN)
        else
            # This shouldn't happen for unary functions
            Float64(data.func(val1, val2))
        end
    elseif F === typeof(exp)
        if val2 == 1.0
            exp(clamp(val1, -700.0, 700.0))
        else
            Float64(data.func(val1, val2))
        end
    elseif F === typeof(sqrt)
        if val2 == 1.0
            val1 ≥ 0.0 ? sqrt(val1) : NaN
        else
            Float64(data.func(val1, val2))
        end
    elseif F === typeof(sin)
        if val2 == 1.0
            sin(val1)
        else
            Float64(data.func(val1, val2))
        end
    elseif F === typeof(cos)
        if val2 == 1.0
            cos(val1)
        else
            Float64(data.func(val1, val2))
        end
    else
        Float64(data.func(val1, val2))
    end
    
    # ALWAYS write to scratch - no branching
    scratch[data.scratch_position] = result
    
    # # println("DEBUG: Intermediate operation $(F) wrote $(result) to scratch[$(data.scratch_position)]")
    
    return nothing
end

function execute_operation!(
    data::FinalBinaryFunctionData{F, T1, T2},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {F, T1, T2}
    
    val1 = get_input_value_zero_alloc(data.input1, output, scratch, input_data, row_idx)
    val2 = get_input_value_zero_alloc(data.input2, output, scratch, input_data, row_idx)
    
    # # println("DEBUG: Final binary operation $(F) on val1=$(val1) ($(data.input1)), val2=$(val2) ($(data.input2))")
    
    # Specialized function execution (identical logic to intermediate)
    result = if F === typeof(+)
        val1 + val2
    elseif F === typeof(-)
        val1 - val2
    elseif F === typeof(*)
        val1 * val2
    elseif F === typeof(/)
        val2 == 0.0 ? (val1 == 0.0 ? NaN : (val1 > 0.0 ? Inf : -Inf)) : val1 / val2
    elseif F === typeof(^)
        if val1 == 0.0 && val2 < 0.0
            Inf
        elseif val1 < 0.0 && !isinteger(val2)
            NaN
        else
            val1^val2
        end
    elseif F === typeof(>)
        Float64(val1 > val2)
    elseif F === typeof(<)
        Float64(val1 < val2)
    elseif F === typeof(>=)
        Float64(val1 >= val2)
    elseif F === typeof(<=)
        Float64(val1 <= val2)
    elseif F === typeof(==)
        Float64(val1 == val2)
    else
        Float64(data.func(val1, val2))
    end
    
    # ALWAYS write to output - no branching
    output[data.output_position] = result
    
    # # println("DEBUG: Final binary operation $(F) wrote result=$(result) to output[$(data.output_position)]")
    
    return nothing
end

###############################################################################
# ZERO-ALLOCATION TUPLE-BASED EXECUTION (PHASE A FIX)
###############################################################################

"""
    execute_operation!(data::SpecializedFunctionData{UT, IT, FT}, op::FunctionOp{N, M, K}, scratch, output, input_data, row_idx)

FIXED: Execute operations in correct dependency order.
The current order (unary → intermediate → final) is wrong because:
- Unary operations might depend on intermediate results (like log(abs(z)))
- We need to execute in true dependency order: intermediate → unary → final
"""
function execute_operation!(
    data::SpecializedFunctionData{UT, IT, FT},
    op::FunctionOp{N, M, K},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {UT, IT, FT, N, M, K}
    
    # # println("DEBUG: FIXED execute_operation! called with $(N) unary, $(M) intermediate, $(K) final")
    
    # FIXED EXECUTION ORDER: Dependencies first!
    # 1. Intermediate operations first (they write to scratch for others to read)
    execute_intermediate_binaries_recursive!(data.intermediate_binaries, scratch, output, input_data, row_idx)
    
    # 2. Unary functions second (they might read from scratch written by intermediates)
    execute_unary_functions_recursive!(data.unary_functions, scratch, output, input_data, row_idx)
    
    # 3. Final binary functions last (they write to final output)
    execute_final_binaries_recursive!(data.final_binaries, scratch, output, input_data, row_idx)
    
    return nothing
end

"""
    execute_unary_functions_recursive!(unary_tuple, scratch, output, input_data, row_idx)

UPDATED: Recursive execution of unary functions with scratch space support.
"""
function execute_unary_functions_recursive!(
    unary_tuple::Tuple{},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

function execute_unary_functions_recursive!(
    unary_tuple::Tuple,
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(unary_tuple) > 0
        execute_operation!(unary_tuple[1], scratch, output, input_data, row_idx)
        
        if length(unary_tuple) > 1
            remaining = Base.tail(unary_tuple)
            execute_unary_functions_recursive!(remaining, scratch, output, input_data, row_idx)
        end
    end
    return nothing
end

"""
    execute_intermediate_binaries_recursive!(intermediate_tuple, scratch, output, input_data, row_idx)

NEW: Recursive execution of intermediate binary functions.
"""
function execute_intermediate_binaries_recursive!(
    intermediate_tuple::Tuple{},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

function execute_intermediate_binaries_recursive!(
    intermediate_tuple::Tuple,
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(intermediate_tuple) > 0
        execute_operation!(intermediate_tuple[1], scratch, output, input_data, row_idx)
        
        if length(intermediate_tuple) > 1
            remaining = Base.tail(intermediate_tuple)
            execute_intermediate_binaries_recursive!(remaining, scratch, output, input_data, row_idx)
        end
    end
    return nothing
end

"""
    execute_final_binaries_recursive!(final_tuple, scratch, output, input_data, row_idx)

NEW: Recursive execution of final binary functions.
"""
function execute_final_binaries_recursive!(
    final_tuple::Tuple{},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

function execute_final_binaries_recursive!(
    final_tuple::Tuple,
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(final_tuple) > 0
        execute_operation!(final_tuple[1], scratch, output, input_data, row_idx)
        
        if length(final_tuple) > 1
            remaining = Base.tail(final_tuple)
            execute_final_binaries_recursive!(remaining, scratch, output, input_data, row_idx)
        end
    end
    return nothing
end

###############################################################################
# ZERO-ALLOCATION ANALYSIS AND COMPILATION (PHASE A FIX)
###############################################################################

"""
    analyze_function_operations_linear(evaluator::CombinedEvaluator) -> (SpecializedFunctionData, FunctionOp)

REVERTED to simpler approach: Fix the core nested function issue without overcomplicating.
"""
function analyze_function_operations_linear(evaluator::CombinedEvaluator)

    function_evaluators = evaluator.function_evaluators
    n_funcs = length(function_evaluators)
    
    # println("DEBUG: analyze_function_operations_linear called with $n_funcs functions")
    
    # DEBUG: Log all function positions
    for (i, func_eval) in enumerate(function_evaluators)
        # println("DEBUG: Function $i ($(func_eval.func)) writes to position $(func_eval.position)")
    end
    
    ##### END DEBUG
    
    if n_funcs == 0
        empty_data = SpecializedFunctionData((), (), ())
        # # println("DEBUG: Returning empty SpecializedFunctionData with 3 tuples")
        return empty_data, FunctionOp(0, 0, 0)
    end
    
    # SIMPLIFIED: Just use the old decomposition for now but with better debugging
    all_operations = LinearizedOperation[]
    temp_allocator = TempAllocator(1)
    
    for func_eval in function_evaluators
        ops = decompose_function_tree_zero_alloc(func_eval, temp_allocator)
        append!(all_operations, ops)
        # # println("DEBUG: Function created $(length(ops)) operations")
    end
    
    # Separate operations
    unary_ops = filter(op -> op.operation_type == :unary, all_operations)
    intermediate_ops = filter(op -> op.operation_type == :intermediate_binary, all_operations)  
    final_ops = filter(op -> op.operation_type == :final_binary, all_operations)
    
    # Handle intermediate_unary by treating as intermediate_binary
    intermediate_unary_ops = filter(op -> op.operation_type == :intermediate_unary, all_operations)
    for op in intermediate_unary_ops
        # Convert to intermediate binary with dummy second input
        converted_op = LinearizedOperation(
            :intermediate_binary,
            op.func,
            Union{Symbol, Int, Float64, ScratchPosition}[op.inputs[1], 1.0],
            op.output_position,
            op.scratch_position
        )
        push!(intermediate_ops, converted_op)
    end
    
    # # println("DEBUG: Operations breakdown: $(length(unary_ops)) unary, $(length(intermediate_ops)) intermediate, $(length(final_ops)) final")
    
    # Create data structures
    unary_data = create_unary_tuple(unary_ops)
    intermediate_data = create_intermediate_binary_tuple(intermediate_ops)
    final_data = create_final_binary_tuple(final_ops)
    
    specialized_data = SpecializedFunctionData(unary_data, intermediate_data, final_data)
    function_op = FunctionOp(length(unary_ops), length(intermediate_ops), length(final_ops))
    
    # # println("DEBUG: Created NEW SpecializedFunctionData with 3 tuples: $(typeof(specialized_data))")
    
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
    create_intermediate_binary_tuple(intermediate_ops::Vector{LinearizedOperation})

NEW: Create compile-time tuple of intermediate binary function data.
"""
function create_intermediate_binary_tuple(intermediate_ops::Vector{LinearizedOperation})
    n_intermediate = length(intermediate_ops)
    
    if n_intermediate == 0
        return ()
    end
    
    return ntuple(n_intermediate) do i
        op = intermediate_ops[i]
        input1 = length(op.inputs) > 0 ? op.inputs[1] : Symbol()
        input2 = length(op.inputs) > 1 ? op.inputs[2] : Symbol()
        IntermediateBinaryFunctionData(op.func, input1, input2, op.scratch_position)
    end
end

"""
    create_final_binary_tuple(final_ops::Vector{LinearizedOperation})

NEW: Create compile-time tuple of final binary function data.
"""
function create_final_binary_tuple(final_ops::Vector{LinearizedOperation})
    n_final = length(final_ops)
    
    if n_final == 0
        return ()
    end
    
    return ntuple(n_final) do i
        op = final_ops[i]
        input1 = length(op.inputs) > 0 ? op.inputs[1] : Symbol()
        input2 = length(op.inputs) > 1 ? op.inputs[2] : Symbol()
        FinalBinaryFunctionData(op.func, input1, input2, op.output_position)
    end
end

# Keep old functions for backward compatibility
function create_binary_tuple(binary_ops::Vector{LinearizedOperation})
    @warn "create_binary_tuple is deprecated, operations are now separated into intermediate and final"
    return create_final_binary_tuple(binary_ops)
end

###############################################################################
# INTERFACE METHODS FOR INTEGRATION
###############################################################################

"""
    Base.isempty(data::SpecializedFunctionData)

UPDATED: Check if function data is empty.
"""
function Base.isempty(data::SpecializedFunctionData)
    return length(data.unary_functions) == 0 && 
           length(data.intermediate_binaries) == 0 &&
           length(data.final_binaries) == 0
end

"""
    Base.length(data::SpecializedFunctionData)

UPDATED: Get total number of functions.
"""
function Base.length(data::SpecializedFunctionData)
    return length(data.unary_functions) + 
           length(data.intermediate_binaries) +
           length(data.final_binaries)
end

"""
    Base.iterate(data::SpecializedFunctionData, state=1)

UPDATED: Iterate over all functions in the data structure.
"""
function Base.iterate(data::SpecializedFunctionData, state=1)
    total_unary = length(data.unary_functions)
    total_intermediate = length(data.intermediate_binaries)
    total_final = length(data.final_binaries)
    total_functions = total_unary + total_intermediate + total_final
    
    if state > total_functions
        return nothing
    end
    
    if state <= total_unary
        return (data.unary_functions[state], state + 1)
    elseif state <= total_unary + total_intermediate
        intermediate_idx = state - total_unary
        return (data.intermediate_binaries[intermediate_idx], state + 1)
    else
        final_idx = state - total_unary - total_intermediate
        return (data.final_binaries[final_idx], state + 1)
    end
end

###############################################################################
# FUNCTION SCRATCH CALCULATION
###############################################################################

"""
    calculate_max_function_scratch_needed(evaluator::CombinedEvaluator) -> Int

Calculate the maximum scratch space needed by simulating the actual decomposition process.
This ensures the scratch space size matches what the decomposition will actually use.
"""
function calculate_max_function_scratch_needed(evaluator::CombinedEvaluator)
    function_evaluators = evaluator.function_evaluators
    
    if isempty(function_evaluators)
        # # println("DEBUG: No function evaluators, returning 0 scratch")
        return 0
    end
    
    # # println("DEBUG: Calculating scratch for $(length(function_evaluators)) functions")
    
    # Simulate the actual decomposition process to get accurate scratch requirements
    max_scratch_position = 0
    
    for (idx, func_eval) in enumerate(function_evaluators)
        # # println("DEBUG: Processing function $idx: $(func_eval.func) with $(length(func_eval.arg_evaluators)) args")
        
        # Create a temporary allocator to simulate decomposition
        temp_allocator = TempAllocator(1)
        
        # Run the same decomposition logic as analyze_function_operations_linear
        try
            operations = decompose_function_tree_zero_alloc(func_eval, temp_allocator)
            # # println("DEBUG: Function $idx created $(length(operations)) operations")
            
            # Find the maximum scratch position used in this decomposition
            for (op_idx, op) in enumerate(operations)
                if op.scratch_position !== nothing
                    # # println("DEBUG: Operation $op_idx uses scratch position $(op.scratch_position)")
                    max_scratch_position = max(max_scratch_position, op.scratch_position)
                end
                
                # Also check any ScratchPosition inputs
                for (inp_idx, input) in enumerate(op.inputs)
                    if input isa ScratchPosition
                        # # println("DEBUG: Operation $op_idx input $inp_idx reads from scratch position $(input.position)")
                        max_scratch_position = max(max_scratch_position, input.position)
                    end
                end
            end
            
            # # println("DEBUG: Function $idx max scratch position: $max_scratch_position")
            
        catch e
            # If decomposition fails, use a conservative estimate
            @warn "Failed to decompose function for scratch calculation: $e"
            # Conservative fallback: assume complex nested structure
            n_args = length(func_eval.arg_evaluators)
            conservative_estimate = n_args * 3  # Very conservative
            max_scratch_position = max(max_scratch_position, conservative_estimate)
            # # println("DEBUG: Using conservative estimate: $conservative_estimate")
        end
    end
    
    # SAFETY: Ensure we have at least some scratch space if there are any functions
    if max_scratch_position == 0 && !isempty(function_evaluators)
        max_scratch_position = 10  # Conservative fallback
        # # println("DEBUG: Using safety fallback: $max_scratch_position")
    end
    
    # # println("DEBUG: Final calculated max function scratch needed: $max_scratch_position")
    return max_scratch_position
end

"""
    simulate_function_decomposition_for_scratch(evaluator::CombinedEvaluator) -> Int

Alternative implementation that more carefully simulates the decomposition.
Use this if the above doesn't work correctly.
"""
function simulate_function_decomposition_for_scratch(evaluator::CombinedEvaluator)
    function_evaluators = evaluator.function_evaluators
    
    if isempty(function_evaluators)
        return 0
    end
    
    # # println("DEBUG: Simulating decomposition for scratch calculation...")
    
    # Track all scratch positions that will be allocated
    all_scratch_positions = Int[]
    
    for (idx, func_eval) in enumerate(function_evaluators)
        # # println("DEBUG: Simulating function $idx with $(length(func_eval.arg_evaluators)) args")
        
        # Create temp allocator starting from where previous functions left off
        start_position = isempty(all_scratch_positions) ? 1 : maximum(all_scratch_positions) + 1
        temp_allocator = TempAllocator(start_position)
        
        # Simulate the decomposition
        scratch_positions = simulate_single_function_scratch(func_eval, temp_allocator)
        append!(all_scratch_positions, scratch_positions)
        
        # # println("DEBUG: Function $idx uses scratch positions: $scratch_positions")
    end
    
    max_scratch = isempty(all_scratch_positions) ? 0 : maximum(all_scratch_positions)
    # # println("DEBUG: Total scratch positions used: $all_scratch_positions")
    # # println("DEBUG: Maximum scratch position: $max_scratch")
    
    return max_scratch
end

"""
    simulate_single_function_scratch(func_eval::FunctionEvaluator, temp_allocator::TempAllocator) -> Vector{Int}

Simulate scratch position allocation for a single function.
"""
function simulate_single_function_scratch(func_eval::FunctionEvaluator, temp_allocator::TempAllocator)
    positions_used = Int[]
    
    # Simulate argument processing
    for arg_eval in func_eval.arg_evaluators
        if arg_eval isa FunctionEvaluator
            # Nested function needs scratch
            nested_scratch_pos = allocate_temp!(temp_allocator)
            push!(positions_used, nested_scratch_pos)
            
            # Recursively simulate the nested function
            nested_positions = simulate_single_function_scratch(arg_eval, temp_allocator)
            append!(positions_used, nested_positions)
        end
    end
    
    # Simulate n-ary decomposition if needed
    n_args = length(func_eval.arg_evaluators)
    if n_args > 2
        # N-ary functions need (n-2) intermediate positions
        for i in 2:(n_args-1)
            intermediate_pos = allocate_temp!(temp_allocator)
            push!(positions_used, intermediate_pos)
        end
    end
    
    return positions_used
end

###############################################################################
# MAIN EXECUTION INTERFACE
###############################################################################

"""
    execute_linear_function_operations!(function_data::SpecializedFunctionData, scratch, output, data, row_idx)

UPDATED: Main execution interface with zero-allocation scratch space handling.
"""
function execute_linear_function_operations!(
    function_data::SpecializedFunctionData,
    scratch::Vector{Float64},
    output::V,  # Type parameter
    data::NamedTuple,
    row_idx::Int
) where {V <: AbstractVector{Float64}}
    # Create operation encoding for dispatch
    n_unary = length(function_data.unary_functions)
    n_intermediate = length(function_data.intermediate_binaries)
    n_final = length(function_data.final_binaries)
    
    op = FunctionOp(n_unary, n_intermediate, n_final)
    
    # Execute using zero-allocation specialized dispatch
    execute_operation!(function_data, op, scratch, output, data, row_idx)
    
    return nothing
end
