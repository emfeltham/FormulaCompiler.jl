# step3_polish_linear.jl
# Linear function execution for efficient function evaluation

###############################################################################
# FUNCTION FLATTENING ALGORITHM
###############################################################################

"""
    flatten_function_to_linear_plan(func_eval::FunctionEvaluator, output_position::Int) -> LinearFunctionData

Convert a function evaluator tree into a linear execution plan.
"""
function flatten_function_to_linear_plan(func_eval::FunctionEvaluator, output_position::Int)
    allocator = FunctionScratchAllocator()
    steps = FunctionExecutionStep[]
    
    # Flatten the function tree into linear steps
    result_position = flatten_function_recursive!(steps, allocator, func_eval)
    
    # Copy final result to output position if needed
    if result_position != output_position
        # We don't have a copy operation, so we'll handle this in execution
    end
    
    return LinearFunctionData(
        steps,
        output_position,
        allocator.next_position - 1  # Total scratch positions used
    )
end

"""
    flatten_function_recursive!(steps::Vector{FunctionExecutionStep}, 
                               allocator::FunctionScratchAllocator,
                               evaluator::AbstractEvaluator) -> Int

Recursively flatten a function/evaluator into linear steps. Returns scratch position of result.
"""
function flatten_function_recursive!(steps::Vector{FunctionExecutionStep}, 
                                    allocator::FunctionScratchAllocator,
                                    evaluator::AbstractEvaluator)
    
    if evaluator isa ConstantEvaluator
        # Load constant value
        scratch_pos = allocate_scratch_position!(allocator)
        push!(steps, FunctionExecutionStep(:load_constant, scratch_pos, evaluator.value))
        return scratch_pos
        
    elseif evaluator isa ContinuousEvaluator
        # Load continuous variable
        scratch_pos = allocate_scratch_position!(allocator)
        push!(steps, FunctionExecutionStep(:load_continuous, scratch_pos, evaluator.column))
        return scratch_pos
        
    elseif evaluator isa FunctionEvaluator
        # Process function with arguments
        func = evaluator.func
        arg_evaluators = evaluator.arg_evaluators
        n_args = length(arg_evaluators)
        
        if n_args == 1
            # Unary function
            arg_pos = flatten_function_recursive!(steps, allocator, arg_evaluators[1])
            result_pos = allocate_scratch_position!(allocator)
            push!(steps, FunctionExecutionStep(:call_unary, func, arg_pos, result_pos))
            return result_pos
            
        elseif n_args == 2
            # Binary function  
            arg1_pos = flatten_function_recursive!(steps, allocator, arg_evaluators[1])
            arg2_pos = flatten_function_recursive!(steps, allocator, arg_evaluators[2])
            result_pos = allocate_scratch_position!(allocator)
            push!(steps, FunctionExecutionStep(:call_binary, func, arg1_pos, arg2_pos, result_pos))
            return result_pos
            
        else
            error("Functions with $(n_args) arguments not yet supported in linear flattening")
        end
        
    else
        error("Unsupported evaluator type in function flattening: $(typeof(evaluator))")
    end
end

###############################################################################
# LINEAR FUNCTION ANALYSIS
###############################################################################

"""
    analyze_function_operations_linear(evaluator::CombinedEvaluator)

Fully specialized analysis that returns compile-time tuples instead of vectors.
"""
function analyze_function_operations_linear(evaluator::CombinedEvaluator)
    println("USED: analyze new")
    function_evaluators = evaluator.function_evaluators
    n_funcs = length(function_evaluators)
    
    if n_funcs == 0
        # No function operations - return empty tuple
        return (), LinearFunctionOp(0)
    end
    
    # Create tuple of specialized function data using ntuple
    function_data = ntuple(n_funcs) do i
        func_eval = function_evaluators[i]
        linear_data = flatten_function_to_linear_plan(func_eval, func_eval.position)
        
        # Convert to specialized type with embedded metadata
        SpecializedLinearFunctionData(
            linear_data.execution_steps,
            linear_data.output_position,
            linear_data.scratch_size
        )
    end
    
    return function_data, LinearFunctionOp(n_funcs)
end

###############################################################################
# LINEAR FUNCTION EXECUTION
###############################################################################

"""
    execute_linear_function!(
    linear_data::LinearFunctionData, 
    scratch::Vector{Float64},
    output::Vector{Float64}, 
    data::NamedTuple, 
    row_idx::Int
)
"""
function execute_linear_function!(linear_data::LinearFunctionData, 
                                  scratch::Vector{Float64},
                                  output::Vector{Float64}, 
                                  data::NamedTuple, 
                                  row_idx::Int)
    
    # Execute each step in sequence using scratch space
    @inbounds for step in linear_data.execution_steps
        if step.operation === :load_constant
            scratch[step.output_position] = step.constant_value
            
        elseif step.operation === :load_continuous
            col = step.column_symbol
            val = get_data_value_specialized(data, col, row_idx)
            scratch[step.output_position] = Float64(val)
            
        elseif step.operation === :call_unary
            input_val = scratch[step.input_positions[1]]
            result = apply_function_direct_single(step.func, input_val)
            scratch[step.output_position] = result
            
        elseif step.operation === :call_binary
            input_val1 = scratch[step.input_positions[1]]
            input_val2 = scratch[step.input_positions[2]]
            result = apply_function_direct_binary(step.func, input_val1, input_val2)
            scratch[step.output_position] = result
            
        else
            error("Unknown operation type: $(step.operation)")
        end
    end
    
    # Copy final result to output (find the last step's output position)
    if !isempty(linear_data.execution_steps)
        final_scratch_pos = linear_data.execution_steps[end].output_position
        output[linear_data.output_position] = scratch[final_scratch_pos]
    end
    
    return nothing
end

###############################################################################
# NEW: Allocation-Free Function Execution with Pre-Allocated Scratch
###############################################################################

"""
    execute_function_in_preallocated_scratch!(
        func_data::SpecializedLinearFunctionData{StepCount, ScratchSize, OutputPos},
        scratch::AbstractVector{Float64},
        output::AbstractVector{Float64},
        data::NamedTuple,
        row_idx::Int,
        scratch_offset::Int = 0
    ) where {StepCount, ScratchSize, OutputPos}

Execute function using pre-allocated scratch space with offset - NO NEW ALLOCATIONS.
"""
function execute_function_in_preallocated_scratch!(
    func_data::SpecializedLinearFunctionData{StepCount, ScratchSize, OutputPos},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int,
    scratch_offset::Int = 0
) where {StepCount, ScratchSize, OutputPos}
    
    # Ensure we have enough scratch space
    required_scratch = scratch_offset + ScratchSize
    if length(scratch) < required_scratch
        error("Insufficient scratch space: need $required_scratch, have $(length(scratch))")
    end
    
    # Execute the linear plan using the pre-allocated scratch space with offset
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

###############################################################################
# RECURSIVE TUPLE PROCESSING FOR FUNCTIONS
###############################################################################

"""
    execute_function_operations_recursive!(
        function_data::Tuple{},
        scratch::AbstractVector{Float64},
        output::AbstractVector{Float64},
        data::NamedTuple,
        row_idx::Int,
        scratch_offset::Int = 0
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
    # Base case: no functions to process
    return nothing
end

"""
    execute_function_operations_recursive!(
        function_data::Tuple,
        scratch::AbstractVector{Float64},
        output::AbstractVector{Float64},
        data::NamedTuple,
        row_idx::Int,
        scratch_offset::Int = 0
    )

Recursive case: process first function, then recursively process the rest.
"""
function execute_function_operations_recursive!(
    function_data::Tuple,
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int,
    scratch_offset::Int = 0
)
    # Handle empty tuple (should be caught by specialized method above)
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
    
    # Recursively process the remaining functions
    if length(function_data) > 1
        remaining_data = Base.tail(function_data)
        execute_function_operations_recursive!(
            remaining_data, scratch, output, data, row_idx, next_scratch_offset
        )
    end
    
    return nothing
end

###############################################################################
# STANDALONE FUNCTION EXECUTION - ALSO USE POSITION MAPPING
###############################################################################

global_call_count = 0

"""
Overwrite the main function execution to use recursive tuple processing.
This replaces the allocation-heavy approach with allocation-free recursion.
"""
function execute_linear_function_operations!(
    function_data::Tuple,
    scratch::Vector{Float64},
    output::Vector{Float64},
    data::NamedTuple,
    row_idx::Int
)

    global global_call_count
    global_call_count += 1
    println("USED: $global_call_count")
    # Use recursive processing instead of loops and allocations
    execute_function_operations_recursive!(function_data, scratch, output, data, row_idx, 0)
    return nothing
end

"""
    execute_function_in_preallocated_scratch(func_data::LinearFunctionData, scratch::Vector{Float64}, data::NamedTuple, row_idx::Int) -> Float64

Execute function using pre-allocated scratch space - NO NEW ALLOCATIONS.
Handles arbitrary nesting through the linear execution plan.
"""
function execute_function_in_preallocated_scratch(
    func_data::FormulaCompiler.LinearFunctionData, 
    scratch::Vector{Float64}, 
    data::NamedTuple, 
    row_idx::Int
)
    # Ensure we have enough scratch space
    if length(scratch) < func_data.scratch_size
        error("Insufficient scratch space: need $(func_data.scratch_size), have $(length(scratch))")
    end
    
    # Execute the linear plan using the pre-allocated scratch space
    # The scratch space layout is: [pos1, pos2, pos3, ...] where each position
    # corresponds to intermediate results in the execution plan
    
    @inbounds for step in func_data.execution_steps
        if step.operation === :load_constant
            scratch[step.output_position] = step.constant_value
            
        elseif step.operation === :load_continuous
            col = step.column_symbol
            val = FormulaCompiler.get_data_value_specialized(data, col, row_idx)
            scratch[step.output_position] = Float64(val)
            
        elseif step.operation === :call_unary
            input_val = scratch[step.input_positions[1]]
            result = FormulaCompiler.apply_function_direct_single(step.func, input_val)
            scratch[step.output_position] = result
            
        elseif step.operation === :call_binary
            input_val1 = scratch[step.input_positions[1]]
            input_val2 = scratch[step.input_positions[2]]
            result = FormulaCompiler.apply_function_direct_binary(step.func, input_val1, input_val2)
            scratch[step.output_position] = result
            
        else
            error("Unknown operation type: $(step.operation)")
        end
    end
    
    # Return the final result (from the last step's output position)
    if !isempty(func_data.execution_steps)
        final_step = func_data.execution_steps[end]
        return scratch[final_step.output_position]
    else
        return 0.0
    end
end

"""
Overwrite the old allocating function with error message.
"""
function execute_function_via_position_mapping(
    func_data::LinearFunctionData,
    data::NamedTuple,
    row_idx::Int
)
    error("execute_function_via_position_mapping should not be called - use allocation-free execution")
end

###############################################################################
# DEBUGGING AND VALIDATION
###############################################################################

"""
    show_function_specialization_info(function_data)

Display information about function specialization for debugging.
"""
function show_function_specialization_info(function_data)
    println("Function Specialization Info:")
    println("  Type: $(typeof(function_data))")
    println("  Count: $(length(function_data))")
    
    if length(function_data) > 0
        total_scratch = 0
        for (i, func_data) in enumerate(function_data)
            println("  Function $i:")
            println("    Type: $(typeof(func_data))")
            println("    Steps: $(length(func_data.execution_steps))")
            println("    Scratch size: $(func_data.scratch_size)")
            println("    Output position: $(func_data.output_position)")
            total_scratch += func_data.scratch_size
        end
        println("  Total scratch needed: $total_scratch")
    end
end

"""
    validate_function_specialization(formula, df, data)

Validate that function specialization produces correct results.
"""
function validate_function_specialization(formula, df, data)
    println("Validating function specialization for: $formula")
    
    # Compile with new specialization
    model = fit(LinearModel, formula, df)
    compiled = compile_formula_specialized(model, data)
    
    println("Compiled type: $(typeof(compiled))")
    show_function_specialization_info(compiled.data.functions)
    
    # Test execution
    output = Vector{Float64}(undef, length(compiled))
    compiled(output, data, 1)
    
    println("Execution successful: $(output)")
    
    # Test allocation performance
    for _ in 1:10
        compiled(output, data, 1)
    end
    
    allocs = @allocated begin
        for i in 1:100
            compiled(output, data, i)
        end
    end
    
    println("Allocations: $(allocs / 100) bytes per call")
    
    return allocs / 100
end
