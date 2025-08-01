# step3_polish_linear.jl
# Linear function execution for efficient function evaluation

###############################################################################
# LINEAR FUNCTION EXECUTION TYPES
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
    # Load constant constructor
    @assert operation === :load_constant
    FunctionExecutionStep(operation, nothing, Int[], output_pos, constant_val, nothing)
end

function FunctionExecutionStep(operation::Symbol, output_pos::Int, col::Symbol)
    # Load continuous constructor  
    @assert operation === :load_continuous
    FunctionExecutionStep(operation, nothing, Int[], output_pos, nothing, col)
end

function FunctionExecutionStep(operation::Symbol, func::Function, input_pos::Int, output_pos::Int)
    # Unary function constructor
    @assert operation === :call_unary
    FunctionExecutionStep(operation, func, [input_pos], output_pos, nothing, nothing)
end

function FunctionExecutionStep(operation::Symbol, func::Function, input_pos1::Int, input_pos2::Int, output_pos::Int)
    # Binary function constructor
    @assert operation === :call_binary
    FunctionExecutionStep(operation, func, [input_pos1, input_pos2], output_pos, nothing, nothing)
end

"""
    LinearFunctionData

Pre-computed linear execution plan for a function.
"""
struct LinearFunctionData
    execution_steps::Vector{FunctionExecutionStep}  # Linear sequence of operations
    output_position::Int                             # Final result destination in model matrix
    scratch_size::Int                               # Number of scratch positions needed
end

"""
    ScratchAllocator

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
# ENHANCED COMPREHENSIVE FORMULA TYPES
###############################################################################

"""
    LinearComprehensiveFormulaOp{ConstOp, ContOp, CatOp, FuncOp}

Combined operation encoding for linear comprehensive formulas.
"""
struct LinearComprehensiveFormulaOp{ConstOp, ContOp, CatOp, FuncOp}
    constants::ConstOp
    continuous::ContOp
    categorical::CatOp
    functions::FuncOp                     # LinearFunctionOp
end

"""
    LinearFunctionOp

Operation encoding for linear functions.
"""
struct LinearFunctionOp end

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
    analyze_function_operations_linear(evaluator::CombinedEvaluator) -> (Vector{LinearFunctionData}, LinearFunctionOp)

Extract function data and convert to linear execution plans.
"""
function analyze_function_operations_linear(evaluator::CombinedEvaluator)
    function_evaluators = evaluator.function_evaluators
    n_funcs = length(function_evaluators)
    
    if n_funcs == 0
        # No function operations - return empty vector
        return LinearFunctionData[], LinearFunctionOp()
    end
    
    # Create vector of LinearFunctionData
    function_data = Vector{LinearFunctionData}(undef, n_funcs)
    
    for i in 1:n_funcs
        func_eval = function_evaluators[i]
        function_data[i] = flatten_function_to_linear_plan(func_eval, func_eval.position)
    end
    
    return function_data, LinearFunctionOp()
end

###############################################################################
# LINEAR FUNCTION EXECUTION
###############################################################################

"""
    execute_linear_function!(linear_data::LinearFunctionData, 
                             scratch::Vector{Float64},
                             output::Vector{Float64}, 
                             data::NamedTuple, 
                             row_idx::Int)

REVERTED: Back to Step 3's original working implementation.
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
# STANDALONE FUNCTION EXECUTION - ALSO USE POSITION MAPPING
###############################################################################

"""
    execute_linear_function_operations!(function_data::Vector{LinearFunctionData}, 
                                       scratch::Vector{Float64},
                                       output::Vector{Float64}, 
                                       data::NamedTuple, 
                                       row_idx::Int)

Standalone functions also use position mappings.
"""
function execute_linear_function_operations!(function_data::Vector{LinearFunctionData}, 
                                           scratch::Vector{Float64},
                                           output::Vector{Float64}, 
                                           data::NamedTuple, 
                                           row_idx::Int)
    # Handle empty case
    if isempty(function_data)
        return nothing
    end
    
    # Execute each function using position mappings
    @inbounds for func_data in function_data
        # Get result via position mapping approach
        result = execute_function_via_position_mapping(func_data, data, row_idx)
        
        # Position mapping: func_data.output_position tells us where to write
        output_pos = func_data.output_position  # Position map!
        output[output_pos] = result
    end
    
    return nothing
end
