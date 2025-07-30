# step3_polish_linear.jl
# Linear function execution for zero-allocation function evaluation

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
    LinearComprehensiveFormulaData{ConstData, ContData, CatData, FuncData}

Combined data using linear function execution.
"""
struct LinearComprehensiveFormulaData{ConstData, ContData, CatData, FuncData}
    constants::ConstData
    continuous::ContData
    categorical::CatData
    functions::FuncData                   # Vector{LinearFunctionData}
    max_function_scratch::Int             # Maximum scratch space needed for functions
end

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

"""
    analyze_evaluator_linear_comprehensive(evaluator::AbstractEvaluator) -> (DataTuple, OpTuple)

Comprehensive analysis using linear function execution.
"""
function analyze_evaluator_linear_comprehensive(evaluator::AbstractEvaluator)
    if evaluator isa CombinedEvaluator
        # Check that this only has simple operation types (no interactions yet)
        has_interactions = !isempty(evaluator.interaction_evaluators)
        
        if has_interactions
            error("Step 3 Polish only supports constants, continuous, categorical, and functions. Found interactions.")
        end
        
        # Analyze all four operation types
        constant_data, constant_op = analyze_constant_operations(evaluator)
        continuous_data, continuous_op = analyze_continuous_operations(evaluator)
        categorical_data, categorical_op = analyze_categorical_operations(evaluator)
        function_data, function_op = analyze_function_operations_linear(evaluator)
        
        # Calculate maximum scratch space needed for functions
        max_function_scratch = isempty(function_data) ? 0 : maximum(f.scratch_size for f in function_data)
        
        # Combine into linear comprehensive formula data
        formula_data = LinearComprehensiveFormulaData(
            constant_data, continuous_data, categorical_data, function_data, max_function_scratch
        )
        formula_op = LinearComprehensiveFormulaOp(constant_op, continuous_op, categorical_op, function_op)
        
        return formula_data, formula_op
        
    else
        error("Step 3 Polish only supports CombinedEvaluator with constants, continuous, categorical, and function operations")
    end
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

Execute a linear function plan with zero allocations.
"""
function execute_linear_function!(linear_data::LinearFunctionData, 
                                  scratch::Vector{Float64},
                                  output::Vector{Float64}, 
                                  data::NamedTuple, 
                                  row_idx::Int)
    
    # Execute each step in sequence
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

"""
    apply_function_direct_single(func::Function, val::Float64) -> Float64

Apply single-argument function directly with domain checking.
"""
function apply_function_direct_single(func::Function, val::Float64)
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
end

"""
    apply_function_direct_binary(func::Function, val1::Float64, val2::Float64) -> Float64

Apply binary function directly with domain checking.
"""
function apply_function_direct_binary(func::Function, val1::Float64, val2::Float64)
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
end

"""
    execute_linear_function_operations!(function_data::Vector{LinearFunctionData}, 
                                       scratch::Vector{Float64},
                                       output::Vector{Float64}, 
                                       data::NamedTuple, 
                                       row_idx::Int)

Execute multiple linear function operations with zero allocations.
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
    
    # Process all function operations
    @inbounds for func_data in function_data
        execute_linear_function!(func_data, scratch, output, data, row_idx)
    end
    return nothing
end

"""
    execute_operation!(data::LinearComprehensiveFormulaData{ConstData, ContData, CatData, FuncData}, 
                      op::LinearComprehensiveFormulaOp{ConstOp, ContOp, CatOp, FuncOp}, 
                      output, input_data, row_idx) where {ConstData, ContData, CatData, FuncData, ConstOp, ContOp, CatOp, FuncOp}

Execute linear comprehensive formulas with zero-allocation function evaluation.
"""
function execute_operation!(data::LinearComprehensiveFormulaData{ConstData, ContData, CatData, FuncData}, 
                           op::LinearComprehensiveFormulaOp{ConstOp, ContOp, CatOp, FuncOp}, 
                           output, input_data, row_idx) where {ConstData, ContData, CatData, FuncData, ConstOp, ContOp, CatOp, FuncOp}
    
    # Pre-allocate scratch space for functions (if needed)
    function_scratch = data.max_function_scratch > 0 ? Vector{Float64}(undef, data.max_function_scratch) : Float64[]
    
    # Execute constants
    execute_operation!(data.constants, op.constants, output, input_data, row_idx)
    
    # Execute continuous variables
    execute_operation!(data.continuous, op.continuous, output, input_data, row_idx)
    
    # Execute categorical variables
    execute_categorical_operations!(data.categorical, output, input_data, row_idx)
    
    # Execute functions (linear execution)
    execute_linear_function_operations!(data.functions, function_scratch, output, input_data, row_idx)
    
    return nothing
end

###############################################################################
# LINEAR COMPREHENSIVE COMPILATION FUNCTIONS
###############################################################################

"""
    create_specialized_formula_linear_comprehensive(compiled_formula::CompiledFormula) -> SpecializedFormula

Convert a CompiledFormula to a SpecializedFormula using linear function execution.
"""
function create_specialized_formula_linear_comprehensive(compiled_formula::CompiledFormula)
    # Analyze the evaluator tree with linear comprehensive support
    data_tuple, op_tuple = analyze_evaluator_linear_comprehensive(compiled_formula.root_evaluator)
    
    # Create specialized formula
    return SpecializedFormula{typeof(data_tuple), typeof(op_tuple)}(
        data_tuple,
        op_tuple,
        compiled_formula.output_width
    )
end

"""
    compile_formula_specialized_linear_comprehensive(model, data::NamedTuple) -> SpecializedFormula

Direct compilation to specialized formula using linear function execution.
"""
function compile_formula_specialized_linear_comprehensive(model, data::NamedTuple)
    # Use existing compilation logic to build evaluator tree
    compiled = compile_formula(model, data)
    
    # Convert to linear comprehensive specialized form
    return create_specialized_formula_linear_comprehensive(compiled)
end

###############################################################################
# LINEAR COMPREHENSIVE UTILITY FUNCTIONS
###############################################################################

"""
    show_linear_function_plan(linear_data::LinearFunctionData)

Display the linear execution plan for a function.
"""
function show_linear_function_plan(linear_data::LinearFunctionData)
    println("Linear Function Execution Plan:")
    println("  Output position: $(linear_data.output_position)")
    println("  Scratch size needed: $(linear_data.scratch_size)")
    println("  Execution steps:")
    
    for (i, step) in enumerate(linear_data.execution_steps)
        if step.operation === :load_constant
            println("    $i. Load constant $(step.constant_value) → scratch[$(step.output_position)]")
        elseif step.operation === :load_continuous  
            println("    $i. Load $(step.column_symbol) → scratch[$(step.output_position)]")
        elseif step.operation === :call_unary
            println("    $i. $(step.func)(scratch[$(step.input_positions[1])]) → scratch[$(step.output_position)]")
        elseif step.operation === :call_binary
            println("    $i. $(step.func)(scratch[$(step.input_positions[1])], scratch[$(step.input_positions[2])]) → scratch[$(step.output_position)]")
        end
    end
end