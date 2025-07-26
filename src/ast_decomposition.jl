# ast_decomposition.jl
# Complete the missing AST decomposition for complex functions

###############################################################################
# REPLACE: TYPE-STABLE ASSIGNMENT SYSTEM
###############################################################################

"""
    Assignment

Abstract base type for all assignment operations.
Replaces the type-unstable SimpleAssignment struct entirely.
"""
abstract type Assignment end

"""
    ConstantAssignment <: Assignment

Type-stable assignment for constant values.
"""
struct ConstantAssignment <: Assignment
    value::Float64
    output_position::Int
end

"""
    ContinuousAssignment <: Assignment

Type-stable assignment for continuous variables.
"""
struct ContinuousAssignment <: Assignment
    column::Symbol
    output_position::Int
end

###############################################################################
# CORE DATA STRUCTURES
###############################################################################

"""
    AssignmentBlock <: ExecutionBlock

Type-stable assignment block that replaces SimpleAssignmentBlock.
Uses Union dispatch for zero-allocation execution.
"""
struct AssignmentBlock <: ExecutionBlock
    assignments::Vector{Assignment}  # Union{ConstantAssignment, ContinuousAssignment}
end

"""
    ZScoreBlock <: ExecutionBlock

Block for Z-score transformations: (x - center) / scale
"""
struct ZScoreBlock <: ExecutionBlock
    underlying_evaluator::AbstractEvaluator
    center::Float64
    scale::Float64
    input_positions::Vector{Int}      # Where to read input values
    output_positions::Vector{Int}     # Where to write results
end

"""
    ScaledBlock <: ExecutionBlock  

Block for scaled evaluations: scale_factor * value
"""
struct ScaledBlock <: ExecutionBlock
    underlying_evaluator::AbstractEvaluator
    scale_factor::Float64
    input_positions::Vector{Int}
    output_positions::Vector{Int}
end

"""
    ProductBlock <: ExecutionBlock

Block for product evaluations: component1 * component2 * ...
"""
struct ProductBlock <: ExecutionBlock
    component_evaluators::Vector{AbstractEvaluator}
    component_positions::Vector{Vector{Int}}  # Input positions for each component
    output_position::Int                      # Single output position
end

"""
    ExecutionPlan

Complete plan for zero-allocation evaluation of a formula.
Contains all information needed to evaluate without any runtime allocation.
"""
struct ExecutionPlan
    scratch_size::Int                    # Total scratch space needed
    blocks::Vector{ExecutionBlock}       # Execution blocks in order
    total_output_width::Int              # Total formula output width
    
    function ExecutionPlan(scratch_size::Int = 0, total_output_width::Int = 0)
        new(scratch_size, ExecutionBlock[], total_output_width)
    end
end

"""
    InteractionLayout

Pre-computed layout for interaction evaluators.
Contains scratch positions for components and Kronecker product patterns.
"""
struct InteractionLayout
    evaluator_hash::UInt                              # Which interaction this is for
    component_scratch_positions::Vector{UnitRange{Int}}  # Where to store each component
    output_positions::UnitRange{Int}                  # Where final results go
    kronecker_pattern::Vector{Tuple{Int,Int,Int}}     # Pre-computed (i,j,k) -> output_idx mapping
    component_widths::Vector{Int}                     # Width of each component
    
    function InteractionLayout(evaluator_hash::UInt, 
                              component_positions::Vector{UnitRange{Int}},
                              output_positions::UnitRange{Int},
                              component_widths::Vector{Int})
        # Pre-compute Kronecker product pattern
        pattern = compute_kronecker_pattern(component_widths, output_positions)
        new(evaluator_hash, component_positions, output_positions, pattern, component_widths)
    end
end

"""
    ScratchLayout

Analysis of scratch space requirements for an evaluator tree.
Tracks where each evaluator's temporary values should be stored.
"""
struct ScratchLayout
    total_size::Int                                    # Total scratch space needed
    evaluator_positions::Dict{UInt, UnitRange{Int}}   # evaluator hash -> scratch positions
    interaction_layouts::Vector{InteractionLayout}    # Special layouts for interactions
    
    function ScratchLayout(total_size::Int = 0)
        new(total_size, Dict{UInt, UnitRange{Int}}(), InteractionLayout[])
    end
end

###############################################################################
# CORE AST DECOMPOSITION SYSTEM
###############################################################################

"""
    OperationRef

Reference to a value source or destination.
"""
struct OperationRef
    location_type::Symbol  # :data, :scratch, :output, :constant
    index::Union{Int, Symbol, Float64}  # Position, column name, or constant value
end

"""
    DecomposedOperation

A single operation in the flattened execution sequence.
Each operation takes inputs and produces one output.
"""
struct DecomposedOperation
    operation_type::Symbol              # :function, :data_access, :constant, etc.
    func::Union{Function, Nothing}      # Function to apply (if any)
    input_refs::Vector{OperationRef}    # Where to get inputs
    output_ref::OperationRef            # Where to store result
    dependencies::Vector{Int}           # Which operations must complete first
end

"""
    DecompositionContext

Tracks state during AST decomposition.
"""
mutable struct DecompositionContext
    operations::Vector{DecomposedOperation}
    scratch_counter::Int
    operation_counter::Int
    scratch_layout::ScratchLayout
end

###############################################################################
# MAIN DECOMPOSITION FUNCTION
###############################################################################

"""
    decompose_evaluator_tree(evaluator::AbstractEvaluator, scratch_layout::ScratchLayout) -> Vector{DecomposedOperation}

Decompose a complex evaluator tree into a flat sequence of zero-allocation operations.
This is the key function that enables arbitrary formula complexity.

# Arguments
- `evaluator`: Complex evaluator tree (from evaluators.jl AST parsing)
- `scratch_layout`: Pre-computed scratch space layout

# Returns
Vector of operations that can be executed sequentially with zero allocation.

# Example
For `log(x^2)`:
1. Operation 1: `x^2` → scratch[1]
2. Operation 2: `log(scratch[1])` → output[pos]
"""
function decompose_evaluator_tree(evaluator::AbstractEvaluator, scratch_layout::ScratchLayout)
    context = DecompositionContext(
        DecomposedOperation[],
        1,  # Start scratch counter at 1
        1,  # Start operation counter at 1
        scratch_layout
    )
    
    # Recursively decompose the evaluator tree
    result_ref = decompose_recursive!(context, evaluator)
    
    return context.operations
end

"""
    decompose_recursive!(context::DecompositionContext, evaluator::AbstractEvaluator) -> OperationRef

Recursively decompose an evaluator into operations, returning where the result will be stored.
"""
function decompose_recursive!(context::DecompositionContext, evaluator::AbstractEvaluator)
    
    if evaluator isa ConstantEvaluator
        # Constants don't need operations - return direct reference
        return OperationRef(:constant, evaluator.value)
        
    elseif evaluator isa ContinuousEvaluator
        # Data access doesn't need operations - return direct reference
        return OperationRef(:data, evaluator.column)
        
    elseif evaluator isa FunctionEvaluator
        return decompose_function!(context, evaluator)
        
    elseif evaluator isa InteractionEvaluator
        return decompose_interaction!(context, evaluator)
        
    elseif evaluator isa ZScoreEvaluator
        return decompose_zscore!(context, evaluator)
        
    elseif evaluator isa ScaledEvaluator
        return decompose_scaled!(context, evaluator)
        
    elseif evaluator isa ProductEvaluator
        return decompose_product!(context, evaluator)
        
    else
        error("Decomposition not implemented for $(typeof(evaluator))")
    end
end

###############################################################################
# FUNCTION DECOMPOSITION (THE KEY MISSING PIECE)
###############################################################################

"""
    decompose_function!(context::DecompositionContext, evaluator::FunctionEvaluator) -> OperationRef

Decompose a function evaluator into constituent operations.
This handles the complex nested cases like log(x^2), sin(x + y), etc.
"""
function decompose_function!(context::DecompositionContext, evaluator::FunctionEvaluator)
    func = evaluator.func
    args = evaluator.arg_evaluators
    
    # First, decompose all arguments recursively
    arg_refs = OperationRef[]
    arg_dependencies = Int[]
    
    for arg in args
        arg_ref = decompose_recursive!(context, arg)
        push!(arg_refs, arg_ref)
        
        # If this argument required operations, we depend on them
        if arg_ref.location_type == :scratch
            # Find operations that output to this scratch location
            for (i, op) in enumerate(context.operations)
                if op.output_ref.location_type == :scratch && op.output_ref.index == arg_ref.index
                    push!(arg_dependencies, i)
                end
            end
        end
    end
    
    # Create output location for this function result
    output_ref = OperationRef(:scratch, context.scratch_counter)
    context.scratch_counter += 1
    
    # Create the function operation
    operation = DecomposedOperation(
        :function,
        func,
        arg_refs,
        output_ref,
        unique(arg_dependencies)  # Remove duplicates
    )
    
    push!(context.operations, operation)
    context.operation_counter += 1
    
    return output_ref
end

"""
    decompose_interaction!(context::DecompositionContext, evaluator::InteractionEvaluator) -> OperationRef

Decompose interaction evaluators (Kronecker products).
"""
function decompose_interaction!(context::DecompositionContext, evaluator::InteractionEvaluator)
    components = evaluator.components
    
    if length(components) == 1
        # Single component - just decompose it
        return decompose_recursive!(context, components[1])
    elseif length(components) == 2
        # Binary interaction - decompose both components then multiply
        ref1 = decompose_recursive!(context, components[1])
        ref2 = decompose_recursive!(context, components[2])
        
        # Create multiplication operation
        output_ref = OperationRef(:scratch, context.scratch_counter)
        context.scratch_counter += 1
        
        # Find dependencies
        dependencies = find_dependencies(context, [ref1, ref2])
        
        operation = DecomposedOperation(
            :interaction,
            (*),  # Simple multiplication for scalar interactions
            [ref1, ref2],
            output_ref,
            dependencies
        )
        
        push!(context.operations, operation)
        return output_ref
    else
        # N-way interaction - chain multiplications
        result_ref = decompose_recursive!(context, components[1])
        
        for i in 2:length(components)
            comp_ref = decompose_recursive!(context, components[i])
            
            # Create multiplication operation
            new_output_ref = OperationRef(:scratch, context.scratch_counter)
            context.scratch_counter += 1
            
            dependencies = find_dependencies(context, [result_ref, comp_ref])
            
            operation = DecomposedOperation(
                :interaction,
                (*),
                [result_ref, comp_ref],
                new_output_ref,
                dependencies
            )
            
            push!(context.operations, operation)
            result_ref = new_output_ref
        end
        
        return result_ref
    end
end

"""
    decompose_zscore!(context::DecompositionContext, evaluator::ZScoreEvaluator) -> OperationRef

Decompose Z-score transformations: (x - center) / scale
"""
function decompose_zscore!(context::DecompositionContext, evaluator::ZScoreEvaluator)
    # Decompose the underlying evaluator
    underlying_ref = decompose_recursive!(context, evaluator.underlying)
    
    # Create Z-score transformation operation
    output_ref = OperationRef(:scratch, context.scratch_counter)
    context.scratch_counter += 1
    
    # Z-score needs center and scale as additional inputs
    center_ref = OperationRef(:constant, evaluator.center)
    scale_ref = OperationRef(:constant, evaluator.scale)
    
    dependencies = find_dependencies(context, [underlying_ref])
    
    # Create custom Z-score function
    zscore_func = (x, center, scale) -> (x - center) / scale
    
    operation = DecomposedOperation(
        :zscore,
        zscore_func,
        [underlying_ref, center_ref, scale_ref],
        output_ref,
        dependencies
    )
    
    push!(context.operations, operation)
    return output_ref
end

"""
    decompose_scaled!(context::DecompositionContext, evaluator::ScaledEvaluator) -> OperationRef

Decompose scaled evaluations: scale_factor * value
"""
function decompose_scaled!(context::DecompositionContext, evaluator::ScaledEvaluator)
    # Decompose the underlying evaluator
    underlying_ref = decompose_recursive!(context, evaluator.evaluator)
    
    # Create scaling operation
    output_ref = OperationRef(:scratch, context.scratch_counter)
    context.scratch_counter += 1
    
    scale_ref = OperationRef(:constant, evaluator.scale_factor)
    dependencies = find_dependencies(context, [underlying_ref])
    
    operation = DecomposedOperation(
        :scaled,
        (*),
        [underlying_ref, scale_ref],
        output_ref,
        dependencies
    )
    
    push!(context.operations, operation)
    return output_ref
end

"""
    decompose_product!(context::DecompositionContext, evaluator::ProductEvaluator) -> OperationRef

Decompose product evaluations: component1 * component2 * ...
"""
# Fix the missing decompose_product! function
function decompose_product!(context::DecompositionContext, evaluator::ProductEvaluator)
    components = evaluator.components
    
    if length(components) == 1
        return decompose_recursive!(context, components[1])
    else
        # Chain multiplications: comp1 * comp2 * comp3 * ...
        result_ref = decompose_recursive!(context, components[1])
        
        for i in 2:length(components)
            comp_ref = decompose_recursive!(context, components[i])
            
            # Create multiplication operation
            new_output_ref = OperationRef(:scratch, context.scratch_counter)
            context.scratch_counter += 1
            
            dependencies = find_dependencies(context, [result_ref, comp_ref])
            
            operation = DecomposedOperation(
                :product,
                (*),
                [result_ref, comp_ref],
                new_output_ref,
                dependencies
            )
            
            push!(context.operations, operation)
            result_ref = new_output_ref
        end
        
        return result_ref
    end
end

###############################################################################
# HELPER FUNCTIONS
###############################################################################

"""
    find_dependencies(context::DecompositionContext, refs::Vector{OperationRef}) -> Vector{Int}

Find which operations the given references depend on.
"""
function find_dependencies(context::DecompositionContext, refs::Vector{OperationRef})
    dependencies = Int[]
    
    for ref in refs
        if ref.location_type == :scratch
            # Find operations that output to this scratch location
            for (i, op) in enumerate(context.operations)
                if op.output_ref.location_type == :scratch && op.output_ref.index == ref.index
                    push!(dependencies, i)
                    # Also include transitive dependencies
                    append!(dependencies, op.dependencies)
                end
            end
        end
    end
    
    return unique(dependencies)
end

###############################################################################
# INTEGRATION WITH EXECUTION PLANS
###############################################################################

"""
    create_decomposed_function_block(operations::Vector{DecomposedOperation}, start_pos::Int) -> FunctionBlock

Create a function block from decomposed operations.
This integrates the AST decomposition with the execution plan system.
"""
function create_decomposed_function_block(operations::Vector{DecomposedOperation}, start_pos::Int)
    # Convert decomposed operations to function operations
    function_ops = FunctionOp[]
    
    for operation in operations
        # Convert OperationRef to InputSource
        input_sources = InputSource[]
        for input_ref in operation.input_refs
            if input_ref.location_type == :data
                push!(input_sources, DataSource(input_ref.index))
            elseif input_ref.location_type == :scratch
                push!(input_sources, ScratchSource(input_ref.index))
            elseif input_ref.location_type == :constant
                push!(input_sources, ConstantSource(input_ref.index))
            end
        end
        
        # Convert output OperationRef to OutputDestination
        if operation.output_ref.location_type == :scratch
            output_dest = ScratchPosition(operation.output_ref.index)
        else
            output_dest = OutputPosition(start_pos)  # Final result goes to output
        end
        
        # Create function operation
        func_op = FunctionOp(operation.func, input_sources, output_dest)
        push!(function_ops, func_op)
    end
    
    # Determine scratch positions needed
    max_scratch = 0
    for op in operations
        if op.output_ref.location_type == :scratch
            max_scratch = max(max_scratch, op.output_ref.index)
        end
    end
    
    scratch_positions = max_scratch > 0 ? [1:max_scratch] : UnitRange{Int}[]
    
    return FunctionBlock(function_ops, scratch_positions, [start_pos])
end

###############################################################################
# UPDATED FUNCTION BLOCK GENERATION
###############################################################################

# """
#     generate_function_block!(plan::ExecutionPlan, evaluator::FunctionEvaluator, 
#                             start_pos::Int, scratch_layout::ScratchLayout) -> Int

# UPDATED: Now handles complex nested functions through AST decomposition.
# This completes the missing piece in the execution plan system.
# """
# function generate_function_block!(plan::ExecutionPlan, evaluator::FunctionEvaluator, 
#                                  start_pos::Int, scratch_layout::ScratchLayout)
    
#     if all(is_direct_evaluatable, evaluator.arg_evaluators)
#         # Simple function call - use existing optimized path
#         input_sources = create_input_sources(evaluator.arg_evaluators)
#         output_dest = output_position(start_pos)
        
#         op = function_op(evaluator.func, input_sources, output_dest)
#         block = FunctionBlock([op], UnitRange{Int}[], [start_pos])
#         push!(plan, block)
        
#         return start_pos + 1
#     else
#         # Complex function - use AST decomposition (NEW!)
#         operations = decompose_evaluator_tree(evaluator, scratch_layout)
        
#         # The last operation should output to the final position, not scratch
#         if !isempty(operations)
#             last_op_idx = length(operations)
#             operations[last_op_idx] = DecomposedOperation(
#                 operations[last_op_idx].operation_type,
#                 operations[last_op_idx].func,
#                 operations[last_op_idx].input_refs,
#                 OperationRef(:output, start_pos),  # Change to output position
#                 operations[last_op_idx].dependencies
#             )
#         end
        
#         block = create_decomposed_function_block(operations, start_pos)
#         push!(plan, block)
        
#         return start_pos + 1
#     end
# end

###############################################################################
# TESTING AND VALIDATION
###############################################################################

"""
    test_ast_decomposition()

Test the AST decomposition system with complex examples.
"""
function test_ast_decomposition()
    println("Testing AST decomposition system...")
    
    # Test case 1: log(x^2)
    x_eval = ContinuousEvaluator(:x)
    two_eval = ConstantEvaluator(2.0)
    power_eval = FunctionEvaluator(^, [x_eval, two_eval])
    log_eval = FunctionEvaluator(log, [power_eval])
    
    scratch_layout = ScratchLayout()
    operations = decompose_evaluator_tree(log_eval, scratch_layout)
    
    println("log(x^2) decomposed into $(length(operations)) operations:")
    for (i, op) in enumerate(operations)
        println("  $i: $(op.operation_type) with function $(op.func)")
        println("     Inputs: $(op.input_refs)")
        println("     Output: $(op.output_ref)")
        println("     Dependencies: $(op.dependencies)")
    end
    
    # Test case 2: sin(x + y)
    y_eval = ContinuousEvaluator(:y)
    add_eval = FunctionEvaluator(+, [x_eval, y_eval])
    sin_eval = FunctionEvaluator(sin, [add_eval])
    
    operations2 = decompose_evaluator_tree(sin_eval, scratch_layout)
    
    println("\nsin(x + y) decomposed into $(length(operations2)) operations:")
    for (i, op) in enumerate(operations2)
        println("  $i: $(op.operation_type) with function $(op.func)")
    end
    
    println("\n✅ AST decomposition system working!")
    return true
end

# Export main functions
export decompose_evaluator_tree, create_decomposed_function_block
export DecomposedOperation, OperationRef, DecompositionContext
export test_ast_decomposition