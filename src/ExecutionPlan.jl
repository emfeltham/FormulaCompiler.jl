# ExecutionPlan.jl

###############################################################################
# TYPE-STABLE ASSIGNMENT SYSTEM
###############################################################################

"""
    Assignment

Abstract base type for all assignment operations.
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

###############################################################################
# Source structs
###############################################################################

"""
    InputSource

Where to get input values for operations.
"""
abstract type InputSource end

struct DataSource <: InputSource
    column::Symbol
end

"""
    data_source(column::Symbol) -> DataSource

Create a data input source.
"""
function data_source(column::Symbol)
    return DataSource(column)
end

struct ScratchSource <: InputSource
    position::Int
end

struct ConstantSource <: InputSource
    value::Float64
end

"""
    constant_source(value::Float64) -> ConstantSource

Create a constant input source.
"""
function constant_source(value::Float64)
    return ConstantSource(value)
end

"""
    OutputDestination

Where to store operation results.
"""
abstract type OutputDestination end

struct OutputPosition <: OutputDestination
    position::Int
end

struct ScratchPosition <: OutputDestination
    position::Int
end

###############################################################################
# LAYOUTS (used within Blocks)
###############################################################################

"""
    CategoricalLayout

Pre-computed layout for a categorical variable.
"""
struct CategoricalLayout
    column::Symbol
    n_levels::Int
    lookup_tables::Vector{Vector{Float64}}  # Pre-computed contrasts for each output
    output_positions::Vector{Int}
end

"""
    InteractionLayout{N}

Layout for interaction evaluation using pre-computed layouts with parametric Kronecker patterns.
Updated to use NTuple{N,Int} for type stability.
"""
struct InteractionLayout{N}
    evaluator_hash::UInt
    component_scratch_positions::Vector{UnitRange{Int}}
    output_positions::UnitRange{Int}
    kronecker_pattern::Vector{NTuple{N,Int}}  # Now parametric!
    component_widths::Vector{Int}
    
    # Updated constructor with parametric pattern
    function InteractionLayout{N}(evaluator_hash::UInt, 
                                 component_scratch_positions::Vector{UnitRange{Int}},
                                 output_positions::UnitRange{Int},
                                 kronecker_pattern::Vector{NTuple{N,Int}},
                                 component_widths::Vector{Int}) where N
        new{N}(evaluator_hash, component_scratch_positions, output_positions, 
               kronecker_pattern, component_widths)
    end
    
    # Convenience constructor that infers N from component_widths
    function InteractionLayout(evaluator_hash::UInt, 
                              component_scratch_positions::Vector{UnitRange{Int}},
                              output_positions::UnitRange{Int},
                              component_widths::Vector{Int})
        N = length(component_widths)
        pattern = compute_kronecker_pattern(component_widths)  # Returns Vector{NTuple{N,Int}}
        InteractionLayout{N}(evaluator_hash, component_scratch_positions, output_positions, 
                            pattern, component_widths)
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
# BLOCKS
###############################################################################

"""
    ExecutionBlock

Abstract base type for different kinds of execution blocks.
Each block represents a group of operations that can be executed together.
"""
abstract type ExecutionBlock end

"""
    AssignmentBlock <: ExecutionBlock

Type-stable assignment. Uses Union dispatch.
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
    FunctionOp

A single function operation with input sources and output destination.
"""
struct FunctionOp
    func::Function
    input_sources::Vector{InputSource}
    output_destination::OutputDestination
end

"""
    function_op(func::Function, inputs::Vector{<:InputSource}, output::OutputDestination) -> FunctionOp

Create a function operation.
"""
function function_op(func::Function, inputs::Vector{<:InputSource}, output::OutputDestination)
    return FunctionOp(func, inputs, output)
end

"""
    FunctionBlock

Block for function evaluations that may require scratch space.
"""
struct FunctionBlock <: ExecutionBlock
    operations::Vector{FunctionOp}
    scratch_positions::Vector{UnitRange{Int}}  # Scratch space for intermediate results
    output_positions::Vector{Int}              # Final output positions
end

"""
    CategoricalBlock

Block for categorical variable evaluation with pre-computed lookup tables.
"""
struct CategoricalBlock <: ExecutionBlock
    layouts::Vector{CategoricalLayout}
end

"""
    InteractionBlock{N}

Block for interaction evaluation using pre-computed layouts.
Updated to use parametric InteractionLayout{N}.
"""
struct InteractionBlock{N} <: ExecutionBlock
    layout::InteractionLayout{N}
    component_evaluators::Vector{AbstractEvaluator}
end

###############################################################################
# ExecutionPlan
###############################################################################

struct ExecutionPlan
    scratch_size::Int
    blocks::Vector{ExecutionBlock}
    total_output_width::Int
    data_length::Int
    validated_columns::Set{Symbol}
    
    # Default constructor
    ExecutionPlan(
        scratch_size, blocks, total_output_width,
        data_length, validated_columns
    ) = new(
        scratch_size, blocks, total_output_width, data_length, validated_columns
    )
end

# Convenience constructor from evaluator + data
function ExecutionPlan(evaluator::AbstractEvaluator, data::NamedTuple)
    scratch_size = max_scratch_needed(evaluator)
    total_output_width = output_width(evaluator)
    data_length = length(first(data))
    validated_columns = Set(keys(data))
    
    # Create empty plan and populate it
    blocks = ExecutionBlock[]
    temp_plan = ExecutionPlan(
        scratch_size, blocks,
        total_output_width, data_length, validated_columns
    )
    generate_blocks!(temp_plan, evaluator)
    validate_plan_against_data!(temp_plan, data)
    
    return temp_plan
end

"""
    Base.length(plan::ExecutionPlan) -> Int

Get the number of execution blocks in a plan.
"""
Base.length(plan::ExecutionPlan) = length(plan.blocks)

"""
    Base.push!(plan::ExecutionPlan, block::ExecutionBlock)

Add an execution block to a plan.
"""
function Base.push!(plan::ExecutionPlan, block::ExecutionBlock)
    push!(plan.blocks, block)
    return plan
end

###############################################################################
# EXECUTION PLAN SYSTEM
###############################################################################

"""
    add_assignment!(plan::ExecutionPlan, assignment::Assignment)

Add a type-stable assignment to the plan.
Replaces add_simple_assignment! entirely.
"""
function add_assignment!(plan::ExecutionPlan, assignment::Assignment)
    # Check if the last block is an AssignmentBlock we can extend
    if !isempty(plan.blocks) && plan.blocks[end] isa AssignmentBlock
        push!(plan.blocks[end].assignments, assignment)
    else
        # Create new AssignmentBlock
        block = AssignmentBlock([assignment])
        push!(plan, block)
    end
end

###############################################################################
# DISPLAY METHODS
###############################################################################

"""
Pretty printing for execution plans and related structures.
"""
function Base.show(io::IO, plan::ExecutionPlan)
    println(io, "ExecutionPlan:")
    println(io, "  Scratch size: $(plan.scratch_size)")
    println(io, "  Output width: $(plan.total_output_width)")
    println(io, "  Execution blocks: $(length(plan.blocks))")
    for (i, block) in enumerate(plan.blocks)
        println(io, "    $i. $(typeof(block).name.name)")
    end
end

function Base.show(io::IO, layout::ScratchLayout)
    println(io, "ScratchLayout:")
    println(io, "  Total size: $(layout.total_size)")
    println(io, "  Evaluator positions: $(length(layout.evaluator_positions))")
    println(io, "  Interaction layouts: $(length(layout.interaction_layouts))")
end

function Base.show(io::IO, layout::InteractionLayout)
    println(io, "InteractionLayout:")
    println(io, "  Components: $(length(layout.component_widths)) (widths: $(layout.component_widths))")
    println(io, "  Output positions: $(layout.output_positions)")
    println(io, "  Kronecker pattern: $(length(layout.kronecker_pattern)) elements")
end

"""
    Base.show(io::IO, block::AssignmentBlock)

Pretty printing for type-stable assignment blocks.
"""
function Base.show(io::IO, block::AssignmentBlock)
    println(io, "AssignmentBlock with $(length(block.assignments)) assignments")
    for assignment in block.assignments
        if assignment isa ConstantAssignment
            println(io, "  constant: $(assignment.value) → output[$(assignment.output_position)]")
        elseif assignment isa ContinuousAssignment
            println(io, "  continuous: $(assignment.column) → output[$(assignment.output_position)]")
        end
    end
end
