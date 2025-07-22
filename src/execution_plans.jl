# execution_plans.jl
# Phase 1: Core Data Structures for Zero-Allocation Evaluation

###############################################################################
# CORE DATA STRUCTURES
###############################################################################

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

"""
    ExecutionBlock

Abstract base type for different kinds of execution blocks.
Each block represents a group of operations that can be executed together.
"""
abstract type ExecutionBlock end

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
    SimpleAssignment

A single simple assignment operation.
"""
struct SimpleAssignment
    type::Symbol              # :constant, :continuous
    source::Any              # value for constant, column symbol for continuous
    output_position::Int     # Where to store the result
end

"""
    SimpleAssignmentBlock

Block for simple assignments: constants, continuous variables, etc.
These require no scratch space and compute directly into output positions.
"""
struct SimpleAssignmentBlock <: ExecutionBlock
    assignments::Vector{SimpleAssignment}
end

"""
    InputSource

Where to get input values for operations.
"""
abstract type InputSource end

struct DataSource <: InputSource
    column::Symbol
end

struct ScratchSource <: InputSource
    position::Int
end

struct ConstantSource <: InputSource
    value::Float64
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
    FunctionBlock

Block for function evaluations that may require scratch space.
"""
struct FunctionBlock <: ExecutionBlock
    operations::Vector{FunctionOp}
    scratch_positions::Vector{UnitRange{Int}}  # Scratch space for intermediate results
    output_positions::Vector{Int}              # Final output positions
end

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
    CategoricalBlock

Block for categorical variable evaluation with pre-computed lookup tables.
"""
struct CategoricalBlock <: ExecutionBlock
    layouts::Vector{CategoricalLayout}
end

"""
    InteractionBlock

Block for interaction evaluation using pre-computed layouts.
"""
struct InteractionBlock <: ExecutionBlock
    layout::InteractionLayout
    component_evaluators::Vector{AbstractEvaluator}  # Original evaluators for components
end

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

"""
    compute_kronecker_pattern(component_widths::Vector{Int}, output_range::UnitRange{Int}) -> Vector{Tuple{Int,Int,Int}}

Pre-compute the Kronecker product indexing pattern for an interaction.
Returns a vector of (comp1_idx, comp2_idx, comp3_idx) tuples corresponding to each output position.
"""
function compute_kronecker_pattern(component_widths::Vector{Int}, output_range::UnitRange{Int})
    n_components = length(component_widths)
    pattern = Tuple{Int,Int,Int}[]
    
    if n_components == 2
        # Binary interaction: A × B
        w1, w2 = component_widths[1], component_widths[2]
        for j in 1:w2
            for i in 1:w1
                push!(pattern, (i, j, 0))  # 0 for unused third component
            end
        end
    elseif n_components == 3
        # Three-way interaction: A × B × C
        w1, w2, w3 = component_widths[1], component_widths[2], component_widths[3]
        for k in 1:w3
            for j in 1:w2
                for i in 1:w1
                    push!(pattern, (i, j, k))
                end
            end
        end
    else
        # For other cases, we'll need a more general approach
        # For now, placeholder implementation
        error("Kronecker patterns for $n_components components not yet implemented")
    end
    
    return pattern
end

"""
    get_evaluator_hash(evaluator::AbstractEvaluator) -> UInt

Generate a unique hash for an evaluator to use as identifier in layouts.
"""
function get_evaluator_hash(evaluator::AbstractEvaluator)
    # Use object_id for now - could be more sophisticated
    return hash(evaluator)
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

"""
    add_scratch_position!(layout::ScratchLayout, evaluator::AbstractEvaluator, positions::UnitRange{Int})

Record scratch positions for an evaluator in the layout.
"""
function add_scratch_position!(layout::ScratchLayout, evaluator::AbstractEvaluator, positions::UnitRange{Int})
    evaluator_hash = get_evaluator_hash(evaluator)
    layout.evaluator_positions[evaluator_hash] = positions
    return layout
end

"""
    get_scratch_position(layout::ScratchLayout, evaluator::AbstractEvaluator) -> Union{UnitRange{Int}, Nothing}

Get the scratch positions for an evaluator, if any.
"""
function get_scratch_position(layout::ScratchLayout, evaluator::AbstractEvaluator)
    evaluator_hash = get_evaluator_hash(evaluator)
    return get(layout.evaluator_positions, evaluator_hash, nothing)
end

"""
    add_interaction_layout!(layout::ScratchLayout, interaction_layout::InteractionLayout)

Add an interaction layout to the scratch layout.
"""
function add_interaction_layout!(layout::ScratchLayout, interaction_layout::InteractionLayout)
    push!(layout.interaction_layouts, interaction_layout)
    return layout
end

###############################################################################
# CONSTRUCTOR HELPERS
###############################################################################

"""
    simple_assignment(type::Symbol, source, output_pos::Int) -> SimpleAssignment

Create a simple assignment.
"""
function simple_assignment(type::Symbol, source, output_pos::Int)
    return SimpleAssignment(type, source, output_pos)
end

"""
    constant_assignment(value::Float64, output_pos::Int) -> SimpleAssignment

Create a constant assignment.
"""
function constant_assignment(value::Float64, output_pos::Int)
    return simple_assignment(:constant, value, output_pos)
end

"""
    continuous_assignment(column::Symbol, output_pos::Int) -> SimpleAssignment

Create a continuous variable assignment.
"""
function continuous_assignment(column::Symbol, output_pos::Int)
    return simple_assignment(:continuous, column, output_pos)
end

"""
    function_op(func::Function, inputs::Vector{<:InputSource}, output::OutputDestination) -> FunctionOp

Create a function operation.
"""
function function_op(func::Function, inputs::Vector{<:InputSource}, output::OutputDestination)
    return FunctionOp(func, inputs, output)
end

"""
    data_source(column::Symbol) -> DataSource

Create a data input source.
"""
function data_source(column::Symbol)
    return DataSource(column)
end

"""
    scratch_source(position::Int) -> ScratchSource

Create a scratch input source.
"""
function scratch_source(position::Int)
    return ScratchSource(position)
end

"""
    constant_source(value::Float64) -> ConstantSource

Create a constant input source.
"""
function constant_source(value::Float64)
    return ConstantSource(value)
end

"""
    output_position(pos::Int) -> OutputPosition

Create an output destination.
"""
function output_position(pos::Int)
    return OutputPosition(pos)
end

"""
    scratch_position(pos::Int) -> ScratchPosition

Create a scratch destination.
"""
function scratch_position(pos::Int)
    return ScratchPosition(pos)
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

function Base.show(io::IO, block::SimpleAssignmentBlock)
    println(io, "SimpleAssignmentBlock with $(length(block.assignments)) assignments")
    for assignment in block.assignments
        println(io, "  $(assignment.type): $(assignment.source) → output[$(assignment.output_position)]")
    end
end

function Base.show(io::IO, layout::InteractionLayout)
    println(io, "InteractionLayout:")
    println(io, "  Components: $(length(layout.component_widths)) (widths: $(layout.component_widths))")
    println(io, "  Output positions: $(layout.output_positions)")
    println(io, "  Kronecker pattern: $(length(layout.kronecker_pattern)) elements")
end

###############################################################################
# VALIDATION FUNCTIONS
###############################################################################

"""
    validate_execution_plan(plan::ExecutionPlan) -> Bool

Validate that an execution plan is well-formed.
"""
function validate_execution_plan(plan::ExecutionPlan)
    # Check that output positions don't overlap inappropriately
    # Check that scratch positions are within bounds
    # Check that all required scratch space is allocated
    # etc.
    
    # For now, basic validation
    return plan.scratch_size >= 0 && plan.total_output_width > 0
end

"""
    validate_scratch_layout(layout::ScratchLayout) -> Bool

Validate that a scratch layout is well-formed.
"""
function validate_scratch_layout(layout::ScratchLayout)
    # Check that positions don't overlap
    # Check that all positions are within total_size
    # etc.
    
    # For now, basic validation
    return layout.total_size >= 0
end
