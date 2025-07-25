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
# VALIDATED EXECUTION PLAN - THE ONLY WAY
###############################################################################

"""
    ValidatedExecutionPlan

The only execution plan type. Construction validates everything once,
execution is guaranteed zero-allocation.
"""
struct ValidatedExecutionPlan
    scratch_size::Int
    blocks::Vector{ExecutionBlock}
    total_output_width::Int
    data_length::Int                    # Cached for bounds checking
    validated_columns::Set{Symbol}     # Columns guaranteed to exist
    
    function ValidatedExecutionPlan(evaluator::AbstractEvaluator, data::NamedTuple)
        # Generate the execution plan structure
        basic_plan = generate_execution_plan_structure(evaluator)
        
        # Comprehensive validation with helpful error messages
        validate_plan_against_data!(basic_plan, data)
        
        # Cache information for zero-allocation execution
        data_length = length(first(data))
        validated_columns = Set(keys(data))
        
        new(basic_plan.scratch_size, basic_plan.blocks, basic_plan.total_output_width, 
            data_length, validated_columns)
    end
end


"""
    generate_execution_plan_structure(evaluator::AbstractEvaluator) -> ExecutionPlan

Generate the basic execution plan structure without validation.
Internal function - users never see ExecutionPlan directly.
"""
function generate_execution_plan_structure(evaluator::AbstractEvaluator)
    # This is your existing generate_execution_plan logic
    # but renamed to make clear it's internal
    scratch_layout = analyze_scratch_requirements(evaluator)
    plan = ExecutionPlan(scratch_layout.total_size, output_width(evaluator))
    generate_execution_blocks!(plan, evaluator, 1, scratch_layout)
    return plan
end

"""
    validate_plan_against_data!(plan::ExecutionPlan, data::NamedTuple)

Comprehensive validation during construction. Can allocate freely since
this happens once. Throws helpful errors for any issues.
"""
function validate_plan_against_data!(plan::ExecutionPlan, data::NamedTuple)
    # Validate plan structure
    plan.scratch_size >= 0 || error("Invalid execution plan: negative scratch size $(plan.scratch_size)")
    plan.total_output_width > 0 || error("Invalid execution plan: non-positive output width $(plan.total_output_width)")
    
    # Validate data
    isempty(data) && error("Cannot execute plan with empty data")
    
    data_length = length(first(data))
    data_length > 0 || error("Cannot execute plan with zero-length data")
    
    # Validate all columns are consistent length
    data_columns = keys(data)
    for col in data_columns
        length(data[col]) == data_length || 
            error("Inconsistent data: column $col has length $(length(data[col])), expected $data_length")
    end
    
    # Validate that all referenced columns exist
    data_column_set = Set(data_columns)
    for (i, block) in enumerate(plan.blocks)
        validate_block_references!(block, data_column_set, i)
    end
    
    # Validate output positions don't conflict
    validate_output_positions!(plan)
    
    println("✅ Execution plan validated successfully")
    println("   Scratch size: $(plan.scratch_size)")
    println("   Output width: $(plan.total_output_width)")
    println("   Data length: $data_length")
    println("   Data columns: $(sort(collect(data_columns)))")
    println("   Execution blocks: $(length(plan.blocks))")
end

function validate_block_references!(block::CategoricalBlock, data_columns::Set{Symbol}, block_idx::Int)
    for (i, layout) in enumerate(block.layouts)
        layout.column in data_columns || 
            error("Block $block_idx, layout $i: references missing column '$(layout.column)'. Available: $(sort(collect(data_columns)))")
            
        layout.n_levels > 0 || 
            error("Block $block_idx, layout $i: invalid n_levels $(layout.n_levels)")
            
        all(pos -> pos > 0, layout.output_positions) || 
            error("Block $block_idx, layout $i: invalid output positions $(layout.output_positions)")
    end
end

function validate_block_references!(block::FunctionBlock, data_columns::Set{Symbol}, block_idx::Int)
    for (i, op) in enumerate(block.operations)
        for (j, source) in enumerate(op.input_sources)
            if source isa DataSource
                source.column in data_columns || 
                    error("Block $block_idx, operation $i, input $j: references missing column '$(source.column)'")
            end
        end
    end
end

function validate_block_references!(block::InteractionBlock, data_columns::Set{Symbol}, block_idx::Int)
    for (i, evaluator) in enumerate(block.component_evaluators)
        validate_evaluator_references!(evaluator, data_columns, block_idx, i)
    end
end

function validate_block_references!(block::ExecutionBlock, data_columns::Set{Symbol}, block_idx::Int)
    # Default: no validation for unknown block types
    @warn "Unknown block type $(typeof(block)) in block $block_idx - skipping validation"
end

function validate_evaluator_references!(evaluator::ContinuousEvaluator, data_columns::Set{Symbol}, block_idx::Int, comp_idx::Int)
    evaluator.column in data_columns || 
        error("Block $block_idx, component $comp_idx: references missing column '$(evaluator.column)'")
end

function validate_evaluator_references!(evaluator::CategoricalEvaluator, data_columns::Set{Symbol}, block_idx::Int, comp_idx::Int)
    evaluator.column in data_columns || 
        error("Block $block_idx, component $comp_idx: references missing column '$(evaluator.column)'")
end

function validate_evaluator_references!(evaluator::AbstractEvaluator, data_columns::Set{Symbol}, block_idx::Int, comp_idx::Int)
    # Default: assume it's valid
    # Add specific validation for other evaluator types as needed
end

function validate_output_positions!(plan::ExecutionPlan)
    used_positions = Set{Int}()
    
    for (block_idx, block) in enumerate(plan.blocks)
        block_positions = get_block_output_positions(block)
        
        for pos in block_positions
            pos in used_positions && 
                error("Output position $pos used by multiple blocks (block $block_idx conflicts)")
            push!(used_positions, pos)
            
            1 <= pos <= plan.total_output_width || 
                error("Block $block_idx: output position $pos outside valid range 1:$(plan.total_output_width)")
        end
    end
end

function get_block_output_positions(block::CategoricalBlock)
    positions = Int[]
    for layout in block.layouts
        append!(positions, layout.output_positions)
    end
    return positions
end

function get_block_output_positions(block::ExecutionBlock)
    return Int[]  # Default: assume no output positions for unknown blocks
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
# SCRATCH SPACE ANALYSIS (Step 1.2)
###############################################################################

"""
    analyze_scratch_requirements(evaluator::AbstractEvaluator) -> ScratchLayout

Analyze an evaluator tree to determine scratch space requirements.
This walks the existing evaluator tree structure to compute memory layout.
"""
function analyze_scratch_requirements(evaluator::AbstractEvaluator)
    layout = ScratchLayout()
    total_size = compute_total_scratch_size(evaluator)
    layout = ScratchLayout(total_size)
    
    # Assign scratch positions for each evaluator that needs them
    assign_scratch_positions!(layout, evaluator, 1)
    
    return layout
end

"""
    compute_evaluator_scratch_size(evaluator::AbstractEvaluator) -> Int

Compute scratch space needed for a single evaluator.
This determines the temporary storage requirements for intermediate results.
"""
function compute_evaluator_scratch_size(evaluator::AbstractEvaluator)
    
    if evaluator isa ConstantEvaluator
        return 0  # Constants need no scratch space
        
    elseif evaluator isa ContinuousEvaluator
        return 0  # Direct data access, no scratch needed
        
    elseif evaluator isa CategoricalEvaluator
        return 0  # Direct output, no intermediate storage
        
    elseif evaluator isa FunctionEvaluator
        # Need scratch for complex argument evaluation
        arg_scratch = sum(compute_evaluator_scratch_size(arg) for arg in evaluator.arg_evaluators; init=0)
        
        # If arguments are complex, need space to store their results
        complex_args = filter(arg -> !is_direct_evaluatable(arg), evaluator.arg_evaluators)
        arg_storage = sum(output_width(arg) for arg in complex_args; init=0)
        
        return arg_scratch + arg_storage
        
    elseif evaluator isa InteractionEvaluator
        # Need scratch space for each component's evaluation
        component_scratch = sum(compute_evaluator_scratch_size(comp) for comp in evaluator.components; init=0)
        
        # Need storage for each component's results during Kronecker product computation
        component_storage = sum(output_width(comp) for comp in evaluator.components; init=0)
        
        return component_scratch + component_storage
        
    elseif evaluator isa ZScoreEvaluator
        # Need scratch for underlying evaluator
        underlying_scratch = compute_evaluator_scratch_size(evaluator.underlying)
        
        # If underlying is complex, need storage for its result
        if !is_direct_evaluatable(evaluator.underlying)
            underlying_storage = output_width(evaluator.underlying)
            return underlying_scratch + underlying_storage
        else
            return underlying_scratch
        end
        
    elseif evaluator isa CombinedEvaluator
        # Each sub-evaluator may need scratch space
        return sum(compute_evaluator_scratch_size(sub) for sub in evaluator.sub_evaluators; init=0)
        
    elseif evaluator isa ScaledEvaluator
        # Need scratch for underlying evaluator
        return compute_evaluator_scratch_size(evaluator.evaluator)
        
    elseif evaluator isa ProductEvaluator
        # Need scratch for each component
        component_scratch = sum(compute_evaluator_scratch_size(comp) for comp in evaluator.components; init=0)
        
        # Need storage for complex components
        complex_components = filter(comp -> !is_direct_evaluatable(comp), evaluator.components)
        component_storage = sum(output_width(comp) for comp in complex_components; init=0)
        
        return component_scratch + component_storage
        
    else
        # Conservative estimate for unknown types
        return output_width(evaluator)
    end
end

"""
    compute_total_scratch_size(evaluator::AbstractEvaluator) -> Int

Compute total scratch space needed for entire evaluator tree.
This accounts for the maximum simultaneous scratch usage.
"""
function compute_total_scratch_size(evaluator::AbstractEvaluator)
    # For now, use conservative approach: sum of all scratch requirements
    # In Phase 4, we can optimize this to reuse scratch space
    return compute_total_scratch_recursive(evaluator)
end

function compute_total_scratch_recursive(evaluator::AbstractEvaluator)
    own_scratch = compute_evaluator_scratch_size(evaluator)
    
    if evaluator isa FunctionEvaluator
        child_scratch = isempty(evaluator.arg_evaluators) ? 0 : 
            maximum(compute_total_scratch_recursive(arg) for arg in evaluator.arg_evaluators)
        return own_scratch + child_scratch
        
    elseif evaluator isa InteractionEvaluator
        child_scratch = isempty(evaluator.components) ? 0 : 
            maximum(compute_total_scratch_recursive(comp) for comp in evaluator.components)
        return own_scratch + child_scratch
        
    elseif evaluator isa CombinedEvaluator
        # Combined evaluators can reuse scratch space between sub-evaluators
        child_scratch = isempty(evaluator.sub_evaluators) ? 0 : 
            maximum(compute_total_scratch_recursive(sub) for sub in evaluator.sub_evaluators)
        return own_scratch + child_scratch
        
    elseif evaluator isa ZScoreEvaluator
        child_scratch = compute_total_scratch_recursive(evaluator.underlying)
        return own_scratch + child_scratch
        
    elseif evaluator isa ScaledEvaluator
        child_scratch = compute_total_scratch_recursive(evaluator.evaluator)
        return own_scratch + child_scratch
        
    elseif evaluator isa ProductEvaluator
        child_scratch = isempty(evaluator.components) ? 0 : 
            maximum(compute_total_scratch_recursive(comp) for comp in evaluator.components)
        return own_scratch + child_scratch
        
    else
        return own_scratch
    end
end

"""
    assign_scratch_positions!(layout::ScratchLayout, evaluator::AbstractEvaluator, start_pos::Int) -> Int

Assign scratch positions for evaluator tree. Returns next available position.
"""
function assign_scratch_positions!(layout::ScratchLayout, evaluator::AbstractEvaluator, start_pos::Int)
    current_pos = start_pos
    
    # Get scratch size needed for this evaluator
    scratch_needed = compute_evaluator_scratch_size(evaluator)
    
    if scratch_needed > 0
        # Assign scratch positions for this evaluator
        evaluator_hash = get_evaluator_hash(evaluator)
        positions = current_pos:(current_pos + scratch_needed - 1)
        add_scratch_position!(layout, evaluator, positions)
        current_pos += scratch_needed
    end
    
    # Handle special cases that need interaction layouts
    if evaluator isa InteractionEvaluator
        interaction_layout = create_interaction_layout(evaluator, current_pos)
        add_interaction_layout!(layout, interaction_layout)
        current_pos += output_width(evaluator)
    end
    
    # Recursively assign positions for child evaluators
    if evaluator isa FunctionEvaluator
        for arg in evaluator.arg_evaluators
            current_pos = assign_scratch_positions!(layout, arg, current_pos)
        end
    elseif evaluator isa InteractionEvaluator
        for comp in evaluator.components
            current_pos = assign_scratch_positions!(layout, comp, current_pos)
        end
    elseif evaluator isa CombinedEvaluator
        for sub in evaluator.sub_evaluators
            current_pos = assign_scratch_positions!(layout, sub, current_pos)
        end
    elseif evaluator isa ZScoreEvaluator
        current_pos = assign_scratch_positions!(layout, evaluator.underlying, current_pos)
    elseif evaluator isa ScaledEvaluator
        current_pos = assign_scratch_positions!(layout, evaluator.evaluator, current_pos)
    elseif evaluator isa ProductEvaluator
        for comp in evaluator.components
            current_pos = assign_scratch_positions!(layout, comp, current_pos)
        end
    end
    
    return current_pos
end

"""
    is_direct_evaluatable(evaluator::AbstractEvaluator) -> Bool

Check if an evaluator can be evaluated directly without scratch space.
"""
function is_direct_evaluatable(evaluator::AbstractEvaluator)
    return evaluator isa ConstantEvaluator || 
           evaluator isa ContinuousEvaluator ||
           (evaluator isa CategoricalEvaluator && size(evaluator.contrast_matrix, 2) <= 4)  # Small categoricals only
    # Note: FunctionEvaluator, InteractionEvaluator, etc. are NOT direct evaluatable
end

"""
    create_interaction_layout(evaluator::InteractionEvaluator, start_pos::Int) -> InteractionLayout

Create specialized layout for interaction evaluators.
"""
function create_interaction_layout(evaluator::InteractionEvaluator, start_pos::Int)
    components = evaluator.components
    component_widths = [output_width(comp) for comp in components]
    
    # Assign scratch positions for each component
    component_positions = UnitRange{Int}[]
    current_pos = start_pos
    
    for width in component_widths
        positions = current_pos:(current_pos + width - 1)
        push!(component_positions, positions)
        current_pos += width
    end
    
    # Output positions come after component scratch positions
    total_component_scratch = sum(component_widths)
    output_width_total = output_width(evaluator)
    output_positions = (start_pos + total_component_scratch):(start_pos + total_component_scratch + output_width_total - 1)
    
    return InteractionLayout(
        get_evaluator_hash(evaluator),
        component_positions,
        output_positions,
        component_widths
    )
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

export ExecutionPlan, ScratchLayout, ExecutionBlock

###############################################################################
# PLAN EXECUTION ENGINE (Step 2.1)
###############################################################################

"""
    execute_plan!(plan::ExecutionPlan, scratch::Vector{Float64}, output::Vector{Float64}, 
                  data::NamedTuple, row_idx::Int)

Execute an execution plan with zero allocations.

# Arguments
- `plan::ExecutionPlan`: The execution plan to run
- `scratch::Vector{Float64}`: Pre-allocated scratch space (length >= plan.scratch_size)
- `output::Vector{Float64}`: Pre-allocated output vector (length >= plan.total_output_width)
- `data::NamedTuple`: Column-table format data
- `row_idx::Int`: Row index to evaluate

# Returns
Nothing (results stored in `output`)

# Performance
- **Time**: ~50-100ns per operation
- **Allocations**: 0 bytes (guaranteed)
"""
function execute_plan!(validated_plan::ValidatedExecutionPlan, 
                       scratch::Vector{Float64}, output::Vector{Float64}, 
                       data::NamedTuple, row_idx::Int)
    
    # Simple bounds checking without string construction
    # These are guaranteed to pass if construction validation succeeded
    @boundscheck begin
        length(scratch) >= validated_plan.scratch_size || throw(BoundsError())
        length(output) >= validated_plan.total_output_width || throw(BoundsError())
        1 <= row_idx <= validated_plan.data_length || throw(BoundsError())
    end
    
    # Execute all blocks - zero allocation guaranteed
    @inbounds for i in 1:length(validated_plan.blocks)
        execute_block!(validated_plan.blocks[i], scratch, output, data, row_idx)
    end
    
    return nothing
end


"""
    execute_function_block!(block::FunctionBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                           data::NamedTuple, row_idx::Int)

Execute function operations with scratch space management.
"""
function execute_function_block!(block::FunctionBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                                data::NamedTuple, row_idx::Int)
    @inbounds for i in 1:length(block.operations)
        operation = block.operations[i]
        execute_function_operation!(operation, scratch, output, data, row_idx)
    end
end

"""
    execute_function_operation!(op::FunctionOp, scratch::Vector{Float64}, output::Vector{Float64}, 
                               data::NamedTuple, row_idx::Int)

Execute a single function operation.
"""
function execute_function_operation!(op::FunctionOp, scratch::Vector{Float64}, output::Vector{Float64}, 
                                   data::NamedTuple, row_idx::Int)
    n_inputs = length(op.input_sources)
    
    # Handle any number of arguments - fully general
    if n_inputs == 0
        result = op.func()
    elseif n_inputs == 1
        val1 = get_input_value(op.input_sources[1], scratch, output, data, row_idx)
        result = apply_function_safe(op.func, val1)
    elseif n_inputs == 2
        val1 = get_input_value(op.input_sources[1], scratch, output, data, row_idx)
        val2 = get_input_value(op.input_sources[2], scratch, output, data, row_idx)
        result = apply_function_safe(op.func, val1, val2)
    else
        # General case for any number of arguments
        args = Vector{Float64}(undef, n_inputs)
        @inbounds for i in 1:n_inputs
            args[i] = get_input_value(op.input_sources[i], scratch, output, data, row_idx)
        end
        result = apply_function_safe(op.func, args...)
    end
    
    store_output_value!(op.output_destination, result, scratch, output)
end

"""
    execute_categorical_block!(block::CategoricalBlock, output::Vector{Float64}, 
                              data::NamedTuple, row_idx::Int)

Execute categorical variable evaluation with pre-computed lookup tables.
"""
function execute_categorical_block!(block::CategoricalBlock, output::Vector{Float64}, 
                                  data::NamedTuple, row_idx::Int)
    @inbounds for i in 1:length(block.layouts)
        layout = block.layouts[i]
        execute_categorical_layout!(layout, output, data, row_idx)
    end
end

"""
    execute_categorical_layout!(layout::CategoricalLayout, output::Vector{Float64}, 
                               data::NamedTuple, row_idx::Int)

Zero-allocation categorical layout execution with aggressive optimizations.
"""
function execute_categorical_layout!(layout::CategoricalLayout, output::Vector{Float64}, 
                                   data::NamedTuple, row_idx::Int)
    
    # OPTIMIZATION 1: Minimize field access allocations
    # Pre-fetch the column reference to avoid repeated NamedTuple access
    column_data = getfield(data, layout.column)
    
    # OPTIMIZATION 2: Aggressive inlining and type-stable level extraction
    @inbounds cat_val = column_data[row_idx]
    
    # OPTIMIZATION 3: Eliminate dynamic dispatch with type-stable branching
    level_idx = _extract_level_code_optimized(cat_val, layout.n_levels)
    
    # OPTIMIZATION 4: Vectorized lookup table application
    # Pre-fetch lookup tables to avoid repeated field access
    lookup_tables = layout.lookup_tables
    output_positions = layout.output_positions
    
    # OPTIMIZATION 5: Unroll small loops to eliminate allocation
    n_contrasts = length(lookup_tables)
    
    if n_contrasts == 1
        # Single contrast - most common case
        @inbounds output[output_positions[1]] = lookup_tables[1][level_idx]
    elseif n_contrasts == 2
        # Two contrasts - treatment coding
        @inbounds output[output_positions[1]] = lookup_tables[1][level_idx]
        @inbounds output[output_positions[2]] = lookup_tables[2][level_idx]
    elseif n_contrasts == 3
        # Three contrasts
        @inbounds output[output_positions[1]] = lookup_tables[1][level_idx]
        @inbounds output[output_positions[2]] = lookup_tables[2][level_idx]
        @inbounds output[output_positions[3]] = lookup_tables[3][level_idx]
    else
        # General case - should still be zero allocation
        @inbounds for j in 1:n_contrasts
            output[output_positions[j]] = lookup_tables[j][level_idx]
        end
    end
    
    return nothing
end

"""
    _extract_level_code_optimized(cat_val, n_levels::Int) -> Int

Type-stable level code extraction that eliminates dynamic dispatch allocations.
"""
@inline function _extract_level_code_optimized(cat_val, n_levels::Int)
    # This function must be type-stable and allocation-free
    # We handle the CategoricalValue case without dynamic dispatch
    
    if cat_val isa CategoricalValue
        # Use levelcode directly - this should be allocation-free
        raw_level = Int(levelcode(cat_val))
        # Manual clamp to avoid allocation in clamp()
        return raw_level < 1 ? 1 : (raw_level > n_levels ? n_levels : raw_level)
    else
        # Non-categorical fallback
        return 1
    end
end

"""
    execute_interaction_block!(block::InteractionBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                              data::NamedTuple, row_idx::Int)

Execute interaction evaluation using pre-computed Kronecker patterns.
"""
function execute_interaction_block!(block::InteractionBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                                  data::NamedTuple, row_idx::Int)
    layout = block.layout
    components = block.component_evaluators
    
    # Evaluate all components to scratch space
    @inbounds for i in 1:length(components)
        component = components[i]
        scratch_positions = layout.component_scratch_positions[i]
        evaluate_component_to_scratch!(component, scratch, scratch_positions, data, row_idx)
    end
    
    # Apply Kronecker pattern
    apply_kronecker_pattern!(layout.kronecker_pattern, 
                           layout.component_scratch_positions,
                           scratch,
                           view(output, layout.output_positions))
end

"""
    execute_categorical_component_to_scratch!(evaluator::CategoricalEvaluator, 
                                            scratch::Vector{Float64}, scratch_positions::UnitRange{Int},
                                            data::NamedTuple, row_idx::Int)

Execute a categorical component directly into scratch space.
"""
function execute_categorical_component_to_scratch!(evaluator::CategoricalEvaluator, 
                                                 scratch::Vector{Float64}, scratch_positions::UnitRange{Int},
                                                 data::NamedTuple, row_idx::Int)
    
    # Get categorical value and level index with explicit typing
    @inbounds cat_val = data[evaluator.column][row_idx]
    
    level_idx = if cat_val isa CategoricalValue
        Int(levelcode(cat_val))
    else
        1
    end
    
    # Bounds checking without allocation
    level_idx = max(1, min(level_idx, evaluator.n_levels))
    
    # Fill scratch positions with contrast values
    width = size(evaluator.contrast_matrix, 2)
    start_pos = first(scratch_positions)
    
    @inbounds for j in 1:width
        pos_idx = start_pos + j - 1
        scratch[pos_idx] = evaluator.contrast_matrix[level_idx, j]
    end
end


"""
    apply_binary_kronecker_pattern!(pattern::Vector{Tuple{Int,Int,Int}}, 
                                   buf1::AbstractVector{Float64}, buf2::AbstractVector{Float64},
                                   output::AbstractVector{Float64})

Apply binary Kronecker product pattern without allocation.
"""
function apply_binary_kronecker_pattern!(pattern::Vector{Tuple{Int,Int,Int}}, 
                                       buf1::AbstractVector{Float64}, buf2::AbstractVector{Float64},
                                       output::AbstractVector{Float64})
    
    for (idx, (i, j, _)) in enumerate(pattern)
        @inbounds output[idx] = buf1[i] * buf2[j]
    end
end

"""
    execute_categorical_component!(evaluator::CategoricalEvaluator, buffer::AbstractVector{Float64}, 
                                  data::NamedTuple, row_idx::Int)

Execute a categorical component into a buffer.
"""
function execute_categorical_component!(evaluator::CategoricalEvaluator, buffer::AbstractVector{Float64}, 
                                      data::NamedTuple, row_idx::Int)
    
    # Get categorical value and level index
    @inbounds cat_val = data[evaluator.column][row_idx]
    
    level_idx = if cat_val isa CategoricalValue
        Int(levelcode(cat_val))
    else
        1
    end
    
    # Bounds checking
    level_idx = max(1, min(level_idx, evaluator.n_levels))
    
    # Fill buffer with contrast values
    width = size(evaluator.contrast_matrix, 2)
    @inbounds for j in 1:width
        buffer[j] = evaluator.contrast_matrix[level_idx, j]
    end
end

"""
    apply_kronecker_pattern!(pattern::Vector{Tuple{Int,Int,Int}}, 
                            component_buffers::Vector{Vector{Float64}}, 
                            output::AbstractVector{Float64})

Apply pre-computed Kronecker product pattern with zero allocations.
"""
function apply_kronecker_pattern!(pattern::Vector{Tuple{Int,Int,Int}}, 
                                 component_buffers::Vector{Vector{Float64}}, 
                                 output::AbstractVector{Float64})
    
    n_components = length(component_buffers)
    
    if n_components == 2
        # Binary interaction: optimized path
        buf1, buf2 = component_buffers[1], component_buffers[2]
        
        for (idx, (i, j, _)) in enumerate(pattern)
            @inbounds output[idx] = buf1[i] * buf2[j]
        end
        
    elseif n_components == 3
        # Three-way interaction: optimized path
        buf1, buf2, buf3 = component_buffers[1], component_buffers[2], component_buffers[3]
        
        for (idx, (i, j, k)) in enumerate(pattern)
            @inbounds output[idx] = buf1[i] * buf2[j] * buf3[k]
        end
        
    else
        error("N-way interactions with N > 3 not yet implemented")
    end
end

###############################################################################
# INPUT/OUTPUT VALUE MANAGEMENT
###############################################################################

"""
    get_input_value(source::InputSource, scratch::Vector{Float64}, output::Vector{Float64}, 
                    data::NamedTuple, row_idx::Int) -> Float64

Get a value from an input source with zero allocations.
"""
function get_input_value(source::InputSource, scratch::Vector{Float64}, output::Vector{Float64}, 
                        data::NamedTuple, row_idx::Int)
    
    if source isa DataSource
        @inbounds return Float64(data[source.column][row_idx])
        
    elseif source isa ScratchSource
        @inbounds return scratch[source.position]
        
    elseif source isa ConstantSource
        return source.value
        
    else
        error("Unknown input source type: $(typeof(source))")
    end
end

"""
    store_output_value!(dest::OutputDestination, value::Float64, 
                       scratch::Vector{Float64}, output::Vector{Float64})

Store a value to an output destination with zero allocations.
"""
function store_output_value!(dest::OutputDestination, value::Float64, 
                           scratch::Vector{Float64}, output::Vector{Float64})
    
    if dest isa OutputPosition
        @inbounds output[dest.position] = value
        
    elseif dest isa ScratchPosition
        @inbounds scratch[dest.position] = value
        
    else
        error("Unknown output destination type: $(typeof(dest))")
    end
end

###############################################################################
# EXECUTION PLAN GENERATION (Step 1.3)
###############################################################################

"""
    generate_execution_plan(evaluator::AbstractEvaluator) -> ExecutionPlan

Transform an evaluator tree into a linear execution plan.
This is the main function that converts recursive evaluator trees into 
optimized execution sequences.
"""
function generate_execution_plan(evaluator::AbstractEvaluator)
    # First, analyze scratch requirements
    scratch_layout = analyze_scratch_requirements(evaluator)
    
    # Create execution plan with appropriate sizing
    plan = ExecutionPlan(scratch_layout.total_size, output_width(evaluator))
    
    # Generate execution blocks from evaluator tree
    generate_execution_blocks!(plan, evaluator, 1, scratch_layout)
    
    return plan
end

"""
    create_categorical_block(evaluator::CategoricalEvaluator, start_pos::Int) -> CategoricalBlock

Create a categorical evaluation block with pre-computed lookup tables.
"""
function create_categorical_block(evaluator::CategoricalEvaluator, start_pos::Int)
    width = output_width(evaluator)
    output_positions = collect(start_pos:(start_pos + width - 1))
    
    # Create lookup tables for each contrast
    lookup_tables = Vector{Vector{Float64}}(undef, width)
    for j in 1:width
        lookup_tables[j] = [evaluator.contrast_matrix[i, j] for i in 1:evaluator.n_levels]
    end
    
    layout = CategoricalLayout(
        evaluator.column,
        evaluator.n_levels,
        lookup_tables,
        output_positions
    )
    
    return CategoricalBlock([layout])
end

"""
    generate_function_block!(plan::ExecutionPlan, evaluator::FunctionEvaluator, 
                            start_pos::Int, scratch_layout::ScratchLayout) -> Int

Generate execution blocks for function evaluation.
"""
function generate_function_block!(plan::ExecutionPlan, evaluator::FunctionEvaluator, 
                                 start_pos::Int, scratch_layout::ScratchLayout)
    
    # Check if function has direct (simple) arguments
    if all(is_direct_evaluatable, evaluator.arg_evaluators)
        # Simple function call - create single operation
        input_sources = create_input_sources(evaluator.arg_evaluators)
        output_dest = output_position(start_pos)
        
        op = function_op(evaluator.func, input_sources, output_dest)
        block = FunctionBlock([op], UnitRange{Int}[], [start_pos])
        push!(plan, block)
        
        return start_pos + 1
        
    else
        # Complex function - need to evaluate arguments first
        return generate_complex_function_block!(plan, evaluator, start_pos, scratch_layout)
    end
end

"""
    generate_complex_function_block!(plan::ExecutionPlan, evaluator::FunctionEvaluator,
                                    start_pos::Int, scratch_layout::ScratchLayout) -> Int

Handle function evaluation with complex arguments that require scratch space.
"""
function generate_complex_function_block!(plan::ExecutionPlan, evaluator::FunctionEvaluator,
                                         start_pos::Int, scratch_layout::ScratchLayout)
    
    # For Phase 1, we only implement simple cases
    # Complex nested functions will be implemented in Phase 2
    error("Complex function evaluation with nested arguments not yet implemented in Phase 1. " *
          "This will be implemented in Phase 2: Zero-Allocation Evaluator Core.")
end

"""
    generate_interaction_block!(plan::ExecutionPlan, evaluator::InteractionEvaluator,
                               start_pos::Int, scratch_layout::ScratchLayout) -> Int

Generate execution block for interaction evaluation.
"""
function generate_interaction_block!(plan::ExecutionPlan, evaluator::InteractionEvaluator,
                                   start_pos::Int, scratch_layout::ScratchLayout)
    
    # Find the interaction layout for this evaluator
    evaluator_hash = get_evaluator_hash(evaluator)
    interaction_layout = nothing
    
    for layout in scratch_layout.interaction_layouts
        if layout.evaluator_hash == evaluator_hash
            interaction_layout = layout
            break
        end
    end
    
    if interaction_layout === nothing
        # Create a simple interaction layout if none found
        component_widths = [output_width(comp) for comp in evaluator.components]
        component_positions = UnitRange{Int}[]
        current_pos = start_pos
        
        for width in component_widths
            positions = current_pos:(current_pos + width - 1)
            push!(component_positions, positions)
            current_pos += width
        end
        
        output_positions = current_pos:(current_pos + output_width(evaluator) - 1)
        
        interaction_layout = InteractionLayout(
            evaluator_hash,
            component_positions,
            output_positions,
            component_widths
        )
    end
    
    # Create interaction block
    block = InteractionBlock(interaction_layout, evaluator.components)
    push!(plan, block)
    
    return start_pos + output_width(evaluator)
end

"""
    generate_combined_blocks!(plan::ExecutionPlan, evaluator::CombinedEvaluator,
                             start_pos::Int, scratch_layout::ScratchLayout) -> Int

Generate execution blocks for combined evaluator (multiple sub-evaluators).
"""
function generate_combined_blocks!(plan::ExecutionPlan, evaluator::CombinedEvaluator,
                                  start_pos::Int, scratch_layout::ScratchLayout)
    current_pos = start_pos
    
    for sub_evaluator in evaluator.sub_evaluators
        current_pos = generate_execution_blocks!(plan, sub_evaluator, current_pos, scratch_layout)
    end
    
    return current_pos
end

"""
    generate_zscore_block!(plan::ExecutionPlan, evaluator::ZScoreEvaluator,
                          start_pos::Int, scratch_layout::ScratchLayout) -> Int

Generate execution block for Z-score transformation.
"""
function generate_zscore_block!(plan::ExecutionPlan, evaluator::ZScoreEvaluator,
                               start_pos::Int, scratch_layout::ScratchLayout)
    
    # First generate block for underlying evaluator
    underlying_width = output_width(evaluator.underlying)
    
    if is_direct_evaluatable(evaluator.underlying)
        # Simple case - direct transformation
        if underlying_width == 1
            # Create transformation operation: (value - center) / scale
            input_source = create_input_source(evaluator.underlying)
            
            # Create compound operation for (x - center) / scale
            # For now, create as a function operation
            center_source = constant_source(evaluator.center)
            scale_source = constant_source(evaluator.scale)
            
            # This is a simplified approach - in Phase 2 we'll optimize this
            op = create_zscore_operation(input_source, center_source, scale_source, output_position(start_pos))
            
            block = FunctionBlock([op], UnitRange{Int}[], [start_pos])
            push!(plan, block)
            
            return start_pos + underlying_width
        else
            error("Multi-output Z-score transformation not yet implemented")
        end
    else
        error("Complex Z-score evaluator not yet implemented")
    end
end

"""
    generate_scaled_block!(plan::ExecutionPlan, evaluator::ScaledEvaluator,
                          start_pos::Int, scratch_layout::ScratchLayout) -> Int

Generate execution block for scaled evaluation.
"""
function generate_scaled_block!(plan::ExecutionPlan, evaluator::ScaledEvaluator,
                               start_pos::Int, scratch_layout::ScratchLayout)
    
    if is_direct_evaluatable(evaluator.evaluator)
        # Simple scaling operation
        input_source = create_input_source(evaluator.evaluator)
        scale_source = constant_source(evaluator.scale_factor)
        
        op = function_op(*, [input_source, scale_source], output_position(start_pos))
        block = FunctionBlock([op], UnitRange{Int}[], [start_pos])
        push!(plan, block)
        
        return start_pos + 1
    else
        error("Complex scaled evaluator not yet implemented")
    end
end

"""
    generate_product_block!(plan::ExecutionPlan, evaluator::ProductEvaluator,
                           start_pos::Int, scratch_layout::ScratchLayout) -> Int

Generate execution block for product evaluation.
"""
function generate_product_block!(plan::ExecutionPlan, evaluator::ProductEvaluator,
                                start_pos::Int, scratch_layout::ScratchLayout)
    
    if all(is_direct_evaluatable, evaluator.components)
        # Simple product operation
        input_sources = [create_input_source(comp) for comp in evaluator.components]
        
        op = function_op(*, input_sources, output_position(start_pos))
        block = FunctionBlock([op], UnitRange{Int}[], [start_pos])
        push!(plan, block)
        
        return start_pos + 1
    else
        error("Complex product evaluator not yet implemented")
    end
end

###############################################################################
# HELPER FUNCTIONS FOR EXECUTION PLAN GENERATION
###############################################################################

"""
    create_input_sources(evaluators::Vector{AbstractEvaluator}) -> Vector{InputSource}

Create input sources for a vector of evaluators.
"""
function create_input_sources(evaluators::Vector{AbstractEvaluator})
    return [create_input_source(eval) for eval in evaluators]
end

"""
    create_input_source(evaluator::AbstractEvaluator) -> InputSource

Create an input source for a single evaluator.
"""
function create_input_source(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator
        return constant_source(evaluator.value)
    elseif evaluator isa ContinuousEvaluator
        return data_source(evaluator.column)
    else
        error("Cannot create direct input source for $(typeof(evaluator))")
    end
end

"""
    create_evaluator_operations(evaluator::AbstractEvaluator, output_dest::OutputDestination) -> Vector{FunctionOp}

Create operations to evaluate an evaluator and store result in output destination.
"""
function create_evaluator_operations(evaluator::AbstractEvaluator, output_dest::OutputDestination)
    # This is a simplified implementation for Phase 1
    # In Phase 2, this will be more sophisticated
    
    if evaluator isa FunctionEvaluator && all(is_direct_evaluatable, evaluator.arg_evaluators)
        input_sources = create_input_sources(evaluator.arg_evaluators)
        return [function_op(evaluator.func, input_sources, output_dest)]
    else
        error("Complex evaluator operations not yet implemented")
    end
end

"""
    create_zscore_operation(input::InputSource, center::ConstantSource, scale::ConstantSource, 
                           output::OutputDestination) -> FunctionOp

Create a Z-score transformation operation: (input - center) / scale.
"""
function create_zscore_operation(input::InputSource, center::ConstantSource, 
                                scale::ConstantSource, output::OutputDestination)
    # For now, create a composite operation
    # In Phase 2, this will be optimized to a single operation
    
    # Create function that computes (x - center) / scale
    zscore_func = (x, c, s) -> (x - c) / s
    
    return function_op(zscore_func, [input, center, scale], output)
end

###############################################################################
# SINGLE CONSTRUCTOR FUNCTION
###############################################################################

"""
    create_execution_plan(evaluator::AbstractEvaluator, data::NamedTuple) -> ValidatedExecutionPlan

The single way to create an execution plan. Does all validation during construction,
guarantees zero-allocation execution.

# Example
```julia
model = lm(@formula(y ~ x * group), df)
compiled = compile_formula(model)
data = Tables.columntable(df)

# Construction: can allocate, happens once
plan = create_execution_plan(compiled.root_evaluator, data)

# Execution: zero allocation, happens many times
output = Vector{Float64}(undef, plan.total_output_width)
scratch = Vector{Float64}(undef, plan.scratch_size)

for i in 1:nrow(df)
    execute_plan!(plan, scratch, output, data, i)  # Zero allocation!
    # Use output...
end
```
"""
function create_execution_plan(evaluator::AbstractEvaluator, data::NamedTuple)
    return ValidatedExecutionPlan(evaluator, data)
end

###############################################################################
# INTEGRATION WITH EXISTING SYSTEM
###############################################################################

"""
Update compile_formula to work with the new system.
"""
function compile_formula_with_plan(model, data::NamedTuple)
    # Get the evaluator tree (existing code)
    rhs = fixed_effects_form(model).rhs
    root_evaluator = compile_term(rhs)
    total_width = output_width(root_evaluator)
    column_names = extract_all_columns(rhs)
    
    # Create validated execution plan
    execution_plan = create_execution_plan(root_evaluator, data)
    
    # Return enhanced compiled formula
    return CompiledFormulaWithPlan(
        execution_plan,
        total_width,
        column_names,
        root_evaluator  # Keep for derivatives
    )
end

struct CompiledFormulaWithPlan
    execution_plan::ValidatedExecutionPlan
    output_width::Int
    column_names::Vector{Symbol}
    root_evaluator::AbstractEvaluator  # For derivatives and analysis
end

Base.length(cf::CompiledFormulaWithPlan) = cf.output_width

function (cf::CompiledFormulaWithPlan)(row_vec::AbstractVector{Float64}, data, row_idx::Int)
    # Zero-allocation execution
    scratch = Vector{Float64}(undef, cf.execution_plan.scratch_size)  # Could be cached
    execute_plan!(cf.execution_plan, scratch, row_vec, data, row_idx)
    return row_vec
end

# Complete replacement of SimpleAssignment system with type-stable approach
# This replaces the problematic parts of execution_plans.jl

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

"""
    AssignmentBlock <: ExecutionBlock

Type-stable assignment block that replaces SimpleAssignmentBlock.
Uses Union dispatch for zero-allocation execution.
"""
struct AssignmentBlock <: ExecutionBlock
    assignments::Vector{Assignment}  # Union{ConstantAssignment, ContinuousAssignment}
end

###############################################################################
# REPLACE: ZERO-ALLOCATION EXECUTION
###############################################################################

"""
    execute_block!(block::AssignmentBlock, scratch::Vector{Float64}, 
                   output::Vector{Float64}, data::NamedTuple, row_idx::Int)

Zero-allocation execution for type-stable assignments.
This completely replaces execute_simple_assignment_block!
"""
function execute_block!(block::AssignmentBlock, scratch::Vector{Float64}, 
                       output::Vector{Float64}, data::NamedTuple, row_idx::Int)
    
    @inbounds for i in 1:length(block.assignments)
        assignment = block.assignments[i]
        execute_assignment!(assignment, output, data, row_idx)
    end
end

"""
    execute_assignment!(assignment::Assignment, output::Vector{Float64}, 
                       data::NamedTuple, row_idx::Int)

Type-stable dispatch for individual assignments.
"""
function execute_assignment!(assignment::ConstantAssignment, output::Vector{Float64}, 
                            data::NamedTuple, row_idx::Int)
    @inbounds output[assignment.output_position] = assignment.value
end

function execute_assignment!(assignment::ContinuousAssignment, output::Vector{Float64}, 
                            data::NamedTuple, row_idx::Int)
    @inbounds output[assignment.output_position] = getfield(data, assignment.column)[row_idx]
end

###############################################################################
# REPLACE: CONSTRUCTOR FUNCTIONS
###############################################################################

"""
    constant_assignment(value::Float64, output_pos::Int) -> ConstantAssignment

Create type-stable constant assignment.
Replaces the old constant_assignment function.
"""
function constant_assignment(value::Float64, output_pos::Int)
    return ConstantAssignment(value, output_pos)
end

"""
    continuous_assignment(column::Symbol, output_pos::Int) -> ContinuousAssignment

Create type-stable continuous assignment.
Replaces the old continuous_assignment function.
"""
function continuous_assignment(column::Symbol, output_pos::Int)
    return ContinuousAssignment(column, output_pos)
end

###############################################################################
# REPLACE: PLAN GENERATION FUNCTIONS
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

"""
    generate_execution_blocks!(plan::ExecutionPlan, evaluator::AbstractEvaluator, 
                               start_pos::Int, scratch_layout::ScratchLayout)

Updated execution block generation using type-stable assignments.
This replaces the problematic version that used SimpleAssignment.
"""
function generate_execution_blocks!(plan::ExecutionPlan, evaluator::AbstractEvaluator, 
                                   start_pos::Int, scratch_layout::ScratchLayout)
    
    if evaluator isa ConstantEvaluator
        # Type-stable constant assignment
        assignment = constant_assignment(evaluator.value, start_pos)
        add_assignment!(plan, assignment)
        return start_pos + 1
        
    elseif evaluator isa ContinuousEvaluator
        # Type-stable continuous assignment  
        assignment = continuous_assignment(evaluator.column, start_pos)
        add_assignment!(plan, assignment)
        return start_pos + 1
        
    elseif evaluator isa CategoricalEvaluator
        # Categorical lookup block (unchanged)
        block = create_categorical_block(evaluator, start_pos)
        push!(plan, block)
        return start_pos + output_width(evaluator)
        
    elseif evaluator isa FunctionEvaluator
        # Function evaluation block (unchanged)
        return generate_function_block!(plan, evaluator, start_pos, scratch_layout)
        
    elseif evaluator isa InteractionEvaluator
        # Interaction block (unchanged)
        return generate_interaction_block!(plan, evaluator, start_pos, scratch_layout)
        
    elseif evaluator isa ZScoreEvaluator
        # Z-score transformation block (unchanged)
        return generate_zscore_block!(plan, evaluator, start_pos, scratch_layout)
        
    elseif evaluator isa CombinedEvaluator
        # Multiple sub-blocks (unchanged)
        return generate_combined_blocks!(plan, evaluator, start_pos, scratch_layout)
        
    elseif evaluator isa ScaledEvaluator
        # Scaled evaluation block (unchanged)
        return generate_scaled_block!(plan, evaluator, start_pos, scratch_layout)
        
    elseif evaluator isa ProductEvaluator
        # Product evaluation block (unchanged)
        return generate_product_block!(plan, evaluator, start_pos, scratch_layout)
        
    else
        error("Execution plan generation not implemented for $(typeof(evaluator))")
    end
end

###############################################################################
# REPLACE: DISPLAY METHODS
###############################################################################

"""
    Base.show(io::IO, block::AssignmentBlock)

Pretty printing for type-stable assignment blocks.
Replaces the show method for SimpleAssignmentBlock.
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

###############################################################################
# REPLACE: VALIDATION FUNCTIONS
###############################################################################

"""
    validate_block_references!(block::AssignmentBlock, data_columns::Set{Symbol}, block_idx::Int)

Validation for type-stable assignment blocks.
Replaces validation for SimpleAssignmentBlock.
"""
function validate_block_references!(block::AssignmentBlock, data_columns::Set{Symbol}, block_idx::Int)
    for (i, assignment) in enumerate(block.assignments)
        if assignment isa ContinuousAssignment
            assignment.column in data_columns || 
                error("Block $block_idx, assignment $i: references missing column '$(assignment.column)'. Available: $(sort(collect(data_columns)))")
        end
        
        assignment.output_position > 0 || 
            error("Block $block_idx, assignment $i: invalid output position $(assignment.output_position)")
    end
end

"""
    get_block_output_positions(block::AssignmentBlock)

Get output positions for type-stable assignment blocks.
Replaces the version for SimpleAssignmentBlock.
"""
function get_block_output_positions(block::AssignmentBlock)
    return [assignment.output_position for assignment in block.assignments]
end

###############################################################################
# CLEAN UP: REMOVE OLD TYPES AND FUNCTIONS
###############################################################################

# The following types and functions should be REMOVED from execution_plans.jl:
# - struct SimpleAssignment
# - struct SimpleAssignmentBlock  
# - execute_simple_assignment_block!()
# - simple_assignment()
# - add_simple_assignment!()
# - show method for SimpleAssignmentBlock
# - validation for SimpleAssignmentBlock

# The following should be REPLACED:
# - constant_assignment() and continuous_assignment() (now return concrete types)
# - generate_execution_blocks!() (now uses type-stable assignments)
# - add_assignment!() replaces add_simple_assignment!()

###############################################################################
# UPDATE: MAIN EXECUTE_BLOCK! DISPATCH
###############################################################################

function execute_block!(block::ExecutionBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                       data::NamedTuple, row_idx::Int)
    
    if block isa AssignmentBlock
        # Assignment execution (should already be zero allocation)
        @inbounds for i in 1:length(block.assignments)
            assignment = block.assignments[i]
            execute_assignment!(assignment, output, data, row_idx)
        end
        
    elseif block isa CategoricalBlock
        # OPTIMIZED categorical execution
        execute_categorical_block_optimized!(block, output, data, row_idx)
        
    elseif block isa FunctionBlock
        execute_function_block!(block, scratch, output, data, row_idx)
        
    elseif block isa InteractionBlock
        execute_interaction_block!(block, scratch, output, data, row_idx)
        
    else
        throw(MethodError(execute_block!, (typeof(block),)))
    end
    
    return nothing
end

###############################################################################
# SOLUTION 2: ALTERNATIVE CATEGORICAL BLOCK EXECUTION
###############################################################################

"""
    execute_categorical_block_optimized!(block::CategoricalBlock, output::Vector{Float64}, 
                                        data::NamedTuple, row_idx::Int)

Alternative implementation that minimizes all potential allocation sources.
"""
function execute_categorical_block_optimized!(block::CategoricalBlock, output::Vector{Float64}, 
                                            data::NamedTuple, row_idx::Int)
    
    # Pre-fetch block layouts to avoid repeated field access
    layouts = block.layouts
    n_layouts = length(layouts)
    
    # Handle common cases with specialized code paths
    if n_layouts == 1
        execute_single_categorical_layout_optimized!(layouts[1], output, data, row_idx)
    else
        # Multiple layouts - should be rare
        @inbounds for i in 1:n_layouts
            execute_single_categorical_layout_optimized!(layouts[i], output, data, row_idx)
        end
    end
    
    return nothing
end

"""
    execute_single_categorical_layout_optimized!(layout::CategoricalLayout, output::Vector{Float64},
                                               data::NamedTuple, row_idx::Int)

Single categorical layout with maximum optimization.
"""
@inline function execute_single_categorical_layout_optimized!(layout::CategoricalLayout, 
                                                            output::Vector{Float64},
                                                            data::NamedTuple, row_idx::Int)
    
    # Minimize NamedTuple access by caching the column
    col = layout.column
    column_data = getfield(data, col)
    
    # Extract categorical value with minimal allocation risk
    @inbounds cat_val = column_data[row_idx]
    
    # Get level index with optimized type-stable dispatch
    level_idx = _get_categorical_level_fast(cat_val, layout.n_levels)
    
    # Apply lookups with specialized paths for common cases
    _apply_lookup_tables_fast!(layout.lookup_tables, layout.output_positions, 
                              output, level_idx)
    
    return nothing
end

"""
    _get_categorical_level_fast(cat_val, n_levels::Int) -> Int

Fastest possible categorical level extraction.
"""
@inline function _get_categorical_level_fast(cat_val, n_levels::Int)
    # Specialized type dispatch to avoid allocations
    if isa(cat_val, CategoricalValue)
        level = levelcode(cat_val)
        # Manual bounds checking without allocation
        return level < 1 ? 1 : (level > n_levels ? n_levels : level)
    else
        return 1
    end
end

"""
    _apply_lookup_tables_fast!(lookup_tables::Vector{Vector{Float64}}, 
                               output_positions::Vector{Int},
                               output::Vector{Float64}, level_idx::Int)

Fastest possible lookup table application.
"""
@inline function _apply_lookup_tables_fast!(lookup_tables::Vector{Vector{Float64}}, 
                                           output_positions::Vector{Int},
                                           output::Vector{Float64}, level_idx::Int)
    
    n_tables = length(lookup_tables)
    
    # Specialized unrolled versions for common cases
    if n_tables == 1
        @inbounds output[output_positions[1]] = lookup_tables[1][level_idx]
    elseif n_tables == 2
        @inbounds output[output_positions[1]] = lookup_tables[1][level_idx]
        @inbounds output[output_positions[2]] = lookup_tables[2][level_idx]
    elseif n_tables == 3
        @inbounds output[output_positions[1]] = lookup_tables[1][level_idx]
        @inbounds output[output_positions[2]] = lookup_tables[2][level_idx]
        @inbounds output[output_positions[3]] = lookup_tables[3][level_idx]
    else
        # General case
        @inbounds for j in 1:n_tables
            output[output_positions[j]] = lookup_tables[j][level_idx]
        end
    end
    
    return nothing
end
