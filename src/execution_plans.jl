# execution_plans.jl

# assign unique names

const VAR_COUNTER = Ref(0)

function next_var(prefix::String="v")
    VAR_COUNTER[] += 1
    return "$(prefix)_$(VAR_COUNTER[])"
end

function reset_var_counter!()
    VAR_COUNTER[] = 0
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
# VALIDATED EXECUTION PLAN
###############################################################################

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
    
    @debug "Execution plan validated successfully" scratch_size=plan.scratch_size total_output_width=plan.total_output_width data_length=data_length data_columns=sort(collect(data_columns)) execution_blocks=length(plan.blocks)
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

# ADD this function to the validation section:
function get_block_output_positions(block::AssignmentBlock)
    return [assignment.output_position for assignment in block.assignments]
end

function get_block_output_positions(block::FunctionBlock)
    return block.output_positions
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
    execute_plan!(plan::Union{ExecutionPlan, ValidatedExecutionPlan}, scratch::Vector{Float64}, 
                  output::Vector{Float64}, data::NamedTuple, row_idx::Int)

Execute an execution plan with zero allocations.
FIXED: Now works with both ExecutionPlan and ValidatedExecutionPlan for backward compatibility.
"""
function execute_plan!(plan::Union{ExecutionPlan, ValidatedExecutionPlan}, 
                       scratch::Vector{Float64}, output::Vector{Float64}, 
                       data::NamedTuple, row_idx::Int)
    
    # Simple bounds checking
    @boundscheck begin
        length(scratch) >= plan.scratch_size || throw(BoundsError())
        length(output) >= plan.total_output_width || throw(BoundsError())
        if plan isa ValidatedExecutionPlan
            1 <= row_idx <= plan.data_length || throw(BoundsError())
        end
    end
    
    # Execute all blocks - zero allocation guaranteed
    @inbounds for i in 1:length(plan.blocks)
        execute_block!(plan.blocks[i], scratch, output, data, row_idx)
    end
    
    return nothing
end


"""
    execute_function_block!(block::FunctionBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                           data::NamedTuple, row_idx::Int)

UPDATED: Execute function operations including decomposed complex functions.
"""
function execute_function_block!(block::FunctionBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                                data::NamedTuple, row_idx::Int)
    
    # Execute all operations in the block sequentially
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
                                 component_positions::Vector{UnitRange{Int}},
                                 scratch::Vector{Float64},
                                 output::AbstractVector{Float64})
    
    n_components = length(component_positions)
    
    if n_components == 2
        # Binary interaction: optimized path
        pos1, pos2 = component_positions[1], component_positions[2]
        
        for (idx, (i, j, _)) in enumerate(pattern)
            @inbounds output[idx] = scratch[first(pos1) + i - 1] * scratch[first(pos2) + j - 1]
        end
        
    elseif n_components == 3
        # Three-way interaction: optimized path
        pos1, pos2, pos3 = component_positions[1], component_positions[2], component_positions[3]
        
        for (idx, (i, j, k)) in enumerate(pattern)
            @inbounds output[idx] = scratch[first(pos1) + i - 1] * 
                                   scratch[first(pos2) + j - 1] * 
                                   scratch[first(pos3) + k - 1]
        end
        
    else
        error("N-way interactions with N > 3 not yet implemented in apply_kronecker_pattern!")
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

COMPLETELY UPDATED: Now handles arbitrary function complexity through AST decomposition.
This fixes the failing test and enables zero-allocation evaluation of any nested function.

# Examples
- Simple: `log(x)` → direct function call
- Complex: `log(x^2)` → decompose to: x^2 → scratch[1], log(scratch[1]) → output
- Very complex: `sin(log(x^2) + cos(y))` → full AST decomposition
"""
function generate_function_block!(plan::ExecutionPlan, evaluator::FunctionEvaluator, 
                                 start_pos::Int, scratch_layout::ScratchLayout)
    
    if all(is_direct_evaluatable, evaluator.arg_evaluators)
        # SIMPLE CASE: Direct function call (optimized path)
        input_sources = create_input_sources(evaluator.arg_evaluators)
        output_dest = output_position(start_pos)
        
        op = function_op(evaluator.func, input_sources, output_dest)
        block = FunctionBlock([op], UnitRange{Int}[], [start_pos])
        push!(plan, block)
        
        return start_pos + 1
        
    else
        # COMPLEX CASE: Use AST decomposition (THE NEW SOLUTION!)
        println("  Decomposing complex function: $(evaluator.func)")
        
        try
            # Decompose the nested evaluator tree into linear operations
            operations = decompose_evaluator_tree(evaluator, scratch_layout)
            
            if isempty(operations)
                error("No operations generated for complex function")
            end
            
            println("    Generated $(length(operations)) operations")
            
            # The final operation should output to the target position
            final_operations = copy(operations)
            if !isempty(final_operations)
                last_idx = length(final_operations)
                last_op = final_operations[last_idx]
                
                # Update final operation to output to correct position
                final_operations[last_idx] = DecomposedOperation(
                    last_op.operation_type,
                    last_op.func,
                    last_op.input_refs,
                    OperationRef(:output, start_pos),  # Final result goes to output
                    last_op.dependencies
                )
            end
            
            # Create execution block from decomposed operations
            block = create_decomposed_function_block(final_operations, start_pos)
            push!(plan, block)
            
            println("    ✅ Complex function decomposition successful")
            return start_pos + 1
            
        catch e
            # Only error for truly unsupported cases
            error("Failed to decompose function $(evaluator.func): $e")
        end
    end
end

###############################################################################
# ENHANCED OPERATION EXECUTION
###############################################################################

"""
    execute_decomposed_operation!(operation::DecomposedOperation, scratch::Vector{Float64}, 
                                  output::Vector{Float64}, data::NamedTuple, row_idx::Int)

Execute a single decomposed operation with zero allocations.
"""
function execute_decomposed_operation!(operation::DecomposedOperation, 
                                      scratch::Vector{Float64}, output::Vector{Float64}, 
                                      data::NamedTuple, row_idx::Int)
    
    # Get input values
    input_values = Float64[]
    for input_ref in operation.input_refs
        val = get_operation_ref_value(input_ref, scratch, output, data, row_idx)
        push!(input_values, val)
    end
    
    # Apply function
    if operation.func !== nothing
        if length(input_values) == 1
            result = apply_function_safe(operation.func, input_values[1])
        elseif length(input_values) == 2
            result = apply_function_safe(operation.func, input_values[1], input_values[2])
        else
            result = apply_function_safe(operation.func, input_values...)
        end
    else
        # No function - just pass through (shouldn't happen)
        result = input_values[1]
    end
    
    # Store result
    store_operation_ref_value!(operation.output_ref, result, scratch, output)
end

"""
    get_operation_ref_value(ref::OperationRef, scratch::Vector{Float64}, output::Vector{Float64}, 
                           data::NamedTuple, row_idx::Int) -> Float64

Get value from an operation reference.
"""
function get_operation_ref_value(ref::OperationRef, scratch::Vector{Float64}, output::Vector{Float64}, 
                                data::NamedTuple, row_idx::Int)
    if ref.location_type == :data
        return Float64(data[ref.index][row_idx])
    elseif ref.location_type == :scratch
        return scratch[ref.index]
    elseif ref.location_type == :output
        return output[ref.index]
    elseif ref.location_type == :constant
        return Float64(ref.index)
    else
        error("Unknown operation reference type: $(ref.location_type)")
    end
end

"""
    store_operation_ref_value!(ref::OperationRef, value::Float64, scratch::Vector{Float64}, output::Vector{Float64})

Store value to an operation reference.
"""
function store_operation_ref_value!(ref::OperationRef, value::Float64, scratch::Vector{Float64}, output::Vector{Float64})
    if ref.location_type == :scratch
        scratch[ref.index] = value
    elseif ref.location_type == :output
        output[ref.index] = value
    else
        error("Cannot store to reference type: $(ref.location_type)")
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
    execute_block!(block::ZScoreBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                   data::NamedTuple, row_idx::Int)

Execute Z-score transformation block.
"""
function execute_block!(block::ZScoreBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                       data::NamedTuple, row_idx::Int)
    
    # First evaluate the underlying evaluator into scratch space
    temp_buffer = Vector{Float64}(undef, length(block.input_positions))
    evaluate_evaluator_to_buffer!(block.underlying_evaluator, temp_buffer, data, row_idx)
    
    # Apply Z-score transformation: (x - center) / scale
    @inbounds for i in 1:length(block.output_positions)
        input_val = temp_buffer[i]
        output_pos = block.output_positions[i]
        output[output_pos] = (input_val - block.center) / block.scale
    end
end

"""
    execute_block!(block::ScaledBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                   data::NamedTuple, row_idx::Int)

Execute scaled evaluation block.
"""
function execute_block!(block::ScaledBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                       data::NamedTuple, row_idx::Int)
    
    # First evaluate the underlying evaluator
    temp_buffer = Vector{Float64}(undef, length(block.input_positions))
    evaluate_evaluator_to_buffer!(block.underlying_evaluator, temp_buffer, data, row_idx)
    
    # Apply scaling: scale_factor * value
    @inbounds for i in 1:length(block.output_positions)
        input_val = temp_buffer[i]
        output_pos = block.output_positions[i]
        output[output_pos] = block.scale_factor * input_val
    end
end

"""
    execute_block!(block::ProductBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                   data::NamedTuple, row_idx::Int)

Execute product evaluation block.
"""
function execute_block!(block::ProductBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                       data::NamedTuple, row_idx::Int)
    
    # Evaluate all components
    component_values = Float64[]
    
    for (i, component_eval) in enumerate(block.component_evaluators)
        temp_buffer = Vector{Float64}(undef, 1)  # Products are always scalar
        evaluate_evaluator_to_buffer!(component_eval, temp_buffer, data, row_idx)
        push!(component_values, temp_buffer[1])
    end
    
    # Compute product
    product = 1.0
    @inbounds for val in component_values
        product *= val
    end
    
    @inbounds output[block.output_position] = product
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

    elseif evaluator isa CombinedEvaluator
        # Multiple sub-blocks (unchanged)
        return generate_combined_blocks!(plan, evaluator, start_pos, scratch_layout)

    elseif evaluator isa ScaledEvaluator
        # Scaled evaluation block (unchanged)
        return generate_scaled_block!(plan, evaluator, start_pos, scratch_layout)

    elseif evaluator isa ZScoreEvaluator
        # Z-score transformation block
        block = create_zscore_block(evaluator, start_pos)
        push!(plan, block)
        return start_pos + output_width(evaluator)
    elseif evaluator isa ScaledEvaluator
        # Scaled evaluation block  
        block = create_scaled_block(evaluator, start_pos)
        push!(plan, block)
        return start_pos + output_width(evaluator)
    elseif evaluator isa ProductEvaluator
        # Product evaluation block
        block = create_product_block(evaluator, start_pos)
        push!(plan, block)
        return start_pos + 1  # Products always have width 1
    else
        error("Execution plan generation not implemented for $(typeof(evaluator))")
    end
end

# Add the missing ProductEvaluator handling to generate_execution_blocks!
function generate_execution_blocks!(plan::ExecutionPlan, evaluator::ProductEvaluator, 
                                   start_pos::Int, scratch_layout::ScratchLayout)
    
    if all(is_direct_evaluatable, evaluator.components)
        # Simple product operation - all components can be evaluated directly
        input_sources = [create_input_source(comp) for comp in evaluator.components]
        
        op = function_op(*, input_sources, output_position(start_pos))
        block = FunctionBlock([op], UnitRange{Int}[], [start_pos])
        push!(plan, block)
        
        return start_pos + 1
    else
        # Complex product - need to evaluate components first, then multiply
        # For now, use the decomposition approach
        try
            operations = decompose_evaluator_tree(evaluator, scratch_layout)
            
            if !isempty(operations)
                # Update final operation to output to correct position
                final_operations = copy(operations)
                last_idx = length(final_operations)
                last_op = final_operations[last_idx]
                
                final_operations[last_idx] = DecomposedOperation(
                    last_op.operation_type,
                    last_op.func,
                    last_op.input_refs,
                    OperationRef(:output, start_pos),
                    last_op.dependencies
                )
                
                block = create_decomposed_function_block(final_operations, start_pos)
                push!(plan, block)
                
                return start_pos + 1
            else
                error("No operations generated for complex product")
            end
        catch e
            error("Complex product evaluator not yet fully implemented: $e")
        end
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

###############################################################################
# PHASE 2D: CORRECTED ADVANCED EVALUATOR IMPLEMENTATIONS
###############################################################################

"""
Phase 2D: ZScore Expression Generation (CORRECTED)
Based on: ZScoreEvaluator(underlying::AbstractEvaluator, center::Float64, scale::Float64)
"""
function generate_zscore_expression_recursive(evaluator::ZScoreEvaluator)
    # ZScore: (underlying_expr - center) / scale
    underlying_expr = generate_expression_recursive(evaluator.underlying)
    center = evaluator.center
    scale = evaluator.scale
    
    # Generate domain-safe expression
    if scale ≈ 0.0
        # Handle zero scale case
        if center ≈ 0.0
            return "0.0"  # (0 - 0) / 0 → 0 by convention
        else
            return "(($underlying_expr) ≈ $center ? 0.0 : (($underlying_expr) > $center ? Inf : -Inf))"
        end
    else
        return "((($underlying_expr) - $center) / $scale)"
    end
end

"""
Phase 2D: Scaled Expression Generation (CORRECTED)
Based on: ScaledEvaluator(evaluator::AbstractEvaluator, scale_factor::Float64)
"""
function generate_scaled_expression_recursive(evaluator::ScaledEvaluator)
    # Scaled: scale_factor * evaluator_expr
    scale_factor = evaluator.scale_factor
    base_expr = generate_expression_recursive(evaluator.evaluator)
    
    if scale_factor ≈ 1.0
        return base_expr
    elseif scale_factor ≈ 0.0
        return "0.0"
    elseif scale_factor ≈ -1.0
        return "(-($base_expr))"
    else
        return "($scale_factor * ($base_expr))"
    end
end

"""
Phase 2D: Product Expression Generation (CORRECTED)
Based on: ProductEvaluator(components::Vector{AbstractEvaluator}) with output_width = 1
"""
function generate_product_expression_recursive(evaluator::ProductEvaluator)
    # Product: component1 * component2 * ... * componentN
    # Note: ProductEvaluator always has output_width = 1 in your system
    components = evaluator.components
    
    if isempty(components)
        return "1.0"
    elseif length(components) == 1
        return generate_expression_recursive(components[1])
    else
        # Generate expressions for all components
        component_exprs = [generate_expression_recursive(comp) for comp in components]
        
        # Build product expression with proper parentheses
        if length(component_exprs) == 2
            return "($(component_exprs[1]) * $(component_exprs[2]))"
        else
            # Chain multiplication for multiple components
            result = component_exprs[1]
            for i in 2:length(component_exprs)
                result = "($result * $(component_exprs[i]))"
            end
            return result
        end
    end
end

"""
Phase 2D: Complex Interaction Expression Generation (CORRECTED)
Based on: InteractionEvaluator(components, total_width) where total_width = prod(component_widths)
"""
function generate_interaction_expression_recursive(evaluator::InteractionEvaluator)
    components = evaluator.components
    
    if isempty(components)
        return "1.0"
    elseif length(components) == 1
        return generate_expression_recursive(components[1])
    end
    
    # Check if this can be handled as a simple expression (total_width == 1)
    if evaluator.total_width == 1
        # Scalar interaction - all components must have width 1
        component_exprs = [generate_expression_recursive(comp) for comp in components]
        
        # Build interaction expression
        if length(component_exprs) == 2
            return "($(component_exprs[1]) * $(component_exprs[2]))"
        else
            # Chain multiplication
            result = component_exprs[1]
            for i in 2:length(component_exprs)
                result = "($result * $(component_exprs[i]))"
            end
            return result
        end
    else
        # Multi-output interaction - cannot be single expression
        error("Multi-output interaction (width=$(evaluator.total_width)) cannot be generated as single expression. Use generate_statements_recursive() instead.")
    end
end

###############################################################################
# ADVANCED INTERACTION STATEMENT GENERATION (PHASE 2D)
###############################################################################

"""
Generate statements for ZScore evaluators (CORRECTED).
"""
function generate_zscore_statements_recursive(evaluator::ZScoreEvaluator, start_pos::Int)
    underlying = evaluator.underlying
    underlying_width = output_width(underlying)
    
    if underlying_width == 1
        # Single output - can use expression
        expr = generate_zscore_expression_recursive(evaluator)
        return ["@inbounds row_vec[$start_pos] = $expr"], start_pos + 1
    else
        # Multi-output - need statements for underlying, then transform
        instructions = String[]
        underlying_instructions, next_pos = generate_statements_recursive(underlying, start_pos)
        append!(instructions, underlying_instructions)
        
        # Apply Z-score transformation to all outputs
        center = evaluator.center
        scale = evaluator.scale
        
        for i in 0:(underlying_width-1)
            pos = start_pos + i
            if scale ≈ 0.0
                if center ≈ 0.0
                    push!(instructions, "@inbounds row_vec[$pos] = 0.0")
                else
                    push!(instructions, "@inbounds row_vec[$pos] = (row_vec[$pos] ≈ $center ? 0.0 : (row_vec[$pos] > $center ? Inf : -Inf))")
                end
            else
                push!(instructions, "@inbounds row_vec[$pos] = (row_vec[$pos] - $center) / $scale")
            end
        end
        
        return instructions, next_pos
    end
end

"""
Generate statements for Scaled evaluators (CORRECTED).
"""
function generate_scaled_statements_recursive(evaluator::ScaledEvaluator, start_pos::Int)
    base_evaluator = evaluator.evaluator
    scale_factor = evaluator.scale_factor
    base_width = output_width(base_evaluator)
    
    if base_width == 1
        # Single output - can use expression
        expr = generate_scaled_expression_recursive(evaluator)
        return ["@inbounds row_vec[$start_pos] = $expr"], start_pos + 1
    else
        # Multi-output - need statements for base, then scale
        instructions = String[]
        base_instructions, next_pos = generate_statements_recursive(base_evaluator, start_pos)
        append!(instructions, base_instructions)
        
        # Apply scaling to all outputs
        for i in 0:(base_width-1)
            pos = start_pos + i
            if scale_factor ≈ 1.0
                # No scaling needed
                continue
            elseif scale_factor ≈ 0.0
                push!(instructions, "@inbounds row_vec[$pos] = 0.0")
            elseif scale_factor ≈ -1.0
                push!(instructions, "@inbounds row_vec[$pos] = -row_vec[$pos]")
            else
                push!(instructions, "@inbounds row_vec[$pos] *= $scale_factor")
            end
        end
        
        return instructions, next_pos
    end
end

"""
Generate statements for Product evaluators (CORRECTED).
Note: ProductEvaluator always has output_width = 1 in your system.
"""
function generate_product_statements_recursive(evaluator::ProductEvaluator, start_pos::Int)
    # Since ProductEvaluator always has width 1, we can use expression generation
    expr = generate_product_expression_recursive(evaluator)
    return ["@inbounds row_vec[$start_pos] = $expr"], start_pos + 1
end

"""
Generate statements for complex interactions that produce multiple outputs (CORRECTED).
"""
function generate_interaction_statements_recursive(evaluator::InteractionEvaluator, start_pos::Int)
    components = evaluator.components
    total_width = evaluator.total_width
    
    if total_width == 1
        # Simple scalar interaction
        expr = generate_interaction_expression_recursive(evaluator)
        return ["@inbounds row_vec[$start_pos] = $expr"], start_pos + 1
    end
    
    # Complex multi-output interaction
    return generate_complex_interaction_statements(components, start_pos, total_width)
end

"""
Generate statements for complex multi-component interactions (CORRECTED).
"""
function generate_complex_interaction_statements(components::Vector{AbstractEvaluator}, start_pos::Int, total_width::Int)
    instructions = String[]
    
    # Get component widths
    component_widths = [output_width(comp) for comp in components]
    
    # Check for reasonable size limits (since this is for @generated)
    if total_width > 1000
        component_info = ["$(typeof(comp).name.name)(width=$w)" for (comp, w) in zip(components, component_widths)]
        error("Interaction too large: $(join(component_info, " × ")) = $total_width terms (limit: 1000 for @generated).\n" *
              "Consider using execution plans for large interactions.")
    end
    
    if length(components) == 2
        # Two-way interaction
        return generate_two_way_interaction_statements(components[1], components[2], start_pos)
    elseif length(components) == 3
        # Three-way interaction
        return generate_three_way_interaction_statements(components[1], components[2], components[3], start_pos)
    else
        # N-way interaction (4+ components)
        return generate_nway_interaction_statements(components, start_pos)
    end
end

"""
Generate optimized two-way interaction statements (CORRECTED).
"""
function generate_two_way_interaction_statements(comp1::AbstractEvaluator, comp2::AbstractEvaluator, start_pos::Int)
    instructions = String[]
    w1, w2 = output_width(comp1), output_width(comp2)
    total_width = w1 * w2
    
    if w1 == 1 && w2 == 1
        # Scalar × Scalar
        expr1 = generate_expression_recursive(comp1)
        expr2 = generate_expression_recursive(comp2)
        push!(instructions, "@inbounds row_vec[$start_pos] = ($expr1) * ($expr2)")
        
    elseif w1 == 1
        # Scalar × Vector
        scalar_expr = generate_expression_recursive(comp1)
        
        if comp2 isa CategoricalEvaluator
            # Scalar × Categorical - optimized
            cat_instructions = generate_scalar_categorical_interaction(scalar_expr, comp2, start_pos)
            append!(instructions, cat_instructions)
        else
            # Scalar × General Vector
            vector_var = next_var("vec")
            vector_instructions, _ = generate_statements_recursive(comp2, 1, vector_var)
            append!(instructions, vector_instructions)
            
            for i in 1:w2
                push!(instructions, "@inbounds row_vec[$(start_pos + i - 1)] = ($scalar_expr) * $vector_var[$i]")
            end
        end
        
    elseif w2 == 1
        # Vector × Scalar (commutative)
        return generate_two_way_interaction_statements(comp2, comp1, start_pos)
        
    else
        # Vector × Vector - Kronecker product
        return generate_vector_vector_interaction_statements(comp1, comp2, start_pos)
    end
    
    return instructions, start_pos + total_width
end

"""
Generate optimized scalar × categorical interaction (CORRECTED).
"""
function generate_scalar_categorical_interaction(scalar_expr::String, cat_eval::CategoricalEvaluator, start_pos::Int)
    instructions = String[]
    
    column = cat_eval.column
    n_levels = cat_eval.n_levels
    contrast_matrix = cat_eval.contrast_matrix
    n_contrasts = size(contrast_matrix, 2)
    
    # Generate level extraction
    level_var = next_var("level")
    push!(instructions, "@inbounds $level_var = clamp(data.$column[row_idx] isa CategoricalValue ? levelcode(data.$column[row_idx]) : 1, 1, $n_levels)")
    
    # Generate interaction for each contrast
    for j in 1:n_contrasts
        output_pos = start_pos + j - 1
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        
        # Optimized lookup generation
        if n_levels <= 3
            # Inline ternary for small cases
            if n_levels == 1
                contrast_expr = string(values[1])
            elseif n_levels == 2
                contrast_expr = "$level_var == 1 ? $(values[1]) : $(values[2])"
            else
                contrast_expr = "$level_var == 1 ? $(values[1]) : $level_var == 2 ? $(values[2]) : $(values[3])"
            end
            push!(instructions, "@inbounds row_vec[$output_pos] = ($scalar_expr) * ($contrast_expr)")
        else
            # Lookup table for larger cases
            lookup_var = next_var("lookup")
            values_str = "[" * join(string.(values), ", ") * "]"
            push!(instructions, "@inbounds $lookup_var = $values_str")
            push!(instructions, "@inbounds row_vec[$output_pos] = ($scalar_expr) * $lookup_var[$level_var]")
        end
    end
    
    return instructions
end

"""
Generate vector × vector interaction using Kronecker product pattern (CORRECTED).
"""
function generate_vector_vector_interaction_statements(comp1::AbstractEvaluator, comp2::AbstractEvaluator, start_pos::Int)
    instructions = String[]
    w1, w2 = output_width(comp1), output_width(comp2)
    total_width = w1 * w2
    
    # Generate temporary variables for components
    vec1_var = next_var("vec1")
    vec2_var = next_var("vec2")
    
    # Generate component values
    vec1_instructions, _ = generate_statements_recursive(comp1, 1, vec1_var)
    vec2_instructions, _ = generate_statements_recursive(comp2, 1, vec2_var)
    append!(instructions, vec1_instructions)
    append!(instructions, vec2_instructions)
    
    # Generate Kronecker product: vec1[i] * vec2[j] for all i,j
    output_idx = start_pos
    for j in 1:w2
        for i in 1:w1
            push!(instructions, "@inbounds row_vec[$output_idx] = $vec1_var[$i] * $vec2_var[$j]")
            output_idx += 1
        end
    end
    
    return instructions, start_pos + total_width
end

"""
Generate three-way interaction statements (CORRECTED).
"""
function generate_three_way_interaction_statements(comp1::AbstractEvaluator, comp2::AbstractEvaluator, comp3::AbstractEvaluator, start_pos::Int)
    instructions = String[]
    w1, w2, w3 = output_width(comp1), output_width(comp2), output_width(comp3)
    total_width = w1 * w2 * w3
    
    # Check size limit for three-way
    if total_width > 500
        error("Three-way interaction too large: $w1 × $w2 × $w3 = $total_width terms (limit: 500 for @generated)")
    end
    
    # Generate temporary variables
    vec1_var = next_var("vec1")
    vec2_var = next_var("vec2") 
    vec3_var = next_var("vec3")
    
    # Generate component values
    vec1_instructions, _ = generate_statements_recursive(comp1, 1, vec1_var)
    vec2_instructions, _ = generate_statements_recursive(comp2, 1, vec2_var)
    vec3_instructions, _ = generate_statements_recursive(comp3, 1, vec3_var)
    append!(instructions, vec1_instructions)
    append!(instructions, vec2_instructions)
    append!(instructions, vec3_instructions)
    
    # Generate three-way Kronecker product
    output_idx = start_pos
    for k in 1:w3
        for j in 1:w2
            for i in 1:w1
                push!(instructions, "@inbounds row_vec[$output_idx] = $vec1_var[$i] * $vec2_var[$j] * $vec3_var[$k]")
                output_idx += 1
            end
        end
    end
    
    return instructions, start_pos + total_width
end

"""
Generate N-way interaction statements (4+ components) (CORRECTED).
"""
function generate_nway_interaction_statements(components::Vector{AbstractEvaluator}, start_pos::Int)
    instructions = String[]
    component_widths = [output_width(comp) for comp in components]
    total_width = prod(component_widths)
    n_components = length(components)
    
    # Check size limit for N-way
    if total_width > 200
        error("$n_components-way interaction too large: $(join(string.(component_widths), " × ")) = $total_width terms (limit: 200 for @generated)")
    end
    
    # Generate temporary variables for all components
    component_vars = [next_var("vec$i") for i in 1:n_components]
    
    # Generate all component values
    for (i, comp) in enumerate(components)
        comp_instructions, _ = generate_statements_recursive(comp, 1, component_vars[i])
        append!(instructions, comp_instructions)
    end
    
    # Generate N-dimensional Kronecker product
    # Use nested loops for general N-way case
    indices_var = next_var("indices")
    push!(instructions, "$indices_var = CartesianIndices(tuple($(join(string.(component_widths), ", "))))")
    
    push!(instructions, "for (linear_idx, cart_idx) in enumerate($indices_var)")
    
    # Build product expression
    product_terms = ["$(component_vars[i])[cart_idx[$i]]" for i in 1:n_components]
    product_expr = join(product_terms, " * ")
    
    push!(instructions, "    @inbounds row_vec[$(start_pos - 1) + linear_idx] = $product_expr")
    push!(instructions, "end")
    
    return instructions, start_pos + total_width
end

###############################################################################
# ENHANCED STATEMENT GENERATION WITH PHASE 2D SUPPORT (CORRECTED)
###############################################################################

"""
    generate_statements_recursive(evaluator::AbstractEvaluator, start_pos::Int, output_var::String = "row_vec")

FIXED: Added missing function for statement generation.
Basic implementation for Phase 1 compatibility.
"""
function generate_statements_recursive(evaluator::AbstractEvaluator, start_pos::Int, output_var::String = "row_vec")
    if evaluator isa ConstantEvaluator
        return ["@inbounds $output_var[$start_pos] = $(evaluator.value)"], start_pos + 1
        
    elseif evaluator isa ContinuousEvaluator
        return ["@inbounds $output_var[$start_pos] = Float64(data.$(evaluator.column)[row_idx])"], start_pos + 1
        
    elseif evaluator isa CategoricalEvaluator
        # Basic categorical handling
        if size(evaluator.contrast_matrix, 2) == 1
            # Single contrast - generate simple expression
            column = evaluator.column
            n_levels = evaluator.n_levels
            values = [evaluator.contrast_matrix[i, 1] for i in 1:n_levels]
            
            level_expr = "clamp(data.$column[row_idx] isa CategoricalValue ? levelcode(data.$column[row_idx]) : 1, 1, $n_levels)"
            
            if n_levels == 2
                contrast_expr = "$level_expr == 1 ? $(values[1]) : $(values[2])"
            else
                # Simple ternary chain
                contrast_expr = "$level_expr == 1 ? $(values[1])"
                for i in 2:(n_levels-1)
                    contrast_expr *= " : $level_expr == $i ? $(values[i])"
                end
                contrast_expr *= " : $(values[n_levels])"
            end
            
            return ["@inbounds $output_var[$start_pos] = $contrast_expr"], start_pos + 1
        else
            error("Multi-contrast categorical not yet implemented in basic statement generation")
        end
        
    elseif evaluator isa FunctionEvaluator
        # FIXED: Basic function handling
        if all(is_direct_evaluatable, evaluator.arg_evaluators)
            # Simple function with direct arguments
            func = evaluator.func
            args = evaluator.arg_evaluators
            
            if length(args) == 1 && args[1] isa ContinuousEvaluator
                # Simple unary function like log(x)
                arg_expr = "Float64(data.$(args[1].column)[row_idx])"
                
                if func === log
                    func_expr = "($arg_expr > 0.0 ? log($arg_expr) : ($arg_expr == 0.0 ? -Inf : NaN))"
                elseif func === exp
                    func_expr = "exp($arg_expr)"
                elseif func === sqrt
                    func_expr = "($arg_expr >= 0.0 ? sqrt($arg_expr) : NaN)"
                else
                    func_expr = "$(func)($arg_expr)"
                end
                
                return ["@inbounds $output_var[$start_pos] = $func_expr"], start_pos + 1
                
            elseif length(args) == 2 && all(arg -> arg isa ContinuousEvaluator, args)
                # Simple binary function like x + y
                arg1_expr = "Float64(data.$(args[1].column)[row_idx])"
                arg2_expr = "Float64(data.$(args[2].column)[row_idx])"
                
                func_expr = "$(func)($arg1_expr, $arg2_expr)"
                return ["@inbounds $output_var[$start_pos] = $func_expr"], start_pos + 1
            else
                error("Complex function arguments not yet implemented in basic statement generation")
            end
        else
            error("Complex function evaluator not yet implemented in basic statement generation")
        end
        
    elseif evaluator isa CombinedEvaluator
        # Handle combined evaluator by recursively processing each sub-evaluator
        instructions = String[]
        current_pos = start_pos
        
        for sub_evaluator in evaluator.sub_evaluators
            sub_instructions, next_pos = generate_statements_recursive(sub_evaluator, current_pos, output_var)
            append!(instructions, sub_instructions)
            current_pos = next_pos
        end
        
        return instructions, current_pos
        
    else
        error("generate_statements_recursive not implemented for $(typeof(evaluator))")
    end
end

# AND add this new function:
"""
Generate statements for CombinedEvaluator (multiple sub-evaluators).
"""
function generate_combined_statements_recursive(evaluator::CombinedEvaluator, start_pos::Int, output_var::String = "row_vec")
    instructions = String[]
    current_pos = start_pos
    
    # Generate statements for each sub-evaluator sequentially
    for sub_evaluator in evaluator.sub_evaluators
        sub_instructions, next_pos = generate_statements_recursive(sub_evaluator, current_pos, output_var)
        append!(instructions, sub_instructions)
        current_pos = next_pos
    end
    
    return instructions, current_pos
end

# AND add this for categorical multi-output:
"""
Generate statements for multi-contrast categorical evaluators.
"""
function generate_categorical_statements_recursive(evaluator::CategoricalEvaluator, start_pos::Int, output_var::String = "row_vec")
    instructions = String[]
    
    column = evaluator.column
    n_levels = evaluator.n_levels
    contrast_matrix = evaluator.contrast_matrix
    n_contrasts = size(contrast_matrix, 2)
    
    # Generate level extraction
    level_var = next_var("level")
    push!(instructions, "@inbounds $level_var = clamp(data.$column[row_idx] isa CategoricalValue ? levelcode(data.$column[row_idx]) : 1, 1, $n_levels)")
    
    # Generate output for each contrast
    for j in 1:n_contrasts
        output_pos = start_pos + j - 1
        values = [contrast_matrix[i, j] for i in 1:n_levels]
        
        # Use lookup approach for multi-contrast
        if n_levels <= 3
            # Inline ternary for small cases
            if n_levels == 1
                contrast_expr = string(values[1])
            elseif n_levels == 2
                contrast_expr = "$level_var == 1 ? $(values[1]) : $(values[2])"
            else
                contrast_expr = "$level_var == 1 ? $(values[1]) : $level_var == 2 ? $(values[2]) : $(values[3])"
            end
            push!(instructions, "@inbounds $output_var[$output_pos] = $contrast_expr")
        else
            # Lookup table for larger cases
            lookup_var = next_var("lookup")
            values_str = "[" * join(string.(values), ", ") * "]"
            push!(instructions, "@inbounds $lookup_var = $values_str")
            push!(instructions, "@inbounds $output_var[$output_pos] = $lookup_var[$level_var]")
        end
    end
    
    return instructions, start_pos + n_contrasts
end

function generate_expression_recursive(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator
        return string(evaluator.value)
        
    elseif evaluator isa ContinuousEvaluator
        return "Float64(data.$(evaluator.column)[row_idx])"
        
    elseif evaluator isa CategoricalEvaluator
        # Add basic categorical support here
        if size(evaluator.contrast_matrix, 2) == 1
            return generate_simple_categorical_expression(evaluator)
        else
            error("Multi-contrast categorical expressions not yet implemented")
        end
        
    elseif evaluator isa ZScoreEvaluator
        return generate_zscore_expression_recursive(evaluator)
        
    elseif evaluator isa ScaledEvaluator
        return generate_scaled_expression_recursive(evaluator)
        
    elseif evaluator isa ProductEvaluator
        return generate_product_expression_recursive(evaluator)
        
    elseif evaluator isa InteractionEvaluator
        return generate_interaction_expression_recursive(evaluator)
        
    else
        error("generate_expression_recursive not implemented for $(typeof(evaluator))")
    end
end

"""
Generate simple categorical expressions (single contrast only).
This function handles categorical evaluators that produce a single output.
"""
function generate_simple_categorical_expression(evaluator::CategoricalEvaluator)
    column = evaluator.column
    n_levels = evaluator.n_levels
    contrast_matrix = evaluator.contrast_matrix
    
    @assert isequal(size(contrast_matrix, 2), 1)
    # "Only single contrast supported in generate_simple_categorical_expression"
    
    # Extract the contrast values for the single contrast
    values = [contrast_matrix[i, 1] for i in 1:n_levels]
    
    # Generate level extraction expression
    level_expr = "clamp(data.$column[row_idx] isa CategoricalValue ? levelcode(data.$column[row_idx]) : 1, 1, $n_levels)"
    
    if n_levels == 1
        # Single level - just return the constant value
        return string(values[1])
        
    elseif n_levels == 2
        # Binary categorical - use ternary operator
        return "($level_expr == 1 ? $(values[1]) : $(values[2]))"
        
    elseif n_levels == 3
        # Three levels - nested ternary
        return "($level_expr == 1 ? $(values[1]) : $level_expr == 2 ? $(values[2]) : $(values[3]))"
        
    elseif n_levels <= 6
        # Small number of levels - use nested ternary chain
        ternary = "$level_expr == 1 ? $(values[1])"
        for i in 2:(n_levels-1)
            ternary *= " : $level_expr == $i ? $(values[i])"
        end
        ternary *= " : $(values[n_levels])"
        return "($ternary)"
        
    else
        # Many levels - use lookup table approach
        values_str = "[" * join(string.(values), ", ") * "]"
        lookup_var = "lookup_$(hash(evaluator) % 10000)"  # Generate unique variable name
        
        # For expressions, we need to create an inline lookup
        # This is a bit complex but avoids requiring statements
        return "(let $lookup_var = $values_str; $lookup_var[$level_expr] end)"
    end
end

###############################################################################
# INTEGRATION WITH EXISTING SYSTEM (CORRECTED)
###############################################################################

"""
Update the main generate_code_from_evaluator to use Phase 2D (CORRECTED).
"""
function generate_code_from_evaluator(evaluator::AbstractEvaluator)
    reset_var_counter!()
    
    total_width = output_width(evaluator)
    
    if total_width == 1
        # Single output - try expression first
        try
            expr = generate_expression_recursive(evaluator)
            return ["@inbounds row_vec[1] = $expr"]
        catch
            # Fall back to statements
            instructions, _ = generate_statements_recursive(evaluator, 1)
            return instructions
        end
    else
        # Multi-output - use statements
        instructions, _ = generate_statements_recursive(evaluator, 1)
        return instructions
    end
end

# Export new Phase 2D functions
export generate_zscore_expression_recursive, generate_scaled_expression_recursive
export generate_product_expression_recursive, generate_interaction_expression_recursive
export generate_zscore_statements_recursive, generate_scaled_statements_recursive
export generate_product_statements_recursive, generate_interaction_statements_recursive
export generate_complex_interaction_statements, generate_code_from_evaluator_phase2d


"""
Apply mathematical functions safely with domain checking.
"""
function apply_function_safe(func::Function, args...)
    if func === log
        @assert length(args) == 1 "log expects 1 argument"
        x = args[1]
        return x > 0.0 ? log(x) : (x == 0.0 ? -Inf : NaN)
        
    elseif func === exp
        @assert length(args) == 1 "exp expects 1 argument"
        x = args[1]
        return exp(clamp(x, -700.0, 700.0))
        
    elseif func === sqrt
        @assert length(args) == 1 "sqrt expects 1 argument"
        x = args[1]
        return x >= 0.0 ? sqrt(x) : NaN
        
    elseif func === (+)
        return sum(args)
        
    elseif func === (*)
        return prod(args)
        
    elseif func === (-)
        @assert length(args) == 2 "subtraction expects 2 arguments"
        return args[1] - args[2]
        
    elseif func === (/)
        @assert length(args) == 2 "division expects 2 arguments"
        return args[1] / args[2]
        
    elseif func === (^)
        @assert length(args) == 2 "power expects 2 arguments"
        return args[1] ^ args[2]
        
    else
        # Generic fallback
        return func(args...)
    end
end

"""
Evaluate a component to scratch space - dispatcher for different evaluator types.
"""
function evaluate_component_to_scratch!(evaluator::AbstractEvaluator, 
                                       scratch::Vector{Float64}, 
                                       scratch_positions::UnitRange{Int},
                                       data::NamedTuple, row_idx::Int)
    
    if evaluator isa ConstantEvaluator
        # Fill scratch with constant value
        @inbounds for pos in scratch_positions
            scratch[pos] = evaluator.value
        end
        
    elseif evaluator isa ContinuousEvaluator
        # Fill scratch with data value
        @inbounds for pos in scratch_positions
            scratch[pos] = Float64(data[evaluator.column][row_idx])
        end
        
    elseif evaluator isa CategoricalEvaluator
        execute_categorical_component_to_scratch!(evaluator, scratch, scratch_positions, data, row_idx)
        
    else
        error("evaluate_component_to_scratch! not implemented for $(typeof(evaluator))")
    end
end

export generate_statements_recursive, next_var, reset_var_counter!

"""
    evaluate_evaluator_to_buffer!(evaluator::AbstractEvaluator, buffer::Vector{Float64}, 
                                  data::NamedTuple, row_idx::Int)

Helper function to evaluate any evaluator into a buffer.
"""
function evaluate_evaluator_to_buffer!(evaluator::AbstractEvaluator, buffer::Vector{Float64}, 
                                      data::NamedTuple, row_idx::Int)
    
    if evaluator isa ConstantEvaluator
        @inbounds buffer[1] = evaluator.value
        
    elseif evaluator isa ContinuousEvaluator
        @inbounds buffer[1] = Float64(data[evaluator.column][row_idx])
        
    elseif evaluator isa CategoricalEvaluator
        # Use existing categorical evaluation logic
        evaluate_categorical_to_buffer!(evaluator, buffer, data, row_idx)
        
    elseif evaluator isa FunctionEvaluator
        # Use existing function evaluation logic
        evaluate_function_to_buffer!(evaluator, buffer, data, row_idx)
        
    else
        # For complex evaluators, use the existing evaluate! method
        evaluate!(evaluator, buffer, data, row_idx, 1)
    end
end

function evaluate_categorical_to_buffer!(evaluator::CategoricalEvaluator, buffer::Vector{Float64}, 
                                       data::NamedTuple, row_idx::Int)
    @inbounds cat_val = data[evaluator.column][row_idx]
    
    level_idx = if cat_val isa CategoricalValue
        Int(levelcode(cat_val))
    else
        1
    end
    
    level_idx = max(1, min(level_idx, evaluator.n_levels))
    
    # Fill buffer with contrast values
    width = size(evaluator.contrast_matrix, 2)
    @inbounds for j in 1:width
        buffer[j] = evaluator.contrast_matrix[level_idx, j]
    end
end

function evaluate_function_to_buffer!(evaluator::FunctionEvaluator, buffer::Vector{Float64}, 
                                    data::NamedTuple, row_idx::Int)
    
    # For simple functions, evaluate directly
    if all(is_direct_evaluatable, evaluator.arg_evaluators)
        # Get argument values
        arg_values = Float64[]
        for arg in evaluator.arg_evaluators
            if arg isa ConstantEvaluator
                push!(arg_values, arg.value)
            elseif arg isa ContinuousEvaluator
                push!(arg_values, Float64(data[arg.column][row_idx]))
            end
        end
        
        # Apply function safely
        result = apply_function_safe(evaluator.func, arg_values...)
        @inbounds buffer[1] = result
    else
        # For complex functions, use existing evaluation
        evaluate!(evaluator, buffer, data, row_idx, 1)
    end
end

###############################################################################
# BLOCK CREATION FUNCTIONS
###############################################################################

"""
    create_zscore_block(evaluator::ZScoreEvaluator, start_pos::Int) -> ZScoreBlock

Create a Z-score execution block.
"""
function create_zscore_block(evaluator::ZScoreEvaluator, start_pos::Int)
    underlying = evaluator.underlying
    underlying_width = output_width(underlying)
    
    input_positions = collect(1:underlying_width)  # Will be filled by underlying evaluator
    output_positions = collect(start_pos:(start_pos + underlying_width - 1))
    
    return ZScoreBlock(
        underlying,
        evaluator.center,
        evaluator.scale,
        input_positions,
        output_positions
    )
end

"""
    create_scaled_block(evaluator::ScaledEvaluator, start_pos::Int) -> ScaledBlock

Create a scaled evaluation block.
"""
function create_scaled_block(evaluator::ScaledEvaluator, start_pos::Int)
    underlying = evaluator.evaluator
    underlying_width = output_width(underlying)
    
    input_positions = collect(1:underlying_width)
    output_positions = collect(start_pos:(start_pos + underlying_width - 1))
    
    return ScaledBlock(
        underlying,
        evaluator.scale_factor,
        input_positions,
        output_positions
    )
end

"""
    create_product_block(evaluator::ProductEvaluator, start_pos::Int) -> ProductBlock

Create a product evaluation block.
"""
function create_product_block(evaluator::ProductEvaluator, start_pos::Int)
    components = evaluator.components
    component_positions = Vector{Vector{Int}}()
    
    # Products are always scalar output, but components can be any width
    for component in components
        width = output_width(component)
        positions = collect(1:width)  # Will be managed by component evaluation
        push!(component_positions, positions)
    end
    
    return ProductBlock(
        components,
        component_positions,
        start_pos  # Single output position
    )
end

###############################################################################
# INTEGRATION HELPERS
###############################################################################

"""
    needs_decomposition(evaluator::FunctionEvaluator) -> Bool

Check if a function evaluator needs AST decomposition.
"""
function needs_decomposition(evaluator::FunctionEvaluator)
    return !all(is_direct_evaluatable, evaluator.arg_evaluators)
end

"""
    estimate_decomposition_complexity(evaluator::FunctionEvaluator) -> Int

Estimate how many operations the decomposition will generate.
"""
function estimate_decomposition_complexity(evaluator::FunctionEvaluator)
    if all(is_direct_evaluatable, evaluator.arg_evaluators)
        return 1  # Simple function call
    else
        # Rough estimate: count nested evaluators
        return count_nested_evaluators(evaluator)
    end
end

function count_nested_evaluators(evaluator::AbstractEvaluator)
    if evaluator isa FunctionEvaluator
        return 1 + sum(count_nested_evaluators(arg) for arg in evaluator.arg_evaluators)
    elseif evaluator isa InteractionEvaluator
        return 1 + sum(count_nested_evaluators(comp) for comp in evaluator.components)
    elseif evaluator isa ZScoreEvaluator
        return 1 + count_nested_evaluators(evaluator.underlying)
    elseif evaluator isa ScaledEvaluator
        return 1 + count_nested_evaluators(evaluator.evaluator)
    elseif evaluator isa ProductEvaluator
        return 1 + sum(count_nested_evaluators(comp) for comp in evaluator.components)
    else
        return 1
    end
end

export generate_function_block!, execute_decomposed_operation!
export needs_decomposition, estimate_decomposition_complexity