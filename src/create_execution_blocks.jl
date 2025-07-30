# create_execution_blocks.jl

###############################################################################
# HELPER FUNCTIONS FOR EXECUTION PLAN GENERATION
###############################################################################

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
    create_input_sources(evaluators::Vector{AbstractEvaluator}) -> Vector{InputSource}

Create input sources for direct evaluatable arguments.
"""
function create_input_sources(evaluators::Vector{AbstractEvaluator})
    sources = InputSource[]

    for eval in evaluators
        push!(sources, create_input_source(eval))
    end
    return sources
end

###############################################################################
# GENERATE BLOCKS
###############################################################################

"""
    generate_blocks!(plan::ExecutionPlan, evaluator::AbstractEvaluator)

Generate execution blocks from self-contained evaluators. NO position calculation needed!
"""
function generate_blocks!(plan::ExecutionPlan, evaluator::AbstractEvaluator)
    
    if evaluator isa ConstantEvaluator
        assignment = ConstantAssignment(evaluator.value, evaluator.position)
        add_assignment!(plan, assignment)
        
    elseif evaluator isa ContinuousEvaluator
        assignment = ContinuousAssignment(evaluator.column, evaluator.position)
        add_assignment!(plan, assignment)
        
    elseif evaluator isa CategoricalEvaluator
        block = create_categorical_block_from_self_contained(evaluator)
        push!(plan, block)
        
    elseif evaluator isa FunctionEvaluator
        block = create_function_block_from_self_contained(evaluator)
        push!(plan, block)
        
    elseif evaluator isa InteractionEvaluator
        block = create_interaction_block_from_self_contained(evaluator)
        push!(plan, block)
        
    elseif evaluator isa ZScoreEvaluator
        block = create_zscore_block_from_self_contained(evaluator)
        push!(plan, block)
        
    elseif evaluator isa ScaledEvaluator
        block = create_scaled_block_from_self_contained(evaluator)
        push!(plan, block)
        
    elseif evaluator isa ProductEvaluator
        block = create_product_block_from_self_contained(evaluator)
        push!(plan, block)
        
    elseif evaluator isa CombinedEvaluator
        # Process sub-evaluators - they have their own positions
        for sub_evaluator in evaluator.sub_evaluators
            generate_blocks!(plan, sub_evaluator)
        end
        
    else
        error("Unknown evaluator type: $(typeof(evaluator))")
    end
    
    return nothing
end

###############################################################################
# CREATE BLOCKS
###############################################################################
"""
    create_categorical_block_from_self_contained(evaluator::CategoricalEvaluator) -> CategoricalBlock

Create categorical block using evaluator's built-in positions.
"""
function create_categorical_block_from_self_contained(evaluator::CategoricalEvaluator)
    # Create lookup tables
    contrast_matrix = evaluator.contrast_matrix
    n_contrasts = length(evaluator.positions)
    lookup_tables = Vector{Vector{Float64}}(undef, n_contrasts)
    
    for j in 1:n_contrasts
        lookup_tables[j] = [contrast_matrix[i, j] for i in 1:evaluator.n_levels]
    end
    
    layout = CategoricalLayout(
        evaluator.column,
        evaluator.n_levels,
        lookup_tables,
        evaluator.positions  # Use built-in positions!
    )
    
    return CategoricalBlock([layout])
end

"""
    create_function_block_from_self_contained(evaluator::FunctionEvaluator) -> FunctionBlock

Create function block using evaluator's built-in scratch positions.
"""
function create_function_block_from_self_contained(evaluator::FunctionEvaluator)
    if all(is_direct_evaluatable, evaluator.arg_evaluators)
        # Simple function - direct evaluation
        input_sources = create_input_sources(evaluator.arg_evaluators)
        output_dest = OutputPosition(evaluator.position)
        
        op = FunctionOp(evaluator.func, input_sources, output_dest)
        return FunctionBlock([op], UnitRange{Int}[], [evaluator.position])
    else
        # Complex function - use evaluator's scratch space mapping
        operations = create_function_operations_from_scratch_map(evaluator)
        
        # Convert scratch positions to ranges for the block
        scratch_ranges = [first(evaluator.scratch_positions):last(evaluator.scratch_positions)]
        
        return FunctionBlock(operations, scratch_ranges, [evaluator.position])
    end
end

"""
    create_interaction_block_from_self_contained(evaluator::InteractionEvaluator{N})

Create interaction block using evaluator's built-in positions.
Updated for parametric types.
"""
function create_interaction_block_from_self_contained(evaluator::InteractionEvaluator{N}) where N
    output_positions = first(evaluator.positions):last(evaluator.positions)
    
    # Create parametric layout
    layout = InteractionLayout{N}(
        hash(evaluator),
        evaluator.component_scratch_map,
        output_positions,
        evaluator.kronecker_pattern,  # Already Vector{NTuple{N,Int}}
        [output_width_structural(comp) for comp in evaluator.components]
    )
    
    return InteractionBlock{N}(layout, evaluator.components)
end

"""
    create_zscore_block_from_self_contained(evaluator::ZScoreEvaluator) -> ZScoreBlock

Create Z-score block using evaluator's built-in positions.
"""
function create_zscore_block_from_self_contained(evaluator::ZScoreEvaluator)
    return ZScoreBlock(
        evaluator.underlying,
        evaluator.center,
        evaluator.scale,
        collect(evaluator.underlying_scratch_map),  # Input positions in scratch
        evaluator.positions  # Output positions in model matrix
    )
end

"""
    create_scaled_block_from_self_contained(evaluator::ScaledEvaluator) -> ScaledBlock

Create scaled block using evaluator's built-in positions.
"""
function create_scaled_block_from_self_contained(evaluator::ScaledEvaluator)
    return ScaledBlock(
        evaluator.evaluator,
        evaluator.scale_factor,
        collect(evaluator.underlying_scratch_map),  # Input positions in scratch
        evaluator.positions  # Output positions in model matrix
    )
end

"""
    create_product_block_from_self_contained(evaluator::ProductEvaluator) -> ProductBlock

Create product block using evaluator's built-in positions.
"""
function create_product_block_from_self_contained(evaluator::ProductEvaluator)
    # Convert scratch ranges to position vectors for each component
    component_positions = Vector{Vector{Int}}()
    
    for scratch_range in evaluator.component_scratch_map
        push!(component_positions, collect(scratch_range))
    end
    
    return ProductBlock(
        evaluator.components,
        component_positions,
        evaluator.position # Single output position
    )
end
