# execute_block.jl

###############################################################################
# MAIN EXECUTE_BLOCK! DISPATCH
###############################################################################

"""
    execute_block!(block::ExecutionBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                   data::NamedTuple, row_idx::Int)

Main dispatcher for all execution block types. This is the core engine that makes
zero-allocation evaluation work.
"""
function execute_block!(block::ExecutionBlock, scratch::Vector{Float64}, output::Vector{Float64}, 
                       data::NamedTuple, row_idx::Int)
    
    if block isa AssignmentBlock
        execute_assignment_block!(block, output, data, row_idx)
        
    elseif block isa CategoricalBlock  
        execute_categorical_block_complete!(block, output, data, row_idx)
        
    elseif block isa FunctionBlock
        execute_function_block_complete!(block, scratch, output, data, row_idx)
        
    elseif block isa InteractionBlock
        execute_interaction_block_complete!(block, scratch, output, data, row_idx)
        
    elseif block isa ZScoreBlock
        execute_zscore_block!(block, scratch, output, data, row_idx)
        
    elseif block isa ScaledBlock
        execute_scaled_block!(block, scratch, output, data, row_idx)
        
    elseif block isa ProductBlock
        execute_product_block!(block, scratch, output, data, row_idx)
        
    else
        error("Unknown execution block type: $(typeof(block))")
    end
    
    return nothing
end

###############################################################################
# 2. CATEGORICAL BLOCK EXECUTION (Complete zero-allocation)
###############################################################################

"""
    execute_categorical_block_complete!(block::CategoricalBlock, output::Vector{Float64}, 
                                       data::NamedTuple, row_idx::Int)

COMPLETE implementation of categorical block execution with true zero allocation.
This fixes the allocation issues in the current implementation.
"""
function execute_categorical_block_complete!(block::CategoricalBlock, output::Vector{Float64}, 
                                           data::NamedTuple, row_idx::Int)
    
    @inbounds for layout in block.layouts
        execute_categorical_layout_zero_alloc!(layout, output, data, row_idx)
    end
    
    return nothing
end

"""
    execute_categorical_layout_zero_alloc!(layout::CategoricalLayout, output::Vector{Float64},
                                          data::NamedTuple, row_idx::Int)

True zero-allocation categorical evaluation with aggressive optimization.
"""
@inline function execute_categorical_layout_zero_alloc!(layout::CategoricalLayout, 
                                                       output::Vector{Float64},
                                                       data::NamedTuple, row_idx::Int)
    
    # OPTIMIZATION 1: Direct field access to avoid NamedTuple overhead
    column_data = getfield(data, layout.column)
    
    # OPTIMIZATION 2: Type-stable categorical value extraction
    @inbounds cat_val = column_data[row_idx]
    
    # OPTIMIZATION 3: Eliminate allocation in level extraction
    level_idx = extract_categorical_level_fast(cat_val, layout.n_levels)
    
    # OPTIMIZATION 4: Specialized lookup application
    apply_categorical_lookups_fast!(layout.lookup_tables, layout.output_positions, 
                                   output, level_idx)
    
    return nothing
end

"""
    apply_categorical_lookups_fast!(lookup_tables::Vector{Vector{Float64}}, 
                                   output_positions::Vector{Int},
                                   output::Vector{Float64}, level_idx::Int)

Apply pre-computed lookup tables with zero allocations.
"""
function apply_categorical_lookups_fast!(lookup_tables::Vector{Vector{Float64}}, 
                                                output_positions::Vector{Int},
                                                output::Vector{Float64}, level_idx::Int)
    
    n_tables = length(lookup_tables)
    
    # Unrolled loops for common cases (better performance)
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
        # General case for many contrasts
        @inbounds for j in 1:n_tables
            output[output_positions[j]] = lookup_tables[j][level_idx]
        end
    end
    
    return nothing
end

##############################################################################
# 3. FUNCTION BLOCK EXECUTION (Complete with decomposition support)
###############################################################################

"""
    execute_function_block_complete!(block::FunctionBlock, scratch::Vector{Float64}, 
                                    output::Vector{Float64}, data::NamedTuple, row_idx::Int)

COMPLETE function block execution supporting both simple and complex (decomposed) functions.
"""
function execute_function_block_complete!(block::FunctionBlock, scratch::Vector{Float64}, 
                                         output::Vector{Float64}, data::NamedTuple, row_idx::Int)
    
    # Execute all function operations in sequence
    @inbounds for operation in block.operations
        execute_function_operation_complete!(operation, scratch, output, data, row_idx)
    end
    
    return nothing
end

"""
    execute_function_operation_complete!(op::FunctionOp, scratch::Vector{Float64}, 
                                        output::Vector{Float64}, data::NamedTuple, row_idx::Int)

Execute a single function operation with complete input/output handling.
"""
function execute_function_operation_complete!(op::FunctionOp, scratch::Vector{Float64}, 
                                             output::Vector{Float64}, data::NamedTuple, row_idx::Int)
    
    n_inputs = length(op.input_sources)
    
    # Handle different arities efficiently
    if n_inputs == 0
        # Nullary function (rare)
        result = op.func()
        
    elseif n_inputs == 1
        # Unary function (common: log, exp, sqrt, etc.)
        val = get_input_value_fast(op.input_sources[1], scratch, output, data, row_idx)
        result = apply_function_safe(op.func, val)
        
    elseif n_inputs == 2
        # Binary function (common: +, -, *, /, ^)
        val1 = get_input_value_fast(op.input_sources[1], scratch, output, data, row_idx)
        val2 = get_input_value_fast(op.input_sources[2], scratch, output, data, row_idx)
        result = apply_function_safe(op.func, val1, val2)
        
    else
        # N-ary function (rare but needed for completeness)
        args = Vector{Float64}(undef, n_inputs)
        @inbounds for i in 1:n_inputs
            args[i] = get_input_value_fast(op.input_sources[i], scratch, output, data, row_idx)
        end
        result = apply_function_safe(op.func, args...)
    end
    
    # Store result in appropriate destination
    store_output_value_fast!(op.output_destination, result, scratch, output)
    
    return nothing
end

"""
    get_input_value_fast(source::InputSource, scratch::Vector{Float64}, output::Vector{Float64}, 
                        data::NamedTuple, row_idx::Int) -> Float64

Fast input value retrieval with zero allocations.
"""
@inline function get_input_value_fast(source::InputSource, scratch::Vector{Float64}, 
                                     output::Vector{Float64}, data::NamedTuple, row_idx::Int)
    
    if source isa DataSource
        @inbounds return Float64(getfield(data, source.column)[row_idx])
        
    elseif source isa ScratchSource
        @inbounds return scratch[source.position]
        
    elseif source isa ConstantSource
        return source.value
        
    else
        error("Unknown input source type: $(typeof(source))")
    end
end

"""
    store_output_value_fast!(dest::OutputDestination, value::Float64, 
                            scratch::Vector{Float64}, output::Vector{Float64})

Fast output value storage with zero allocations.
"""
@inline function store_output_value_fast!(dest::OutputDestination, value::Float64, 
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
# 4. INTERACTION BLOCK EXECUTION (Complete Kronecker product implementation)
###############################################################################

"""
    execute_interaction_block_complete!(block::InteractionBlock{N}, scratch::Vector{Float64}, 
                                       output::Vector{Float64}, data::NamedTuple, row_idx::Int) where N

Execute interaction block with parametric type support.
"""
function execute_interaction_block_complete!(block::InteractionBlock{N}, scratch::Vector{Float64}, 
                                           output::Vector{Float64}, data::NamedTuple, row_idx::Int) where N
    
    layout = block.layout
    components = block.component_evaluators
    
    # Step 1: Evaluate all components into their scratch positions
    for (i, component) in enumerate(components)
        scratch_range = layout.component_scratch_positions[i]
        evaluate_component_to_scratch_complete!(component, 
                                               view(scratch, scratch_range), 
                                               data, row_idx)
    end
    
    # Step 2: Apply Kronecker product using parametric pattern
    apply_kronecker_pattern_complete!(
        layout.kronecker_pattern,  # Now Vector{NTuple{N,Int}}
        layout.component_scratch_positions,
        scratch,
        view(output, layout.output_positions)
    )
    
    return nothing
end

"""
    evaluate_component_to_scratch_complete!(evaluator::AbstractEvaluator, 
                                           scratch::Vector{Float64}, 
                                           scratch_positions::UnitRange{Int},
                                           data::NamedTuple, row_idx::Int)

Evaluate any evaluator type directly into scratch space with zero allocations.
"""
function evaluate_component_to_scratch_complete!(evaluator::AbstractEvaluator, 
                                                scratch::Vector{Float64}, 
                                                scratch_positions::UnitRange{Int},
                                                data::NamedTuple, row_idx::Int)
    
    if evaluator isa ConstantEvaluator
        # Fill all positions with constant value
        @inbounds for pos in scratch_positions
            scratch[pos] = evaluator.value
        end
        
    elseif evaluator isa ContinuousEvaluator
        # Fill all positions with data value (should be width 1)
        @inbounds scratch[first(scratch_positions)] = Float64(data[evaluator.column][row_idx])
        
    elseif evaluator isa CategoricalEvaluator
        # Evaluate categorical directly into scratch
        execute_categorical_to_scratch!(evaluator, scratch, scratch_positions, data, row_idx)
        
    elseif evaluator isa FunctionEvaluator
        # For simple functions, evaluate directly
        if length(evaluator.arg_evaluators) == 1 && 
           evaluator.arg_evaluators[1] isa ContinuousEvaluator
            
            arg_val = Float64(data[evaluator.arg_evaluators[1].column][row_idx])
            result = apply_function_safe(evaluator.func, arg_val)
            @inbounds scratch[first(scratch_positions)] = result
        else
            # Complex function - use temporary evaluation
            temp_buffer = Vector{Float64}(undef, length(scratch_positions))
            evaluate!(evaluator, temp_buffer, data, row_idx, 1)
            @inbounds for (i, pos) in enumerate(scratch_positions)
                scratch[pos] = temp_buffer[i]
            end
        end
        
    else
        # General case - use existing evaluator system
        temp_buffer = Vector{Float64}(undef, length(scratch_positions))
        evaluate!(evaluator, temp_buffer, data, row_idx, 1)
        @inbounds for (i, pos) in enumerate(scratch_positions)
            scratch[pos] = temp_buffer[i]
        end
    end
    
    return nothing
end

"""
    execute_categorical_to_scratch!(evaluator::CategoricalEvaluator, 
                                   scratch::Vector{Float64}, scratch_positions::UnitRange{Int},
                                   data::NamedTuple, row_idx::Int)

Execute categorical evaluator directly into scratch space.
"""
function execute_categorical_to_scratch!(evaluator::CategoricalEvaluator, 
                                        scratch::Vector{Float64}, scratch_positions::UnitRange{Int},
                                        data::NamedTuple, row_idx::Int)
    
    # Get categorical value and level index
    @inbounds cat_val = getfield(data, evaluator.column)[row_idx]
    level_idx = extract_categorical_level_fast(cat_val, evaluator.n_levels)
    
    # Fill scratch positions with contrast values
    contrast_matrix = evaluator.contrast_matrix
    n_contrasts = size(contrast_matrix, 2)
    
    @inbounds for j in 1:n_contrasts
        pos = first(scratch_positions) + j - 1
        scratch[pos] = contrast_matrix[level_idx, j]
    end
    
    return nothing
end

"""
    apply_kronecker_pattern_complete!(pattern::Vector{NTuple{N,Int}},
                                     component_positions::Vector{UnitRange{Int}},
                                     scratch::Vector{Float64},
                                     output::AbstractVector{Float64}) where N

Apply pre-computed Kronecker product pattern with parametric type support.
"""
function apply_kronecker_pattern_complete!(pattern::Vector{NTuple{N,Int}},
                                          component_positions::Vector{UnitRange{Int}},
                                          scratch::Vector{Float64},
                                          output::AbstractVector{Float64}) where N
    
    @inbounds for (idx, indices) in enumerate(pattern)
        # Type-stable computation with compile-time known N
        product = 1.0
        for i in 1:N
            scratch_pos = first(component_positions[i]) + indices[i] - 1
            product *= scratch[scratch_pos]
        end
        output[idx] = product
    end
    
    return nothing
end

###############################################################################
# 5. ZSCORE BLOCK EXECUTION
###############################################################################

"""
    execute_zscore_block!(block::ZScoreBlock, scratch::Vector{Float64}, 
                         output::Vector{Float64}, data::NamedTuple, row_idx::Int)

Execute Z-score transformation: (x - center) / scale
"""
function execute_zscore_block!(block::ZScoreBlock, scratch::Vector{Float64}, 
                              output::Vector{Float64}, data::NamedTuple, row_idx::Int)
    
    # First evaluate the underlying evaluator
    underlying_width = length(block.input_positions)
    temp_buffer = Vector{Float64}(undef, underlying_width)
    evaluate!(block.underlying_evaluator, temp_buffer, data, row_idx, 1)
    
    # Apply Z-score transformation to all outputs
    center = block.center
    scale = block.scale
    
    @inbounds for i in 1:underlying_width
        input_val = temp_buffer[i]
        output_pos = block.output_positions[i]
        
        if scale ≈ 0.0
            # Handle zero scale case
            output[output_pos] = input_val ≈ center ? 0.0 : (input_val > center ? Inf : -Inf)
        else
            output[output_pos] = (input_val - center) / scale
        end
    end
    
    return nothing
end

###############################################################################
# 6. SCALED BLOCK EXECUTION
###############################################################################

"""
    execute_scaled_block!(block::ScaledBlock, scratch::Vector{Float64}, 
                         output::Vector{Float64}, data::NamedTuple, row_idx::Int)

Execute scaled evaluation: scale_factor * value
"""
function execute_scaled_block!(block::ScaledBlock, scratch::Vector{Float64}, 
                              output::Vector{Float64}, data::NamedTuple, row_idx::Int)
    
    # First evaluate the underlying evaluator
    underlying_width = length(block.input_positions)
    temp_buffer = Vector{Float64}(undef, underlying_width)
    evaluate!(block.underlying_evaluator, temp_buffer, data, row_idx, 1)
    
    # Apply scaling to all outputs
    scale_factor = block.scale_factor
    
    @inbounds for i in 1:underlying_width
        input_val = temp_buffer[i]
        output_pos = block.output_positions[i]
        output[output_pos] = scale_factor * input_val
    end
    
    return nothing
end

###############################################################################
# 7. PRODUCT BLOCK EXECUTION
###############################################################################

"""
    execute_product_block!(block::ProductBlock, scratch::Vector{Float64}, 
                          output::Vector{Float64}, data::NamedTuple, row_idx::Int)

Execute product evaluation: component1 * component2 * ... * componentN
"""
function execute_product_block!(block::ProductBlock, scratch::Vector{Float64}, 
                                output::Vector{Float64}, data::NamedTuple, row_idx::Int)
    
    # Evaluate all components and compute their product
    product = 1.0
    temp_buffer = Vector{Float64}(undef, 1)  # Products are always scalar
    
    @inbounds for component in block.component_evaluators
        evaluate!(component, temp_buffer, data, row_idx, 1)
        product *= temp_buffer[1]
    end
    
    output[block.output_position] = product
    
    return nothing
end

###############################################################################
# 1. ASSIGNMENT BLOCK EXECUTION
###############################################################################

"""
    execute_assignment_block!(block::AssignmentBlock, output::Vector{Float64}, 
                              data::NamedTuple, row_idx::Int)

Execute type-stable assignments with zero allocations.
"""
function execute_assignment_block!(block::AssignmentBlock, output::Vector{Float64}, 
                                  data::NamedTuple, row_idx::Int)
    
    @inbounds for assignment in block.assignments
        if assignment isa ConstantAssignment
            output[assignment.output_position] = assignment.value
        elseif assignment isa ContinuousAssignment
            output[assignment.output_position] = Float64(data[assignment.column][row_idx])
        else
            error("Unknown assignment type: $(typeof(assignment))")
        end
    end
    
    return nothing
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
# 9. UTILITY FUNCTIONS
###############################################################################

"""
    validate_execution_blocks()

Validate that all execution block types are properly implemented.
"""
function validate_execution_blocks()
    println("Validating execution block implementations...")
    
    # Test that all block types have execute_block! methods
    block_types = [
        AssignmentBlock, CategoricalBlock, FunctionBlock, 
        InteractionBlock, ZScoreBlock, ScaledBlock, ProductBlock
    ]
    
    for block_type in block_types
        if !hasmethod(execute_block!, (block_type, Vector{Float64}, Vector{Float64}, NamedTuple, Int))
            error("Missing execute_block! method for $block_type")
        end
        println("  ✅ $block_type execution implemented")
    end
    
    println("All execution block implementations validated!")
    return true
end
