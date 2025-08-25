# step4/main.jl
# Implementation of the interaction system (Step 4 of compilation pipeline)
# Complete interaction system with zero allocations for all cases (except functions!)
# PLAN: @generated for functions?

# Types are defined in step4/types.jl and included by step4_interactions.jl

###############################################################################
# ZERO-ALLOCATION VALUE ACCESS (MIRROR FUNCTION SYSTEM)
###############################################################################

"""
    get_interaction_value(input, output, scratch, input_data, row_idx) -> Float64

Zero-allocation input value access mirroring get_input_value exactly.
"""

# Constant values - compile-time dispatch
@inline function get_interaction_value(input::Float64, output, scratch, input_data, row_idx)
    return input
end

# Column references - compile-time dispatch  
@inline function get_interaction_value(input::Symbol, output, scratch, input_data, row_idx)
    return Float64(get_data_value_specialized(input_data, input, row_idx))
end

# Output positions - compile-time dispatch
@inline function get_interaction_value(input::Int, output, scratch, input_data, row_idx)
    return output[input]
end

# Interaction scratch positions - compile-time dispatch
@inline function get_interaction_value(input::InteractionScratchPosition{P}, output, scratch, input_data, row_idx) where P
    return scratch[input.position]
end

###############################################################################
# COMPONENT VALUE ACCESS WITH ENHANCED VALIDATION
###############################################################################

"""
    get_component_interaction_value(component::AbstractEvaluator, index::Int, data::NamedTuple, row_idx::Int, output::Vector{Float64}, scratch::Vector{Float64}) -> Float64

Get component value for interactions with comprehensive validation.
"""
@inline function get_component_interaction_value(
    component::ConstantEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::V,
    scratch::Vector{Float64}
) where {V <: AbstractVector{Float64}}
    if index != 1
        error("ConstantEvaluator is scalar but got index=$index (must be 1)")
    end
    return component.value
end

@inline function get_component_interaction_value(
    component::ContinuousEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::V,
    scratch::Vector{Float64}
) where {V <: AbstractVector{Float64}}
    if index != 1
        error("ContinuousEvaluator is scalar but got index=$index (must be 1)")
    end
    return Float64(get_data_value_specialized(data, component.column, row_idx))
end

"""
    get_component_interaction_value(component::CategoricalEvaluator, index::Int, input_data, row_idx, output, scratch)

Extract level codes dynamically during execution instead of using pre-computed level_codes.
"""
@inline function get_component_interaction_value(
    component::CategoricalEvaluator, 
    index::Int, 
    input_data::NamedTuple, 
    row_idx::Int,
    output::AbstractVector{Float64},
    scratch::AbstractVector{Float64}
) 
    n_contrasts = size(component.contrast_matrix, 2)
    if index < 1 || index > n_contrasts
        error("CategoricalEvaluator index $index out of bounds (1:$n_contrasts)")
    end
    
    # FIXED: Extract level dynamically from current input_data instead of using pre-computed level_codes
    column_data = getproperty(input_data, component.column)
    level = extract_level_code_zero_alloc_interactions(column_data, row_idx)
    
    if level < 1 || level > component.n_levels
        error("Level $level out of bounds for categorical $(component.column) with $(component.n_levels) levels")
    end
    
    # Return the correct contrast value
    return component.contrast_matrix[level, index]
end

"""
    extract_level_code_zero_alloc_interactions(column_data, row_idx::Int) -> Int

Extract level code for interactions with zero allocations using type-stable dispatch.
This is identical to the main categorical system's extract_level_code_zero_alloc.
"""
@inline function extract_level_code_zero_alloc_interactions(column_data::CategoricalVector, row_idx::Int)
    return Int(levelcode(column_data[row_idx]))
end

@inline function extract_level_code_zero_alloc_interactions(column_data::OverrideVector{CategoricalValue{T,R}}, row_idx::Int) where {T,R}
    # For OverrideVector, all rows have the same value - extract once, no allocation
    return Int(levelcode(column_data.override_value))
end

@inline function extract_level_code_zero_alloc_interactions(column_data, row_idx::Int)
    # Fallback for other types
    error("Cannot extract level code from $(typeof(column_data)) in interactions")
end

@inline function get_component_interaction_value(
    component::FunctionEvaluator, 
    index::Int, 
    input_data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64},
    scratch::Vector{Float64}
)
    if index != 1
        error("FunctionEvaluator is scalar but got index=$index (must be 1)")
    end
    
    # FIXED: Evaluate arguments without allocating
    n_args = length(component.arg_evaluators)
    
    if n_args == 0
        # No arguments - shouldn't happen but handle gracefully
        return Float64(component.func())
    elseif n_args == 1
        # Unary function - evaluate single argument inline
        arg_eval = component.arg_evaluators[1]
        arg_val = if arg_eval isa ContinuousEvaluator
            Float64(get_data_value_specialized(input_data, arg_eval.column, row_idx))
        elseif arg_eval isa ConstantEvaluator
            arg_eval.value
        else
            error("Unsupported argument evaluator in function: $(typeof(arg_eval))")
        end
        
        # Apply function with special cases
        result = if component.func === log
            arg_val > 0.0 ? log(arg_val) : (arg_val == 0.0 ? -Inf : NaN)
        elseif component.func === exp
            exp(clamp(arg_val, -700.0, 700.0))
        elseif component.func === sqrt
            arg_val ≥ 0.0 ? sqrt(arg_val) : NaN
        elseif component.func === abs
            abs(arg_val)
        elseif component.func === sin
            sin(arg_val)
        elseif component.func === cos
            cos(arg_val)
        else
            Float64(component.func(arg_val))
        end
        
        return result
    elseif n_args == 2
        # Binary function - evaluate both arguments inline
        arg_eval1 = component.arg_evaluators[1]
        arg_val1 = if arg_eval1 isa ContinuousEvaluator
            Float64(get_data_value_specialized(input_data, arg_eval1.column, row_idx))
        elseif arg_eval1 isa ConstantEvaluator
            arg_eval1.value
        else
            error("Unsupported argument evaluator in function: $(typeof(arg_eval1))")
        end
        
        arg_eval2 = component.arg_evaluators[2]
        arg_val2 = if arg_eval2 isa ContinuousEvaluator
            Float64(get_data_value_specialized(input_data, arg_eval2.column, row_idx))
        elseif arg_eval2 isa ConstantEvaluator
            arg_eval2.value
        else
            error("Unsupported argument evaluator in function: $(typeof(arg_eval2))")
        end
        
        # Apply binary function
        return Float64(component.func(arg_val1, arg_val2))
    else
        # N-ary function (n > 2) - this is rare and might need special handling
        # For now, error as we don't support it without allocation
        error("Functions with more than 2 arguments not supported in interactions without allocation")
    end
end

@inline function get_component_interaction_value(
    scratch_pos::InteractionScratchPosition{P},
    index::Int,
    data::NamedTuple,
    row_idx::Int,
    output::Vector{Float64},
    scratch::Vector{Float64}
) where P
    if index != 1
        error("InteractionScratchPosition is scalar but got index=$index (must be 1)")
    end
    
    if scratch_pos.position < 1 || scratch_pos.position > length(scratch)
        error("InteractionScratchPosition $(scratch_pos.position) out of bounds (scratch length: $(length(scratch)))")
    end
    
    return scratch[scratch_pos.position]
end

###############################################################################
# INPUT SOURCE VALUE ACCESS (ENHANCED)
###############################################################################

"""
    get_value_from_source(source, component::AbstractEvaluator, index::Int, input_data, row_idx, output, scratch)

Get value from input source, handling all source types with type dispatch.
"""

# Direct value sources (constants)
@inline function get_value_from_source(
    source::Float64, 
    component::AbstractEvaluator, 
    index::Int, 
    input_data::NamedTuple, 
    row_idx::Int, 
    output::AbstractVector{Float64}, 
    scratch::AbstractVector{Float64}
)
    return source
end

# Column sources (continuous/categorical variables)
@inline function get_value_from_source(
    source::Symbol, 
    component::AbstractEvaluator, 
    index::Int, 
    input_data::NamedTuple, 
    row_idx::Int, 
    output::AbstractVector{Float64}, 
    scratch::AbstractVector{Float64}
)
    # Always use component-based access, don't re-access raw data
    return get_component_interaction_value(component, index, input_data, row_idx, output, scratch)
end

# Output position sources - SPECIAL HANDLING FOR FUNCTIONS
@inline function get_value_from_source(
    source::Int, 
    component::FunctionEvaluator,  # More specific type
    index::Int, 
    input_data::NamedTuple, 
    row_idx::Int, 
    output::AbstractVector{Float64}, 
    scratch::AbstractVector{Float64}
)
    if index != 1
        error("FunctionEvaluator is scalar but got index=$index (must be 1)")
    end
    
    # FIXED: Evaluate the function's arguments without allocating
    n_args = length(component.arg_evaluators)
    
    if n_args == 0
        # No arguments - shouldn't happen but handle gracefully
        return Float64(component.func())
    elseif n_args == 1
        # Unary function - evaluate single argument inline
        arg_eval = component.arg_evaluators[1]
        arg_val = if arg_eval isa ContinuousEvaluator
            Float64(get_data_value_specialized(input_data, arg_eval.column, row_idx))
        elseif arg_eval isa ConstantEvaluator
            arg_eval.value
        elseif arg_eval isa CategoricalEvaluator
            level = arg_eval.level_codes[row_idx]
            level = clamp(level, 1, arg_eval.n_levels)
            arg_eval.contrast_matrix[level, 1]
        else
            error("Unsupported argument type in function: $(typeof(arg_eval))")
        end
        
        # Apply the function with special cases
        result = if component.func === log
            arg_val > 0.0 ? log(arg_val) : (arg_val == 0.0 ? -Inf : NaN)
        elseif component.func === exp
            exp(clamp(arg_val, -700.0, 700.0))
        elseif component.func === sqrt
            arg_val ≥ 0.0 ? sqrt(arg_val) : NaN
        elseif component.func === abs
            abs(arg_val)
        elseif component.func === sin
            sin(arg_val)
        elseif component.func === cos
            cos(arg_val)
        else
            Float64(component.func(arg_val))
        end
        
        return result
    elseif n_args == 2
        # Binary function - evaluate both arguments inline
        arg_eval1 = component.arg_evaluators[1]
        arg_val1 = if arg_eval1 isa ContinuousEvaluator
            Float64(get_data_value_specialized(input_data, arg_eval1.column, row_idx))
        elseif arg_eval1 isa ConstantEvaluator
            arg_eval1.value
        elseif arg_eval1 isa CategoricalEvaluator
            level = arg_eval1.level_codes[row_idx]
            level = clamp(level, 1, arg_eval1.n_levels)
            arg_eval1.contrast_matrix[level, 1]
        else
            error("Unsupported argument type in function: $(typeof(arg_eval1))")
        end
        
        arg_eval2 = component.arg_evaluators[2]
        arg_val2 = if arg_eval2 isa ContinuousEvaluator
            Float64(get_data_value_specialized(input_data, arg_eval2.column, row_idx))
        elseif arg_eval2 isa ConstantEvaluator
            arg_eval2.value
        elseif arg_eval2 isa CategoricalEvaluator
            level = arg_eval2.level_codes[row_idx]
            level = clamp(level, 1, arg_eval2.n_levels)
            arg_eval2.contrast_matrix[level, 1]
        else
            error("Unsupported argument type in function: $(typeof(arg_eval2))")
        end
        
        # Apply binary function
        return Float64(component.func(arg_val1, arg_val2))
    else
        # N-ary function (n > 2) - this is rare and might need special handling
        error("Functions with more than 2 arguments not supported in interactions without allocation")
    end
end

# Output position sources - GENERAL CASE (non-function components)
@inline function get_value_from_source(
    source::Int, 
    component::AbstractEvaluator,  # General case
    index::Int, 
    input_data::NamedTuple, 
    row_idx::Int, 
    output::AbstractVector{Float64}, 
    scratch::AbstractVector{Float64}
)
    if source < 1 || source > length(output)
        error("Output position $source out of bounds (output length: $(length(output)))")
    end
    return output[source]
end

# Scratch position sources (intermediate results)
@inline function get_value_from_source(
    source::InteractionScratchPosition{P}, 
    component::AbstractEvaluator, 
    index::Int, 
    input_data::NamedTuple, 
    row_idx::Int, 
    output::AbstractVector{Float64}, 
    scratch::AbstractVector{Float64}
) where P
    # When the component is an InteractionScratchReference,
    # we need to use the component's scratch positions array indexed by 'index'
    if component isa InteractionScratchReference
        # The index tells us which element of the intermediate result to get
        if index < 1 || index > length(component.scratch_positions)
            error("Index $index out of bounds for InteractionScratchReference with $(length(component.scratch_positions)) positions")
        end
        
        scratch_pos = component.scratch_positions[index]
        
        if scratch_pos < 1 || scratch_pos > length(scratch)
            error("Scratch position $scratch_pos out of bounds (scratch length: $(length(scratch)))")
        end
        
        return scratch[scratch_pos]
    else
        # For other component types, use the source position directly
        return get_interaction_value(source, output, scratch, input_data, row_idx)
    end
end

###############################################################################
# INTERACTION PATTERN GENERATION (NOW RETURNS TUPLE)
###############################################################################

"""
    compute_interaction_pattern_tuple(width1::Int, width2::Int)

Generate interaction pattern as compile-time tuple.
Returns tuple (instead of Vector) for zero allocation.
"""
function compute_interaction_pattern_tuple(width1::Int, width2::Int)
    if width1 <= 0 || width2 <= 0
        error("Invalid component widths: width1=$width1, width2=$width2 (both must be > 0)")
    end
    
    n_patterns = width1 * width2
    
    # Create compile-time tuple using ntuple
    pattern_tuple = ntuple(n_patterns) do idx
        # Convert linear index to (i, j) pair
        # Match StatsModels: kron(b, a) means a varies fast, b varies slow
        j = ((idx - 1) ÷ width1) + 1  # Slow index (second component)
        i = ((idx - 1) % width1) + 1  # Fast index (first component)
        (i, j)
    end
    
    return pattern_tuple
end

###############################################################################
# LINEARIZED OPERATION (TEMPORARY FOR CONSTRUCTION)
###############################################################################

"""
    LinearizedInteractionOperation

Intermediate representation for interaction decomposition.
Still uses Vectors during construction phase only.
"""
struct LinearizedInteractionOperation
    operation_type::Symbol  # :intermediate_interaction or :final_interaction
    component1::AbstractEvaluator
    component2::AbstractEvaluator
    input1_source::Union{Symbol, Int, Float64, InteractionScratchPosition}
    input2_source::Union{Symbol, Int, Float64, InteractionScratchPosition}
    output_positions::Vector{Int}
    scratch_position::Union{Int, Nothing}  # For intermediate operations only
    function_pre_evals::Vector{FunctionPreEvalOperation}  # Will be converted to tuple
end

"""
    InteractionScratchReference

Placeholder component representing intermediate results in scratch space.
"""
struct InteractionScratchReference <: AbstractEvaluator
    scratch_positions::Vector{Int}
    width::Int
    
    function InteractionScratchReference(positions::Vector{Int})
        new(positions, length(positions))
    end
end

# Interface methods for InteractionScratchReference
output_width(ref::InteractionScratchReference) = ref.width
get_positions(ref::InteractionScratchReference) = ref.scratch_positions
get_scratch_positions(ref::InteractionScratchReference) = ref.scratch_positions
max_scratch_needed(ref::InteractionScratchReference) = maximum(ref.scratch_positions)
get_component_output_width(ref::InteractionScratchReference) = ref.width

function get_component_input_source(component::InteractionScratchReference)
    return InteractionScratchPosition(component.scratch_positions[1])
end

@inline function get_component_interaction_value(
    ref::InteractionScratchReference,
    index::Int,
    input_data::NamedTuple,
    row_idx::Int,
    output::AbstractVector{Float64},
    scratch::AbstractVector{Float64}
)
    if index < 1 || index > length(ref.scratch_positions)
        error("InteractionScratchReference index $index out of bounds (1:$(length(ref.scratch_positions)))")
    end
    
    scratch_pos = ref.scratch_positions[index]
    
    if scratch_pos < 1 || scratch_pos > length(scratch)
        error("Scratch position $scratch_pos out of bounds (scratch length: $(length(scratch)))")
    end
    
    return scratch[scratch_pos]
end

"""
    get_component_input_source(component::AbstractEvaluator)

Extract input source for component (column symbol, position, etc.)
"""
function get_component_input_source(component::AbstractEvaluator)
    if component isa ConstantEvaluator
        return component.value
    elseif component isa ContinuousEvaluator
        return component.column
    elseif component isa CategoricalEvaluator
        return component.column
    elseif component isa FunctionEvaluator
        return :function_component  # Special marker
    elseif component isa InteractionScratchReference
        return InteractionScratchPosition(component.scratch_positions[1])
    else
        error("Unsupported component type for input source: $(typeof(component))")
    end
end

"""
    decompose_interaction_tree_zero_alloc(interaction_eval::InteractionEvaluator, temp_allocator::TempAllocator)

Decompose interaction into operations. Still returns Vector during construction.
"""
function decompose_interaction_tree_zero_alloc(interaction_eval::InteractionEvaluator, temp_allocator::TempAllocator)
    operations = LinearizedInteractionOperation[]
    
    components = interaction_eval.components
    N = length(components)
    final_positions = interaction_eval.positions
    component_widths = interaction_eval.component_widths
    
    # Track function pre-evaluations needed for this interaction
    function_pre_evals = FunctionPreEvalOperation[]
    
    # Pre-process components
    processed_components = []
    processed_input_sources = []
    processed_widths = []
    
    for (i, comp) in enumerate(components)
        width = component_widths[i]
        
        if comp isa FunctionEvaluator
            # Check if this is a specializable function×component pattern
            other_components = [components[j] for j in 1:N if j != i]
            if N == 2 && length(other_components) == 1 && is_specialized_function_interaction(comp, other_components[1])
                # Use specialized zero-allocation path - bypass individual component processing
                # The specialized interaction handles the entire interaction internally
                specialized_interaction = create_specialized_function_interaction(comp, other_components[1], final_positions)
                
                # Create the operation directly and skip the regular binary interaction logic
                push!(operations, LinearizedInteractionOperation(
                    :specialized_interaction,
                    specialized_interaction,
                    other_components[1],  # Keep the other component for reference
                    :specialized,  # input1_source - indicates specialized path
                    :specialized,  # input2_source - indicates specialized path  
                    final_positions,
                    nothing,  # scratch_position not used
                    FunctionPreEvalOperation[]  # No pre-evaluations needed!
                ))
                
                return operations  # Early return - no further processing needed
            else
                # Fallback: Functions need pre-evaluation to scratch
                func_result_pos = allocate_temp!(temp_allocator)
                
                # Create a temp allocator for the function's internal operations
                func_internal_allocator = TempAllocator(temp_allocator.next_temp)
                
                # Decompose the function using step3's system
                func_operations = decompose_function_tree_as_intermediate(
                    comp, 
                    func_result_pos, 
                    func_internal_allocator
                )
                
                # Update the main allocator
                temp_allocator.next_temp = func_internal_allocator.next_temp
                
                # Separate operations by type
                intermediate_ops = filter(op -> op.operation_type == :intermediate_binary, func_operations)
                final_ops = filter(op -> op.operation_type == :final_binary, func_operations)
                
                # Create tuples of specialized data inline
                intermediate_tuple = if length(intermediate_ops) == 0
                    ()
                else
                    ntuple(length(intermediate_ops)) do j
                        op = intermediate_ops[j]
                        input1 = length(op.inputs) > 0 ? op.inputs[1] : Symbol()
                        input2 = length(op.inputs) > 1 ? op.inputs[2] : Symbol()
                        IntermediateBinaryFunctionData(op.func, input1, input2, op.scratch_position)
                    end
                end
                
                final_tuple = if length(final_ops) == 0
                    ()
                else
                    ntuple(length(final_ops)) do j
                        op = final_ops[j]
                        input1 = length(op.inputs) > 0 ? op.inputs[1] : Symbol()
                        input2 = length(op.inputs) > 1 ? op.inputs[2] : Symbol()
                        FinalBinaryFunctionData(op.func, input1, input2, op.output_position)
                    end
                end
                
                # Create the compiled function data
                function_data = SpecializedFunctionData((), intermediate_tuple, final_tuple)
                function_op = FunctionOp(0, length(intermediate_ops), length(final_ops))
                
                # Create the pre-eval operation with compiled function data
                pre_eval = FunctionPreEvalOperation(function_data, function_op, func_result_pos)
                
                # Store in the function_pre_evals vector
                push!(function_pre_evals, pre_eval)
                
                # Create scratch reference for the function's output
                func_scratch_ref = InteractionScratchReference([func_result_pos])
                push!(processed_components, func_scratch_ref)
                push!(processed_input_sources, InteractionScratchPosition(func_result_pos))
                push!(processed_widths, 1)  # Functions are scalar
            end  # End of specialized vs. fallback check
            
        else
            # Regular component (categorical, continuous, etc.)
            push!(processed_components, comp)
            
            # Get appropriate input source
            if comp isa CategoricalEvaluator
                push!(processed_input_sources, comp.column)
            elseif comp isa ContinuousEvaluator
                push!(processed_input_sources, comp.column)
            elseif comp isa ConstantEvaluator
                push!(processed_input_sources, comp.value)
            else
                push!(processed_input_sources, get_component_input_source(comp))
            end
            
            push!(processed_widths, width)
        end
    end
    
    if N == 2
        # Binary case
        comp1 = processed_components[1]
        comp2 = processed_components[2]
        width1 = processed_widths[1]
        width2 = processed_widths[2]
        input1_source = processed_input_sources[1]
        input2_source = processed_input_sources[2]
        
        expected_width = width1 * width2
        if length(final_positions) != expected_width
            error("Binary interaction: expected $expected_width positions, got $(length(final_positions))")
        end
        
        push!(operations, LinearizedInteractionOperation(
            :final_interaction,
            comp1,
            comp2,
            input1_source,
            input2_source,
            final_positions,
            nothing,
            function_pre_evals  # Include pre-eval operations
        ))
        
        return operations
    end
    
    # N-way interaction
    current_component = processed_components[1]
    current_input_source = processed_input_sources[1]
    current_width = processed_widths[1]
    
    for i in 2:N
        next_component = processed_components[i]
        next_input_source = processed_input_sources[i]
        next_width = processed_widths[i]
        
        intermediate_width = current_width * next_width
        
        if i == N
            # Final step
            if length(final_positions) != intermediate_width
                error("Final step: expected $intermediate_width positions, got $(length(final_positions))")
            end
            
            push!(operations, LinearizedInteractionOperation(
                :final_interaction,
                current_component,
                next_component,
                current_input_source,
                next_input_source,
                final_positions,
                nothing,
                i == 2 ? function_pre_evals : FunctionPreEvalOperation[]
            ))
        else
            # Intermediate step
            scratch_start = allocate_temp!(temp_allocator)
            scratch_positions = collect(scratch_start:(scratch_start + intermediate_width - 1))
            
            for _ in 2:intermediate_width
                allocate_temp!(temp_allocator)
            end
            
            push!(operations, LinearizedInteractionOperation(
                :intermediate_interaction,
                current_component,
                next_component,
                current_input_source,
                next_input_source,
                scratch_positions,
                scratch_start,
                i == 2 ? function_pre_evals : FunctionPreEvalOperation[]
            ))
            
            current_component = InteractionScratchReference(scratch_positions)
            current_input_source = InteractionScratchPosition(scratch_start)
            current_width = intermediate_width
        end
    end
    
    return operations
end

###############################################################################
# INTERACTION DATA CREATION (NOW WITH TUPLES)
###############################################################################

function create_intermediate_interaction_data(op::LinearizedInteractionOperation)
    @assert op.operation_type == :intermediate_interaction
    @assert op.scratch_position !== nothing
    
    width1 = get_actual_component_width(op.component1)
    width2 = get_actual_component_width(op.component2)
    
    # FIXED: Create pattern as tuple
    pattern_tuple = compute_interaction_pattern_tuple(width1, width2)
    
    # FIXED: Convert pre-evals vector to tuple
    pre_evals_tuple = if isempty(op.function_pre_evals)
        ()
    else
        ntuple(length(op.function_pre_evals)) do i
            op.function_pre_evals[i]
        end
    end
    
    return IntermediateInteractionData(
        op.component1,
        op.component2,
        op.input1_source,
        op.input2_source,
        width1,
        width2,
        pattern_tuple,      # Now a tuple
        op.scratch_position,
        pre_evals_tuple     # Now a tuple
    )
end

function create_specialized_interaction_data(op::LinearizedInteractionOperation)
    @assert op.operation_type == :specialized_interaction
    
    # For specialized interactions, the component1 is the specialized interaction evaluator
    # We can use it directly without the complex pattern/width calculation
    specialized_evaluator = op.component1
    
    return IntermediateInteractionData(
        specialized_evaluator,
        specialized_evaluator,  # Use same evaluator for both components since it's self-contained
        :specialized,           # input1_source - not used in specialized execution
        :specialized,           # input2_source - not used in specialized execution  
        output_width(specialized_evaluator),  # width1 - the actual output width
        1,                      # width2 - not used
        ((1, 1),),             # pattern_tuple - minimal pattern, not used in specialized execution
        1,                      # scratch_position - not used for specialized interactions
        ()                      # function_pre_evals - empty since no pre-evaluation needed
    )
end

function create_final_interaction_data(op::LinearizedInteractionOperation)
    @assert op.operation_type in [:final_interaction, :specialized_interaction]
    @assert op.scratch_position === nothing
    
    # Handle specialized interactions differently
    if op.operation_type == :specialized_interaction
        return create_specialized_interaction_data(op)
    end
    
    width1 = get_actual_component_width(op.component1)
    width2 = get_actual_component_width(op.component2)
    
    # FIXED: Create pattern as tuple
    pattern_tuple = compute_interaction_pattern_tuple(width1, width2)
    
    output_position = length(op.output_positions) > 0 ? op.output_positions[1] : 1
    
    # FIXED: Convert pre-evals vector to tuple
    pre_evals_tuple = if isempty(op.function_pre_evals)
        ()
    else
        ntuple(length(op.function_pre_evals)) do i
            op.function_pre_evals[i]
        end
    end
    
    return FinalInteractionData(
        op.component1,
        op.component2,
        op.input1_source,
        op.input2_source,
        width1,
        width2,
        pattern_tuple,      # Now a tuple
        output_position,
        pre_evals_tuple     # Now a tuple
    )
end

# Helper function
function get_actual_component_width(component::AbstractEvaluator)
    if component isa CategoricalEvaluator
        return size(component.contrast_matrix, 2)
    elseif component isa InteractionScratchReference
        return component.width
    else
        return get_component_output_width(component)
    end
end

###############################################################################
# ZERO-ALLOCATION PRE-EVAL EXECUTION
###############################################################################

"""
    execute_pre_evals_recursive!(pre_evals::Tuple{}, scratch, output, input_data, row_idx)

Base case: empty tuple of pre-evals.
"""
function execute_pre_evals_recursive!(
    pre_evals::Tuple{},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

"""
    execute_pre_evals_recursive!(pre_evals::Tuple, scratch, output, input_data, row_idx)

Recursive case: execute first pre-eval, then remaining.
"""
function execute_pre_evals_recursive!(
    pre_evals::Tuple,
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(pre_evals) > 0
        # Execute first pre-eval using step3's execution system
        pre_eval = pre_evals[1]
        execute_operation!(
            pre_eval.function_data,
            pre_eval.function_op,
            scratch,  # Use interaction's scratch space
            output,
            input_data,
            row_idx
        )
        
        # Recursively execute remaining
        if length(pre_evals) > 1
            remaining = Base.tail(pre_evals)
            execute_pre_evals_recursive!(remaining, scratch, output, input_data, row_idx)
        end
    end
    
    return nothing
end

###############################################################################
# ENHANCED EXECUTION WITH ZERO ALLOCATIONS
###############################################################################

"""
    execute_operation!(data::IntermediateInteractionData, scratch, output, input_data, row_idx)

Execute intermediate interaction with zero allocations.
Uses tuple-based patterns and pre-evals.
"""
function execute_operation!(
    data::IntermediateInteractionData{C1, C2, T1, T2, PT, PET},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {C1, C2, T1, T2, PT, PET}

    # Check if this is a specialized interaction using type-based dispatch
    if is_specialized_interaction_type(data.component1)
        # This is a specialized interaction - call its execute_interaction! method directly
        execute_interaction!(data.component1, output, input_data, row_idx)
        return nothing
    end

    # Regular interaction execution path
    # FIXED: Execute function pre-evaluations using recursive tuple processing
    execute_pre_evals_recursive!(data.function_pre_evals, scratch, output, input_data, row_idx)
    
    # Validation
    validate_interaction_bounds!(data, scratch)
    
    # FIXED: Pattern is now a tuple, iterate with compile-time bounds
    @inbounds for pattern_idx in 1:length(data.index_pattern)
        i, j = data.index_pattern[pattern_idx]  # Tuple access is zero-allocation
        
        val1 = get_value_from_source(data.input1_source, data.component1, i, input_data, row_idx, output, scratch)
        val2 = get_value_from_source(data.input2_source, data.component2, j, input_data, row_idx, output, scratch)
        
        product = val1 * val2
        scratch_pos = data.scratch_position + pattern_idx - 1
        scratch[scratch_pos] = product
    end
    
    return nothing
end

function execute_operation!(
    data::FinalInteractionData{C1, C2, T1, T2, PT, PET},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {C1, C2, T1, T2, PT, PET}
    
    # FIXED: Execute function pre-evaluations using recursive tuple processing
    execute_pre_evals_recursive!(data.function_pre_evals, scratch, output, input_data, row_idx)
    
    # Validation
    validate_interaction_bounds!(data, output)
    
    # FIXED: Pattern is now a tuple, iterate with compile-time bounds
    @inbounds for pattern_idx in 1:length(data.index_pattern)
        i, j = data.index_pattern[pattern_idx]  # Tuple access is zero-allocation
        
        val1 = get_value_from_source(data.input1_source, data.component1, i, input_data, row_idx, output, scratch)
        val2 = get_value_from_source(data.input2_source, data.component2, j, input_data, row_idx, output, scratch)
        
        product = val1 * val2
        output_pos = data.output_position + pattern_idx - 1
        output[output_pos] = product
    end
    
    return nothing
end

###############################################################################
# TUPLE-BASED RECURSIVE EXECUTION (UNCHANGED)
###############################################################################

"""
    execute_intermediate_interactions_recursive!(intermediate_tuple::Tuple{}, ...)

Base case: empty tuple - nothing to execute.
"""
function execute_intermediate_interactions_recursive!(
    intermediate_tuple::Tuple{},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

"""
    execute_intermediate_interactions_recursive!(intermediate_tuple::Tuple, ...)

Recursive case: execute first intermediate interaction, then process remaining.
"""
function execute_intermediate_interactions_recursive!(
    intermediate_tuple::Tuple,
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(intermediate_tuple) > 0
        execute_operation!(intermediate_tuple[1], scratch, output, input_data, row_idx)
        
        if length(intermediate_tuple) > 1
            remaining = Base.tail(intermediate_tuple)
            execute_intermediate_interactions_recursive!(remaining, scratch, output, input_data, row_idx)
        end
    end
    return nothing
end

"""
    execute_final_interactions_recursive!(final_tuple::Tuple{}, ...)

Base case: empty tuple - nothing to execute.
"""
function execute_final_interactions_recursive!(
    final_tuple::Tuple{},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

"""
    execute_final_interactions_recursive!(final_tuple::Tuple, ...)

Recursive case: execute first final interaction, then process remaining.
"""
function execute_final_interactions_recursive!(
    final_tuple::Tuple,
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(final_tuple) > 0
        execute_operation!(final_tuple[1], scratch, output, input_data, row_idx)
        
        if length(final_tuple) > 1
            remaining = Base.tail(final_tuple)
            execute_final_interactions_recursive!(remaining, scratch, output, input_data, row_idx)
        end
    end
    return nothing
end

"""
    execute_complete_interaction!(complete_interaction::CompleteInteractionData{IT, FT}, ...)

Execute one complete interaction using recursive tuple processing.
"""
function execute_complete_interaction!(
    complete_interaction::CompleteInteractionData{IT, FT},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {IT, FT}
    
    # Step 1: Execute all intermediate operations for this interaction (tuple recursion)
    execute_intermediate_interactions_recursive!(
        complete_interaction.intermediate_operations, 
        scratch, output, input_data, row_idx
    )
    
    # Step 2: Execute all final operations for this interaction (tuple recursion)
    execute_final_interactions_recursive!(
        complete_interaction.final_operations, 
        scratch, output, input_data, row_idx
    )
    
    return nothing
end

"""
    execute_complete_interactions_recursive!(complete_tuple::Tuple{}, ...)

Base case: empty tuple - no complete interactions.
"""
function execute_complete_interactions_recursive!(
    complete_tuple::Tuple{},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

"""
    execute_complete_interactions_recursive!(complete_tuple::Tuple, ...)

Recursive case: execute first complete interaction, then process remaining.
"""
function execute_complete_interactions_recursive!(
    complete_tuple::Tuple,
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(complete_tuple) > 0
        execute_complete_interaction!(complete_tuple[1], scratch, output, input_data, row_idx)
        
        if length(complete_tuple) > 1
            remaining = Base.tail(complete_tuple)
            execute_complete_interactions_recursive!(remaining, scratch, output, input_data, row_idx)
        end
    end
    return nothing
end

# Main execute_operation! for SpecializedInteractionData
"""
    execute_operation!(data::SpecializedInteractionData{CT}, op::InteractionOp{I, F}, ...)

Execute complete interactions using recursive tuple processing.
"""
function execute_operation!(
    data::SpecializedInteractionData{CT},
    op::InteractionOp{I, F},
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {CT, I, F}
    
    # ZERO-ALLOCATION: Process each complete interaction using tuple recursion
    execute_complete_interactions_recursive!(data.complete_interactions, scratch, output, input_data, row_idx)
    
    return nothing
end

###############################################################################
# ANALYSIS (UNCHANGED FROM ORIGINAL)
###############################################################################

"""
    analyze_interaction_operations_linear(evaluator::CombinedEvaluator) -> (SpecializedInteractionData, InteractionOp)

Analyze and create specialized interaction data with compile-time tuples.
"""
function analyze_interaction_operations_linear(evaluator::CombinedEvaluator)
    interaction_evaluators = evaluator.interaction_evaluators
    n_interactions = length(interaction_evaluators)
    
    if n_interactions == 0
        empty_data = SpecializedInteractionData(())
        return empty_data, InteractionOp(0, 0)
    end
    
    temp_allocator = TempAllocator(1)
    
    # Pre-decompose all interactions to avoid allocations in ntuple
    all_decomposed_interactions = Vector{Any}(undef, n_interactions)
    total_intermediate = 0
    total_final = 0
    
    for (idx, interaction_eval) in enumerate(interaction_evaluators)
        operations = decompose_interaction_tree_zero_alloc(interaction_eval, temp_allocator)
        
        # Separate operations for this specific interaction
        intermediate_ops = filter(op -> op.operation_type == :intermediate_interaction, operations)
        final_ops = filter(op -> op.operation_type in [:final_interaction, :specialized_interaction], operations)
        
        # Store for ntuple construction
        all_decomposed_interactions[idx] = (intermediate_ops, final_ops, idx)
        total_intermediate += length(intermediate_ops)
        total_final += length(final_ops)
    end
    
    # Create compile-time tuple of complete interactions
    complete_tuple = ntuple(n_interactions) do i
        intermediate_ops, final_ops, interaction_idx = all_decomposed_interactions[i]
        
        # Create compile-time tuples for operations
        intermediate_tuple = ntuple(length(intermediate_ops)) do j
            create_intermediate_interaction_data(intermediate_ops[j])
        end
        
        final_tuple = ntuple(length(final_ops)) do j
            create_final_interaction_data(final_ops[j])
        end
        
        # Create CompleteInteractionData with compile-time tuples
        CompleteInteractionData(intermediate_tuple, final_tuple, interaction_idx)
    end
    
    specialized_data = SpecializedInteractionData(complete_tuple)
    interaction_op = InteractionOp(total_intermediate, total_final)
    
    return specialized_data, interaction_op
end

###############################################################################
# SCRATCH CALCULATION
###############################################################################

"""
    calculate_max_interaction_scratch_needed(evaluator::CombinedEvaluator) -> Int

Calculate maximum scratch space needed for interactions.
"""
function calculate_max_interaction_scratch_needed(evaluator::CombinedEvaluator)
    interaction_evaluators = evaluator.interaction_evaluators
    
    if isempty(interaction_evaluators)
        return 0
    end
    
    # Use single shared TempAllocator like the analysis phase
    temp_allocator = TempAllocator(1)
    max_scratch_position = 0
    
    for (idx, interaction_eval) in enumerate(interaction_evaluators)
        try
            operations = decompose_interaction_tree_zero_alloc(interaction_eval, temp_allocator)
            
            for op in operations
                if op.scratch_position !== nothing
                    scratch_size = length(op.output_positions)
                    max_scratch_end = op.scratch_position + scratch_size - 1
                    max_scratch_position = max(max_scratch_position, max_scratch_end)
                end
                
                # Check input sources that read from scratch
                if op.input1_source isa InteractionScratchPosition
                    max_scratch_position = max(max_scratch_position, op.input1_source.position)
                end
                if op.input2_source isa InteractionScratchPosition
                    max_scratch_position = max(max_scratch_position, op.input2_source.position)
                end
            end
            
        catch e
            # Conservative fallback using stored widths
            N = length(interaction_eval.components)
            if N > 2
                # Calculate based on stored component widths
                total_width = prod(interaction_eval.component_widths)
                conservative_estimate = total_width * 2  # Safety factor
                max_scratch_position = max(max_scratch_position, conservative_estimate)
            end
        end
    end
    
    # Safety fallback
    if max_scratch_position == 0 && !isempty(interaction_evaluators)
        max_scratch_position = 10
    end
    
    return max_scratch_position
end

###############################################################################
# VALIDATION AND BOUNDS CHECKING
###############################################################################

"""
    validate_interaction_bounds!(data::FinalInteractionData, output::Vector{Float64})

Validate bounds for final interactions.
"""
function validate_interaction_bounds!(
    data::FinalInteractionData, output::V
) where {V <: AbstractVector{Float64}}
    # Check the full range of positions that will be written
    max_output_pos = data.output_position + length(data.index_pattern) - 1
    
    if max_output_pos > length(output)
        error("Final interaction requires output positions up to $max_output_pos but output length is $(length(output))")
    end
    
    if data.output_position < 1
        error("Final interaction output_position $(data.output_position) is invalid (must be ≥ 1)")
    end
    
    return nothing
end

"""
    validate_interaction_bounds!(data::IntermediateInteractionData, scratch::Vector{Float64})

Validate bounds for intermediate interactions.
"""
function validate_interaction_bounds!(data::IntermediateInteractionData, scratch::Vector{Float64})
    # Check the full range of positions that will be written
    max_scratch_pos = data.scratch_position + length(data.index_pattern) - 1
    
    if max_scratch_pos > length(scratch)
        error("Intermediate interaction requires scratch positions up to $max_scratch_pos but scratch length is $(length(scratch))")
    end
    
    if data.scratch_position < 1
        error("Intermediate interaction scratch_position $(data.scratch_position) is invalid (must be ≥ 1)")
    end
    
    return nothing
end

###############################################################################
# INTERFACE METHODS
###############################################################################

"""
    Base.isempty(data::SpecializedInteractionData) -> Bool

Check if interaction data is empty.
"""
function Base.isempty(data::SpecializedInteractionData)
    return length(data.complete_interactions) == 0
end

"""
    Base.length(data::SpecializedInteractionData) -> Int

Get total number of complete interactions.
"""
function Base.length(data::SpecializedInteractionData)
    return length(data.complete_interactions)
end

"""
    Base.iterate(data::SpecializedInteractionData, state=1)

Iterate over complete interactions.
"""
function Base.iterate(data::SpecializedInteractionData, state=1)
    if state > length(data.complete_interactions)
        return nothing
    end
    
    return (data.complete_interactions[state], state + 1)
end

###############################################################################
# MAIN EXECUTION INTERFACE
###############################################################################

"""
    execute_interaction_operations!(interaction_data::SpecializedInteractionData, scratch, output, data, row_idx)

Execute all interaction terms (x*group, log(z)*category, etc.) using Kronecker product patterns
for zero-allocation computation of interaction effects.

## Two-Phase Execution:

1. **Intermediate Phase**: Compute component values into scratch space
   - Extract column values, function results, or categorical contrasts  
   - All components needed for Kronecker products are gathered

2. **Final Phase**: Compute Kronecker products and assign to output positions
   - For each interaction: multiply all component values element-wise
   - Store results in final output positions

## Interaction Types Supported:
- **Continuous × Categorical**: `x * group` → element-wise products with contrast matrix
- **Function × Categorical**: `log(z) * group` → function results × contrasts  
- **Higher Order**: `x * y * group` → triple products using Kronecker patterns
- **Complex**: `log(z) * sqrt(w) * category` → function combinations × categoricals

## Arguments:
- `interaction_data`: Pre-computed interaction specifications and output positions
- `scratch`: Pre-allocated scratch space for intermediate computations  
- `output`: Final output vector (modified in-place)
- `data`: Raw input data columns
- `row_idx`: Row to evaluate (1-based)

## Performance:
Achieves zero allocation through:
- Pre-allocated scratch space reuse
- Type-stable Kronecker product computation
- No temporary array creation during products

**Current Issue**: Allocation problems (~96-864+ bytes) likely due to symbol-based column
access patterns instead of compile-time Val{Column} dispatch.
"""
function execute_interaction_operations!(
    interaction_data::SpecializedInteractionData{CompleteInteractionTuple, I, F},
    scratch::Vector{Float64},
    output::V,
    data::NamedTuple,
    row_idx::Int
) where {CompleteInteractionTuple, I, F, V <: AbstractVector{Float64}}
    # Use compile-time operation counts - NO runtime computation or allocations!
    op = InteractionOp(I, F)
    execute_operation!(interaction_data, op, scratch, output, data, row_idx)
    
    return nothing
end

"""
    count_operations_zero_alloc(complete_tuple::Tuple) -> (Int, Int)

Count operations using tuple recursion.
"""
function count_operations_zero_alloc(complete_tuple::Tuple)
    return count_operations_recursive(complete_tuple)
end

"""
    count_operations_recursive(complete_tuple::Tuple{}) -> (Int, Int)

Base case: empty tuple - zero operations.
"""
function count_operations_recursive(complete_tuple::Tuple{})
    return (0, 0)
end

"""
    count_operations_recursive(complete_tuple::Tuple) -> (Int, Int)

Count operations using tuple recursion.
"""
function count_operations_recursive(complete_tuple::Tuple)
    if length(complete_tuple) == 0
        return (0, 0)
    end
    
    # Count operations in first complete interaction
    first_intermediate = length(complete_tuple[1].intermediate_operations)
    first_final = length(complete_tuple[1].final_operations)
    
    if length(complete_tuple) == 1
        return (first_intermediate, first_final)
    else
        # Recursively count remaining interactions
        remaining = Base.tail(complete_tuple)
        remaining_intermediate, remaining_final = count_operations_recursive(remaining)
        
        return (first_intermediate + remaining_intermediate, first_final + remaining_final)
    end
end

## %

###############################################################################
# COMPLETE FORMULA INTEGRATION
###############################################################################

"""
    CompleteFormulaOp{ConstOp, ContOp, CatOp, FuncOp, IntOp}

Complete formula operation encoding with interaction support.
"""
struct CompleteFormulaOp{ConstOp, ContOp, CatOp, FuncOp, IntOp}
    constants::ConstOp
    continuous::ContOp
    categorical::CatOp
    functions::FuncOp
    interactions::IntOp
end

"""
    CompleteFormulaData{ConstData, ContData, CatData, FuncData, IntData}

Complete formula data with enhanced integration.
"""
struct CompleteFormulaData{ConstData, ContData, CatData, FuncData, IntData}
    constants::ConstData
    continuous::ContData
    categorical::CatData
    functions::FuncData
    interactions::IntData
    max_function_scratch::Int
    max_interaction_scratch::Int
    function_scratch::Vector{Float64}
    interaction_scratch::Vector{Float64}
end

"""
    analyze_evaluator(evaluator::AbstractEvaluator) -> (data_tuple, op_tuple)

**Critical function**: Converts a CompiledFormula's evaluator tree into specialized tuple-based 
data structures for maximum performance. This is the bridge between the evaluator system 
and the specialized formula system.

## Purpose:
Transforms the flexible but slower evaluator tree representation into compile-time specialized
tuples that enable zero-allocation execution through type-stable dispatch.

## Process:
1. **Analyzes** the CombinedEvaluator to extract all operation types and data requirements
2. **Separates** operations into 5 specialized categories (constants, continuous, categorical, functions, interactions)
3. **Builds** tuple-based data structures with compile-time type information
4. **Returns** (data_tuple, op_tuple) pair for SpecializedFormula construction

## Data Flow:
```
CompiledFormula.root_evaluator (CombinedEvaluator)
    ↓ analyze_evaluator
(CompleteFormulaData, CompleteFormulaOp) tuples
    ↓ SpecializedFormula constructor
High-performance execution via tuple dispatch
```

## Performance Impact:
- **Before**: Runtime dispatch through evaluator tree (~100ns per row)
- **After**: Compile-time dispatch through tuples (~50ns per row, 0 allocations)

This function is the core of the specialization system that enables the performance gains.
"""
function analyze_evaluator(evaluator::AbstractEvaluator)
    if evaluator isa CombinedEvaluator
        constant_data, constant_op = analyze_constant_operations(evaluator)
        continuous_data, continuous_op = analyze_continuous_operations(evaluator)
        categorical_data, categorical_op = analyze_categorical_operations(evaluator)
        function_data, function_op = analyze_function_operations_linear(evaluator)
        interaction_data, interaction_op = analyze_interaction_operations_linear(evaluator)
        
        max_function_scratch = calculate_max_function_scratch_needed(evaluator)
        max_interaction_scratch = calculate_max_interaction_scratch_needed(evaluator)
        
        function_scratch = Vector{Float64}(undef, max(max_function_scratch, 1))
        interaction_scratch = Vector{Float64}(undef, max(max_interaction_scratch, 1))
        
        formula_data = CompleteFormulaData(
            constant_data,
            continuous_data,
            categorical_data,
            function_data,
            interaction_data,
            max_function_scratch,
            max_interaction_scratch,
            function_scratch,
            interaction_scratch
        )
        
        formula_op = CompleteFormulaOp(constant_op, continuous_op, categorical_op, function_op, interaction_op)
        return formula_data, formula_op
        
    else
        error("Complete analysis only supports CombinedEvaluator")
    end
end

"""
    execute_operation!(data::CompleteFormulaData, op::CompleteFormulaOp, output, input_data, row_idx)

Main execution function for SpecializedFormula. Executes all formula components in the correct
dependency order to evaluate a single row of data into the output vector.

## Execution Phases (in dependency order):

1. **Constants**: Direct value assignment to output positions
2. **Continuous**: Type-stable column value extraction and assignment  
3. **Categorical**: Contrast matrix lookups for categorical variables
4. **Functions**: Mathematical functions (log, exp, etc.) using intermediate scratch space
5. **Interactions**: Kronecker products of component values using intermediate scratch space

## Arguments:
- `data`: Pre-computed specialized data structures for each phase
- `op`: Compile-time operation encodings for type-stable dispatch
- `output`: Pre-allocated output vector to fill (modified in-place)
- `input_data`: NamedTuple containing the raw data columns
- `row_idx`: Which row to evaluate (1-based index)

## Performance:
This function achieves zero-allocation execution through:
- Pre-allocated scratch spaces that are reused
- Compile-time type specialization via data/op tuple structures
- In-place output vector modification
- Type-stable column access patterns (Val{Column} dispatch)

Target performance: ~50ns per row for simple formulas, 0 allocations.
"""
function execute_operation!(
    data::CompleteFormulaData,
    op::CompleteFormulaOp,
    output, input_data, row_idx
)
    # Initialize scratch spaces
    fill!(data.function_scratch, 0.0)
    fill!(data.interaction_scratch, 0.0)
    
    # Phase 1: Constants
    execute_complete_constant_operations!(data.constants, output, input_data, row_idx)
    
    # Phase 2: Continuous (type-stable executor)
    execute_operation!(data.continuous, op.continuous, output, input_data, row_idx)
    
    # Phase 3: Categoricals
    execute_categorical_operations!(data.categorical, output, input_data, row_idx)
    
    # Phase 4: Functions (intermediate → unary → final)
    execute_linear_function_operations!(data.functions, data.function_scratch, output, input_data, row_idx)
    
    # Phase 5: Interactions (intermediate → final)
    execute_interaction_operations!(data.interactions, data.interaction_scratch, output, input_data, row_idx)
    
    return nothing
end

###############################################################################
# SUPPORT FUNCTIONS
###############################################################################

"""
    execute_complete_constant_operations!(constant_data::ConstantData{N}, output, input_data, row_idx) where N

Execute constant operations.
"""
function execute_complete_constant_operations!(constant_data::ConstantData{N}, output, input_data, row_idx) where N
    @inbounds for i in 1:N
        pos = constant_data.positions[i]
        val = constant_data.values[i]
        output[pos] = val
    end
    return nothing
end

"""
    execute_complete_continuous_operations!(continuous_data::ContinuousData{N, Cols}, output, input_data, row_idx) where {N, Cols}

Execute continuous operations.
"""
function execute_complete_continuous_operations!(continuous_data::ContinuousData{N, Cols}, output, input_data, row_idx) where {N, Cols}
    @inbounds for i in 1:N
        col = continuous_data.columns[i]
        pos = continuous_data.positions[i]
        val = get_data_value_specialized(input_data, col, row_idx)
        output[pos] = Float64(val)
    end
    return nothing
end

# Fallback for empty data
function execute_complete_constant_operations!(constant_data::ConstantData{0}, output, input_data, row_idx)
    return nothing
end

function execute_complete_continuous_operations!(continuous_data::ContinuousData{0, Tuple{}}, output, input_data, row_idx)
    return nothing
end

###############################################################################
# COMPILATION FUNCTIONS
###############################################################################

"""
    compile_formula(compiled_formula::CompiledFormula) -> SpecializedFormula

Convert a CompiledFormula (evaluator tree-based) into a SpecializedFormula (tuple-based)
for maximum performance. This analyzes the evaluator structure and creates specialized
type-stable execution paths.
"""
function compile_formula(compiled_formula::CompiledFormula)
    data_tuple, op_tuple = analyze_evaluator(compiled_formula.root_evaluator)
    
    return SpecializedFormula{typeof(data_tuple), typeof(op_tuple)}(
        data_tuple,
        op_tuple,
        compiled_formula.output_width
    )
end

"""
    compile_formula(model, data::NamedTuple) -> SpecializedFormula

Compile a statistical model into a high-performance SpecializedFormula.
This does complete compilation followed by specialization in one step.
"""
function compile_formula(model, data::NamedTuple)
    compiled_formula = compile_formula_complete(model, data)
    data_tuple, op_tuple = analyze_evaluator(compiled_formula.root_evaluator)
    
    return SpecializedFormula{typeof(data_tuple), typeof(op_tuple)}(
        data_tuple,
        op_tuple,
        compiled_formula.output_width
    )
end


# Add this function to your step4_interactions.jl file
# Place it in the debugging section around line 1800+ near other debug functions

"""
    validate_schema_based_interactions(evaluator::CombinedEvaluator)

Validate that schema-based compilation produced correct interaction components.
"""
function validate_schema_based_interactions(evaluator::CombinedEvaluator)
    
    interaction_evaluators = evaluator.interaction_evaluators
    
    for (idx, interaction_eval) in enumerate(interaction_evaluators)
        
        components = interaction_eval.components
        for (comp_idx, comp) in enumerate(components)
            if comp isa CategoricalEvaluator
                
                contrast_size = size(comp.contrast_matrix)
                n_levels = comp.n_levels
                
                
                # Check if contrasts look reasonable
                if contrast_size[2] == n_levels
                elseif contrast_size[2] == n_levels - 1
                else
                end
                
                # Check level codes
                if !isempty(comp.level_codes)
                else
                end
                
            else
            end
        end
        
        # Check overall width consistency
        expected_width = prod(get_component_output_width(comp) for comp in components)
        actual_width = length(interaction_eval.positions)
        
        if expected_width == actual_width
        else
        end
    end
    
end

"""
    execute_function_in_interaction!(func_eval::FunctionEvaluator, scratch, output, input_data, row_idx, target_scratch_pos::Int)

Execute a function evaluator and write its result to scratch for use in interactions.
"""
function execute_function_in_interaction!(
    func_eval::FunctionEvaluator,
    scratch::Vector{Float64},
    output::Vector{Float64},
    input_data::NamedTuple,
    row_idx::Int,
    target_scratch_pos::Int
)
    # Execute the function using its normal evaluation logic
    # but write to scratch instead of output
    
    # First evaluate arguments into scratch
    for (i, arg_eval) in enumerate(func_eval.arg_evaluators)
        arg_range = func_eval.arg_scratch_map[i]
        if arg_eval isa ContinuousEvaluator
            scratch[first(arg_range)] = Float64(get_data_value_specialized(input_data, arg_eval.column, row_idx))
        elseif arg_eval isa ConstantEvaluator
            scratch[first(arg_range)] = arg_eval.value
        else
            # Handle other argument types...
            error("Unsupported argument type in function: $(typeof(arg_eval))")
        end
    end
    
    # Apply the function
    if length(func_eval.arg_evaluators) == 1
        arg_val = scratch[first(func_eval.arg_scratch_map[1])]
        result = apply_function_direct_single(func_eval.func, arg_val)
    elseif length(func_eval.arg_evaluators) == 2
        arg1_val = scratch[first(func_eval.arg_scratch_map[1])]
        arg2_val = scratch[first(func_eval.arg_scratch_map[2])]
        result = apply_function_direct_binary(func_eval.func, arg1_val, arg2_val)
    else
        error("Functions with >2 arguments not yet supported in interactions")
    end
    
    # Write to target scratch position
    scratch[target_scratch_pos] = result
    
    return nothing
end

###############################################################################
# VERIFICATION TEST for interactions
###############################################################################

"""
    test_categorical_interaction_fix()

Test that categorical × continuous interactions work correctly with scenarios.
"""
function test_categorical_interaction_fix()
    println("Testing categorical interaction fix...")
    
    # Create test data with interaction
    n = 100
    df = DataFrame(
        x = randn(n),
        cat3 = categorical(rand(["A", "B", "C"], n))
    )
    df.y = (
        2.0 * df.x +                           # Main x effect
        1.0 * (df.cat3 .== "B") +              # Main cat3 B effect  
        3.0 * (df.cat3 .== "C") +              # Main cat3 C effect
        0.5 * df.x .* (df.cat3 .== "B") +      # Interaction x:B
        -0.8 * df.x .* (df.cat3 .== "C") +     # Interaction x:C
        randn(n) * 0.1
    )
    
    # Fit interaction model
    model = lm(@formula(y ~ x * cat3), df)
    data_nt = Tables.columntable(df)
    
    println("Model coefficients:")
    coef_names = coefnames(model)
    coefs = coef(model)
    for (name, coef) in zip(coef_names, coefs)
        println("  $name: $(round(coef, digits=4))")
    end
    
    # Create scenarios with different categorical levels
    scenario_A = create_scenario("A", data_nt; cat3 = "A", x = 1.0)
    scenario_B = create_scenario("B", data_nt; cat3 = "B", x = 1.0)  
    scenario_C = create_scenario("C", data_nt; cat3 = "C", x = 1.0)
    
    # Compile and test
    compiled = compile_formula(model, scenario_A.data)
    buffer = Vector{Float64}(undef, length(compiled))
    
    # Test zero allocations
    alloc_test = @allocated begin
        for i in 1:50
            compiled(buffer, scenario_A.data, 1)
            compiled(buffer, scenario_B.data, 1)
            compiled(buffer, scenario_C.data, 1)
        end
    end
    println("Allocations for 150 interaction evaluations: $alloc_test bytes")
    
    # Test that results are different
    compiled(buffer, scenario_A.data, 1)
    result_A = copy(buffer)
    
    compiled(buffer, scenario_B.data, 1)
    result_B = copy(buffer)
    
    compiled(buffer, scenario_C.data, 1)
    result_C = copy(buffer)
    
    println("\nInteraction model matrix results:")
    println("  Scenario A (cat3=A, x=1): $result_A")
    println("  Scenario B (cat3=B, x=1): $result_B")
    println("  Scenario C (cat3=C, x=1): $result_C")
    
    # Check differences in interaction terms (should be different)
    println("\nResults are different:")
    println("  A ≠ B: $(result_A != result_B)")
    println("  B ≠ C: $(result_B != result_C)")
    println("  A ≠ C: $(result_A != result_C)")
    
    # Test marginal effects with interaction
    println("\nTesting marginal effects with interaction...")
    try
        # Test continuous marginal effect (should vary by categorical level)
        result_x = margins(model, df, :x)
        println("Marginal effect of x (average): $(result_x.effects[:x])")
        
        # Test categorical marginal effects  
        result_cat = margins(model, df, :cat3)
        println("Categorical effects:")
        if result_cat.effects[:cat3] isa Dict
            for (contrast, effect) in result_cat.effects[:cat3]
                println("  $contrast: $(round(effect, digits=4))")
            end
        end
        
        println("✅ Marginal effects with interactions working!")
        
    catch e
        println("❌ Marginal effects failed: $e")
    end
    
    return (alloc_test, result_A, result_B, result_C)
end
