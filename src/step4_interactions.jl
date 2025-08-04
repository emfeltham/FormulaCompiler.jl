# step4_interactions.jl - CLEAN FINAL VERSION
# N-way interaction decomposition integrated with existing scratch system

###############################################################################
# CORE BINARY INTERACTION DATA TYPES
###############################################################################

struct BinaryInteractionData{Comp1Type, Comp2Type, Pattern}
    component1::Comp1Type                    
    component2::Comp2Type                    
    width1::Int                              
    width2::Int                              
    index_pattern::Pattern                   
    output_positions::Vector{Int}            
    
    function BinaryInteractionData(
        comp1::C1, comp2::C2, w1::Int, w2::Int, 
        pattern::P, positions::Vector{Int}
    ) where {C1, C2, P}
        new{C1, C2, P}(comp1, comp2, w1, w2, pattern, positions)
    end
end

struct SpecializedInteractionData{BinaryTuple}
    binary_interactions::BinaryTuple         
    
    function SpecializedInteractionData(binary_tuple::T) where T
        new{T}(binary_tuple)
    end
end

struct InteractionOp{M}
    function InteractionOp(n_binary::Int)
        new{n_binary}()
    end
end

InteractionOp() = InteractionOp(0)

###############################################################################
# INTERACTION SCRATCH EVALUATOR (Following Function Pattern)
###############################################################################

struct InteractionScratchEvaluator <: AbstractEvaluator
    scratch_positions::Vector{Int}
    width::Int
    
    function InteractionScratchEvaluator(scratch_positions::Vector{Int})
        new(scratch_positions, length(scratch_positions))
    end
end

output_width(eval::InteractionScratchEvaluator) = eval.width
get_positions(eval::InteractionScratchEvaluator) = eval.scratch_positions
get_scratch_positions(eval::InteractionScratchEvaluator) = eval.scratch_positions
max_scratch_needed(eval::InteractionScratchEvaluator) = maximum(eval.scratch_positions)
get_component_output_width(eval::InteractionScratchEvaluator) = eval.width

###############################################################################
# SCRATCH POSITION MANAGEMENT
###############################################################################

mutable struct InteractionScratchAllocator
    next_position::Int
    allocated_positions::Vector{Int}
    
    function InteractionScratchAllocator()
        new(1, Int[])
    end
end

function allocate_interaction_scratch!(allocator::InteractionScratchAllocator, size::Int)
    positions = collect(allocator.next_position:(allocator.next_position + size - 1))
    allocator.next_position += size
    append!(allocator.allocated_positions, positions)
    return positions
end

###############################################################################
# COMPONENT VALUE ACCESS WITH SCRATCH SUPPORT
###############################################################################

@inline function get_component_value(
    component::ConstantEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64},
    scratch::Vector{Float64}
)
    return component.value
end

@inline function get_component_value(
    component::ContinuousEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64},
    scratch::Vector{Float64}
)
    return Float64(get_data_value_specialized(data, component.column, row_idx))
end

@inline function get_component_value(
    component::CategoricalEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64},
    scratch::Vector{Float64}
)
    level = component.level_codes[row_idx]
    level = clamp(level, 1, component.n_levels)
    return component.contrast_matrix[level, index]
end

@inline function get_component_value(
    component::FunctionEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64},
    scratch::Vector{Float64}
)
    return output[component.position]
end

@inline function get_component_value(
    component::InteractionScratchEvaluator,
    index::Int,
    data::NamedTuple,
    row_idx::Int,
    output::Vector{Float64},
    scratch::Vector{Float64}
)
    scratch_pos = component.scratch_positions[index]
    return scratch[scratch_pos]
end

###############################################################################
# BINARY INTERACTION PATTERN GENERATION
###############################################################################

function compute_binary_interaction_pattern(width1::Int, width2::Int)
    pattern = Tuple{Int, Int}[]
    sizehint!(pattern, width1 * width2)
    
    for i in 1:width1
        for j in 1:width2
            push!(pattern, (i, j))
        end
    end
    
    return pattern
end

function create_binary_interaction_data(
    comp1::AbstractEvaluator, 
    comp2::AbstractEvaluator, 
    output_positions::Vector{Int}
)
    width1 = get_component_output_width(comp1)
    width2 = get_component_output_width(comp2)
    
    expected_width = width1 * width2
    if length(output_positions) != expected_width
        error("Output positions length $(length(output_positions)) != expected width $expected_width")
    end
    
    pattern = compute_binary_interaction_pattern(width1, width2)
    
    return BinaryInteractionData(comp1, comp2, width1, width2, pattern, output_positions)
end

###############################################################################
# N-WAY DECOMPOSITION WITH SCRATCH
###############################################################################

function decompose_nway_with_scratch(
    interaction_eval::InteractionEvaluator{N},
    scratch_allocator::InteractionScratchAllocator
) where N
    
    # println("    DECOMPOSE_NWAY_WITH_SCRATCH DEBUG:")
    # println("      N = $N")
    # println("      Final positions: $(interaction_eval.positions)")
    
    if N == 2
        comp1, comp2 = interaction_eval.components
        return [create_binary_interaction_data(comp1, comp2, interaction_eval.positions)]
    end
    
    components = interaction_eval.components
    final_positions = interaction_eval.positions
    
    binary_ops = BinaryInteractionData[]
    current_component = components[1]
    
    for i in 2:N
        next_component = components[i]
        
        current_width = get_component_output_width(current_component)
        next_width = get_component_output_width(next_component)
        
        # println("      Step $(i-1): $(current_width) × $(next_width)")
        
        if i == N
            output_positions = final_positions
            # println("        FINAL → output positions $output_positions")
        else
            scratch_size = current_width * next_width
            scratch_positions = allocate_interaction_scratch!(scratch_allocator, scratch_size)
            output_positions = scratch_positions
            # println("        INTERMEDIATE → scratch positions $output_positions")
        end
        
        binary_data = create_binary_interaction_data(
            current_component,
            next_component,
            output_positions
        )
        push!(binary_ops, binary_data)
        
        if i < N
            current_component = InteractionScratchEvaluator(output_positions)
            # println("        Next iteration: InteractionScratchEvaluator with scratch positions $output_positions")
        end
    end
    
    # println("      Created $(length(binary_ops)) binary operations")
    return binary_ops
end

###############################################################################
# BINARY INTERACTION EXECUTION
###############################################################################

function execute_operation!(
    data::BinaryInteractionData{C1, C2, P},
    output::AbstractVector{Float64},
    scratch::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {C1, C2, P}
    
    @inbounds for output_idx in 1:length(data.index_pattern)
        i, j = data.index_pattern[output_idx]
        
        val1 = get_component_value(data.component1, i, input_data, row_idx, output, scratch)
        val2 = get_component_value(data.component2, j, input_data, row_idx, output, scratch)
        
        product = val1 * val2
        output_pos = data.output_positions[output_idx]
        
        # Write to scratch for intermediate results, output for final results
        if data.component1 isa InteractionScratchEvaluator || data.component2 isa InteractionScratchEvaluator
            if output_pos <= length(scratch)
                scratch[output_pos] = product
            else
                output[output_pos] = product
            end
        else
            output[output_pos] = product
        end
    end
    
    return nothing
end

###############################################################################
# RECURSIVE EXECUTION WITH SCRATCH
###############################################################################

function execute_binary_interactions_recursive!(
    binary_tuple::Tuple{},
    output::AbstractVector{Float64},
    scratch::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

function execute_binary_interactions_recursive!(
    binary_tuple::Tuple,
    output::AbstractVector{Float64},
    scratch::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(binary_tuple) > 0
        execute_operation!(binary_tuple[1], output, scratch, input_data, row_idx)
        
        if length(binary_tuple) > 1
            remaining = Base.tail(binary_tuple)
            execute_binary_interactions_recursive!(remaining, output, scratch, input_data, row_idx)
        end
    end
    return nothing
end

function execute_operation!(
    data::SpecializedInteractionData{BT},
    op::InteractionOp{M},
    output::AbstractVector{Float64},
    scratch::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {BT, M}
    
    execute_binary_interactions_recursive!(data.binary_interactions, output, scratch, input_data, row_idx)
    return nothing
end

###############################################################################
# MAIN EXECUTION INTERFACE
###############################################################################

function execute_interaction_operations!(
    interaction_data::SpecializedInteractionData,
    scratch::Vector{Float64},
    output::Vector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    n_binary = length(interaction_data.binary_interactions)
    op = InteractionOp(n_binary)
    
    execute_operation!(interaction_data, op, output, scratch, data, row_idx)
    return nothing
end

###############################################################################
# ANALYSIS FUNCTION
###############################################################################

function analyze_interaction_operations(evaluator::CombinedEvaluator)
    interaction_evaluators = evaluator.interaction_evaluators
    n_interactions = length(interaction_evaluators)
    
    # println("ANALYZE_INTERACTION_OPERATIONS WITH SCRATCH DEBUG:")
    # println("  Number of interactions: $n_interactions")
    
    if n_interactions == 0
        empty_data = SpecializedInteractionData(())
        return empty_data, InteractionOp(0)
    end
    
    scratch_allocator = InteractionScratchAllocator()
    all_binary_interactions = BinaryInteractionData[]
    
    for (idx, interaction_eval) in enumerate(interaction_evaluators)
        N = length(interaction_eval.components)
        # println("  Processing interaction $idx: $(N)-way")
        
        if N == 2
            comp1, comp2 = interaction_eval.components
            binary_data = create_binary_interaction_data(comp1, comp2, interaction_eval.positions)
            push!(all_binary_interactions, binary_data)
            # println("    Binary: positions $(interaction_eval.positions)")
        elseif N > 2
            # println("    Decomposing $(N)-way interaction using scratch...")
            binary_sequence = decompose_nway_with_scratch(interaction_eval, scratch_allocator)
            append!(all_binary_interactions, binary_sequence)
            # println("    Created $(length(binary_sequence)) binary operations")
        else
            @warn "Skipping interaction with $(N) components (< 2)"
        end
    end
    
    max_scratch_needed = isempty(scratch_allocator.allocated_positions) ? 0 : maximum(scratch_allocator.allocated_positions)
    # println("  Maximum scratch position needed: $max_scratch_needed")
    
    n_binary = length(all_binary_interactions)
    
    if n_binary == 0
        empty_data = SpecializedInteractionData(())
        return empty_data, InteractionOp(0)
    end
    
    binary_tuple = ntuple(n_binary) do i
        all_binary_interactions[i]
    end
    
    specialized_data = SpecializedInteractionData(binary_tuple)
    interaction_op = InteractionOp(n_binary)
    
    return specialized_data, interaction_op
end

###############################################################################
# SCRATCH CALCULATION FUNCTIONS
###############################################################################

function calculate_max_interaction_scratch_needed(evaluator::CombinedEvaluator)
    interaction_evaluators = evaluator.interaction_evaluators
    
    if isempty(interaction_evaluators)
        return 0
    end
    
    scratch_allocator = InteractionScratchAllocator()
    
    for interaction_eval in interaction_evaluators
        N = length(interaction_eval.components)
        
        if N > 2
            decompose_nway_with_scratch(interaction_eval, scratch_allocator)
        end
    end
    
    return isempty(scratch_allocator.allocated_positions) ? 0 : maximum(scratch_allocator.allocated_positions)
end

function calculate_total_temp_positions(interaction_evaluators::Vector{InteractionEvaluator})
    if isempty(interaction_evaluators)
        return 0
    end
    
    scratch_allocator = InteractionScratchAllocator()
    
    for interaction_eval in interaction_evaluators
        N = length(interaction_eval.components)
        
        if N > 2
            components = interaction_eval.components
            current_width = get_component_output_width(components[1])
            
            for i in 2:(N-1)
                next_width = get_component_output_width(components[i])
                scratch_size = current_width * next_width
                allocate_interaction_scratch!(scratch_allocator, scratch_size)
                current_width = scratch_size
            end
        end
    end
    
    return isempty(scratch_allocator.allocated_positions) ? 0 : maximum(scratch_allocator.allocated_positions)
end

###############################################################################
# UPDATED COMPLETE FORMULA DATA TYPES
###############################################################################

struct CompleteFormulaOp{ConstOp, ContOp, CatOp, FuncOp, IntOp}
    constants::ConstOp
    continuous::ContOp
    categorical::CatOp
    functions::FuncOp
    interactions::IntOp
end

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

###############################################################################
# UPDATED ANALYZE_EVALUATOR WITH SCRATCH CALCULATION
###############################################################################

function analyze_evaluator(evaluator::AbstractEvaluator)
    if evaluator isa CombinedEvaluator
        constant_data, constant_op = analyze_constant_operations(evaluator)
        continuous_data, continuous_op = analyze_continuous_operations(evaluator)
        categorical_data, categorical_op = analyze_categorical_operations(evaluator)
        function_data, function_op = analyze_function_operations_linear(evaluator)
        
        interaction_data, interaction_op = analyze_interaction_operations(evaluator)
        
        max_function_scratch = 0
        max_interaction_scratch = calculate_max_interaction_scratch_needed(evaluator)
        
        function_scratch = Vector{Float64}(undef, max_function_scratch)
        interaction_scratch = Vector{Float64}(undef, max_interaction_scratch)
        
        # println("ANALYZE_EVALUATOR SCRATCH DEBUG:")
        # println("  Function scratch needed: $max_function_scratch")
        # println("  Interaction scratch needed: $max_interaction_scratch")
        
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

###############################################################################
# EXECUTION ORDER MANAGEMENT
###############################################################################

function execute_operation!(
    data::CompleteFormulaData,
    op::CompleteFormulaOp,
    output, input_data, row_idx
)
    # Phase 1: Constants and Continuous (no scratch needed)
    execute_complete_constant_operations!(data.constants, output, input_data, row_idx)
    execute_complete_continuous_operations!(data.continuous, output, input_data, row_idx)
    
    # Phase 2: Functions (use function scratch)
    execute_linear_function_operations!(data.functions, data.function_scratch, output, input_data, row_idx)
    
    # Phase 3: Categoricals (no scratch needed)
    execute_categorical_operations!(data.categorical, output, input_data, row_idx)
    
    # Phase 4: Interactions (use interaction scratch - FOLLOWS function pattern)
    execute_interaction_operations!(data.interactions, data.interaction_scratch, output, input_data, row_idx)
    
    return nothing
end

###############################################################################
# COMPILATION FUNCTIONS
###############################################################################

function create_specialized_formula(compiled_formula::CompiledFormula)
    data_tuple, op_tuple = analyze_evaluator(compiled_formula.root_evaluator)
    
    return SpecializedFormula{typeof(data_tuple), typeof(op_tuple)}(
        data_tuple,
        op_tuple,
        compiled_formula.output_width
    )
end

function compile_formula_specialized(model, data::NamedTuple)
    compiled = compile_formula(model, data)
    return create_specialized_formula(compiled)
end

###############################################################################
# EXECUTION FUNCTIONS FOR CONSTANTS AND CONTINUOUS
###############################################################################

function execute_complete_constant_operations!(constant_data::ConstantData{N}, output, input_data, row_idx) where N
    @inbounds for i in 1:N
        pos = constant_data.positions[i]
        val = constant_data.values[i]
        output[pos] = val
    end
    return nothing
end

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
# INTERFACE METHODS
###############################################################################

Base.isempty(data::SpecializedInteractionData) = length(data.binary_interactions) == 0
Base.length(data::SpecializedInteractionData) = length(data.binary_interactions)

function Base.iterate(data::SpecializedInteractionData, state=1)
    if state > length(data.binary_interactions)
        return nothing
    end
    return (data.binary_interactions[state], state + 1)
end

###############################################################################
# REST OF EXISTING FUNCTIONS (unchanged)
###############################################################################
