# step4_interactions.jl - COMPLETE REPLACEMENT
# Replace the old interaction system entirely with the new binary system

###############################################################################
# IMPORT BINARY INTERACTION SYSTEM
###############################################################################

# All the binary interaction types and functions from phase1_binary_interactions.jl
# (Copy the entire contents here or include the file)

###############################################################################
# CORE BINARY INTERACTION DATA TYPES
###############################################################################

"""
    BinaryInteractionData{Comp1Type, Comp2Type, Pattern}

Compile-time specialized binary interaction with known component types.
"""
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

"""
    SpecializedInteractionData{BinaryTuple}

Compile-time specialized interaction data with tuple of binary interactions.
"""
struct SpecializedInteractionData{BinaryTuple}
    binary_interactions::BinaryTuple         
    
    function SpecializedInteractionData(binary_tuple::T) where T
        new{T}(binary_tuple)
    end
end

"""
    InteractionOp{M}

Compile-time encoding of interaction operations.
"""
struct InteractionOp{M}
    function InteractionOp(n_binary::Int)
        new{n_binary}()
    end
end

# Backward compatibility
InteractionOp() = InteractionOp(0)

###############################################################################
# COMPONENT VALUE ACCESS
###############################################################################

###############################################################################
# TEMP RESULT EVALUATOR FOR N-WAY DECOMPOSITION
###############################################################################

"""
    TempResultEvaluator <: AbstractEvaluator

Represents intermediate results in N-way interaction decomposition.
"""
struct TempResultEvaluator <: AbstractEvaluator
    positions::Vector{Int}    
    width::Int               
    
    function TempResultEvaluator(positions::Vector{Int})
        new(positions, length(positions))
    end
end

# Interface methods for TempResultEvaluator
output_width(eval::TempResultEvaluator) = eval.width
get_positions(eval::TempResultEvaluator) = eval.positions
get_scratch_positions(eval::TempResultEvaluator) = Int[]
max_scratch_needed(eval::TempResultEvaluator) = 0
get_component_output_width(eval::TempResultEvaluator) = eval.width

@inline function get_component_value(
    component::TempResultEvaluator,
    index::Int,
    data::NamedTuple,
    row_idx::Int,
    output::Vector{Float64}
)
    return output[component.positions[index]]
end

###############################################################################
# TEMPORARY POSITION ALLOCATION
###############################################################################

mutable struct TempPositionAllocator
    next_position::Int
    TempPositionAllocator(start_position::Int) = new(start_position)
end

function allocate_temp_positions!(allocator::TempPositionAllocator, width::Int)
    positions = collect(allocator.next_position:(allocator.next_position + width - 1))
    allocator.next_position += width
    return positions
end

###############################################################################
# N-WAY DECOMPOSITION FUNCTIONS
###############################################################################

function decompose_nway_to_binary(
    interaction_eval::InteractionEvaluator{N},
    temp_allocator::TempPositionAllocator
) where N
    
    println("    DECOMPOSE_NWAY_TO_BINARY DEBUG:")
    println("      N = $N")
    println("      Final positions: $(interaction_eval.positions)")
    
    if N < 2
        error("Cannot decompose interaction with < 2 components")
    elseif N == 2
        comp1, comp2 = interaction_eval.components
        return [create_binary_interaction_data(comp1, comp2, interaction_eval.positions)]
    end
    
    components = interaction_eval.components
    final_positions = interaction_eval.positions
    
    # Debug component widths
    for (i, comp) in enumerate(components)
        width = get_component_output_width(comp)
        println("      Component $i: $(typeof(comp)), width = $width")
    end
    
    binary_ops = BinaryInteractionData[]
    current_component = components[1]
    
    for i in 2:N
        next_component = components[i]
        
        current_width = get_component_output_width(current_component)
        next_width = get_component_output_width(next_component)
        
        println("      Step $(i-1): $(current_width) × $(next_width)")
        
        if i == N
            output_positions = final_positions
            println("        FINAL → positions $output_positions")
        else
            temp_width = current_width * next_width
            println("        Allocating $temp_width temp positions starting at $(temp_allocator.next_position)")
            output_positions = allocate_temp_positions!(temp_allocator, temp_width)
            println("        TEMP → positions $output_positions")
        end
        
        binary_data = create_binary_interaction_data(
            current_component, 
            next_component, 
            output_positions
        )
        push!(binary_ops, binary_data)
        
        if i < N
            current_component = TempResultEvaluator(output_positions)
            println("        Next iteration: TempResultEvaluator with positions $output_positions")
        end
    end
    
    println("      Created $(length(binary_ops)) binary operations")
    return binary_ops
end

function calculate_total_temp_positions(interaction_evaluators::Vector{InteractionEvaluator})
    total_temp = 0
    
    for interaction_eval in interaction_evaluators
        N = length(interaction_eval.components)
        
        if N > 2
            components = interaction_eval.components
            current_width = get_component_output_width(components[1])
            
            for i in 2:(N-1)
                next_width = get_component_output_width(components[i])
                temp_width = current_width * next_width
                total_temp += temp_width
                current_width = temp_width
            end
        end
    end
    
    return total_temp
end

###############################################################################
# COMPONENT VALUE ACCESS (UPDATED WITH CONSISTENT SIGNATURES)
###############################################################################

@inline function get_component_value(
    component::ConstantEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64}  # Unused but consistent signature
)
    return component.value
end

@inline function get_component_value(
    component::ContinuousEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64}  # Unused but consistent signature
)
    return Float64(get_data_value_specialized(data, component.column, row_idx))
end

@inline function get_component_value(
    component::CategoricalEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64}  # Unused but consistent signature
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
    output::Vector{Float64}  # NOW USED: Read pre-computed function result
)
    # FIXED: Read pre-computed function result from Phase 2 execution
    # No inline computation, no allocations!
    return output[component.position]
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
# BINARY INTERACTION EXECUTION
###############################################################################

function execute_operation!(
    data::BinaryInteractionData{C1, C2, P},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {C1, C2, P}
    
    @inbounds for output_idx in 1:length(data.index_pattern)
        i, j = data.index_pattern[output_idx]
        
        # UPDATED: Pass output array to component value access
        val1 = get_component_value(data.component1, i, input_data, row_idx, output)
        val2 = get_component_value(data.component2, j, input_data, row_idx, output)
        
        product = val1 * val2
        
        output_pos = data.output_positions[output_idx]
        output[output_pos] = product
    end
    
    return nothing
end

###############################################################################
# RECURSIVE EXECUTION
###############################################################################

function execute_binary_interactions_recursive!(
    binary_tuple::Tuple{},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

function execute_binary_interactions_recursive!(
    binary_tuple::Tuple,
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(binary_tuple) > 0
        execute_operation!(binary_tuple[1], output, input_data, row_idx)
        
        if length(binary_tuple) > 1
            remaining = Base.tail(binary_tuple)
            execute_binary_interactions_recursive!(remaining, output, input_data, row_idx)
        end
    end
    return nothing
end

function execute_operation!(
    data::SpecializedInteractionData{BT},
    op::InteractionOp{M},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {BT, M}
    
    execute_binary_interactions_recursive!(data.binary_interactions, output, input_data, row_idx)
    return nothing
end

###############################################################################
# UPDATED ANALYSIS FUNCTION (NOW HANDLES N-WAY INTERACTIONS)
###############################################################################

"""
    analyze_interaction_operations(evaluator::CombinedEvaluator) -> (SpecializedInteractionData, InteractionOp)

UPDATED: Now handles N-way interactions by decomposing them into binary sequences.
DEBUGGING: Added extensive logging to trace bounds error.
"""
function analyze_interaction_operations(evaluator::CombinedEvaluator)
    interaction_evaluators = evaluator.interaction_evaluators
    n_interactions = length(interaction_evaluators)
    
    println("ANALYZE_INTERACTION_OPERATIONS DEBUG:")
    println("  Number of interactions: $n_interactions")
    
    if n_interactions == 0
        empty_data = SpecializedInteractionData(())
        return empty_data, InteractionOp(0)
    end
    
    # Calculate temporary position requirements
    total_temp_positions = calculate_total_temp_positions(interaction_evaluators)
    println("  Total temp positions needed: $total_temp_positions")
    
    # Find maximum final position to determine where temps start
    max_final_position = 0
    for interaction_eval in interaction_evaluators
        if !isempty(interaction_eval.positions)
            max_pos = maximum(interaction_eval.positions)
            max_final_position = max(max_final_position, max_pos)
            println("  Interaction positions: $(interaction_eval.positions), max: $max_pos")
        end
    end
    
    # ALSO check other evaluators to find the true maximum position
    # Get the total output width of the evaluator (before temp extension)
    original_output_width = output_width(evaluator)
    println("  Max final position: $max_final_position")
    println("  Original output width: $original_output_width")
    
    # Start temps after the original output width, not after max interaction position
    temp_start_position = max(max_final_position + 1, original_output_width + 1)
    temp_allocator = TempPositionAllocator(temp_start_position)
    println("  Temp start position: $temp_start_position")
    
    # Convert all interactions (both 2-way and N-way) to binary operations
    all_binary_interactions = BinaryInteractionData[]
    
    for (idx, interaction_eval) in enumerate(interaction_evaluators)
        N = length(interaction_eval.components)
        println("  Processing interaction $idx: $(N)-way")
        
        if N == 2
            # Direct binary conversion (existing path)
            comp1, comp2 = interaction_eval.components
            binary_data = create_binary_interaction_data(comp1, comp2, interaction_eval.positions)
            push!(all_binary_interactions, binary_data)
            println("    Binary: positions $(interaction_eval.positions)")
            
        elseif N > 2
            # NEW: Decompose N-way into binary sequence
            println("    Decomposing $(N)-way interaction...")
            binary_sequence = decompose_nway_to_binary(interaction_eval, temp_allocator)
            append!(all_binary_interactions, binary_sequence)
            println("    Created $(length(binary_sequence)) binary operations")
            
            # Log the positions used
            for (i, binary_op) in enumerate(binary_sequence)
                max_pos = maximum(binary_op.output_positions)
                println("      Binary op $i: positions $(binary_op.output_positions), max: $max_pos")
            end
            
        else
            @warn "Skipping interaction with $(N) components (< 2)"
        end
    end
    
    # Check final position usage
    max_position_used = 0
    for binary_op in all_binary_interactions
        if !isempty(binary_op.output_positions)
            max_pos = maximum(binary_op.output_positions)
            max_position_used = max(max_position_used, max_pos)
        end
    end
    
    println("  Maximum position used across all binary ops: $max_position_used")
    
    n_binary = length(all_binary_interactions)
    
    if n_binary == 0
        empty_data = SpecializedInteractionData(())
        return empty_data, InteractionOp(0)
    end
    
    # Create compile-time tuple
    binary_tuple = ntuple(n_binary) do i
        all_binary_interactions[i]
    end
    
    specialized_data = SpecializedInteractionData(binary_tuple)
    interaction_op = InteractionOp(n_binary)
    
    return specialized_data, interaction_op
end

###############################################################################
# NEW EXECUTION FUNCTION (Replaces old execute_interaction_operations!)
###############################################################################

"""
    execute_interaction_operations!(
        interaction_data::SpecializedInteractionData,
        scratch::Vector{Float64},
        output::Vector{Float64},
        data::NamedTuple,
        row_idx::Int
    )

NEW: Execute binary interactions using the new system.
Completely replaces the old function that took Vector{InteractionEvaluator}.
"""
function execute_interaction_operations!(
    interaction_data::SpecializedInteractionData,
    scratch::Vector{Float64},  # Maintained for interface compatibility
    output::Vector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    n_binary = length(interaction_data.binary_interactions)
    op = InteractionOp(n_binary)
    
    execute_operation!(interaction_data, op, output, data, row_idx)
    return nothing
end

###############################################################################
# UPDATED COMPLETE FORMULA DATA TYPES
###############################################################################

"""
    CompleteFormulaOp{ConstOp, ContOp, CatOp, FuncOp, IntOp}

Complete operation encoding including interactions.
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

Updated to use SpecializedInteractionData instead of Vector{InteractionEvaluator}.
"""
struct CompleteFormulaData{ConstData, ContData, CatData, FuncData, IntData}
    constants::ConstData
    continuous::ContData
    categorical::CatData
    functions::FuncData
    interactions::IntData               # Now SpecializedInteractionData{BinaryTuple}
    max_function_scratch::Int
    max_interaction_scratch::Int
    function_scratch::Vector{Float64}
    interaction_scratch::Vector{Float64}
end

###############################################################################
# UPDATED ANALYZE_EVALUATOR
###############################################################################

"""
    analyze_evaluator(evaluator::AbstractEvaluator) -> (DataTuple, OpTuple)

Updated to use new binary interaction system.
"""
function analyze_evaluator(evaluator::AbstractEvaluator)
    if evaluator isa CombinedEvaluator
        constant_data, constant_op = analyze_constant_operations(evaluator)
        continuous_data, continuous_op = analyze_continuous_operations(evaluator)
        categorical_data, categorical_op = analyze_categorical_operations(evaluator)
        function_data, function_op = analyze_function_operations_linear(evaluator)
        
        # Use NEW binary interaction analysis
        interaction_data, interaction_op = analyze_interaction_operations(evaluator)
        
        max_function_scratch = 0
        max_interaction_scratch = 0
        
        function_scratch = Float64[]
        interaction_scratch = Float64[]
        
        formula_data = CompleteFormulaData(
            constant_data,
            continuous_data,
            categorical_data,
            function_data,
            interaction_data,  # Now SpecializedInteractionData
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
# UPDATED EXECUTION ORDER
###############################################################################

"""
    execute_operation!(data::CompleteFormulaData, op::CompleteFormulaOp, output, input_data, row_idx)

Updated execution with new binary interaction system.
"""
function execute_operation!(
    data::CompleteFormulaData,
    op::CompleteFormulaOp,
    output, input_data, row_idx
)
    # Phase 1: Constants and Continuous
    execute_complete_constant_operations!(data.constants, output, input_data, row_idx)
    execute_complete_continuous_operations!(data.continuous, output, input_data, row_idx)
    
    # Phase 2: Functions 
    execute_linear_function_operations!(data.functions, data.function_scratch, output, input_data, row_idx)
    
    # Phase 3: Categoricals
    execute_categorical_operations!(data.categorical, output, input_data, row_idx)
    
    # Phase 4: Binary Interactions (NEW)
    execute_interaction_operations!(data.interactions, data.interaction_scratch, output, input_data, row_idx)
    
    return nothing
end

###############################################################################
# OUTPUT WIDTH CALCULATION FOR N-WAY INTERACTIONS
###############################################################################

"""
    calculate_extended_output_width(evaluator::CombinedEvaluator) -> Int

Calculate total output width including temporary positions for N-way decomposition.
DEBUGGING: Added logging to track calculation.
"""
function calculate_extended_output_width(evaluator::CombinedEvaluator)
    original_width = output_width(evaluator)
    temp_positions = calculate_total_temp_positions(evaluator.interaction_evaluators)
    extended = original_width + temp_positions
    
    println("CALCULATE_EXTENDED_OUTPUT_WIDTH DEBUG:")
    println("  Original width: $original_width")
    println("  Temp positions: $temp_positions") 
    println("  Extended width: $extended")
    
    return extended
end

"""
    update_output_width_for_nway(compiled_formula::CompiledFormula) -> Int

Update output width calculation to include temporary positions.
"""
function update_output_width_for_nway(compiled_formula::CompiledFormula)
    if compiled_formula.root_evaluator isa CombinedEvaluator
        return calculate_extended_output_width(compiled_formula.root_evaluator)
    else
        return compiled_formula.output_width
    end
end

###############################################################################
# COMPILATION FUNCTIONS
###############################################################################

"""
    create_specialized_formula(compiled_formula::CompiledFormula) -> SpecializedFormula

Updated to work with N-way interaction system including temporary positions.
DEBUGGING: Added logging to track output width calculation.
"""
function create_specialized_formula(compiled_formula::CompiledFormula)
    data_tuple, op_tuple = analyze_evaluator(compiled_formula.root_evaluator)
    
    # Use extended output width for N-way interactions
    original_width = compiled_formula.output_width
    extended_width = update_output_width_for_nway(compiled_formula)
    
    println("CREATE_SPECIALIZED_FORMULA DEBUG:")
    println("  Original width: $original_width")
    println("  Extended width: $extended_width")
    
    return SpecializedFormula{typeof(data_tuple), typeof(op_tuple)}(
        data_tuple,
        op_tuple,
        extended_width
    )
end

"""
    compile_formula_specialized(model, data::NamedTuple) -> SpecializedFormula

Updated direct compilation to specialized formula with enhanced interaction support.
"""
function compile_formula_specialized(model, data::NamedTuple)
    # Use existing compilation logic to build evaluator tree
    compiled = compile_formula(model, data)
    # Convert to complete specialized form with enhanced interactions
    return create_specialized_formula(compiled)
end

###############################################################################
# EXECUTION FUNCTIONS FOR CONSTANTS AND CONTINUOUS
###############################################################################

"""
    execute_complete_constant_operations!(constant_data, output, input_data, row_idx)

Execute constant operations for complete formulas.
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
    execute_complete_continuous_operations!(continuous_data, output, input_data, row_idx)

Execute continuous operations for complete formulas.
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

# Fallback for empty constant data
function execute_complete_constant_operations!(constant_data::ConstantData{0}, output, input_data, row_idx)
    return nothing
end

# Fallback for empty continuous data  
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
# REMOVE ALL OLD FUNCTIONS
###############################################################################

# Delete these functions entirely:
# - Old execute_interaction_operation! for InteractionEvaluator
# - execute_interaction_recursive 
# - compute_interaction_product
# - All the old recursive tuple processing

# The new binary interaction system replaces all of this