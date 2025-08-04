# phase1_binary_interactions.jl
# Core binary interaction system following the proven function pattern

###############################################################################
# CORE BINARY INTERACTION DATA TYPES (Following Function Pattern)
###############################################################################

"""
    BinaryInteractionData{Comp1Type, Comp2Type, Pattern}

Compile-time specialized binary interaction with known component types.
Follows the exact pattern of BinaryFunctionData{F, T1, T2}.

# Type Parameters
- `Comp1Type`: Type of first component (ConstantEvaluator, ContinuousEvaluator, etc.)
- `Comp2Type`: Type of second component 
- `Pattern`: Type of pre-computed index pattern

# Fields
- `component1`: First component evaluator
- `component2`: Second component evaluator
- `width1`: First component width (compile-time constant)
- `width2`: Second component width (compile-time constant)
- `index_pattern`: Pre-computed (i,j) pairs for iteration
- `output_positions`: Where interaction results go in output array
"""
struct BinaryInteractionData{Comp1Type, Comp2Type, Pattern}
    component1::Comp1Type                    # First component evaluator
    component2::Comp2Type                    # Second component evaluator
    width1::Int                              # Component 1 width (compile-time known)
    width2::Int                              # Component 2 width (compile-time known) 
    index_pattern::Pattern                   # Pre-computed (i,j) pairs
    output_positions::Vector{Int}            # Where results go in output array
    
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
Follows the exact pattern of SpecializedFunctionData{UnaryTuple, BinaryTuple}.
"""
struct SpecializedInteractionData{BinaryTuple}
    binary_interactions::BinaryTuple         # NTuple{M, BinaryInteractionData{...}}
    
    function SpecializedInteractionData(binary_tuple::T) where T
        new{T}(binary_tuple)
    end
end

"""
    InteractionOp{M}

Compile-time encoding of interaction operations.
Follows the pattern of FunctionOp{N, M}.
"""
struct InteractionOp{M}
    function InteractionOp(n_binary::Int)
        new{n_binary}()
    end
end

# Backward compatibility for old code that calls InteractionOp()
InteractionOp() = InteractionOp(0)

###############################################################################
# COMPONENT VALUE ACCESS (Following Function get_input_value Pattern)
###############################################################################

"""
    get_component_value(component::ConstantEvaluator, index::Int, data::NamedTuple, row_idx::Int, output::Vector{Float64}) -> Float64

Get value from constant component (output parameter unused but kept for consistency).
"""
@inline function get_component_value(
    component::ConstantEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64}  # Unused but consistent signature
)
    return component.value
end

"""
    get_component_value(component::ContinuousEvaluator, index::Int, data::NamedTuple, row_idx::Int, output::Vector{Float64}) -> Float64

Get value from continuous component (output parameter unused but kept for consistency).
"""
@inline function get_component_value(
    component::ContinuousEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64}  # Unused but consistent signature
)
    return Float64(get_data_value_specialized(data, component.column, row_idx))
end

"""
    get_component_value(component::CategoricalEvaluator, index::Int, data::NamedTuple, row_idx::Int, output::Vector{Float64}) -> Float64

Get value from categorical component (output parameter unused but kept for consistency).
"""
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

"""
    get_component_value(component::FunctionEvaluator, index::Int, data::NamedTuple, row_idx::Int, output::Vector{Float64}) -> Float64

Get value from function component by reading pre-computed result from output array.
FIXED: No longer computes function inline - reads from Phase 2 execution results.
"""
@inline function get_component_value(
    component::FunctionEvaluator, 
    index::Int, 
    data::NamedTuple, 
    row_idx::Int,
    output::Vector{Float64}  # NOW USED: Read pre-computed function result
)
    # Function result was computed in Phase 2 and stored in output array
    # Just read the pre-computed value - zero allocations!
    return output[component.position]
end

###############################################################################
# BINARY INTERACTION PATTERN GENERATION
###############################################################################

"""
    compute_binary_interaction_pattern(width1::Int, width2::Int) -> Vector{Tuple{Int, Int}}

Generate all (i,j) index pairs for binary interaction.
Pre-computes the iteration pattern to avoid runtime generation.

# Example
```julia
# For width1=2, width2=3:
# Returns: [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
```
"""
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

"""
    create_binary_interaction_data(
        comp1::AbstractEvaluator, 
        comp2::AbstractEvaluator, 
        output_positions::Vector{Int}
    ) -> BinaryInteractionData

Create compile-time specialized binary interaction data.
"""
function create_binary_interaction_data(
    comp1::AbstractEvaluator, 
    comp2::AbstractEvaluator, 
    output_positions::Vector{Int}
)
    width1 = get_component_output_width(comp1)
    width2 = get_component_output_width(comp2)
    
    # Validate output positions
    expected_width = width1 * width2
    if length(output_positions) != expected_width
        error("Output positions length $(length(output_positions)) != expected width $expected_width")
    end
    
    # Pre-compute interaction pattern
    pattern = compute_binary_interaction_pattern(width1, width2)
    
    return BinaryInteractionData(comp1, comp2, width1, width2, pattern, output_positions)
end

###############################################################################
# BINARY INTERACTION EXECUTION (Following Function Pattern)
###############################################################################

"""
    execute_operation!(
        data::BinaryInteractionData{C1, C2, P}, 
        output::AbstractVector{Float64}, 
        input_data::NamedTuple, 
        row_idx::Int
    ) where {C1, C2, P}

Execute binary interaction with compile-time specialization.
Follows exact pattern of execute_operation! for BinaryFunctionData.

# Key Features
- Zero allocations: Direct scalar computation
- Type stability: All types known at compile time
- No intermediate arrays: Each product computed and stored directly
"""
function execute_operation!(
    data::BinaryInteractionData{C1, C2, P},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {C1, C2, P}
    
    # Direct computation for each interaction term (no intermediate arrays)
    @inbounds for output_idx in 1:length(data.index_pattern)
        i, j = data.index_pattern[output_idx]
        
        # Get component values (scalar operations)
        val1 = get_component_value(data.component1, i, input_data, row_idx)  # Scalar
        val2 = get_component_value(data.component2, j, input_data, row_idx)  # Scalar
        
        # Compute interaction product (scalar)
        product = val1 * val2
        
        # Store result directly (no intermediate storage)
        output_pos = data.output_positions[output_idx]
        output[output_pos] = product
    end
    
    return nothing
end

###############################################################################
# TUPLE-BASED RECURSIVE EXECUTION (Following Function Pattern)
###############################################################################

"""
    execute_binary_interactions_recursive!(
        binary_tuple::Tuple{}, 
        output::AbstractVector{Float64}, 
        input_data::NamedTuple, 
        row_idx::Int
    )

Base case: empty tuple - nothing to execute.
"""
function execute_binary_interactions_recursive!(
    binary_tuple::Tuple{},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    return nothing
end

"""
    execute_binary_interactions_recursive!(
        binary_tuple::Tuple, 
        output::AbstractVector{Float64}, 
        input_data::NamedTuple, 
        row_idx::Int
    )

Recursive case: execute first binary interaction, then process remaining.
Follows exact pattern of execute_binary_functions_recursive!.
"""
function execute_binary_interactions_recursive!(
    binary_tuple::Tuple,
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
)
    if length(binary_tuple) > 0
        # Execute first binary interaction
        execute_operation!(binary_tuple[1], output, input_data, row_idx)
        
        # Recursively process remaining binary interactions
        if length(binary_tuple) > 1
            remaining = Base.tail(binary_tuple)
            execute_binary_interactions_recursive!(remaining, output, input_data, row_idx)
        end
    end
    return nothing
end

"""
    execute_operation!(
        data::SpecializedInteractionData{BT}, 
        op::InteractionOp{M}, 
        output::AbstractVector{Float64}, 
        input_data::NamedTuple, 
        row_idx::Int
    ) where {BT, M}

Execute all binary interactions using recursive tuple processing.
Follows exact pattern of execute_operation! for SpecializedFunctionData.
"""
function execute_operation!(
    data::SpecializedInteractionData{BT},
    op::InteractionOp{M},
    output::AbstractVector{Float64},
    input_data::NamedTuple,
    row_idx::Int
) where {BT, M}
    
    # Execute all binary interactions using recursive processing
    execute_binary_interactions_recursive!(data.binary_interactions, output, input_data, row_idx)
    
    return nothing
end

###############################################################################
# ANALYSIS AND COMPILATION (Following Function Pattern)
###############################################################################

"""
    analyze_binary_interactions(evaluator::CombinedEvaluator) -> (SpecializedInteractionData, InteractionOp)

Analyze interaction evaluators and create specialized binary interaction data.
Phase 1: Only handles existing 2-way InteractionEvaluators directly.
Phase 2 will add decomposition of N-way interactions.
"""
function analyze_binary_interactions(evaluator::CombinedEvaluator)
    interaction_evaluators = evaluator.interaction_evaluators
    n_interactions = length(interaction_evaluators)
    
    if n_interactions == 0
        empty_data = SpecializedInteractionData(())
        return empty_data, InteractionOp(0)
    end
    
    # Phase 1: Filter for binary interactions only
    binary_interactions = AbstractEvaluator[]
    
    for interaction_eval in interaction_evaluators
        if length(interaction_eval.components) == 2
            # Convert InteractionEvaluator to BinaryInteractionData
            comp1, comp2 = interaction_eval.components
            binary_data = create_binary_interaction_data(comp1, comp2, interaction_eval.positions)
            push!(binary_interactions, binary_data)
        else
            @warn "Phase 1: Skipping non-binary interaction with $(length(interaction_eval.components)) components"
        end
    end
    
    n_binary = length(binary_interactions)
    
    if n_binary == 0
        empty_data = SpecializedInteractionData(())
        return empty_data, InteractionOp(0)
    end
    
    # Create compile-time tuple of binary interactions
    binary_tuple = ntuple(n_binary) do i
        binary_interactions[i]
    end
    
    specialized_data = SpecializedInteractionData(binary_tuple)
    interaction_op = InteractionOp(n_binary)
    
    return specialized_data, interaction_op
end

###############################################################################
# INTERFACE METHODS (Following Function Pattern)
###############################################################################

"""
    Base.isempty(data::SpecializedInteractionData) -> Bool

Check if interaction data is empty (no interactions to execute).
"""
function Base.isempty(data::SpecializedInteractionData)
    return length(data.binary_interactions) == 0
end

"""
    Base.length(data::SpecializedInteractionData) -> Int

Get total number of binary interactions.
"""
function Base.length(data::SpecializedInteractionData)
    return length(data.binary_interactions)
end

"""
    Base.iterate(data::SpecializedInteractionData, state=1)

Iterate over all binary interactions.
Required for isempty() to work properly.
"""
function Base.iterate(data::SpecializedInteractionData, state=1)
    if state > length(data.binary_interactions)
        return nothing
    end
    
    return (data.binary_interactions[state], state + 1)
end

###############################################################################  
# MAIN EXECUTION INTERFACE (Following Function Pattern)
###############################################################################

"""
    execute_binary_interaction_operations!(
        interaction_data::SpecializedInteractionData, 
        scratch::Vector{Float64},  # Not used in specialized version
        output::Vector{Float64}, 
        data::NamedTuple, 
        row_idx::Int
    )

Main execution interface - maintains compatibility with existing system.
Follows pattern of execute_linear_function_operations!.
"""
function execute_binary_interaction_operations!(
    interaction_data::SpecializedInteractionData,
    scratch::Vector{Float64},  # Not used in specialized version
    output::Vector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Create operation encoding for dispatch
    n_binary = length(interaction_data.binary_interactions)
    op = InteractionOp(n_binary)
    
    # Execute using specialized dispatch
    execute_operation!(interaction_data, op, output, data, row_idx)
    
    return nothing
end

###############################################################################
# INTEGRATION WITH EXISTING SYSTEM
###############################################################################

"""
    update_step4_interactions!(
        data::CompleteFormulaData,
        op::CompleteFormulaOp,
        output, input_data, row_idx
    )

Integration point: Replace the current interaction execution in step4_interactions.jl
with binary interaction system.

This function should replace execute_interaction_operations! in the main execution pipeline.
"""
function execute_interaction_operations!(
    interaction_data::SpecializedInteractionData,
    scratch::Vector{Float64},  # Maintained for interface compatibility
    output::Vector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    execute_binary_interaction_operations!(interaction_data, scratch, output, data, row_idx)
    return nothing
end

"""
    update_analyze_evaluator_for_binary_interactions(evaluator::CombinedEvaluator)

Integration point: Update analyze_evaluator in step4_interactions.jl to use
binary interaction analysis instead of current approach.

This replaces the analyze_interaction_operations call.
"""
function analyze_interaction_operations_binary(evaluator::CombinedEvaluator)
    return analyze_binary_interactions(evaluator)
end

###############################################################################
# TESTING AND DEBUGGING UTILITIES
###############################################################################

"""
    test_binary_interaction_pattern()

Test binary interaction pattern generation.
"""
function test_binary_interaction_pattern()
    println("Testing binary interaction patterns:")
    
    test_cases = [
        (1, 1, "Scalar × Scalar"),
        (2, 1, "Binary × Scalar"), 
        (1, 3, "Scalar × Ternary"),
        (2, 3, "Binary × Ternary"),
        (3, 3, "Ternary × Ternary")
    ]
    
    for (w1, w2, desc) in test_cases
        pattern = compute_binary_interaction_pattern(w1, w2)
        expected_length = w1 * w2
        
        println("  $desc: $(length(pattern)) terms (expected $expected_length)")
        println("    Pattern: $pattern")
        
        @assert length(pattern) == expected_length "Pattern length mismatch"
        @assert all(1 ≤ i ≤ w1 && 1 ≤ j ≤ w2 for (i,j) in pattern) "Invalid indices"
    end
    
    println("✅ All pattern tests passed!")
end

"""
    trace_binary_interaction_execution(data::SpecializedInteractionData, description="")

Debug function to trace binary interaction execution.
"""
function trace_binary_interaction_execution(data::SpecializedInteractionData, description="")
    println("Tracing binary interaction execution: $description")
    println("  Input data type: $(typeof(data))")
    println("  Number of binary interactions: $(length(data.binary_interactions))")
    
    for (idx, binary_data) in enumerate(data.binary_interactions)
        println("  Binary interaction $idx:")
        println("    Component 1: $(typeof(binary_data.component1))")
        println("    Component 2: $(typeof(binary_data.component2))") 
        println("    Widths: $(binary_data.width1) × $(binary_data.width2)")
        println("    Pattern length: $(length(binary_data.index_pattern))")
        println("    Output positions: $(length(binary_data.output_positions))")
    end
    
    return nothing
end
