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
# NEW ANALYSIS FUNCTION (Replaces old analyze_interaction_operations)
###############################################################################

"""
    analyze_interaction_operations(evaluator::CombinedEvaluator) -> (SpecializedInteractionData, InteractionOp)

NEW: Analyze interaction evaluators and create specialized binary interaction data.
Completely replaces the old function that returned Vector{InteractionEvaluator}.
"""
function analyze_interaction_operations(evaluator::CombinedEvaluator)
    interaction_evaluators = evaluator.interaction_evaluators
    n_interactions = length(interaction_evaluators)
    
    if n_interactions == 0
        empty_data = SpecializedInteractionData(())
        return empty_data, InteractionOp(0)
    end
    
    # Convert InteractionEvaluators to BinaryInteractionData
    binary_interactions = BinaryInteractionData[]
    
    for interaction_eval in interaction_evaluators
        if length(interaction_eval.components) == 2
            # Convert 2-way InteractionEvaluator to BinaryInteractionData
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
    
    # Create compile-time tuple
    binary_tuple = ntuple(n_binary) do i
        binary_interactions[i]
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
# COMPILATION FUNCTIONS
###############################################################################

"""
    create_specialized_formula(compiled_formula::CompiledFormula) -> SpecializedFormula

Updated to work with enhanced interaction system.
"""
function create_specialized_formula(compiled_formula::CompiledFormula)
    # Analyze the evaluator tree with complete support including enhanced interactions
    data_tuple, op_tuple = analyze_evaluator(compiled_formula.root_evaluator)
    
    # Create specialized formula
    return SpecializedFormula{typeof(data_tuple), typeof(op_tuple)}(
        data_tuple,
        op_tuple,
        compiled_formula.output_width
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
