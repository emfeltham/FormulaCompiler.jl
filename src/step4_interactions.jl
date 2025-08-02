# step4_interactions.jl
# Complete interaction support with full precomputation - OPTIMIZED VERSION
# This is a COMPLETE REPLACEMENT for the original step4_interactions.jl

###############################################################################
# COMPONENT CONVERSION FUNCTIONS - USE STEP 1-3 OPTIMIZATIONS
###############################################################################


###############################################################################
# INTERACTION ANALYSIS
###############################################################################

"""
    analyze_interaction_operations(evaluator::CombinedEvaluator) -> (Vector{InteractionEvaluator}, InteractionOp)

VERIFIED: Returns InteractionEvaluator objects directly, not InteractionData.
"""
function analyze_interaction_operations(evaluator::CombinedEvaluator)
    interaction_evaluators = evaluator.interaction_evaluators
    
    if isempty(interaction_evaluators)
        return InteractionEvaluator[], InteractionOp()
    end
    
    # CRITICAL: Return the InteractionEvaluator objects directly
    # Do NOT convert them to InteractionData or any other format
    return interaction_evaluators, InteractionOp()
end

"""
    InteractionOp

Operation encoding for interactions.
"""
struct InteractionOp end

###############################################################################
# COMPLETE FORMULA DATA TYPES
###############################################################################

"""
    CompleteFormulaData{ConstData, ContData, CatData, FuncData, IntData}

Complete formula data including interactions.
"""
struct CompleteFormulaData{ConstData, ContData, CatData, FuncData, IntData}
    constants::ConstData
    continuous::ContData
    categorical::CatData
    functions::FuncData
    interactions::IntData               # Vector{InteractionData}
    max_function_scratch::Int
    max_interaction_scratch::Int
    function_scratch::Vector{Float64}   # Pre-allocated function scratch
    interaction_scratch::Vector{Float64}# Pre-allocated interaction scratch
end

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

###############################################################################
# COMPLETE ANALYSIS FUNCTION
###############################################################################

"""
    analyze_evaluator(evaluator::AbstractEvaluator) -> (DataTuple, OpTuple)

Complete analysis for all operation types including interactions using Step 1-3 optimizations.
"""
function analyze_evaluator(evaluator::AbstractEvaluator)
    if evaluator isa CombinedEvaluator
        # Analyze all operation types (Steps 1-3)
        constant_data, constant_op = analyze_constant_operations(evaluator)
        continuous_data, continuous_op = analyze_continuous_operations(evaluator)
        categorical_data, categorical_op = analyze_categorical_operations(evaluator)
        function_data, function_op = analyze_function_operations_linear(evaluator)
        # Analyze interaction types (Step 4)
        interaction_evaluators, interaction_op = analyze_interaction_operations(evaluator)
        
        max_function_scratch = isempty(function_data) ? 0 : maximum(f.scratch_size for f in function_data)
        max_interaction_scratch = isempty(interaction_evaluators) ? 0 : maximum(i.total_scratch_needed for i in interaction_evaluators)

        # Pre-allocate once
        function_scratch = max_function_scratch > 0 ? Vector{Float64}(undef, max_function_scratch) : Float64[]
        interaction_scratch = max_interaction_scratch > 0 ? Vector{Float64}(undef, max_interaction_scratch) : Float64[]

        # Construct data with optimized interactions
        formula_data = CompleteFormulaData(
            constant_data,
            continuous_data,
            categorical_data,
            function_data,
            interaction_evaluators,  # Direct InteractionEvaluator objects
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
# INTERACTION EXECUTION FUNCTIONS - USE STEP 1-3 OPTIMIZATIONS
###############################################################################

"""
    execute_interaction_operation!(interaction::InteractionEvaluator{N},
                                  scratch::AbstractVector{Float64},  # <-- Changed
                                  output::AbstractVector{Float64},   # <-- Changed  
                                  data::NamedTuple,
                                  row_idx::Int) where N

Execute interaction using recursive scratch planning (zero allocations).
"""
function execute_interaction_operation!(
    interaction::InteractionEvaluator{N},
    scratch::AbstractVector{Float64},  # <-- Changed to AbstractVector
    output::AbstractVector{Float64},   # <-- Changed to AbstractVector
    data::NamedTuple,
    row_idx::Int
) where N
    # Phase 1: Execute all components into their assigned scratch regions
    @inbounds for (i, component) in enumerate(interaction.components)
        output_range = interaction.component_scratch_map[i]
        internal_range = interaction.component_internal_scratch_map[i]
        
        execute_component_in_assigned_scratch!(
            component, output_range, internal_range, scratch, data, row_idx
        )
    end
    
    # Phase 2: Apply Kronecker pattern using component outputs
    @inbounds for (result_idx, pattern_indices) in enumerate(interaction.kronecker_pattern)
        if result_idx <= length(interaction.positions)
            product = 1.0
            
            for (comp_idx, pattern_val) in enumerate(pattern_indices)
                comp_output_range = interaction.component_scratch_map[comp_idx]
                value_pos = first(comp_output_range) + pattern_val - 1
                product *= scratch[value_pos]
            end
            
            output_pos = interaction.positions[result_idx]
            output[output_pos] = product
        end
    end
    
    return nothing
end

"""
    execute_interaction_operations!(
        interaction_evaluators::Vector{InteractionEvaluator},  # More specific type
        scratch::AbstractVector{Float64},
        output::AbstractVector{Float64},
        data::NamedTuple,
        row_idx::Int
    )

Execute multiple interactions - simplified type signature.
"""
function execute_interaction_operations!(
    interaction_evaluators::Vector,  # Even more general - let Julia figure it out
    scratch::AbstractVector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # Handle empty case
    if isempty(interaction_evaluators)
        return nothing
    end
    
    # Execute each interaction
    @inbounds for interaction in interaction_evaluators
        if interaction isa InteractionEvaluator
            execute_interaction_operation!(interaction, scratch, output, data, row_idx)
        end
    end
    
    return nothing
end

"""
    calculate_component_scratch_recursive(component::AbstractEvaluator) -> Int

Calculate total scratch space needed by a component, including all internal computation.
ENHANCED: Better handling of FunctionEvaluator scratch needs.
"""
function calculate_component_scratch_recursive(component::AbstractEvaluator)
    if component isa ConstantEvaluator
        return 0  # No scratch needed
    elseif component isa ContinuousEvaluator
        return 0  # No scratch needed
    elseif component isa CategoricalEvaluator
        return 0  # No scratch needed (uses pre-computed lookup)
    elseif component isa FunctionEvaluator
        # For FunctionEvaluator, we need scratch for:
        # 1. All argument evaluations
        # 2. The function computation itself
        
        total_scratch = 0
        
        # Add scratch needed by all arguments
        for arg_eval in component.arg_evaluators
            arg_scratch = calculate_component_scratch_recursive(arg_eval)
            total_scratch += arg_scratch
        end
        
        # Add scratch for the function's own computation
        # This should be at least the length of scratch_positions if available
        function_own_scratch = if !isempty(component.scratch_positions)
            maximum(component.scratch_positions)
        else
            # Fallback: minimum scratch for argument storage + result
            length(component.arg_evaluators) + 1
        end
        
        total_scratch = max(total_scratch, function_own_scratch)
        return total_scratch
        
    elseif component isa InteractionEvaluator
        return component.total_scratch_needed
    elseif component isa ZScoreEvaluator
        return max_scratch_needed(component.underlying)
    else
        return max_scratch_needed(component)
    end
end

###############################################################################
# COMPLETE EXECUTION - FIXED TO AVOID RECURSION
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

"""
    execute_operation!(data::CompleteFormulaData{ConstData, ContData, CatData, FuncData, IntData}, 
                      op::CompleteFormulaOp{ConstOp, ContOp, CatOp, FuncOp, IntOp}, 
                      output, input_data, row_idx) where {ConstData, ContData, CatData, FuncData, IntData, ConstOp, ContOp, CatOp, FuncOp, IntOp}

Execute complete formulas with all operation types including interactions using Step 1-3 optimizations.
"""
function execute_operation!(data::CompleteFormulaData{ConstData,ContData,CatData,FuncData,IntData},
                            op::CompleteFormulaOp{ConstOp,ContOp,CatOp,FuncOp,IntOp},
                            output, input_data, row_idx) where {ConstData,ContData,CatData,FuncData,IntData,ConstOp,ContOp,CatOp,FuncOp,IntOp}
    # Reuse pre-allocated buffers
    fs = data.function_scratch
    is = data.interaction_scratch

    # Execute constants, continuous, categorical as before (these were already optimized)
    execute_complete_constant_operations!(data.constants, output, input_data, row_idx)
    execute_complete_continuous_operations!(data.continuous, output, input_data, row_idx)
    execute_categorical_operations!(data.categorical, output, input_data, row_idx)

    # Functions use optimized execution (Step 3 was already good)
    execute_linear_function_operations!(data.functions, fs, output, input_data, row_idx)
    
    # Interactions now use OPTIMIZED Step 1-3 component execution
    execute_interaction_operations!(data.interactions, data.interaction_scratch, output, input_data, row_idx)
    
    return nothing
end

###############################################################################
# COMPLETE COMPILATION FUNCTIONS
###############################################################################

"""
    create_specialized_formula(compiled_formula::CompiledFormula) -> SpecializedFormula

Convert a CompiledFormula to a SpecializedFormula with complete interaction support using Step 1-3 optimizations.
"""
function create_specialized_formula(compiled_formula::CompiledFormula)
    # Analyze the evaluator tree with complete support using Step 1-3 optimizations
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

Direct compilation to specialized formula with complete interaction support using Step 1-3 optimizations.
"""
function compile_formula_specialized(model, data::NamedTuple)
    # Use existing compilation logic to build evaluator tree
    compiled = compile_formula(model, data)
    # Convert to complete specialized form with optimizations
    return create_specialized_formula(compiled)
end

###############################################################################
# COMPLETE UTILITY FUNCTIONS
###############################################################################

###############################################################################
# ADD TO step4_interactions.jl - NEW EXECUTION FUNCTIONS
###############################################################################

"""
    execute_component_in_assigned_scratch!(
        component::AbstractEvaluator,
        output_range::UnitRange{Int},
        internal_range::UnitRange{Int},
        scratch::AbstractVector{Float64},  # <-- Changed to AbstractVector
        data::NamedTuple,
        row_idx::Int
    )

Execute component using pre-assigned scratch space regions.
Accepts both Vector{Float64} and SubArray (views).
"""
function execute_component_in_assigned_scratch!(
    component::AbstractEvaluator,
    output_range::UnitRange{Int},
    internal_range::UnitRange{Int},
    scratch::AbstractVector{Float64},  # <-- Changed to AbstractVector
    data::NamedTuple,
    row_idx::Int
)
    if component isa ConstantEvaluator
        # Direct assignment to output position
        if !isempty(output_range)
            @inbounds scratch[first(output_range)] = component.value
        end
        
    elseif component isa ContinuousEvaluator
        # Data lookup and assignment
        if !isempty(output_range)
            val = get_data_value_specialized(data, component.column, row_idx)
            @inbounds scratch[first(output_range)] = Float64(val)
        end
        
    elseif component isa CategoricalEvaluator
        # Categorical contrast lookup
        if !isempty(output_range)
            level = component.level_codes[row_idx]
            level = clamp(level, 1, component.n_levels)
            
            for (i, pos) in enumerate(output_range)
                @inbounds scratch[pos] = component.contrast_matrix[level, i]
            end
        end
        
    elseif component isa FunctionEvaluator
        # Execute function using internal scratch space
        execute_function_in_assigned_scratch!(
            component, output_range, internal_range, scratch, data, row_idx
        )
        
    else
        error("Unsupported component type in interaction: $(typeof(component))")
    end
    
    return nothing
end

"""
    execute_function_in_assigned_scratch!(
        func_eval::FunctionEvaluator,
        output_range::UnitRange{Int},
        internal_range::UnitRange{Int},
        scratch::AbstractVector{Float64},
        data::NamedTuple,
        row_idx::Int
    )

SIMPLIFIED: Execute function evaluator using assigned scratch space.
Avoids complex fallback paths that might allocate.
"""
function execute_function_in_assigned_scratch!(
    func_eval::FunctionEvaluator,
    output_range::UnitRange{Int},
    internal_range::UnitRange{Int},
    scratch::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # For simple functions with one continuous argument (most common case)
    if length(func_eval.arg_evaluators) == 1 && 
       func_eval.arg_evaluators[1] isa ContinuousEvaluator &&
       !isempty(output_range)
        
        # Direct path: load data -> apply function -> store result
        arg_eval = func_eval.arg_evaluators[1]
        val = get_data_value_specialized(data, arg_eval.column, row_idx)
        result = apply_function_direct_single(func_eval.func, Float64(val))
        @inbounds scratch[first(output_range)] = result
        return nothing
    end
    
    # For more complex functions, use internal scratch space
    if !isempty(internal_range)
        # Use the internal range as working space
        working_scratch = @view scratch[internal_range]
        
        # Execute arguments into working scratch
        for (i, arg_eval) in enumerate(func_eval.arg_evaluators)
            if i <= length(working_scratch)
                execute_component_in_assigned_scratch!(
                    arg_eval, i:i, 1:0, working_scratch, data, row_idx
                )
            end
        end
        
        # Apply function using working scratch values
        if !isempty(output_range)
            if length(func_eval.arg_evaluators) == 1 && length(working_scratch) >= 1
                result = apply_function_direct_single(func_eval.func, working_scratch[1])
                @inbounds scratch[first(output_range)] = result
            elseif length(func_eval.arg_evaluators) == 2 && length(working_scratch) >= 2
                result = apply_function_direct_binary(func_eval.func, working_scratch[1], working_scratch[2])
                @inbounds scratch[first(output_range)] = result
            elseif length(working_scratch) >= length(func_eval.arg_evaluators)
                arg_values = working_scratch[1:length(func_eval.arg_evaluators)]
                result = apply_function_direct_varargs(func_eval.func, arg_values...)
                @inbounds scratch[first(output_range)] = result
            end
        end
    else
        # If no internal scratch available, fall back to direct computation
        # This should only happen for very simple cases
        if length(func_eval.arg_evaluators) == 1 && 
           func_eval.arg_evaluators[1] isa ContinuousEvaluator &&
           !isempty(output_range)
            
            arg_eval = func_eval.arg_evaluators[1]
            val = get_data_value_specialized(data, arg_eval.column, row_idx)
            result = apply_function_direct_single(func_eval.func, Float64(val))
            @inbounds scratch[first(output_range)] = result
        end
    end
    
    return nothing
end

###############################################################################
# ADD TO step4_interactions.jl - DEBUGGING UTILITIES
###############################################################################

"""
    show_scratch_planning_info(interaction::InteractionEvaluator)

Display scratch space planning information for debugging.
"""
function show_scratch_planning_info(interaction::InteractionEvaluator)
    println("Interaction Scratch Planning:")
    println("  Total scratch needed: $(interaction.total_scratch_needed)")
    println("  Components: $(length(interaction.components))")
    
    for (i, component) in enumerate(interaction.components)
        comp_type = typeof(component).__name__
        output_range = interaction.component_scratch_map[i]
        internal_range = interaction.component_internal_scratch_map[i]
        
        println("    Component $i ($comp_type):")
        println("      Output range: $output_range")
        println("      Internal range: $internal_range")
    end
end

"""
    validate_scratch_planning(interaction::InteractionEvaluator) -> Bool

Validate that scratch space planning is consistent and non-overlapping.
"""
function validate_scratch_planning(interaction::InteractionEvaluator)
    all_positions = Set{Int}()
    
    # Check that all scratch positions are unique
    for range in interaction.component_scratch_map
        for pos in range
            if pos in all_positions
                @warn "Overlapping scratch positions detected: $pos"
                return false
            end
            push!(all_positions, pos)
        end
    end
    
    for range in interaction.component_internal_scratch_map
        for pos in range
            if pos in all_positions
                @warn "Overlapping scratch positions detected: $pos"
                return false
            end
            push!(all_positions, pos)
        end
    end
    
    # Check that total scratch needed matches allocated positions
    expected_max = isempty(all_positions) ? 0 : maximum(all_positions)
    if expected_max != interaction.total_scratch_needed
        @warn "Scratch planning mismatch: expected $expected_max, got $(interaction.total_scratch_needed)"
        return false
    end
    
    return true
end
