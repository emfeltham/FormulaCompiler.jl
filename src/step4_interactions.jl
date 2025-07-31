# step4_interactions.jl
# Complete interaction support with full precomputation - OPTIMIZED VERSION
# This is a COMPLETE REPLACEMENT for the original step4_interactions.jl

###############################################################################
# UNIFIED INTERACTION COMPONENT SYSTEM
###############################################################################

"""
    InteractionComponentData

Unified data for any interaction component type that WRAPS Step 1-3 optimizations.
"""
struct InteractionComponentData
    component_type::Symbol              # :constant, :continuous, :categorical, :function
    optimized_data::Any                 # Step 1-3 optimized data structures
    optimized_op::Any                   # Step 1-3 optimized operations
    scratch_range::UnitRange{Int}       # Where this component's results go in scratch
    output_width::Int                   # How many values this component produces
end

"""
    InteractionData

Complete interaction data with pre-computed Kronecker patterns.
"""
struct InteractionData
    components::Vector{InteractionComponentData}  # Now wraps optimized data
    component_widths::Vector{Int}                 # Width of each component  
    kronecker_pattern::Vector{Vector{Int}}        # Pre-computed, always
    output_positions::Vector{Int}                 # Where results go in model matrix
    total_scratch_needed::Int                     # Scratch space for components
end

###############################################################################
# COMPONENT CONVERSION FUNCTIONS - USE STEP 1-3 OPTIMIZATIONS
###############################################################################

"""
    convert_component_to_unified(component::AbstractEvaluator, scratch_start::Int) -> (InteractionComponentData, Int)

Convert any evaluator component to unified format that WRAPS Step 1-3 optimizations.
"""
function convert_component_to_unified(component::AbstractEvaluator, scratch_start::Int)
    if component isa ConstantEvaluator
        # Wrap Step 1 constant optimization
        const_data = ConstantData((component.value,), (1,))  # Dummy position
        const_op = ConstantOp(const_data)
        
        return InteractionComponentData(
            :constant,
            const_data,
            const_op,
            scratch_start:scratch_start,
            1
        ), scratch_start + 1
        
    elseif component isa ContinuousEvaluator
        # Wrap Step 1 continuous optimization
        cont_data = ContinuousData((component.column,), (1,))  # Dummy position
        cont_op = ContinuousOp(cont_data)
        
        return InteractionComponentData(
            :continuous,
            cont_data,
            cont_op,
            scratch_start:scratch_start,
            1
        ), scratch_start + 1
        
    elseif component isa CategoricalEvaluator
        # Wrap Step 2 categorical optimization
        # Use correct CategoricalData constructor (contrast_matrix, level_codes, positions, n_levels)
        cat_data = CategoricalData(
            component.contrast_matrix,
            component.level_codes,
            collect(1:length(component.positions)),  # Dummy positions
            component.n_levels
        )
        cat_op = CategoricalOp()
        
        n_contrasts = length(component.positions)
        scratch_end = scratch_start + n_contrasts - 1
        
        return InteractionComponentData(
            :categorical,
            cat_data,
            cat_op,
            scratch_start:scratch_end,
            n_contrasts
        ), scratch_end + 1
        
    elseif component isa FunctionEvaluator
        # Wrap Step 3 function optimization
        linear_func_data = flatten_function_to_linear_plan(component, 1)  # Dummy position
        func_op = LinearFunctionOp()
        
        return InteractionComponentData(
            :function,
            linear_func_data,
            func_op,
            scratch_start:scratch_start,
            1
        ), scratch_start + 1
        
    else
        error("Unsupported component type for interactions: $(typeof(component))")
    end
end

###############################################################################
# KRONECKER PATTERN COMPUTATION
###############################################################################

"""
    compute_generalized_kronecker_pattern(component_widths::Vector{Int}) -> Vector{Vector{Int}}

Compute full Kronecker pattern for any N-way interaction. Always precomputes everything.
"""
function compute_generalized_kronecker_pattern(component_widths::Vector{Int})
    N = length(component_widths)
    
    if N == 0
        return Vector{Vector{Int}}[]
    elseif N == 1
        # Single component - trivial pattern
        return [[i] for i in 1:component_widths[1]]
    end
    
    total_terms = prod(component_widths)
    
    # Pre-allocate the full pattern
    pattern = Vector{Vector{Int}}(undef, total_terms)
    
    # Generate all combinations using Cartesian indices
    ranges = Tuple(1:w for w in component_widths)
    
    idx = 1
    for combo in Iterators.product(ranges...)
        pattern[idx] = collect(combo)  # Convert tuple to vector
        idx += 1
    end
    
    return pattern
end

###############################################################################
# INTERACTION ANALYSIS
###############################################################################

"""
    analyze_interaction_operations(evaluator::CombinedEvaluator) -> (Vector{InteractionData}, InteractionOp)

Extract and convert all interaction evaluators using Step 1-3 optimizations.
"""
function analyze_interaction_operations(evaluator::CombinedEvaluator)
    interaction_evaluators = evaluator.interaction_evaluators
    n_interactions = length(interaction_evaluators)
    
    if n_interactions == 0
        # No interactions
        return InteractionData[], InteractionOp()
    end
    
    interaction_data = Vector{InteractionData}(undef, n_interactions)
    
    for (i, interaction_eval) in enumerate(interaction_evaluators)
        # Convert all components using OPTIMIZED Step 1-3 wrappers
        components = Vector{InteractionComponentData}()
        component_widths = Int[]
        current_scratch_pos = 1
        
        for component in interaction_eval.components
            # Convert component using optimized wrappers
            component_data, next_scratch_pos = convert_component_to_unified(component, current_scratch_pos)
            
            push!(components, component_data)
            push!(component_widths, component_data.output_width)
            
            # Update scratch position for next component
            current_scratch_pos = next_scratch_pos
        end
        
        total_scratch_needed = current_scratch_pos - 1
        
        # Precompute the full Kronecker pattern (no limits!)
        kronecker_pattern = compute_generalized_kronecker_pattern(component_widths)
        
        # Create interaction data with optimized components
        interaction_data[i] = InteractionData(
            components,
            component_widths,
            kronecker_pattern,
            collect(interaction_eval.positions),
            total_scratch_needed
        )
        
        # Debug info for large interactions
        total_terms = length(kronecker_pattern)
        if total_terms > 10_000
            println("⚠️  Large interaction detected: $(total_terms) terms, estimated memory: $(round(total_terms * length(component_widths) * 8 / 1024^2, digits=1)) MB")
        end
    end
    
    return interaction_data, InteractionOp()
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
        # Analyze all operation types (Steps 1-3 unchanged)
        constant_data, constant_op = analyze_constant_operations(evaluator)
        continuous_data, continuous_op = analyze_continuous_operations(evaluator)
        categorical_data, categorical_op = analyze_categorical_operations(evaluator)
        function_data, function_op = analyze_function_operations_linear(evaluator)
        
        # Use OPTIMIZED interaction analysis
        interaction_data, interaction_op = analyze_interaction_operations(evaluator)
        
        max_function_scratch = isempty(function_data) ? 0 : maximum(f.scratch_size for f in function_data)
        max_interaction_scratch = isempty(interaction_data) ? 0 : maximum(i.total_scratch_needed for i in interaction_data)

        # Pre-allocate once
        function_scratch = max_function_scratch > 0 ? Vector{Float64}(undef, max_function_scratch) : Float64[]
        interaction_scratch = max_interaction_scratch > 0 ? Vector{Float64}(undef, max_interaction_scratch) : Float64[]

        # Construct data with optimized interactions
        formula_data = CompleteFormulaData(
            constant_data,
            continuous_data,
            categorical_data,
            function_data,
            interaction_data,  # Now contains optimized Step 1-3 wrappers
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
    evaluate_unified_component!(component::InteractionComponentData,
                               scratch::Vector{Float64},
                               data::NamedTuple,
                               row_idx::Int)

Use pre-computed position mappings, zero manual arithmetic.
"""
function evaluate_unified_component!(component::InteractionComponentData,
                                   scratch::Vector{Float64},
                                   data::NamedTuple,
                                   row_idx::Int)
    
    if component.component_type === :constant
        # Position mapping: component.scratch_range tells us exactly where to write
        const_data = component.optimized_data::ConstantData
        output_pos = first(component.scratch_range)  # Position map!
        @inbounds scratch[output_pos] = const_data.values[1]
        
    elseif component.component_type === :continuous
        # Position mapping: component.scratch_range tells us exactly where to write
        cont_data = component.optimized_data::ContinuousData
        output_pos = first(component.scratch_range)  # Position map!
        col = cont_data.columns[1]
        val = get_data_value_specialized(data, col, row_idx)
        @inbounds scratch[output_pos] = Float64(val)
        
    elseif component.component_type === :categorical
        # Position mapping: component.scratch_range tells us exactly where to write
        cat_data = component.optimized_data::CategoricalData
        level = cat_data.level_codes[row_idx]
        level = clamp(level, 1, cat_data.n_levels)
        
        # Write to positions specified by the position map
        @inbounds for i in 1:component.output_width
            output_pos = first(component.scratch_range) + i - 1  # Position map!
            scratch[output_pos] = cat_data.contrast_matrix[level, i]
        end
        
    elseif component.component_type === :function
        # POSITION MAPPING: Execute function, use position map for result placement
        func_data = component.optimized_data::LinearFunctionData
        output_pos = first(component.scratch_range)  # Position map tells us where result goes!
        
        # Execute function to get result, place using position map
        result = execute_function_via_position_mapping(func_data, data, row_idx)
        @inbounds scratch[output_pos] = result  # Position map handles placement!
        
    else
        error("Unknown component type: $(component.component_type)")
    end
    
    return nothing
end

"""
    execute_function_via_position_mapping(func_data::LinearFunctionData, data::NamedTuple, row_idx::Int) -> Float64

Execute function and return result - position mapping handles placement.
"""
function execute_function_via_position_mapping(func_data::LinearFunctionData, data::NamedTuple, row_idx::Int)
    # Create minimal scratch space for function's internal execution
    # This is the ONLY allocation, but it's minimal and isolated
    if func_data.scratch_size > 0
        internal_scratch = Vector{Float64}(undef, func_data.scratch_size)
    else
        internal_scratch = Float64[]
    end
    
    # Execute function steps in internal scratch space (positions 1, 2, 3...)
    @inbounds for step in func_data.execution_steps
        if step.operation === :load_constant
            internal_scratch[step.output_position] = step.constant_value
            
        elseif step.operation === :load_continuous
            col = step.column_symbol
            val = get_data_value_specialized(data, col, row_idx)
            internal_scratch[step.output_position] = Float64(val)
            
        elseif step.operation === :call_unary
            input_val = internal_scratch[step.input_positions[1]]
            result = apply_function_direct_single(step.func, input_val)
            internal_scratch[step.output_position] = result
            
        elseif step.operation === :call_binary
            input_val1 = internal_scratch[step.input_positions[1]]
            input_val2 = internal_scratch[step.input_positions[2]]
            result = apply_function_direct_binary(step.func, input_val1, input_val2)
            internal_scratch[step.output_position] = result
            
        else
            error("Unknown operation type: $(step.operation)")
        end
    end
    
    # Return the final result (last step's output position)
    if !isempty(func_data.execution_steps)
        final_step = func_data.execution_steps[end]
        return internal_scratch[final_step.output_position]
    else
        return 0.0
    end
end

"""
    execute_interaction_operation!(interaction_data::InteractionData,
                                  scratch::Vector{Float64},
                                  output::Vector{Float64},
                                  data::NamedTuple,
                                  row_idx::Int)

ALL-IN POSITION MAPPING: Use position mappings throughout Kronecker application.
"""
function execute_interaction_operation!(interaction_data::InteractionData,
                                       scratch::Vector{Float64},
                                       output::Vector{Float64},
                                       data::NamedTuple,
                                       row_idx::Int)
    
    # Step 1: Execute all components using position mappings
    @inbounds for component in interaction_data.components
        evaluate_unified_component!(component, scratch, data, row_idx)
    end
    
    # Step 2: Apply Kronecker pattern using position mappings
    @inbounds for (result_idx, pattern_indices) in enumerate(interaction_data.kronecker_pattern)
        if result_idx <= length(interaction_data.output_positions)
            product = 1.0
            
            # Use position mappings to find component values
            @inbounds for (comp_idx, pattern_val) in enumerate(pattern_indices)
                component = interaction_data.components[comp_idx]
                # Position mapping: component.scratch_range tells us where to read
                value_pos = first(component.scratch_range) + pattern_val - 1  # Position map!
                product *= scratch[value_pos]
            end
            
            # Position mapping: interaction_data.output_positions tells us where to write
            output_pos = interaction_data.output_positions[result_idx]  # Position map!
            output[output_pos] = product
        end
    end
    
    return nothing
end

"""
    execute_interaction_operations!(interaction_data::Vector{InteractionData},
                                   scratch::Vector{Float64},
                                   output::Vector{Float64},
                                   data::NamedTuple,
                                   row_idx::Int)

"""
function execute_interaction_operations!(interaction_data::Vector{InteractionData},
                                        scratch::Vector{Float64},
                                        output::Vector{Float64},
                                        data::NamedTuple,
                                        row_idx::Int)
    # Handle empty case
    if isempty(interaction_data)
        return nothing
    end
    
    # Process all interactions
    @inbounds for interaction in interaction_data
        execute_interaction_operation!(interaction, scratch, output, data, row_idx)
    end
    
    return nothing
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
    execute_interaction_operations!(data.interactions, is, output, input_data, row_idx)
    
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

"""
    show_interaction_info(interaction_data::InteractionData)

Display information about an interaction.
"""
function show_interaction_info(interaction_data::InteractionData)
    println("Interaction Information:")
    println("  Components: $(length(interaction_data.components))")
    println("  Component widths: $(interaction_data.component_widths)")
    println("  Total terms: $(length(interaction_data.kronecker_pattern))")
    println("  Scratch space needed: $(interaction_data.total_scratch_needed)")
    println("  Output positions: $(interaction_data.output_positions)")
    
    # Memory estimate
    pattern_memory = length(interaction_data.kronecker_pattern) * length(interaction_data.component_widths) * 8
    println("  Pattern memory: $(round(pattern_memory / 1024, digits=1)) KB")
    
    if pattern_memory > 1024^2
        println("  ⚠️  Large interaction: $(round(pattern_memory / 1024^2, digits=1)) MB")
    end
end

"""
    execute_function_component_in_scratch!(func_data::LinearFunctionData,
                                          scratch::Vector{Float64},
                                          output_pos::Int,
                                          data::NamedTuple,
                                          row_idx::Int)

Execute function component using direct scratch space.
"""
function execute_function_component_in_scratch!(func_data::LinearFunctionData,
                                               scratch::Vector{Float64},
                                               output_pos::Int,
                                               data::NamedTuple,
                                               row_idx::Int)
    
    # Calculate scratch space offset for this function's operations
    # Use positions after the output position for intermediate calculations
    scratch_base = output_pos + 1
    
    # Execute each step directly in scratch space
    @inbounds for step in func_data.execution_steps
        # Map function scratch positions to actual scratch positions
        actual_pos = if step.output_position == 1
            output_pos  # Final result goes to component output position
        else
            scratch_base + step.output_position - 2  # Intermediate results
        end
        
        if step.operation === :load_constant
            scratch[actual_pos] = step.constant_value
            
        elseif step.operation === :load_continuous
            col = step.column_symbol
            val = get_data_value_specialized(data, col, row_idx)
            scratch[actual_pos] = Float64(val)
            
        elseif step.operation === :call_unary
            input_pos = if step.input_positions[1] == 1
                output_pos
            else
                scratch_base + step.input_positions[1] - 2
            end
            input_val = scratch[input_pos]
            result = apply_function_direct_single(step.func, input_val)
            scratch[actual_pos] = result
            
        elseif step.operation === :call_binary
            input_pos1 = if step.input_positions[1] == 1
                output_pos
            else
                scratch_base + step.input_positions[1] - 2
            end
            input_pos2 = if step.input_positions[2] == 1
                output_pos
            else
                scratch_base + step.input_positions[2] - 2
            end
            
            input_val1 = scratch[input_pos1]
            input_val2 = scratch[input_pos2]
            result = apply_function_direct_binary(step.func, input_val1, input_val2)
            scratch[actual_pos] = result
            
        else
            error("Unknown operation type: $(step.operation)")
        end
    end
    
    return nothing
end

"""
    execute_linear_function_in_place!(func_data::LinearFunctionData,
                                     scratch::Vector{Float64},
                                     output_pos::Int,
                                     data::NamedTuple,
                                     row_idx::Int)

Execute linear function directly in the provided scratch space without any allocations.
"""
function execute_linear_function_in_place!(func_data::LinearFunctionData,
                                          scratch::Vector{Float64},
                                          output_pos::Int,
                                          data::NamedTuple,
                                          row_idx::Int)
    
    # Use scratch space directly for function execution
    # Map function scratch positions to actual scratch positions
    scratch_offset = output_pos - 1  # Offset for this function's scratch space
    
    # Execute each step directly in the scratch space
    @inbounds for step in func_data.execution_steps
        actual_output_pos = scratch_offset + step.output_position
        
        if step.operation === :load_constant
            scratch[actual_output_pos] = step.constant_value
            
        elseif step.operation === :load_continuous
            col = step.column_symbol
            val = get_data_value_specialized(data, col, row_idx)
            scratch[actual_output_pos] = Float64(val)
            
        elseif step.operation === :call_unary
            actual_input_pos = scratch_offset + step.input_positions[1]
            input_val = scratch[actual_input_pos]
            result = apply_function_direct_single(step.func, input_val)
            scratch[actual_output_pos] = result
            
        elseif step.operation === :call_binary
            actual_input_pos1 = scratch_offset + step.input_positions[1]
            actual_input_pos2 = scratch_offset + step.input_positions[2]
            input_val1 = scratch[actual_input_pos1]
            input_val2 = scratch[actual_input_pos2]
            result = apply_function_direct_binary(step.func, input_val1, input_val2)
            scratch[actual_output_pos] = result
            
        else
            error("Unknown operation type: $(step.operation)")
        end
    end
    
    # Final result should already be in the correct position
    # No copying needed since we used the scratch space directly
    
    return nothing
end
