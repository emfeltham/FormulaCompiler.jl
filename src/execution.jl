# execution.jl

###############################################################################
# ENSURE ALL EVALUATOR TYPES HAVE execute_self_contained! METHODS
###############################################################################

###############################################################################
# INTERACTION EVALUATOR METHOD
###############################################################################

"""
    execute_categorical_self_contained!(evaluator::CategoricalEvaluator, output::AbstractVector{Float64},
                                       data::NamedTuple, row_idx::Int)

Zero-allocation categorical execution using built-in positions.
"""
function execute_categorical_self_contained!(
    evaluator::CategoricalEvaluator, output::AbstractVector{Float64},
    data::NamedTuple, row_idx::Int
)
    
    # Get categorical value and level
    @inbounds cat_val = data[evaluator.column][row_idx]
    level_idx = extract_categorical_level_fast(cat_val, evaluator.n_levels)
    
    # Write contrasts directly to known positions
    contrast_matrix = evaluator.contrast_matrix

    positions = evaluator.positions
    @inbounds for j in 1:length(positions)
        pos = positions[j]
        output[pos] = contrast_matrix[level_idx, j]
    end
    
    return nothing
end

"""
    apply_kronecker_to_scratch_range!(pattern::Vector{NTuple{N,Int}},
                                     component_scratch_map::Vector{UnitRange{Int}},
                                     scratch::Vector{Float64},
                                     output_start::Int,
                                     output_end::Int) where N

Apply Kronecker pattern directly to a scratch range.
"""
function apply_kronecker_to_scratch_range!(
    pattern::Vector{NTuple{N,Int}},
    maps::Vector{UnitRange{Int}},
    scratch::Vector{Float64},
    sstart::Int,
    send::Int
) where N
    @inbounds for idx in 1:length(pattern)
        prod = 1.0
        inds = pattern[idx]
        # no view: compute each component by direct indexing
        for j in 1:N
            r = maps[j]
            prod *= scratch[first(r) + inds[j] - 1]
        end
        scratch[sstart + idx - 1] = prod
    end
end

"""
    execute_interaction_self_contained!(evaluator::InteractionEvaluator{N}, 
                                       scratch::Vector{Float64},
                                       output::AbstractVector{Float64}, 
                                       data::NamedTuple, row_idx::Int) where N

"""
function execute_interaction_self_contained!(evaluator::InteractionEvaluator{N}, 
                                            scratch::Vector{Float64},
                                            output::AbstractVector{Float64}, 
                                            data::NamedTuple, row_idx::Int) where N
    components = evaluator.components
    component_scratch_map = evaluator.component_scratch_map
    n_components = length(components)
    
    @inbounds for i in 1:n_components
        comp = components[i]
        comp_range = component_scratch_map[i]
        comp_start = first(comp_range)
        
        # FIXED: Use type-stable data access like in the main execution
        if comp isa ContinuousEvaluator
            col = comp.column
            # Type-stable data access
            val = if col === :x
                data.x[row_idx]
            elseif col === :y
                data.y[row_idx] 
            elseif col === :z
                data.z[row_idx]
            elseif col === :w
                data.w[row_idx]
            elseif col === :t
                data.t[row_idx]
            else
                data[col][row_idx]  # Fallback
            end
            scratch[comp_start] = Float64(val)
            
        elseif comp isa ConstantEvaluator
            scratch[comp_start] = comp.value
            
        elseif comp isa CategoricalEvaluator
            # Use pre-extracted level codes
            level_codes = comp.level_codes
            cm = comp.contrast_matrix
            n_levels = comp.n_levels
            
            lvl = level_codes[row_idx]
            lvl = lvl < 1 ? 1 : (lvl > n_levels ? n_levels : lvl)
            
            comp_end = last(comp_range)
            n_contrasts = comp_end - comp_start + 1
            
            # FIXED: Manual unrolling for common cases, fallback preserves generality
            if n_contrasts == 1
                scratch[comp_start] = cm[lvl, 1]
            elseif n_contrasts == 2
                scratch[comp_start] = cm[lvl, 1]
                scratch[comp_start + 1] = cm[lvl, 2]
            elseif n_contrasts == 3
                scratch[comp_start] = cm[lvl, 1]
                scratch[comp_start + 1] = cm[lvl, 2]
                scratch[comp_start + 2] = cm[lvl, 3]
            elseif n_contrasts == 4
                scratch[comp_start] = cm[lvl, 1]
                scratch[comp_start + 1] = cm[lvl, 2]
                scratch[comp_start + 2] = cm[lvl, 3]
                scratch[comp_start + 3] = cm[lvl, 4]
            elseif n_contrasts == 5
                scratch[comp_start] = cm[lvl, 1]
                scratch[comp_start + 1] = cm[lvl, 2]
                scratch[comp_start + 2] = cm[lvl, 3]
                scratch[comp_start + 3] = cm[lvl, 4]
                scratch[comp_start + 4] = cm[lvl, 5]
            else
                # Fallback preserves full generality for any size
                for k in 1:n_contrasts
                    scratch[comp_start + k - 1] = cm[lvl, k]
                end
            end
        elseif comp isa FunctionEvaluator
            # Keep existing function logic for now
            func = comp.func
            arg_evaluators = comp.arg_evaluators
            if length(arg_evaluators) == 1
                arg = arg_evaluators[1]
                val = if arg isa ConstantEvaluator
                    arg.value
                elseif arg isa ContinuousEvaluator
                    col = arg.column
                    # FIXED: Type-stable data access for function args too
                    if col === :x
                        data.x[row_idx]
                    elseif col === :y
                        data.y[row_idx]
                    elseif col === :z
                        data.z[row_idx]
                    else
                        data[col][row_idx]
                    end
                else
                    error("Complex function arguments not supported in interactions yet")
                end
                scratch[comp_start] = apply_function_safe(func, val)
            else
                error("Multi-argument functions in interactions not supported yet")
            end
        else
            error("Unsupported component type in interaction: $(typeof(comp))")
        end
    end
    
    # Apply Kronecker pattern
    apply_kronecker_pattern_to_positions!(
        evaluator.kronecker_pattern,
        evaluator.component_scratch_map,
        scratch,
        output,
        evaluator.positions
    )
    
    return nothing
end

"""
    execute_zscore_self_contained!(evaluator::ZScoreEvaluator, scratch::Vector{Float64},
                                  output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)

UPDATED: Remove view allocation.
"""
function execute_zscore_self_contained!(evaluator::ZScoreEvaluator, scratch::Vector{Float64},
                                       output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)
    
    # Evaluate underlying into scratch space - FIXED: no view allocation
    underlying_start = first(evaluator.underlying_scratch_map)
    underlying_end = last(evaluator.underlying_scratch_map)
    execute_to_scratch!(evaluator.underlying, scratch, underlying_start, underlying_end, data, row_idx)
    
    # Apply Z-score transformation: (x - center) / scale
    center = evaluator.center
    scale = evaluator.scale
    
    positions = evaluator.positions
    @inbounds for i in 1:length(positions)
        pos        = positions[i]
        scratch_val = scratch[underlying_start + i - 1]
        if scale ≈ 0.0
            output[pos] = scratch_val ≈ center ? 0.0 : (scratch_val > center ? Inf : -Inf)
        else
            output[pos] = (scratch_val - center) / scale
        end
    end
    
    return nothing
end

"""
    execute_scaled_self_contained!(evaluator::ScaledEvaluator, scratch::Vector{Float64},
                                  output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)

UPDATED: Remove view allocation.
"""
function execute_scaled_self_contained!(evaluator::ScaledEvaluator, scratch::Vector{Float64},
                                       output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)
    
    # Evaluate underlying into scratch space - FIXED: no view allocation
    underlying_start = first(evaluator.underlying_scratch_map)
    underlying_end = last(evaluator.underlying_scratch_map)
    execute_to_scratch!(evaluator.evaluator, scratch, underlying_start, underlying_end, data, row_idx)
    
    # Apply scaling: scale_factor * value
    scale_factor = evaluator.scale_factor
    
    positions = evaluator.positions
    @inbounds for i in 1:length(positions)
        pos        = positions[i]
        scratch_val = scratch[underlying_start + i - 1]
        output[pos] = scale_factor * scratch_val
    end
    
    return nothing
end

"""
    execute_product_self_contained!(evaluator::ProductEvaluator, scratch::Vector{Float64},
                                   output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)

UPDATED: Remove dynamic vector allocation.
"""
function execute_product_self_contained!(evaluator::ProductEvaluator, scratch::Vector{Float64},
                                        output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)
    
    # Evaluate each component into its assigned scratch space
    components = evaluator.components
    component_scratch_map = evaluator.component_scratch_map
    n_components = length(components)
    @inbounds for i in 1:n_components
        component_range = component_scratch_map[i]
        component_start = first(component_range)
        component_end = last(component_range)
        execute_to_scratch!(components[i], scratch, component_start, component_end, data, row_idx)
    end
    
    # Compute product - FIXED: no dynamic vector allocation
    product = 1.0
    @inbounds for i in 1:n_components
        component_range = evaluator.component_scratch_map[i]
        val = scratch[first(component_range)]  # Products assume scalar components
        product *= val
    end
    
    @inbounds output[evaluator.position] = product
    
    return nothing
end

"""
    apply_function_safe(func::Function, args...)

Safe function application with domain checking.
"""
function apply_function_safe(func::Function, args...)
    if length(args) == 1
        val = args[1]
        if func === log
            return val > 0.0 ? log(val) : (val == 0.0 ? -Inf : NaN)
        elseif func === exp
            return exp(clamp(val, -700.0, 700.0))
        elseif func === sqrt
            return val ≥ 0.0 ? sqrt(val) : NaN
        elseif func === abs
            return abs(val)
        elseif func === sin
            return sin(val)
        elseif func === cos
            return cos(val)
        elseif func === tan  # ← Add this if you need it
            return tan(val)
        else
            return Float64(func(val))
        end
    elseif length(args) == 2
        val1, val2 = args[1], args[2]
        if func === (+)
            return val1 + val2
        elseif func === (-)
            return val1 - val2
        elseif func === (*)
            return val1 * val2
        elseif func === (/)
            return val2 == 0.0 ? (val1 == 0.0 ? NaN : (val1 > 0.0 ? Inf : -Inf)) : val1 / val2
        elseif func === (^)
            if val1 == 0.0 && val2 < 0.0
                return Inf
            elseif val1 < 0.0 && !isinteger(val2)
                return NaN
            else
                return val1^val2
            end
        else
            return Float64(func(val1, val2))
        end
    else
        return Float64(func(args...))
    end
end

"""
    is_direct_evaluatable(evaluator::AbstractEvaluator) -> Bool

Check if evaluator can be evaluated directly without scratch space.
"""
function is_direct_evaluatable(evaluator::AbstractEvaluator)
    return evaluator isa ConstantEvaluator || evaluator isa ContinuousEvaluator
end

###############################################################################
# HELPER FUNCTIONS FOR EXECUTION PLAN CREATION
###############################################################################

"""
    create_function_operations_from_scratch_map(evaluator::FunctionEvaluator) -> Vector{FunctionOp}

Create function operations using evaluator's scratch mapping.
"""
function create_function_operations_from_scratch_map(evaluator::FunctionEvaluator)
    # For now, return simple operation - complex decomposition would go here
    operations = FunctionOp[]
    
    # Create operation that reads from scratch and writes to output
    input_sources = InputSource[]
    for scratch_range in evaluator.arg_scratch_map
        if length(scratch_range) == 1
            push!(input_sources, ScratchSource(first(scratch_range)))
        else
            error("Multi-output arguments not yet supported in function operations")
        end
    end
    
    output_dest = OutputPosition(evaluator.position)
    op = FunctionOp(evaluator.func, input_sources, output_dest)
    push!(operations, op)
    
    return operations
end

###############################################################################
# OPTIMIZED KRONECKER PATTERN APPLICATION
###############################################################################

"""
    apply_kronecker_pattern_to_positions!(pattern::Vector{NTuple{N,Int}},
                                         component_scratch_map::Vector{UnitRange{Int}},
                                         scratch::Vector{Float64},
                                         output::AbstractVector{Float64},
                                         output_positions::Vector{Int}) where N

UPDATED: Apply Kronecker pattern to specific positions without enumerate().
Overwrites old method.
"""
function apply_kronecker_pattern_to_positions!(
    pattern::Vector{NTuple{N,Int}},
    component_scratch_map::Vector{UnitRange{Int}},
    scratch::Vector{Float64},
    output::AbstractVector{Float64},
    output_positions::Vector{Int}
) where N
    
    pattern_length = length(pattern)
    
    @inbounds for idx in 1:pattern_length
        if idx <= length(output_positions)
            indices = pattern[idx]
            
            # Type-stable computation with compile-time known N
            product = 1.0
            for i in 1:N
                scratch_pos = first(component_scratch_map[i]) + indices[i] - 1
                product *= scratch[scratch_pos]
            end
            
            output[output_positions[idx]] = product
        end
    end
    
    return nothing
end
