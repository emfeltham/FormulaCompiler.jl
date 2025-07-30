# Remove ALL existing execute_self_contained! methods and replace with these:

function execute_self_contained!(evaluator::CombinedEvaluator, scratch::Vector{Float64},
                                 output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)
    
    # Zero-allocation loops using precomputed operations
    @inbounds for op in evaluator.constant_ops
        output[op.position] = op.value
    end
    
    # FIXED: Type-stable data access using direct field access
    @inbounds for op in evaluator.continuous_ops
        col = op.column
        pos = op.position
        
        # Type-stable data access - eliminates Union return types
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
        elseif col === :response
            data.response[row_idx]
        elseif col === :flag
            data.flag[row_idx]
        else
            # Fallback for any other columns (will still cause union but rarely used)
            data[col][row_idx]
        end
        
        output[pos] = Float64(val)
    end
    
    @inbounds for eval in evaluator.categorical_evaluators
        level_codes = eval.level_codes
        cm = eval.contrast_matrix
        positions = eval.positions
        n_levels = eval.n_levels
        
        lvl = level_codes[row_idx]
        lvl = lvl < 1 ? 1 : (lvl > n_levels ? n_levels : lvl)
        
        n_contrasts = length(positions)
        
        # FIXED: Manual unrolling for common cases
        if n_contrasts == 1
            output[positions[1]] = cm[lvl, 1]
        elseif n_contrasts == 2
            output[positions[1]] = cm[lvl, 1]
            output[positions[2]] = cm[lvl, 2]
        elseif n_contrasts == 3
            output[positions[1]] = cm[lvl, 1]
            output[positions[2]] = cm[lvl, 2]
            output[positions[3]] = cm[lvl, 3]
        elseif n_contrasts == 4
            output[positions[1]] = cm[lvl, 1]
            output[positions[2]] = cm[lvl, 2]
            output[positions[3]] = cm[lvl, 3]
            output[positions[4]] = cm[lvl, 4]
        elseif n_contrasts == 5
            output[positions[1]] = cm[lvl, 1]
            output[positions[2]] = cm[lvl, 2]
            output[positions[3]] = cm[lvl, 3]
            output[positions[4]] = cm[lvl, 4]
            output[positions[5]] = cm[lvl, 5]
        else
            # Fallback preserves full generality
            for j in 1:n_contrasts
                output[positions[j]] = cm[lvl, j]
            end
        end
    end
    
    @inbounds for eval in evaluator.function_evaluators
        execute_function_self_contained!(eval, scratch, output, data, row_idx)
    end
    
    @inbounds for eval in evaluator.interaction_evaluators
        execute_interaction_self_contained!(eval, scratch, output, data, row_idx)
    end
    
    return nothing
end

@inline function execute_self_contained!(
    evaluator::ConstantEvaluator, 
    scratch::Vector{Float64},
    output::AbstractVector{Float64}, 
    data::NamedTuple, 
    row_idx::Int
)
    pos = evaluator.position
    val = evaluator.value
    @inbounds output[pos] = val
    return nothing
end

@inline function execute_self_contained!(
    evaluator::ContinuousEvaluator, 
    scratch::Vector{Float64},
    output::AbstractVector{Float64}, 
    data::NamedTuple, 
    row_idx::Int
)
    col = evaluator.column
    pos = evaluator.position
    @inbounds output[pos] = Float64(data[col][row_idx])
    return nothing
end

@inline function execute_self_contained!(
    evaluator::CategoricalEvaluator,
    scratch::Vector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    level_codes = evaluator.level_codes
    cm = evaluator.contrast_matrix
    positions = evaluator.positions
    n_levels = evaluator.n_levels
    
    @inbounds begin
        lvl = level_codes[row_idx]
        lvl = lvl < 1 ? 1 : (lvl > n_levels ? n_levels : lvl)
        
        for j in 1:length(positions)
            output[positions[j]] = cm[lvl, j]
        end
    end
    return nothing
end

@inline function execute_function_self_contained!(
    evaluator::FunctionEvaluator, 
    scratch::Vector{Float64},
    output::AbstractVector{Float64}, 
    data::NamedTuple, 
    row_idx::Int
)
    # Cache all field accesses at the top for type stability
    func = evaluator.func
    arg_evaluators = evaluator.arg_evaluators
    pos = evaluator.position
    
    n_args = length(arg_evaluators)
    
    if n_args == 1
        arg = arg_evaluators[1]
        val = if arg isa ConstantEvaluator
            arg.value
        elseif arg isa ContinuousEvaluator
            col = arg.column  # Cache field access
            Float64(data[col][row_idx])
        else
            error("Complex function arguments not supported")
        end
        result = apply_function_safe(func, val)
        @inbounds output[pos] = result
        
    elseif n_args == 2
        arg1, arg2 = arg_evaluators
        val1 = if arg1 isa ConstantEvaluator
            arg1.value
        elseif arg1 isa ContinuousEvaluator
            col1 = arg1.column
            Float64(data[col1][row_idx])
        else
            error("Complex function arguments not supported")
        end
        
        val2 = if arg2 isa ConstantEvaluator
            arg2.value
        elseif arg2 isa ContinuousEvaluator
            col2 = arg2.column
            Float64(data[col2][row_idx])
        else
            error("Complex function arguments not supported")
        end
        
        result = apply_function_safe(func, val1, val2)
        @inbounds output[pos] = result
        
    else
        error("Functions with $n_args arguments not supported")
    end
    
    return nothing
end

@inline function execute_self_contained!(
    evaluator::InteractionEvaluator{N}, 
    scratch::Vector{Float64},
    output::AbstractVector{Float64}, 
    data::NamedTuple, 
    row_idx::Int
) where N
    components = evaluator.components
    component_scratch_map = evaluator.component_scratch_map
    n_components = length(components)
    
    @inbounds for i in 1:n_components
        component_range = component_scratch_map[i]
        component_start = first(component_range)
        component_end = last(component_range)
        execute_to_scratch!(components[i], scratch, component_start, component_end, data, row_idx)
    end
    
    apply_kronecker_pattern_to_positions!(
        evaluator.kronecker_pattern,
        evaluator.component_scratch_map,
        scratch,
        output,
        evaluator.positions
    )
    
    return nothing
end
