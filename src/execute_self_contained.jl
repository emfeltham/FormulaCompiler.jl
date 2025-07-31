# Remove ALL existing execute_self_contained! methods and replace with these:

function execute_self_contained!(evaluator::CombinedEvaluator, scratch::Vector{Float64},
                                 output::AbstractVector{Float64}, data::NamedTuple, row_idx::Int)
    
    # PRE-EXTRACT ALL FIELDS at function entry (eliminates field access allocations)
    constant_ops = evaluator.constant_ops
    continuous_ops = evaluator.continuous_ops
    categorical_evaluators = evaluator.categorical_evaluators
    function_evaluators = evaluator.function_evaluators
    interaction_evaluators = evaluator.interaction_evaluators
    
    # Now use the local variables (no more field access in loops)
    @inbounds for op in constant_ops
        output[op.position] = op.value
    end
    
    @inbounds for op in continuous_ops
        col = op.column
        pos = op.position
        
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
            data[col][row_idx]
        end
        
        output[pos] = Float64(val)
    end
    
    @inbounds for eval in categorical_evaluators
        level_codes = eval.level_codes
        cm = eval.contrast_matrix
        positions = eval.positions
        n_levels = eval.n_levels
        
        lvl = level_codes[row_idx]
        lvl = lvl < 1 ? 1 : (lvl > n_levels ? n_levels : lvl)
        
        n_contrasts = length(positions)
        
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
            for j in 1:n_contrasts
                output[positions[j]] = cm[lvl, j]
            end
        end
    end
    
    @inbounds for eval in function_evaluators
        execute_function_self_contained!(eval, scratch, output, data, row_idx)
    end
    
    @inbounds for eval in interaction_evaluators
        execute_interaction_self_contained!(eval, scratch, output, data, row_idx)
    end
    
    return nothing
end

"""
    execute_interaction_self_contained!(evaluator::InteractionEvaluator{N}, 
                                       scratch::Vector{Float64},
                                       output::AbstractVector{Float64}, 
                                       data::NamedTuple, row_idx::Int) where N

"""
@inline function execute_interaction_self_contained!(ev::InteractionEvaluator{N},
                                                     scratch::Vector{Float64},
                                                     output::Vector{Float64},
                                                     data::NamedTuple,
                                                     row_idx::Int) where N
    # 1. evaluate every component into its assigned scratch block
    @inbounds for i in eachindex(ev.components)
        r = ev.component_scratch_map[i]
        execute_to_scratch!(ev.components[i], scratch,
                            first(r), last(r), data, row_idx)
    end

    # 2. build the interaction columns from the scratch results
    apply_kronecker_pattern_to_positions!(ev.kronecker_pattern,
                                          ev.component_scratch_map,
                                          scratch,
                                          output,
                                          ev.positions)
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
    row_idx::Int)

    func            = evaluator.func
    arg_evaluators  = evaluator.arg_evaluators
    arg_scratch_map = evaluator.arg_scratch_map
    pos             = evaluator.position
    n_args          = length(arg_evaluators)

    # Gather (and if needed, compute) each argument value --------------------
    vals = ntuple(i -> begin
        arg = arg_evaluators[i]
        if arg isa ConstantEvaluator
            arg.value
        elseif arg isa ContinuousEvaluator
            Float64(data[arg.column][row_idx])
        else
            r = arg_scratch_map[i]                 # pre-allocated range
            execute_to_scratch!(arg, scratch,
                                first(r), last(r),
                                data, row_idx)     # recurse
            scratch[first(r)]                      # scalar argument → take 1st
        end
    end, n_args)

    # Apply, store -----------------------------------------------------------
    @inbounds output[pos] = apply_function_safe(func, vals...)
    return nothing
end

execute_function_self_contained!(pf::ParametricFunctionEvaluator, scratch, output, data, row_idx) =
    execute_function_self_contained!(FunctionEvaluator(pf.func,
                                                       collect(pf.arg_evaluators),
                                                       0,                      # dummy position
                                                       Int[],                  # no own scratch
                                                       collect(pf.arg_scratch_map)),
                                     scratch, output, data, row_idx)


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

"""
    execute_categorical_self_contained!(evaluator::CategoricalEvaluator, output::AbstractVector{Float64},
                                       data::NamedTuple, row_idx::Int)

Categorical execution using pre-extracted level codes.
"""
function execute_categorical_self_contained!(
    evaluator::CategoricalEvaluator, output::AbstractVector{Float64},
    data::NamedTuple, row_idx::Int
)
    # Use pre-extracted level codes (no data access!)
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