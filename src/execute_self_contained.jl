# execute_self_contained.jl

"""
    execute_self_contained!(
        evaluator::CombinedEvaluator,
        scratch::Vector{Float64},
        output::AbstractVector{Float64},
        data::NamedTuple,
        row_idx::Int
    ) -> Nothing

Perform a full, in-place evaluation of a compiled evaluator tree (`CombinedEvaluator`),
writing directly into the `output` vector for the given data row.

# Arguments
- `evaluator::CombinedEvaluator`: holds pre-extracted lists of all evaluator operations:
  - `constant_ops` (positions+values)
  - `continuous_ops` (column+positions)
  - `categorical_evaluators`
  - `function_evaluators`
  - `interaction_evaluators`
- `scratch::Vector{Float64}`: working buffer for intermediate values (e.g. function args,
  interaction components).
- `output::AbstractVector{Float64}`: the model matrix row being built.
- `data::NamedTuple`: row-wise access to all columns (e.g. `data.x[row_idx]`).
- `row_idx::Int`: index of the current observation.

# Behavior
1. Pre-extract evaluator lists to locals to avoid field allocations.
2. Fill all constant positions.
3. Loop continuous ops: fetch & cast values.
4. Loop categorical evaluators: clamp codes, copy contrasts.
5. Loop function evaluators: delegate to `execute_function_self_contained!`.
6. Loop interaction evaluators: delegate to `execute_interaction_self_contained!`.

Returns `nothing`.
"""
function execute_self_contained!(
    evaluator::CombinedEvaluator,
    scratch::Vector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    # PRE-EXTRACT ALL FIELDS at function entry (eliminates field access allocations)
    constant_ops = evaluator.constant_ops
    continuous_ops = evaluator.continuous_ops
    categorical_evaluators = evaluator.categorical_evaluators
    function_evaluators = evaluator.function_evaluators
    interaction_evaluators = evaluator.interaction_evaluators

    @inbounds for op in constant_ops
        output[op.position] = op.value
    end

    @inbounds for op in continuous_ops
        val = get_data_value_specialized(data, op.column, row_idx)
        output[op.position] = Float64(val)
    end

    @inbounds for ev in categorical_evaluators
        execute_categorical_self_contained!(ev, output, row_idx)
    end

    @inbounds for ev in function_evaluators
        execute_function_self_contained!(ev, scratch, output, data, row_idx)
    end

    @inbounds for ev in interaction_evaluators
        execute_interaction_self_contained!(ev, scratch, output, data, row_idx)
    end

    return nothing
end

"""
    execute_interaction_self_contained!(
        ev::InteractionEvaluator{N},
        scratch::Vector{Float64},
        output::AbstractVector{Float64},
        data::NamedTuple,
        row_idx::Int
    ) where N -> Nothing

Evaluate each component into `scratch`, then assemble the N-way interaction
terms directly into `output` using a Kronecker pattern.

# Arguments
- `ev::InteractionEvaluator{N}`: contains:
  - `components::Vector{AbstractEvaluator}`
  - `component_scratch_map::Vector{UnitRange{Int}}`
  - `kronecker_pattern::Vector{NTuple{N,Int}}`
  - `positions::Vector{Int}` output indices for the interaction terms
- `scratch`: buffer for intermediate values.
- `output`: target vector for model columns.
- `data`, `row_idx`: as above.

# Behavior
1. Recursively `execute_to_scratch!` each component into its scratch range.
2. Call `apply_kronecker_to_output!` to multiply and scatter into `output`.

Returns `nothing`.
"""
@inline function execute_interaction_self_contained!(
    ev::InteractionEvaluator{N},
    scratch::Vector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
) where N
    @inbounds for i in eachindex(ev.components)
        r = ev.component_scratch_map[i]
        execute_to_scratch!(ev.components[i], scratch, first(r), last(r), data, row_idx)
    end
    apply_kronecker_to_output!(
        ev.kronecker_pattern,
        ev.component_scratch_map,
        scratch,
        output,
        ev.positions
    )
    return nothing
end

"""
    execute_function_self_contained!(
        evaluator::FunctionEvaluator,
        scratch::Vector{Float64},
        output::AbstractVector{Float64},
        data::NamedTuple,
        row_idx::Int
    ) -> Nothing

Evaluate a custom function evaluator and store its result.
Uses execute_to_scratch! for all arguments consistently.
"""
@inline function execute_function_self_contained!(
    evaluator::FunctionEvaluator,
    scratch::Vector{Float64},
    output::AbstractVector{Float64},
    data::NamedTuple,
    row_idx::Int
)
    n_args = length(evaluator.arg_evaluators)
    
    # Evaluate all arguments into scratch using consistent approach
    @inbounds for i in 1:n_args
        r = evaluator.arg_scratch_map[i]
        execute_to_scratch!(evaluator.arg_evaluators[i], scratch, first(r), last(r), data, row_idx)
    end
    
    # Apply function using direct methods (no varargs)
    if n_args == 1
        r = evaluator.arg_scratch_map[1]
        arg_val = scratch[first(r)]
        @inbounds output[evaluator.position] = apply_function_direct_single(evaluator.func, arg_val)
    elseif n_args == 2
        r1 = evaluator.arg_scratch_map[1]
        r2 = evaluator.arg_scratch_map[2]
        arg1_val = scratch[first(r1)]
        arg2_val = scratch[first(r2)]
        @inbounds output[evaluator.position] = apply_function_direct_binary(evaluator.func, arg1_val, arg2_val)
    else
        # For 3+ arguments, fall back to varargs but with better error message
        args = ntuple(n_args) do i
            r = evaluator.arg_scratch_map[i]
            scratch[first(r)]
        end
        @inbounds output[evaluator.position] = apply_function_direct_varargs(evaluator.func, args...)
    end
    
    return nothing
end

"""
    execute_categorical_self_contained!(
        evaluator::CategoricalEvaluator,
        pos::AbstractVector{Float64},
        row_idx::Int
    ) -> Nothing

This conceptually overlaps with the nested evaluate_to_scrach, but is
more direct for this non-nested context.
"""
@inline function execute_categorical_self_contained!(
    evaluator::CategoricalEvaluator,
    output::AbstractVector{Float64},
    row_idx::Int
)

    lvl = clamp(evaluator.level_codes[row_idx], 1, evaluator.n_levels)
    @inbounds for (j, p) in enumerate(evaluator.positions)
        output[p] = evaluator.contrast_matrix[lvl, j]
    end

    return nothing
end
