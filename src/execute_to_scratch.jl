# execute_to_scratch.jl

"""
    execute_to_scratch!()

# Behavior
1. **Constants**: write a single value at `scratch_start`.
2. **Continuous**: read one element from `data` and write at `scratch_start`.
3. **Categorical**: look up the pre-extracted level code, clamp to `[1,n_levels]`,
   then copy the appropriate row of the contrast matrix into
   `scratch[scratch_start:scratch_end]`.
4. **Functions**: recursively compute all argument values (using `execute_to_scratch!`
   for any nested evaluators) into their assigned subranges, then call
   `apply_function_*(func, args...)` and write the scalar result at
   `scratch_start`. (NO LONGER TRUE)
5. **Interactions**: first recurse on each component, writing into their
   individual scratch subranges; then invoke
   `apply_kronecker_to_scratch_range!` to build all cross-terms.

Mutating, returns `nothing`.
Designed for efficient, type-stable inner loops.
"""

#=== Fallback for unknown evaluators ===#

"""
    execute_to_scratch!(evaluator::AbstractEvaluator, scratch, s0, s1, data, row_idx)

Catch-all definition: throws an error if no subtype-specific method exists.

# Arguments
- `evaluator::AbstractEvaluator`: Any evaluator subtype not explicitly handled.
- `scratch`, `s0`, `s1`, `data`, `row_idx`: As in subtype methods.

# Throws
`ErrorException` stating method is unimplemented for the given type.
"""
function execute_to_scratch!(
    evaluator::AbstractEvaluator,
    scratch::Vector{Float64},
    s0::Int,
    s1::Int,
    data::NamedTuple,
    row_idx::Int
)
    error("execute_to_scratch! not implemented for $(typeof(evaluator))")
end

#=== Concrete evaluator methods ===#

"""
    execute_to_scratch!(ev::ConstantEvaluator, scratch, s0, s1, data, row_idx)

Write a constant value into the scratch buffer.

# Arguments
- `ev::ConstantEvaluator`: Holds a scalar `value`.
- `scratch[s0]` is set to `ev.value`.

# Example
```julia
execute_to_scratch!(ConstantEvaluator(3.14), scratch, 2, 2, data, 1)
# scratch[2] == 3.14
```
"""
@inline function execute_to_scratch!(
    ev::ConstantEvaluator,
    scratch::Vector{Float64},
    s0::Int,
    s1::Int,
    data::NamedTuple,
    row_idx::Int
)
    @inbounds scratch[s0] = ev.value
    return nothing
end

"""
    execute_to_scratch!(ev::ContinuousEvaluator, scratch, s0, s1, data, row_idx)

Fetch and cast a continuous predictor value from `data` into scratch.

# Arguments
- `ev::ContinuousEvaluator`: Contains `column::Symbol`.
- Reads `data[column][row_idx]` via `get_data_value_specialized`.
- Stores `Float64(val)` at `scratch[s0]`.
"""
@inline function execute_to_scratch!(
    ev::ContinuousEvaluator,
    scratch::Vector{Float64},
    s0::Int,
    s1::Int,
    data::NamedTuple,
    row_idx::Int
)
    val = get_data_value_specialized(data, ev.column, row_idx)
    @inbounds scratch[s0] = Float64(val)
    return nothing
end

"""
    execute_to_scratch!(ev::CategoricalEvaluator, scratch, s0, s1, data, row_idx)

Lookup a level code, clamp it, and copy the corresponding row of the contrast matrix.

# Arguments
- `ev::CategoricalEvaluator`: Fields:
  - `level_codes::Vector{Int}`
  - `contrast_matrix::AbstractMatrix{Float64}`
  - `n_levels::Int`
- Clamps `level_codes[row_idx]` between 1 and `n_levels`.
- Invokes `write_contrasts!` to fill `scratch[s0:s1]`.
"""
@inline function execute_to_scratch!(
    ev::CategoricalEvaluator,
    scratch::Vector{Float64},
    s0::Int,
    s1::Int,
    data::NamedTuple,
    row_idx::Int
)
    lvl = clamp(ev.level_codes[row_idx], 1, ev.n_levels)
    write_contrasts!(scratch, s0, s1, ev.contrast_matrix, lvl)
    return nothing
end

"""
    write_contrasts!(scratch, s0::Int, s1::Int, cm::AbstractMatrix{<:Float64}, lvl::Int)

Copy a slice of one row of the contrast matrix into a contiguous region of `scratch`.

# Arguments
- `scratch::Vector{Float64}`: Pre-allocated buffer for intermediate values.
- `s0::Int`, `s1::Int`: Inclusive start and end indices in `scratch`.
- `cm::AbstractMatrix{<:Float64}`: Contrast matrix, rows = levels, columns = contrasts.
- `lvl::Int`: The clamped level index (1 ≤ lvl ≤ size(cm,1)).

# Behavior
Writes `cm[lvl, 1:(s1-s0+1)]` into `scratch[s0:s1]` with inbounds indexing.

# Example
```julia
write_contrasts!(scratch, 5, 7, cm, 2)
# copies cm[2,1], cm[2,2], cm[2,3] into scratch[5], scratch[6], scratch[7]
```
"""
@inline function write_contrasts!(
    scratch::Vector{Float64},
    s0::Int,
    s1::Int,
    cm::AbstractMatrix{<:Float64},
    lvl::Int
)
    n = s1 - s0 + 1
    @inbounds for j in 1:n
        scratch[s0 + j - 1] = cm[lvl, j]
    end
    return nothing
end

"""
    execute_to_scratch!(ev::FunctionEvaluator, scratch, s0, s1, data, row_idx)

Compute all argument evaluators, then apply a user-defined function.
SIMPLIFIED: Uses execute_to_scratch! for all arguments consistently.

# Arguments
- `ev::FunctionEvaluator`: Fields:
  - `func::Function`
  - `arg_evaluators::Vector{AbstractEvaluator}`
  - `arg_scratch_map::Vector{UnitRange{Int}}`

# Behavior
1. Recursively evaluates each argument into its scratch region.
2. Gathers scalars and calls `apply_function_safe(func, args...)`.
3. Writes the result at `scratch[s0]`.
"""
function execute_to_scratch!(
    ev::FunctionEvaluator,
    scratch::Vector{Float64},
    s0::Int,
    s1::Int,
    data::NamedTuple,
    row_idx::Int
)
    n_args = length(ev.arg_evaluators)
    
    if n_args == 1
        # Single argument - most common case, avoid ntuple
        arg = ev.arg_evaluators[1]
        arg_val = if arg isa ConstantEvaluator
            arg.value
        elseif arg isa ContinuousEvaluator
            Float64(get_data_value_specialized(data, arg.column, row_idx))
        else
            r = ev.arg_scratch_map[1]
            execute_to_scratch!(arg, scratch, first(r), last(r), data, row_idx)
            scratch[first(r)]
        end
        @inbounds scratch[s0] = apply_function_direct_single(ev.func, arg_val)
        
    elseif n_args == 2
        # Binary function - avoid ntuple
        arg1 = ev.arg_evaluators[1]
        arg1_val = if arg1 isa ConstantEvaluator
            arg1.value
        elseif arg1 isa ContinuousEvaluator
            Float64(get_data_value_specialized(data, arg1.column, row_idx))
        else
            r = ev.arg_scratch_map[1]
            execute_to_scratch!(arg1, scratch, first(r), last(r), data, row_idx)
            scratch[first(r)]
        end
        
        arg2 = ev.arg_evaluators[2]
        arg2_val = if arg2 isa ConstantEvaluator
            arg2.value
        elseif arg2 isa ContinuousEvaluator
            Float64(get_data_value_specialized(data, arg2.column, row_idx))
        else
            r = ev.arg_scratch_map[2]
            execute_to_scratch!(arg2, scratch, first(r), last(r), data, row_idx)
            scratch[first(r)]
        end
        
        @inbounds scratch[s0] = apply_function_direct_binary(ev.func, arg1_val, arg2_val)
        
    else
        # 3+ arguments - fall back to ntuple but use direct varargs
        vals = ntuple(i -> begin
            arg = ev.arg_evaluators[i]
            if arg isa ConstantEvaluator
                arg.value
            elseif arg isa ContinuousEvaluator
                Float64(get_data_value_specialized(data, arg.column, row_idx))
            else
                r = ev.arg_scratch_map[i]
                execute_to_scratch!(arg, scratch, first(r), last(r), data, row_idx)
                scratch[first(r)]
            end
        end, n_args)

        @inbounds scratch[s0] = apply_function_direct_varargs(ev.func, vals...)
    end
    
    return nothing
end

"""
    execute_to_scratch!(ev::InteractionEvaluator, scratch, s0, s1, data, row_idx)

Compute component evaluators, then produce N-way interactions via Kronecker products.

# Arguments
- `ev::InteractionEvaluator`: Fields:
  - `components::Vector{AbstractEvaluator}`
  - `component_scratch_map::Vector{UnitRange{Int}}`
  - `kronecker_pattern::Vector{NTuple{N,Int}}`

# Behavior
1. Recursively evaluate each component into its scratch range.
2. Call `apply_kronecker_to_scratch_range!` to fill `scratch[s0:s1]`.
"""
function execute_to_scratch!(
    ev::InteractionEvaluator,
    scratch::Vector{Float64},
    s0::Int,
    s1::Int,
    data::NamedTuple,
    row_idx::Int
)
    for (i, comp) in enumerate(ev.components)
        r = ev.component_scratch_map[i]
        execute_to_scratch!(comp, scratch, first(r), last(r), data, row_idx)
    end
    apply_kronecker_to_scratch_range!(
        ev.kronecker_pattern,
        ev.component_scratch_map,
        scratch,
        s0,
        s1
    )
    return nothing
end
