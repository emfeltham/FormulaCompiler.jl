# contrast_evaluator.jl - Zero-allocation categorical contrast evaluator

"""
    ContrastEvaluator{T, Ops, S, O, NTMerged, CounterfactualTuple}

Zero-allocation evaluator for categorical and binary variable contrasts.

Provides efficient discrete marginal effects computation by pre-allocating all buffers
and pre-computing categorical level mappings. Eliminates the ~2KB allocation overhead
of the basic `contrast_modelrow!` function for batch contrast operations.

Uses typed counterfactual vectors for type-stable, zero-allocation performance.

# Fields
- `compiled`: Base compiled formula evaluator
- `vars`: Variables available for contrast computation
- `data_counterfactual`: Counterfactual data structure for variable substitution
- `counterfactuals`: Tuple of typed CounterfactualVector{T} subtypes for each variable
- `y_from_buf`: Pre-allocated buffer for "from" level evaluation
- `y_to_buf`: Pre-allocated buffer for "to" level evaluation
- `row`: Current row being processed

# Performance
- **Zero allocations** after construction for all contrast operations
- **Type stability** via typed counterfactual vectors
- **Buffer reuse** across multiple contrasts and rows
- **Type specialization** for compiled formula operations

# Usage
```julia
# One-time setup
evaluator = contrastevaluator(compiled, data, [:treatment, :education])
contrast_buf = Vector{Float64}(undef, length(compiled))

# Fast repeated contrasts (zero allocations)
for row in 1:n_rows
    contrast_modelrow!(contrast_buf, evaluator, row, :treatment, "Control", "Drug")
    # Process contrast_buf...
end
```
"""
mutable struct ContrastEvaluator{T, Ops, S, O, NTMerged, CounterfactualTuple}
    compiled::UnifiedCompiled{T, Ops, S, O}
    vars::Vector{Symbol}
    data_counterfactual::NTMerged
    counterfactuals::CounterfactualTuple  # Tuple of CounterfactualVector{T} subtypes

    # Pre-allocated buffers for zero-allocation performance
    y_from_buf::Vector{Float64}
    y_to_buf::Vector{Float64}

    # Pre-computed categorical level mappings for zero-allocation categorical contrasts
    categorical_level_maps::Dict{Symbol, Dict{String, CategoricalValue}}

    # Current processing state
    row::Int
end

Base.length(ce::ContrastEvaluator) = length(ce.compiled)

"""
    contrastevaluator(compiled, data, vars) -> ContrastEvaluator

Construct a ContrastEvaluator for efficient categorical and binary contrast computation.

Pre-allocates all necessary buffers and pre-computes categorical level mappings to
eliminate allocations during contrast evaluation.

# Arguments
- `compiled`: Result from `compile_formula(model, data)`
- `data`: Column-table data as NamedTuple
- `vars`: Vector of variable symbols available for contrasts

# Returns
ContrastEvaluator configured for zero-allocation contrast computation.

# Performance Notes
- **One-time cost**: Setup involves building override structures and categorical mappings
- **Categorical optimization**: Level mappings computed once, reused for all contrasts
- **Memory efficiency**: Buffers sized exactly for the compiled formula

# Example
```julia
# Setup for categorical contrasts
evaluator = contrastevaluator(compiled, data, [:group, :region, :binary_var])

# Zero-allocation usage
contrast_buf = Vector{Float64}(undef, length(compiled))
contrast_modelrow!(contrast_buf, evaluator, 1, :group, "Control", "Treatment")
```
"""
function contrastevaluator(compiled, data, vars::Vector{Symbol})
    # Use Float64 counterfactuals for contrast evaluation
    data_counterfactual, counterfactuals = build_counterfactual_data(data, vars, 1, Float64)

    # Pre-compute categorical level mappings for zero-allocation performance
    categorical_level_maps = Dict{Symbol, Dict{String, CategoricalValue}}()
    for (i, var) in enumerate(vars)
        cf_vec = counterfactuals[i]
        if cf_vec isa CategoricalCounterfactualVector
            # Build level map: String → CategoricalValue for this variable
            level_map = Dict{String, CategoricalValue}()
            base_array = cf_vec.base
            for level_str in levels(base_array)
                # Find a CategoricalValue instance for this level
                matching_idx = findfirst(x -> string(x) == level_str, base_array)
                if matching_idx !== nothing
                    level_map[level_str] = base_array[matching_idx]
                end
            end
            categorical_level_maps[var] = level_map
        end
    end

    return ContrastEvaluator(
        compiled,
        vars,
        data_counterfactual,
        counterfactuals,
        Vector{Float64}(undef, length(compiled)),
        Vector{Float64}(undef, length(compiled)),
        categorical_level_maps,
        1
    )
end

"""
    contrast_modelrow!(Δ, evaluator, row, var, from, to) -> Δ

Compute discrete contrast using pre-allocated ContrastEvaluator (zero allocations).

Evaluates `Δ = X(var=to) - X(var=from)` using the evaluator's pre-allocated buffers
and pre-computed categorical mappings for optimal performance.

# Arguments
- `Δ::AbstractVector{Float64}`: Output contrast vector (modified in-place)
- `evaluator::ContrastEvaluator`: Pre-configured contrast evaluator
- `row::Int`: Row index to evaluate
- `var::Symbol`: Variable to contrast (must be in `evaluator.vars`)
- `from`: Reference level (baseline)
- `to`: Target level (comparison)

# Performance
- **Zero allocations** - uses pre-allocated buffers from evaluator
- **Categorical optimization** - uses pre-computed level mappings
- **Type specialization** - compiled formula operations fully optimized

# Error Handling
- Validates that `var` exists in evaluator's variable list
- Handles both categorical and numeric variable types
- Provides clear error messages for invalid level specifications

# Example
```julia
evaluator = contrastevaluator(compiled, data, [:treatment])
contrast_buf = Vector{Float64}(undef, length(compiled))

# Zero-allocation contrast computation
contrast_modelrow!(contrast_buf, evaluator, 1, :treatment, "Control", "Drug")
# contrast_buf now contains the discrete effect vector
```
"""
function contrast_modelrow!(
    Δ::AbstractVector{Float64},
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to
)
    # Update counterfactual for this variable
    update_counterfactual_for_var!(evaluator.counterfactuals, evaluator.vars, var, row, from, evaluator.categorical_level_maps)
    evaluator.compiled(evaluator.y_from_buf, evaluator.data_counterfactual, row)

    update_counterfactual_for_var!(evaluator.counterfactuals, evaluator.vars, var, row, to, evaluator.categorical_level_maps)
    evaluator.compiled(evaluator.y_to_buf, evaluator.data_counterfactual, row)

    Δ .= evaluator.y_to_buf .- evaluator.y_from_buf
end

# Convenience method for single contrast computation
function contrast_modelrow(
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to
)
    Δ = Vector{Float64}(undef, length(evaluator))
    contrast_modelrow!(Δ, evaluator, row, var, from, to)
    return Δ
end