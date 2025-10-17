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
mutable struct ContrastEvaluator{T, Ops, S, O, NTMerged, CounterfactualTuple, CatLevelMaps, VarsTuple, VarMap}
    compiled::UnifiedCompiled{T, Ops, S, O}
    vars::VarsTuple  # Tuple of symbols for zero-allocation lookups
    var_map::VarMap  # NamedTuple for compile-time index mapping
    data_counterfactual::NTMerged
    counterfactuals::CounterfactualTuple  # Tuple of CounterfactualVector{T} subtypes

    # Pre-allocated buffers for zero-allocation performance
    y_from_buf::Vector{Float64}
    y_to_buf::Vector{Float64}

    # Tuple of CategoricalLevelMap structs (similar to UnifiedCompiled.ops pattern)
    categorical_level_maps::CatLevelMaps

    # Gradient computation buffers for parameter gradients and uncertainty quantification
    gradient_buffer::Vector{Float64}  # Buffer for parameter gradients ∂(discrete_effect)/∂β
    xrow_from_buf::Vector{Float64}    # Model row buffer for "from" level (X₀)
    xrow_to_buf::Vector{Float64}      # Model row buffer for "to" level (X₁)

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
    # Preserve natural types for zero-allocation performance
    # Initialize with row=0 so CounterfactualVectors don't shadow actual data initially
    data_counterfactual, counterfactuals = build_counterfactual_data(data, vars, 0)

    # Build categorical level maps as tuple of CategoricalLevelMap structs
    # (mirrors UnifiedCompiled.ops pattern)
    level_map_structs = []

    for (i, var) in enumerate(vars)
        cf_vec = counterfactuals[i]
        if cf_vec isa CategoricalCounterfactualVector
            base_array = cf_vec.base
            levs = levels(base_array)  # Returns Vector{T} where T is level type (String, Int64, etc.)

            # Build tuple of (level, ref_index) pairs using reference indices instead of CategoricalValue
            # CRITICAL: CategoricalValue is not a bitstype - copying it allocates!
            # Solution: Store reference indices (UInt8/UInt32/UInt64) which ARE bitstypes (zero allocations)
            # IMPORTANT: Use the actual reference type R from the CategoricalArray, not hardcoded UInt32
            R = typeof(cf_vec).parameters[2]  # Extract reference type (UInt8, UInt32, etc.)
            pairs_vec = []
            for (ref_idx, level) in enumerate(levs)
                # ref_idx is 1-based, matches CategoricalArray.refs encoding
                # Convert to the actual reference type R for type consistency
                push!(pairs_vec, (level, R(ref_idx)))
            end
            level_pairs = Tuple(pairs_vec)

            # Create CategoricalLevelMap struct with variable name and level tuple as type parameters
            # This is exactly like creating ContrastOp{:group, (4,5)}(matrix)
            level_map = CategoricalLevelMap{var, typeof(level_pairs)}(level_pairs)
            push!(level_map_structs, level_map)
        end
    end

    # Convert to tuple for type stability (exactly like UnifiedCompiled.ops)
    categorical_level_maps = Tuple(level_map_structs)

    # Convert vars to tuple for type stability and zero-allocation lookups
    vars_tuple = Tuple(vars)

    # Build NamedTuple index map for compile-time variable lookup
    # Maps variable names to indices: NamedTuple{(:treatment, :education)}((1, 2))
    var_map = NamedTuple{vars_tuple}(ntuple(i -> i, length(vars)))

    return ContrastEvaluator(
        compiled,
        vars_tuple,
        var_map,
        data_counterfactual,
        counterfactuals,
        Vector{Float64}(undef, length(compiled)),                # y_from_buf
        Vector{Float64}(undef, length(compiled)),                # y_to_buf
        categorical_level_maps,
        Vector{Float64}(undef, length(compiled)),                # gradient_buffer
        Vector{Float64}(undef, length(compiled)),                # xrow_from_buf
        Vector{Float64}(undef, length(compiled)),                # xrow_to_buf
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
    # Hot path: assume inputs are valid (validation moved to construction time)
    # For debug/safety, validation can be enabled with @boundscheck

    # CRITICAL: Reset ALL counterfactuals to inactive state (row=0) before evaluation
    # This ensures each contrast is independent and prevents state bleeding between calls
    # Bug fix: Without this, previous contrast_modelrow! calls contaminate subsequent ones
    reset_all_counterfactuals!(evaluator.counterfactuals)

    # Evaluate with "from" value
    update_counterfactual_for_var!(evaluator, var, row, from)
    evaluator.compiled(evaluator.y_from_buf, evaluator.data_counterfactual, row)

    # Reset again before "to" evaluation
    reset_all_counterfactuals!(evaluator.counterfactuals)

    # Evaluate with "to" value
    update_counterfactual_for_var!(evaluator, var, row, to)
    evaluator.compiled(evaluator.y_to_buf, evaluator.data_counterfactual, row)

    # Compute contrast
    Δ .= evaluator.y_to_buf .- evaluator.y_from_buf

    # Clean up: Reset counterfactuals to inactive state after use
    reset_all_counterfactuals!(evaluator.counterfactuals)

    return Δ
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

"""
    _validate_contrast_inputs!(evaluator, var, from, to)

Comprehensive validation for contrast inputs with clear error messages.

Validates:
- Variable exists in evaluator
- Variable type compatibility with contrast values
- Proper binary vs categorical variable usage
- Type compatibility for from/to values

Throws descriptive ErrorExceptions for invalid specifications.
"""
function _validate_contrast_inputs!(evaluator::ContrastEvaluator, var::Symbol, from, to)
    # 1. Check if variable exists in evaluator
    if var ∉ evaluator.vars
        available_vars = join(evaluator.vars, ", ")
        error("Variable :$var not found in ContrastEvaluator. Available variables: $available_vars")
    end

    # 2. Get the counterfactual vector for this variable to check its type
    var_idx = findfirst(==(var), evaluator.vars)
    cf_vec = evaluator.counterfactuals[var_idx]

    # 3. Type compatibility validation
    _validate_type_compatibility!(cf_vec, var, from, to)
end

# Constant tuples to avoid allocations in validation functions
const BINARY_CONTRAST_VALUES = (0, 1, true, false, "0", "1", "true", "false", "TRUE", "FALSE")
const VALID_BOOLEAN_VALUES = (true, false, 0, 1, "true", "false", "TRUE", "FALSE", "0", "1")

"""
    _is_valid_boolean_or_probability(val) -> Bool

Check if value is a valid boolean value or probability in [0.0, 1.0].
"""
function _is_valid_boolean_or_probability(val)
    # Check against constant tuple first (fast path)
    if val ∈ VALID_BOOLEAN_VALUES
        return true
    end
    # Check if it's a probability value
    if val isa Real && 0.0 <= val <= 1.0
        return true
    end
    return false
end

"""
    _looks_like_binary_contrast(from, to) -> Bool

Detect if user is trying to use binary syntax (0/1, true/false) on a non-binary variable.
"""
function _looks_like_binary_contrast(from, to)
    return from in BINARY_CONTRAST_VALUES || to in BINARY_CONTRAST_VALUES
end

"""
    _describe_variable_type(cf_vec) -> String

Return a human-readable description of the variable type.
"""
function _describe_variable_type(cf_vec)
    if cf_vec isa CategoricalCounterfactualVector
        return "categorical"
    elseif cf_vec isa NumericCounterfactualVector
        return "numeric (but not binary)"
    elseif cf_vec isa BoolCounterfactualVector
        return "boolean"
    elseif cf_vec isa CategoricalMixtureCounterfactualVector
        return "categorical mixture"
    else
        return "non-binary"
    end
end

"""
    _validate_type_compatibility!(cf_vec, var, from, to)

Validate that from/to values are compatible with the variable's data type.
"""
function _validate_type_compatibility!(cf_vec, var::Symbol, from, to)
    # For categorical variables, check if levels exist
    if cf_vec isa CategoricalCounterfactualVector
        _validate_categorical_levels!(cf_vec, var, from, to)
    elseif cf_vec isa NumericCounterfactualVector
        _validate_numeric_values!(cf_vec, var, from, to)
    elseif cf_vec isa BoolCounterfactualVector
        _validate_boolean_values!(cf_vec, var, from, to)
    elseif cf_vec isa CategoricalMixtureCounterfactualVector
        _validate_mixture_values!(cf_vec, var, from, to)
    end
end

"""
    _validate_categorical_levels!(cf_vec, var, from, to)

Validate that categorical levels exist in the variable's level set.
"""
function _validate_categorical_levels!(cf_vec::CategoricalCounterfactualVector, var::Symbol, from, to)
    available_levels = levels(cf_vec.base)

    # Check 'from' level
    if string(from) ∉ available_levels
        available_str = join(available_levels, ", ")
        error("Level \"$from\" not found in categorical variable :$var. Available levels: $available_str")
    end

    # Check 'to' level
    if string(to) ∉ available_levels
        available_str = join(available_levels, ", ")
        error("Level \"$to\" not found in categorical variable :$var. Available levels: $available_str")
    end
end

"""
    _validate_numeric_values!(cf_vec, var, from, to)

Validate that numeric values are reasonable for the variable.
"""
function _validate_numeric_values!(cf_vec::NumericCounterfactualVector, var::Symbol, from, to)
    # Check if values are reasonable numeric types first
    if !(from isa Real) || !(to isa Real)
        error("Variable :$var requires real numeric values for contrasts. Got: $from ($(typeof(from))) → $to ($(typeof(to)))")
    end

    # Check if values can be converted to the numeric type
    try
        convert(eltype(cf_vec.base), from)
        convert(eltype(cf_vec.base), to)
    catch e
        error("Cannot convert contrast values ($from, $to) to $(eltype(cf_vec.base)) for numeric variable :$var")
    end
end

"""
    _validate_boolean_values!(cf_vec, var, from, to)

Validate that boolean values are valid true/false variants or probabilities in [0.0, 1.0].
"""
function _validate_boolean_values!(cf_vec::BoolCounterfactualVector, var::Symbol, from, to)
    if !_is_valid_boolean_or_probability(from)
        error("Invalid boolean value \"$from\" for boolean variable :$var. " *
              "Use true/false, 0/1, string equivalents, or probabilities in [0.0, 1.0]")
    end

    if !_is_valid_boolean_or_probability(to)
        error("Invalid boolean value \"$to\" for boolean variable :$var. " *
              "Use true/false, 0/1, string equivalents, or probabilities in [0.0, 1.0]")
    end
end

"""
    _validate_mixture_values!(cf_vec, var, from, to)

Validate mixture variable contrast values for categorical mixture variables.

Ensures that `from` and `to` mixture contrasts:
- Use the same mixture type as the compiled data column
- Include identical level sets as the baseline mixture
- Provide finite, non-negative weights that sum to 1.0

This validation mirrors compile-time mixture checks but guards the contrast API
so users receive immediate feedback when supplying incompatible mixture values.
"""
function _validate_mixture_values!(cf_vec::CategoricalMixtureCounterfactualVector, var::Symbol, from, to)
    isempty(cf_vec.base) && error("Mixture variable :$var does not have baseline data for validation")

    baseline_value = cf_vec.base[1]
    expected_type = typeof(baseline_value)

    _validate_single_mixture_value(expected_type, var, "from", from)
    _validate_single_mixture_value(expected_type, var, "to", to)

    baseline_spec = extract_mixture_spec(baseline_value)
    base_levels = _normalize_mixture_levels(baseline_spec.levels)
    base_level_set = Set(base_levels)

    _validate_mixture_spec(var, "from", from, base_levels, base_level_set)
    _validate_mixture_spec(var, "to", to, base_levels, base_level_set)
end

@inline function _validate_single_mixture_value(expected_type::Type, var::Symbol, label::AbstractString, value)
    if !(hasproperty(value, :levels) && hasproperty(value, :weights))
        error("Mixture variable :$var $label contrast must be provided as a mixture object (e.g., mix(\"A\" => 0.5, \"B\" => 0.5)). Got $(typeof(value))")
    end

    if !(value isa expected_type)
        error("Mixture variable :$var uses mixture type $(expected_type). The $label contrast must use the same type, got $(typeof(value)).")
    end

    # Extraction happens later during spec validation.
end

@inline function _normalize_mixture_levels(levels)
    normalized = Vector{String}(undef, length(levels))
    @inbounds for i in eachindex(levels)
        level = levels[i]
        normalized[i] = level isa String ? level :
                        level isa Symbol ? String(level) :
                        string(level)
    end
    return normalized
end

@inline function _validate_mixture_spec(var::Symbol, label::AbstractString, value, base_levels::Vector{String}, base_level_set::Set{String})
    spec = extract_mixture_spec(value)
    levels = _normalize_mixture_levels(spec.levels)
    weights = spec.weights

    spec_set = Set(levels)
    if length(spec_set) != length(levels)
        error("Mixture variable :$var $label contrast contains duplicate levels: $(join(levels, ", ")).")
    end

    if length(weights) != length(levels)
        error("Mixture variable :$var $label contrast must specify a weight for each level. Got $(length(weights)) weights for $(length(levels)) levels.")
    end

    extra_levels = setdiff(spec_set, base_level_set)
    if !isempty(extra_levels)
        available_str = join(base_levels, ", ")
        error("Mixture variable :$var $label contrast contains unknown levels: $(_format_level_list(extra_levels)). Available levels: $available_str")
    end

    missing_levels = setdiff(base_level_set, spec_set)
    if !isempty(missing_levels)
        available_str = join(base_levels, ", ")
        error("Mixture variable :$var $label contrast is missing levels: $(_format_level_list(missing_levels)). All contrasts must include the same levels as the data mixture ($available_str)")
    end

    _validate_mixture_weights(var, label, weights)
end

@inline function _validate_mixture_weights(var::Symbol, label::AbstractString, weights)
    isempty(weights) && error("Mixture variable :$var $label contrast must include at least one weight")

    sum_weights = 0.0
    @inbounds for w in weights
        (w isa Real && isfinite(w)) ||
            error("Mixture variable :$var $label contrast has non-finite weight $w")
        w < 0.0 &&
            error("Mixture variable :$var $label contrast has negative weight $w")
        sum_weights += Float64(w)
    end

    if !(isapprox(sum_weights, 1.0; atol=1e-10))
        error("Mixture variable :$var $label contrast weights must sum to 1.0 (got $sum_weights)")
    end
end

@inline function _format_level_list(levels)
    collected = collect(levels)
    sort!(collected)
    return join(collected, ", ")
end


# Gradient Computation Functions for Discrete Effects

"""
    contrast_gradient!(∇β, evaluator, row, var, from, to, β, [link]) -> ∇β

Compute parameter gradients for discrete effects: ∂(discrete_effect)/∂β - zero allocations.

Computes the gradient of discrete marginal effects with respect to model parameters
using the mathematical formula:
- **Linear scale (η)**: ∇β = ΔX = X₁ - X₀ (contrast vector)
- **Response scale (μ)**: ∇β = g'(η₁) × X₁ - g'(η₀) × X₀ (chain rule with link derivatives)

This enables uncertainty quantification via the delta method: SE = √(∇β' Σ ∇β).

# Arguments
- `∇β::AbstractVector{Float64}`: Output gradient vector (modified in-place)
- `evaluator::ContrastEvaluator`: Pre-configured contrast evaluator
- `row::Int`: Row index to evaluate
- `var::Symbol`: Variable to contrast (must be in `evaluator.vars`)
- `from`: Reference level (baseline)
- `to`: Target level (comparison)
- `β::AbstractVector{<:Real}`: Model coefficients (used only for response-scale computation)
- `link`: GLM link function (optional, defaults to linear scale)

# Returns
- `∇β`: The same vector passed in, containing parameter gradients ∂(discrete_effect)/∂β

# Performance
- **Zero allocations** - uses pre-allocated buffers from evaluator
- **Link function support** - handles all GLM links (Identity, Log, Logit, etc.)
- **Type flexibility** - accepts any Real coefficient type, converts internally

# Mathematical Method
**Linear Scale (default)**:
```
discrete_effect = η₁ - η₀ = (X₁'β) - (X₀'β) = (X₁ - X₀)'β = ΔX'β
∇β = ΔX = X₁ - X₀
```

**Response Scale (with link function)**:
```
discrete_effect = μ₁ - μ₀ = g⁻¹(η₁) - g⁻¹(η₀)
∇β = g'(η₁) × X₁ - g'(η₀) × X₀  (chain rule)
```

# Example
```julia
evaluator = contrastevaluator(compiled, data, [:treatment])
∇β = Vector{Float64}(undef, length(compiled))

# Linear scale gradients (η = Xβ scale)
contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Drug", β)

# Response scale gradients (μ = g⁻¹(η) scale)
link = GLM.LogitLink()
contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Drug", β, link)

# Delta method standard error
se = sqrt(∇β' * vcov_matrix * ∇β)
```

# Integration with Delta Method
Parameter gradients enable uncertainty quantification:
```julia
# Compute discrete effect + gradient simultaneously
discrete_effect = contrast_modelrow(evaluator, row, var, from, to)
contrast_gradient!(∇β, evaluator, row, var, from, to, β, link)

# Delta method confidence intervals
variance = ∇β' * vcov_matrix * ∇β
se = sqrt(variance)
ci_lower = discrete_effect - 1.96 * se
ci_upper = discrete_effect + 1.96 * se
```
"""
function contrast_gradient!(
    ∇β::AbstractVector{Float64},
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to,
    β::AbstractVector{<:Real},
    link=nothing
)
    # Hot path: assume inputs are valid (validation at construction time)
    # Keep dimension checks as they're cheap and prevent memory corruption
    length(∇β) == length(evaluator.compiled) || throw(DimensionMismatch("Gradient buffer size mismatch"))
    length(β) == length(evaluator.compiled) || throw(DimensionMismatch("Coefficient vector size mismatch"))

    if link === nothing
        # Linear scale: ∇β = ΔX = X₁ - X₀ (contrast vector)
        _contrast_gradient_linear_scale!(∇β, evaluator, row, var, from, to)
    else
        # Response scale: ∇β = g'(η₁) × X₁ - g'(η₀) × X₀ (chain rule)
        # Let Julia's method dispatch handle unsupported links naturally
        _contrast_gradient_response_scale!(∇β, evaluator, row, var, from, to, β, link)
    end

    return ∇β
end

"""
    contrast_gradient(evaluator, row, var, from, to, β, [link]) -> Vector{Float64}

Convenience version that allocates and returns the gradient vector.
"""
function contrast_gradient(
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to,
    β::AbstractVector{<:Real},
    link=nothing
)
    ∇β = Vector{Float64}(undef, length(evaluator))
    contrast_gradient!(∇β, evaluator, row, var, from, to, β, link)
    return ∇β
end

# =============================================================================
# REMOVED (2025-10-07): delta_method_se migrated to Margins.jl v2.0
# Use: `using Margins; delta_method_se(...)`
# =============================================================================

# Internal implementation functions

"""
    _contrast_gradient_linear_scale!(∇β, evaluator, row, var, from, to)

Compute parameter gradient for linear scale discrete effects: ∇β = ΔX = X₁ - X₀.
"""
function _contrast_gradient_linear_scale!(
    ∇β::AbstractVector{Float64},
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to
)
    # General path: compute X₁ - X₀ using model matrix evaluation

    # Reset all counterfactuals to prevent state bleeding
    reset_all_counterfactuals!(evaluator.counterfactuals)

    # Compute X₀ (baseline model matrix row)
    update_counterfactual_for_var!(evaluator, var, row, from)
    evaluator.compiled(evaluator.xrow_from_buf, evaluator.data_counterfactual, row)

    # Reset again before computing X₁
    reset_all_counterfactuals!(evaluator.counterfactuals)

    # Compute X₁ (counterfactual model matrix row)
    update_counterfactual_for_var!(evaluator, var, row, to)
    evaluator.compiled(evaluator.xrow_to_buf, evaluator.data_counterfactual, row)

    # ΔX = X₁ - X₀
    @inbounds @fastmath for i in eachindex(∇β)
        ∇β[i] = evaluator.xrow_to_buf[i] - evaluator.xrow_from_buf[i]
    end

    # Clean up: Reset counterfactuals to inactive state
    reset_all_counterfactuals!(evaluator.counterfactuals)

    return ∇β
end

"""
    _contrast_gradient_response_scale!(∇β, evaluator, row, var, from, to, β, link)

Compute parameter gradient for response scale discrete effects using mathematically correct chain rule.

**Mathematical Formula**: ∇β = g'(η₁) × X₁ - g'(η₀) × X₀

Where:
- η₁ = X₁'β (linear predictor at counterfactual level)
- η₀ = X₀'β (linear predictor at baseline level)
- g'(η) = dμ/dη (link function derivative)

**No computational shortcuts are used** - mathematical correctness is paramount.
"""
function _contrast_gradient_response_scale!(
    ∇β::AbstractVector{Float64},
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to,
    β::AbstractVector{<:Real},
    link
)
    # Reset all counterfactuals to prevent state bleeding
    @inbounds for i in 1:length(evaluator.counterfactuals)
        update_counterfactual_row!(evaluator.counterfactuals[i], 0)
    end

    # Step 1: Compute X₀ and η₀ = X₀'β
    update_counterfactual_for_var!(evaluator, var, row, from)
    evaluator.compiled(evaluator.xrow_from_buf, evaluator.data_counterfactual, row)
    η₀ = dot(β, evaluator.xrow_from_buf)

    # Reset again before computing X₁
    @inbounds for i in 1:length(evaluator.counterfactuals)
        update_counterfactual_row!(evaluator.counterfactuals[i], 0)
    end

    # Step 2: Compute X₁ and η₁ = X₁'β
    update_counterfactual_for_var!(evaluator, var, row, to)
    evaluator.compiled(evaluator.xrow_to_buf, evaluator.data_counterfactual, row)
    η₁ = dot(β, evaluator.xrow_to_buf)

    # Step 3: Compute link function derivatives (exact evaluation)
    g_prime_η₀ = _dmu_deta(link, η₀)  # g'(η₀) - exact
    g_prime_η₁ = _dmu_deta(link, η₁)  # g'(η₁) - exact

    # Step 4: Apply mathematically correct chain rule formula
    # ∇β = g'(η₁) × X₁ - g'(η₀) × X₀
    @inbounds @fastmath for i in eachindex(∇β)
        ∇β[i] = g_prime_η₁ * evaluator.xrow_to_buf[i] - g_prime_η₀ * evaluator.xrow_from_buf[i]
    end

    # Clean up: Reset counterfactuals to inactive state
    @inbounds for i in 1:length(evaluator.counterfactuals)
        update_counterfactual_row!(evaluator.counterfactuals[i], 0)
    end

    return ∇β
end

# Link Function Support
# Note: Link function support is determined by the existence of _dmu_deta methods.
# Julia's method dispatch will naturally error if an unsupported link is used.

"""
    supported_link_functions() -> Vector{String}

Return list of GLM link functions with implemented _dmu_deta methods.

Note: Link function support is now determined by Julia's method dispatch.
Any link function with a _dmu_deta method will work automatically.
This function provides a convenience list of commonly tested functions.

# Example
```julia
links = supported_link_functions()
println("Common GLM links: ", join(links, ", "))
```
"""
function supported_link_functions()
    links = [
        "GLM.IdentityLink",
        "GLM.LogLink",
        "GLM.LogitLink",
        "GLM.ProbitLink",
        "GLM.CloglogLink",
        "GLM.CauchitLink",
        "GLM.InverseLink",
        "GLM.SqrtLink"
    ]

    # Add InverseSquareLink if available
    if isdefined(GLM, :InverseSquareLink)
        push!(links, "GLM.InverseSquareLink")
    end

    return links
end
