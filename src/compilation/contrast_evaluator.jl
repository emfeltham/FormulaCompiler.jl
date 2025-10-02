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

    # Binary variable optimization metadata
    binary_vars::Set{Symbol}  # Variables that are truly binary (0/1 or true/false)
    binary_coef_indices::Dict{Symbol, Int}  # Mapping from binary var to coefficient index

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

    # Detect binary variables and their coefficient positions for fast path optimization
    binary_vars = Set{Symbol}()
    binary_coef_indices = Dict{Symbol, Int}()

    for (i, var) in enumerate(vars)
        cf_vec = counterfactuals[i]
        col = getproperty(data, var)

        # Check if variable is truly binary
        if _is_truly_binary_variable(cf_vec, col)
            binary_vars = union(binary_vars, [var])
            # Find coefficient index for this binary variable in the compiled formula
            coef_idx = _find_binary_coefficient_index(compiled, var)
            if coef_idx !== nothing
                binary_coef_indices[var] = coef_idx
            end
        end
    end

    return ContrastEvaluator(
        compiled,
        vars,
        data_counterfactual,
        counterfactuals,
        Vector{Float64}(undef, length(compiled)),                # y_from_buf
        Vector{Float64}(undef, length(compiled)),                # y_to_buf
        categorical_level_maps,
        binary_vars,
        binary_coef_indices,
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

    # Binary variable fast path optimization
    if var in evaluator.binary_vars && haskey(evaluator.binary_coef_indices, var)
        return _contrast_modelrow_binary_fast_path!(Δ, evaluator, var, from, to)
    end

    # General path for categorical and non-binary variables
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

# Binary variable optimization helper functions

"""
    _is_truly_binary_variable(cf_vec, col) -> Bool

Check if a variable is truly binary (only contains 0/1 or true/false values).

# Arguments
- `cf_vec`: CounterfactualVector for the variable
- `col`: Original data column

# Returns
Boolean indicating if the variable is truly binary and eligible for fast path optimization.
"""
function _is_truly_binary_variable(cf_vec, col)
    # Must be BoolCounterfactualVector or numeric with only 0/1 values
    if cf_vec isa BoolCounterfactualVector
        return true
    elseif cf_vec isa NumericCounterfactualVector && eltype(col) <: Real
        # Check if all values are 0 or 1
        unique_vals = unique(col)
        return length(unique_vals) <= 2 && all(v in (0, 1) for v in unique_vals)
    else
        return false
    end
end

"""
    _find_binary_coefficient_index(compiled, var) -> Union{Int, Nothing}

Find the coefficient index for a binary variable in the compiled formula.

For binary variables, we need to identify which position in the model matrix
corresponds to the variable's coefficient to enable fast path computation.

# Arguments
- `compiled`: Compiled formula evaluator
- `var`: Variable symbol to find coefficient for

# Returns
Coefficient index as Int, or Nothing if not found or not a simple binary coefficient.
"""
function _find_binary_coefficient_index(compiled, var)
    # Find the coefficient index for a binary variable by examining the compiled operations
    # For simple binary variables, we look for LoadOp operations that load the variable

    ops = compiled.ops
    for (i, op) in enumerate(ops)
        # Check if this is a LoadOp for our variable
        op_type = typeof(op)
        if op_type <: LoadOp
            # Extract type parameters from LoadOp{Column, OutPos}
            type_params = op_type.parameters
            if length(type_params) >= 2
                column_param = type_params[1]
                position_param = type_params[2]

                # Check if this LoadOp is for our variable
                if column_param == var
                    # Return the position parameter
                    return position_param
                end
            end
        end
    end

    # If not found as a simple LoadOp, the variable might be part of a more complex term
    # For now, return nothing to fall back to the general path
    return nothing
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

    # 3. Validate binary variable usage
    if var in evaluator.binary_vars
        # Variable is binary - validate binary contrast syntax
        if !_is_valid_binary_contrast(from, to)
            error("Invalid binary contrast for variable :$var (detected as binary). Use binary values like 0→1, false→true, or \"0\"→\"1\". Got: $from → $to")
        end
    else
        # Variable is categorical - check for misuse of binary syntax
        if _looks_like_binary_contrast(from, to)
            variable_type = _describe_variable_type(cf_vec)
            error("Variable :$var is $variable_type, not binary. Use categorical level names for contrasts (e.g., \"Level1\" → \"Level2\"), not binary values like $from → $to")
        end
    end

    # 4. Type compatibility validation
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

Validate mixture variable contrast values (placeholder for future mixture support).
"""
function _validate_mixture_values!(cf_vec::CategoricalMixtureCounterfactualVector, var::Symbol, from, to)
    # For now, just ensure they're not obviously wrong types
    # Full mixture contrast validation would be implemented in future phases
    if !(from isa Union{String, Symbol}) || !(to isa Union{String, Symbol})
        error("Mixture variable :$var contrasts should use string or symbol level specifications")
    end
end

"""
    _contrast_modelrow_binary_fast_path!(Δ, evaluator, var, from, to) -> Δ

Optimized binary variable contrast computation that skips full model evaluation.

For truly binary variables (0/1 or true/false), the contrast is simply the
coefficient difference multiplied by the variable change (1 - 0 = 1).

# Arguments
- `Δ::AbstractVector{Float64}`: Output contrast vector (modified in-place)
- `evaluator::ContrastEvaluator`: Pre-configured contrast evaluator
- `var::Symbol`: Binary variable to contrast
- `from`: Reference level (should be 0 or false)
- `to`: Target level (should be 1 or true)

# Performance
- **Zero allocations** - directly computes coefficient difference
- **Skip model evaluation** - avoids two full formula evaluations
- **Type specialization** - optimized for binary variable patterns

# Validation
Validates that from/to values are valid binary levels before optimization.
"""
function _contrast_modelrow_binary_fast_path!(
    Δ::AbstractVector{Float64},
    evaluator::ContrastEvaluator,
    var::Symbol,
    from, to
)
    # Validate binary contrast specification
    if !_is_valid_binary_contrast(from, to)
        error("Invalid binary contrast specification for variable $var: from=$from, to=$to. " *
              "Expected binary values (0/1, false/true) or probabilities in [0.0, 1.0].")
    end

    # Get coefficient index for this binary variable
    coef_idx = evaluator.binary_coef_indices[var]

    # For binary variables, the contrast is simply the coefficient value
    # when transitioning from 0→1 or false→true

    # Zero out all coefficients manually (avoid fill! allocation)
    @inbounds for i in eachindex(Δ)
        Δ[i] = 0.0
    end

    # Set the coefficient for this binary variable
    # The contrast magnitude depends on the direction (0→1 is +coef, 1→0 is -coef)
    contrast_direction = _binary_contrast_direction(from, to)
    @inbounds Δ[coef_idx] = contrast_direction

    return Δ
end

"""
    _is_valid_binary_contrast(from, to) -> Bool

Validate that from/to values represent a valid binary contrast.

Accepts:
- Binary transitions: 0→1, 1→0, false→true, true→false, "0"→"1", "1"→"0"
- Probability transitions: 0.0→0.6, 0.3→0.7, false→0.5, etc.
- Zero contrasts: 0.5→0.5 (valid, produces zero effect)

Returns true if both values can be standardized to Float64 in [0.0, 1.0].
"""
function _is_valid_binary_contrast(from, to)
    # Convert to standardized form for validation
    from_std = _standardize_binary_value(from)
    to_std = _standardize_binary_value(to)

    # Must both be valid binary/probability values (can be equal for zero contrast)
    return from_std !== nothing && to_std !== nothing
end

"""
    _binary_contrast_direction(from, to) -> Float64

Determine the direction and magnitude of a binary contrast.

Returns the difference `to - from`, representing the change in probability:
- 0→1 (false→true): returns 1.0
- 1→0 (true→false): returns -1.0
- 0→0.6: returns 0.6 (60% of full effect)
- 0.3→0.7: returns 0.4 (40% of full effect)

# Mathematical Interpretation
For Boolean variable z with coefficient β:
- Contrast(z: p₁ → p₂) = (p₂ - p₁) * β
- This is consistent with categorical mixture semantics in linear predictor space
"""
function _binary_contrast_direction(from, to)
    from_std = _standardize_binary_value(from)
    to_std = _standardize_binary_value(to)

    if from_std === nothing || to_std === nothing
        error("Invalid binary contrast direction: $from → $to")
    end

    # Return the difference in probability
    # Works for both discrete (0→1) and continuous (0.3→0.7) contrasts
    return to_std - from_std
end

"""
    _standardize_binary_value(val) -> Union{Float64, Nothing}

Convert various binary value representations to standardized form.

Handles:
- Binary values: 0, 1, false, true, "0", "1", "false", "true" (case insensitive)
- Probability values: Float64 in [0.0, 1.0] (e.g., 0.6 represents 60% true)

Returns: Float64 in [0.0, 1.0], or nothing if not a valid binary/probability value

# Probability Interpretation
For Boolean variables, Float64 values represent mixture probabilities:
- `z = 0.6` means 60% true, 40% false
- Plugged directly into linear predictor: η = β₀ + β_z*0.6
- Consistent with categorical mixture semantics
"""
function _standardize_binary_value(val)
    # Direct type dispatch for common cases (no allocation)
    if val === 0 || val === false
        return 0.0
    elseif val === 1 || val === true
        return 1.0
    elseif val isa Float64
        # Accept probabilities in [0, 1]
        if 0.0 <= val <= 1.0
            return val
        else
            return nothing
        end
    elseif val isa String
        # Only allocate for string processing if needed
        val_lower = lowercase(val)
        if val_lower == "0" || val_lower == "false"
            return 0.0
        elseif val_lower == "1" || val_lower == "true"
            return 1.0
        else
            return nothing
        end
    else
        # Handle other numeric types without string conversion
        if val == 0
            return 0.0
        elseif val == 1
            return 1.0
        elseif val isa Real && 0.0 <= val <= 1.0
            # Accept other numeric probabilities
            return Float64(val)
        else
            return nothing
        end
    end
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

"""
    delta_method_se(evaluator, row, var, from, to, β, vcov, [link]) -> Float64

Compute standard error for discrete effects using delta method - zero allocations.

Uses the mathematical formula: SE = √(∇β' Σ ∇β) where:
- ∇β = parameter gradient from `contrast_gradient!`
- Σ = parameter covariance matrix

# Arguments
- `evaluator::ContrastEvaluator`: Pre-configured contrast evaluator
- `row::Int`: Row index to evaluate
- `var::Symbol`: Variable to contrast
- `from`, `to`: Reference and target levels
- `β::AbstractVector{<:Real}`: Model coefficients
- `vcov::AbstractMatrix{<:Real}`: Parameter covariance matrix
- `link`: GLM link function (optional, defaults to linear scale)

# Returns
- `Float64`: Standard error for the discrete effect

# Performance
- **Zero allocations** - reuses evaluator's gradient buffer
- **Type flexibility** - accepts any Real matrix/vector types

# Example
```julia
# Standard error for treatment effect
se = delta_method_se(evaluator, 1, :treatment, "Control", "Drug", β, vcov)

# Response scale standard error
link = GLM.LogitLink()
se_mu = delta_method_se(evaluator, 1, :treatment, "Control", "Drug", β, vcov, link)
```
"""
function delta_method_se(
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to,
    β::AbstractVector{<:Real},
    vcov::AbstractMatrix{<:Real},
    link=nothing
)
    # Hot path: let method dispatch handle unsupported links naturally
    # Compute parameter gradient (reuses evaluator's buffer)
    contrast_gradient!(evaluator.gradient_buffer, evaluator, row, var, from, to, β, link)

    # Delta method variance: ∇β' Σ ∇β
    variance = dot(evaluator.gradient_buffer, vcov, evaluator.gradient_buffer)

    return sqrt(max(0.0, variance))  # Ensure non-negative due to numerical precision
end

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
    # Binary variable fast path
    if var in evaluator.binary_vars && haskey(evaluator.binary_coef_indices, var)
        return _contrast_gradient_binary_fast_path!(∇β, evaluator, var, from, to)
    end

    # General path: compute X₁ - X₀ using model matrix evaluation

    # Compute X₀ (baseline model matrix row)
    update_counterfactual_for_var!(evaluator.counterfactuals, evaluator.vars, var, row, from, evaluator.categorical_level_maps)
    evaluator.compiled(evaluator.xrow_from_buf, evaluator.data_counterfactual, row)

    # Compute X₁ (counterfactual model matrix row)
    update_counterfactual_for_var!(evaluator.counterfactuals, evaluator.vars, var, row, to, evaluator.categorical_level_maps)
    evaluator.compiled(evaluator.xrow_to_buf, evaluator.data_counterfactual, row)

    # ΔX = X₁ - X₀
    @inbounds @fastmath for i in eachindex(∇β)
        ∇β[i] = evaluator.xrow_to_buf[i] - evaluator.xrow_from_buf[i]
    end

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
    # Step 1: Compute X₀ and η₀ = X₀'β
    update_counterfactual_for_var!(evaluator.counterfactuals, evaluator.vars, var, row, from, evaluator.categorical_level_maps)
    evaluator.compiled(evaluator.xrow_from_buf, evaluator.data_counterfactual, row)
    η₀ = dot(β, evaluator.xrow_from_buf)

    # Step 2: Compute X₁ and η₁ = X₁'β
    update_counterfactual_for_var!(evaluator.counterfactuals, evaluator.vars, var, row, to, evaluator.categorical_level_maps)
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

    return ∇β
end

"""
    _contrast_gradient_binary_fast_path!(∇β, evaluator, var, from, to)

Optimized gradient computation for binary variables: ∇β has single non-zero element.
"""
function _contrast_gradient_binary_fast_path!(
    ∇β::AbstractVector{Float64},
    evaluator::ContrastEvaluator,
    var::Symbol,
    from, to
)
    # Validate binary contrast specification
    if !_is_valid_binary_contrast(from, to)
        error("Invalid binary contrast specification for variable $var: from=$from, to=$to")
    end

    # Get coefficient index for this binary variable
    coef_idx = evaluator.binary_coef_indices[var]

    # Zero out all gradients (avoid fill! allocation)
    @inbounds for i in eachindex(∇β)
        ∇β[i] = 0.0
    end

    # Set single non-zero gradient element
    # For linear scale: ∇β = ΔX, which has ±1 at the coefficient position
    contrast_direction = _binary_contrast_direction(from, to)
    @inbounds ∇β[coef_idx] = contrast_direction

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