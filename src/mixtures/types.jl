# types.jl - Core mixture types for FormulaCompiler.jl
# Adapted from Margins.jl src/features/categorical_mixtures.jl

"""
    CategoricalMixture{T}

Represents a mixture of categorical levels with associated weights for statistical analysis.
Used to specify population composition scenarios and marginal effects computation.

# Fields
- `levels::Vector{T}`: Categorical levels (strings, symbols, booleans, or other types)
- `weights::Vector{Float64}`: Associated weights (must sum to 1.0)

# Example
```julia
# Educational composition mixture
edu_mix = CategoricalMixture(["high_school", "college"], [0.6, 0.4])

# Using the convenient mix() constructor
treatment_mix = mix("control" => 0.4, "treatment" => 0.6)
boolean_mix = mix(false => 0.3, true => 0.7)
```

# Validation
- Levels and weights must have the same length
- All weights must be non-negative
- Weights must sum to 1.0 (within tolerance)
- Levels must be unique

# Integration with FormulaCompiler
CategoricalMixture objects are automatically detected by FormulaCompiler's compilation
system and compiled into efficient zero-allocation evaluators using MixtureContrastOp.
"""
struct CategoricalMixture{T}
    levels::Vector{T}
    weights::Vector{Float64}

    function CategoricalMixture(levels::Vector{T}, weights::Vector{Float64}) where T
        length(levels) == length(weights) ||
            throw(ArgumentError("levels and weights must have same length"))
        validate_mixture_levels(levels)
        validate_mixture_weights(weights)
        new{T}(levels, weights)
    end
end

"""
    validate_mixture_weights(weights; atol=1e-10)

Throw an `ArgumentError` unless the mixture weights are non-negative and sum to 1
(within `atol`). Used by the `CategoricalMixture` constructor.

```julia
validate_mixture_weights([0.3, 0.7])           # ok
validate_mixture_weights([0.3, 0.8])           # error: does not sum to 1
validate_mixture_weights([0.3, -0.7, 1.4])     # error: negative weight
```
"""
function validate_mixture_weights(weights::AbstractVector{<:Real}; atol::Real=1e-10)
    if any(w < 0 for w in weights)
        throw(ArgumentError("Mixture weights must be non-negative: $weights"))
    end
    weight_sum = sum(weights)
    if !isapprox(weight_sum, 1.0, atol=atol)
        throw(ArgumentError("Mixture weights must sum to 1.0 (±$atol): got $weights (sum = $weight_sum)"))
    end
    return nothing
end

"""
    validate_mixture_levels(levels)

Throw an `ArgumentError` unless the mixture levels are non-empty and unique. Used by the
`CategoricalMixture` constructor.

```julia
validate_mixture_levels(["A", "B", "C"])    # ok
validate_mixture_levels(["A", "A", "B"])    # error: duplicate level "A"
validate_mixture_levels(String[])           # error: empty
```
"""
function validate_mixture_levels(levels::AbstractVector)
    if isempty(levels)
        throw(ArgumentError("Mixture levels cannot be empty"))
    end
    if length(unique(levels)) != length(levels)
        duplicates = [level for level in unique(levels) if count(==(level), levels) > 1]
        throw(ArgumentError("Mixture levels must be unique. Duplicates found: $duplicates"))
    end
    return nothing
end

# Support for indexing and iteration
Base.length(m::CategoricalMixture) = length(m.levels)
Base.getindex(m::CategoricalMixture, i) = (m.levels[i], m.weights[i])
Base.iterate(m::CategoricalMixture, state=1) = state > length(m) ? nothing : (m[state], state + 1)

"""
    MixtureWithLevels{T}

Wrapper that includes original categorical levels with the mixture for FormulaCompiler processing.
This type provides proper type-safe access to mixture components for the compilation system.

# Fields
- `mixture::CategoricalMixture{T}`: The core mixture specification
- `original_levels::Vector{String}`: Original levels from the data column

# Usage
This type is used internally by FormulaCompiler's scenario system to provide type-safe 
mixture processing with access to both mixture specifications and original data structure.

```julia
# Usually created automatically by FormulaCompiler's scenario system
mixture = mix("A" => 0.3, "B" => 0.7)
original_levels = ["A", "B", "C"]  # From the actual data column
wrapper = MixtureWithLevels(mixture, original_levels)

# Direct property access
wrapper.mixture.levels     # Access to mixture levels
wrapper.mixture.weights    # Access to mixture weights
wrapper.original_levels    # Access to original data levels
```
"""
struct MixtureWithLevels{T}
    mixture::CategoricalMixture{T}
    original_levels::Vector{String}
end

# Provide property access interface for FormulaCompiler compatibility
Base.getproperty(mwl::MixtureWithLevels, sym::Symbol) = 
    sym === :levels ? getfield(mwl, :mixture).levels :
    sym === :weights ? getfield(mwl, :mixture).weights :
    sym === :original_levels ? getfield(mwl, :original_levels) :
    sym === :mixture ? mwl :  # Self-reference for boolean mixture compatibility
    getfield(mwl, sym)

Base.hasproperty(::MixtureWithLevels, sym::Symbol) = 
    sym in (:levels, :weights, :original_levels, :mixture)