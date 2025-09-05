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
        # Validation
        length(levels) == length(weights) || 
            throw(ArgumentError("levels and weights must have same length"))
        all(weights .≥ 0.0) || 
            throw(ArgumentError("all weights must be non-negative"))
        abs(sum(weights) - 1.0) < 1e-10 || 
            throw(ArgumentError("weights must sum to 1.0, got $(sum(weights))"))
        length(unique(levels)) == length(levels) || 
            throw(ArgumentError("levels must be unique"))
            
        new{T}(levels, weights)
    end
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