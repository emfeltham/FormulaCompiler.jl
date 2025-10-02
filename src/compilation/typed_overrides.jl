# typed_overrides.jl - Type-stable counterfactual vector system

"""
Typed Counterfactual Vector System

This file implements a type-stable counterfactual vector system for efficient statistical analysis.
Provides type-stable counterfactual vectors with concrete typed variants.

Counterfactual vectors provide efficient single-row variable substitution for statistical
formula evaluation without data copying, enabling counterfactual analysis, sensitivity testing,
and marginal effects computation.
"""

using CategoricalArrays

# Phase 1: Abstract type hierarchy and concrete implementations

"""
    CounterfactualVector{T} <: AbstractVector{T}

Abstract supertype for all typed counterfactual vectors.

Counterfactual vectors provide efficient single-row variable substitution for statistical
formula evaluation without data copying. Used for counterfactual analysis, sensitivity testing,
and marginal effects computation.

Provides single-row substitution for statistical analysis.

All subtypes must implement:
- `base::AbstractVector{T}`: Original data column
- `row::Int`: Row index to override
- `replacement::T`: Replacement value for the specified row
"""
abstract type CounterfactualVector{T} <: AbstractVector{T} end

# Common interface for all CounterfactualVector subtypes
@inline Base.getindex(v::CounterfactualVector, i::Int) = (i == v.row ? v.replacement : v.base[i])
Base.size(v::CounterfactualVector) = size(v.base)
Base.length(v::CounterfactualVector) = length(v.base)
Base.IndexStyle(::Type{<:CounterfactualVector}) = IndexLinear()
Base.eltype(::Type{CounterfactualVector{T}}) where {T} = T

"""
    BoolCounterfactualVector <: CounterfactualVector{Bool}

Type-stable counterfactual vector for Boolean variables.

Provides efficient single-row substitution for boolean data without copying the
entire column. Used for binary contrast evaluation and boolean variable scenarios.

# Fields
- `base::Vector{Bool}`: Original boolean data
- `row::Int`: Row index to override (1-based)
- `replacement::Bool`: Boolean value to substitute

# Example
```julia
data = [true, false, true, false]
cf_vec = BoolCounterfactualVector(data, 2, true)

cf_vec[1]  # → true (original)
cf_vec[2]  # → true (counterfactual)
cf_vec[3]  # → true (original)
cf_vec[4]  # → false (original)
```
"""
mutable struct BoolCounterfactualVector <: CounterfactualVector{Bool}
    base::Vector{Bool}
    row::Int
    replacement::Bool
end

"""
    NumericCounterfactualVector{T<:Real} <: CounterfactualVector{T}

Type-stable counterfactual vector for numeric variables.

Provides efficient single-row substitution for numeric data with concrete element
type T. Supports any Real number type with consistent performance characteristics.

# Type Parameters
- `T<:Real`: Element type (Int64, Float64, Float32, etc.)

# Fields
- `base::Vector{T}`: Original numeric data
- `row::Int`: Row index to override (1-based)
- `replacement::T`: Numeric value to substitute

# Example
```julia
data = [1.0, 2.0, 3.0, 4.0]
cf_vec = NumericCounterfactualVector{Float64}(data, 3, 99.0)

cf_vec[1]  # → 1.0 (original)
cf_vec[2]  # → 2.0 (original)
cf_vec[3]  # → 99.0 (counterfactual)
cf_vec[4]  # → 4.0 (original)
```
"""
mutable struct NumericCounterfactualVector{T<:Real} <: CounterfactualVector{T}
    base::Vector{T}
    row::Int
    replacement::T
end

"""
    CategoricalCounterfactualVector{T,R} <: CounterfactualVector{CategoricalValue{T,R}}

Type-stable counterfactual vector for categorical variables.

Provides efficient single-row substitution for CategoricalArray data while
preserving all categorical metadata (levels, ordering, reference type).

# Type Parameters
- `T`: Level type (String, Symbol, Int, etc.)
- `R`: Reference type (UInt8, UInt16, UInt32)

# Fields
- `base::CategoricalArray{T,1,R}`: Original categorical data
- `row::Int`: Row index to override (1-based)
- `replacement::CategoricalValue{T,R}`: Categorical value to substitute

# Example
```julia
using CategoricalArrays
data = categorical(["A", "B", "A", "B"])
level_B = CategoricalValue("B", data[1].pool)
cf_vec = CategoricalCounterfactualVector{String,UInt32}(data, 1, level_B)

cf_vec[1]  # → CategoricalValue("B") (counterfactual)
cf_vec[2]  # → CategoricalValue("B") (original)
cf_vec[3]  # → CategoricalValue("A") (original)
cf_vec[4]  # → CategoricalValue("B") (original)
```
"""
mutable struct CategoricalCounterfactualVector{T,R} <: CounterfactualVector{CategoricalValue{T,R}}
    base::CategoricalArray{T,1,R}
    row::Int
    replacement::CategoricalValue{T,R}
end

"""
    TypedCounterfactualVector{T,V<:AbstractVector{T}} <: CounterfactualVector{T}

Generic type-stable counterfactual vector for other vector types.

Fallback implementation that works with any AbstractVector subtype while
preserving type stability. Used for vector types that don't have specialized
counterfactual vector implementations.

# Type Parameters
- `T`: Element type
- `V<:AbstractVector{T}`: Specific vector type

# Fields
- `base::V`: Original data vector
- `row::Int`: Row index to override (1-based)
- `replacement::T`: Value to substitute

# Example
```julia
using InlineStrings
data = InlineString15.(["short", "text", "here"])
cf_vec = TypedCounterfactualVector{InlineString15,typeof(data)}(data, 2, "REPLACED")

cf_vec[1]  # → "short" (original)
cf_vec[2]  # → "REPLACED" (counterfactual)
cf_vec[3]  # → "here" (original)
```
"""
mutable struct TypedCounterfactualVector{T,V<:AbstractVector{T}} <: CounterfactualVector{T}
    base::V
    row::Int
    replacement::T
end

"""
    CategoricalMixtureCounterfactualVector{T} <: CounterfactualVector{T}

Type-stable counterfactual vector for categorical mixture variables.

Provides efficient single-row substitution for categorical mixture data while
maintaining mixture specifications and supporting derivative computation.

# Type Parameters
- `T`: Mixture object type (CategoricalMixture, MixtureWithLevels, etc.)

# Fields
- `base::Vector{T}`: Original mixture data column
- `row::Int`: Row index to override (1-based)
- `replacement::T`: Mixture object to substitute

# Example
```julia
using FormulaCompiler

# Original mixture column
mixtures = [mix("A" => 0.3, "B" => 0.7), mix("A" => 0.3, "B" => 0.7)]
replacement_mix = mix("A" => 0.8, "B" => 0.2)

cf_vec = CategoricalMixtureCounterfactualVector(mixtures, 1, replacement_mix)

cf_vec[1]  # → mix("A" => 0.8, "B" => 0.2) (counterfactual)
cf_vec[2]  # → mix("A" => 0.3, "B" => 0.7) (original)
```

# Applications
- Single-row mixture perturbations for sensitivity analysis
- Derivative computation with respect to mixture weights
- Individual observation mixture effects
"""
mutable struct CategoricalMixtureCounterfactualVector{T} <: CounterfactualVector{T}
    base::Vector{T}
    row::Int
    replacement::T
end

# Phase 2: Type-stable construction functions

"""
    counterfactualvector(col::AbstractVector, row::Int) -> CounterfactualVector

Create an appropriate typed counterfactual vector for the given column type.

Dispatches on the column type to create the most efficient counterfactual vector
implementation. All returned counterfactual vectors are concrete subtypes of
CounterfactualVector{T} for type stability.

# Arguments
- `col::AbstractVector`: Data column to create counterfactual for
- `row::Int`: Initial row index (can be updated later)

# Returns
Appropriate CounterfactualVector subtype based on column type:
- `BoolCounterfactualVector` for `Vector{Bool}`
- `NumericCounterfactualVector{T}` for `Vector{T<:Real}`
- `CategoricalCounterfactualVector{T,R}` for `CategoricalArray{T,1,R}`
- `TypedCounterfactualVector{T,V}` for other `AbstractVector{T}` types

# Example
```julia
bool_col = [true, false, true]
bool_cf = counterfactualvector(bool_col, 1)  # BoolCounterfactualVector

float_col = [1.0, 2.0, 3.0]
numeric_cf = counterfactualvector(float_col, 1)  # NumericCounterfactualVector{Float64}

cat_col = categorical(["A", "B", "C"])
cat_cf = counterfactualvector(cat_col, 1)  # CategoricalCounterfactualVector{String,UInt32}
```
"""
function counterfactualvector end

# Step 2.1: Type Dispatch Constructor Functions
# Basic dispatch (preserves original types) - for general/contrast usage

counterfactualvector(col::Vector{Bool}, row::Int) = BoolCounterfactualVector(col, row, false)

counterfactualvector(col::Vector{T}, row::Int) where {T<:Real} = NumericCounterfactualVector{T}(col, row, zero(T))

counterfactualvector(col::CategoricalArray{T,1,R}, row::Int) where {T,R} =
    CategoricalCounterfactualVector{T,R}(col, row, col[1])  # Use first level as default

counterfactualvector(col::AbstractVector{T}, row::Int) where {T} =
    TypedCounterfactualVector{T,typeof(col)}(col, row, col[1])

# Categorical mixture dispatch
function counterfactualvector(col::Vector{T}, row::Int) where {T}
    # Check if T is a mixture type (has levels and weights fields)
    if hasfield(T, :levels) && hasfield(T, :weights)
        return CategoricalMixtureCounterfactualVector{T}(col, row, col[1])
    else
        return TypedCounterfactualVector{T,typeof(col)}(col, row, col[1])
    end
end

# Backend-specialized dispatch (for ADEvaluator type stability - FDEvaluator uses basic dispatch)
# Bool: Convert to Float64 for ContrastEvaluator (supports probabilistic contrasts)
# Categorical: Always backend-invariant (never change type)
counterfactualvector(col::Vector{Bool}, row::Int, ::Type{S}) where S<:AbstractFloat =
    NumericCounterfactualVector{S}(convert(Vector{S}, col), row, zero(S))

counterfactualvector(col::Vector{Bool}, row::Int, ::Type{S}) where S =
    BoolCounterfactualVector(col, row, false)

# Numeric types: convert for ADEvaluator type stability (uniform Dual types required)
counterfactualvector(col::Vector{T}, row::Int, ::Type{S}) where {T<:Union{AbstractFloat,Signed,Unsigned}, S<:Number} =
    NumericCounterfactualVector{S}(convert(Vector{S}, col), row, zero(S))

counterfactualvector(col::CategoricalArray{T,1,R}, row::Int, ::Type{S}) where {T,R,S} =
    CategoricalCounterfactualVector{T,R}(col, row, col[1])

# Categorical mixture dispatch (backend-invariant - mixtures don't change type for AD)
function counterfactualvector(col::Vector{T}, row::Int, ::Type{S}) where {T,S}
    # Check if T is a mixture type (has levels and weights fields)
    if hasfield(T, :levels) && hasfield(T, :weights)
        return CategoricalMixtureCounterfactualVector{T}(col, row, col[1])
    else
        return TypedCounterfactualVector{T,typeof(col)}(col, row, col[1])
    end
end

# Generic fallback (backend-invariant)
counterfactualvector(col::AbstractVector{T}, row::Int, ::Type{S}) where {T,S} =
    TypedCounterfactualVector{T,typeof(col)}(col, row, col[1])

# Step 2.2: Generic Update Functions
# Generic functions that work with any CounterfactualVector subtype

function update_counterfactual_row!(cv::CounterfactualVector, new_row::Int)
    cv.row = new_row
    return cv
end

function update_counterfactual_replacement!(cv::CounterfactualVector{T}, replacement::T) where T
    cv.replacement = replacement
    return cv
end

# Step 2.3: Backend-Specific Data Builder Functions

# Basic builder function (preserves original types - for FDEvaluator, ContrastEvaluator, and general use)
function build_counterfactual_data(base::NamedTuple, vars::Vector{Symbol}, row::Int)
    counterfactual_vecs = map(vars) do var
        col = getproperty(base, var)
        counterfactualvector(col, row)
    end

    # Create typed merged data
    pairs = [var => cv for (var, cv) in zip(vars, counterfactual_vecs)]
    data_counterfactual = merge(base, NamedTuple(pairs))

    return data_counterfactual, Tuple(counterfactual_vecs)
end

# Backend-specialized builder function (for ADEvaluator type stability)
function build_counterfactual_data(base::NamedTuple, vars::Vector{Symbol}, row::Int, ::Type{T}) where T
    counterfactual_vecs = map(vars) do var
        col = getproperty(base, var)
        counterfactualvector(col, row, T)  # Converts numeric types to uniform T (e.g., Dual)
    end

    pairs = [var => cv for (var, cv) in zip(vars, counterfactual_vecs)]
    data_counterfactual = merge(base, NamedTuple(pairs))

    return data_counterfactual, Tuple(counterfactual_vecs)
end

# Step 2.4: Type-Safe Accessors
# Safe accessors for working with counterfactual tuples

function get_counterfactual_for_var(counterfactuals::Tuple, vars::Vector{Symbol}, var::Symbol)
    idx = findfirst(==(var), vars)
    idx === nothing && error("Variable $var not found in counterfactual vars")
    return counterfactuals[idx]
end

function update_counterfactual_for_var!(counterfactuals::Tuple, vars::Vector{Symbol}, var::Symbol,
                                       row::Int, replacement)
    cf_vec = get_counterfactual_for_var(counterfactuals, vars, var)
    update_counterfactual_row!(cf_vec, row)

    # Handle categorical value conversion
    if cf_vec isa CategoricalCounterfactualVector
        # Find an existing CategoricalValue with the desired level
        base_array = cf_vec.base
        matching_idx = findfirst(x -> string(x) == string(replacement), base_array)
        if matching_idx === nothing
            error("Level $replacement not found in categorical variable $var")
        end
        cat_val = base_array[matching_idx]
        update_counterfactual_replacement!(cf_vec, cat_val)
    else
        # Non-categorical variables (boolean, numeric) - handle type conversion
        if cf_vec isa NumericCounterfactualVector{T} where T
            # Convert replacement to the correct numeric type
            converted_replacement = convert(eltype(cf_vec), replacement)
            update_counterfactual_replacement!(cf_vec, converted_replacement)
        else
            # Direct replacement for other types (boolean, etc.)
            update_counterfactual_replacement!(cf_vec, replacement)
        end
    end

    return cf_vec
end

# Optimized version with pre-computed categorical level mappings
function update_counterfactual_for_var!(counterfactuals::Tuple, vars::Vector{Symbol}, var::Symbol,
                                       row::Int, replacement, categorical_level_maps::Dict{Symbol, Dict{String, CategoricalValue}})
    cf_vec = get_counterfactual_for_var(counterfactuals, vars, var)
    update_counterfactual_row!(cf_vec, row)

    # Use pre-computed level mapping for categorical variables (zero allocations)
    if cf_vec isa CategoricalCounterfactualVector && haskey(categorical_level_maps, var)
        level_map = categorical_level_maps[var]
        replacement_str = string(replacement)
        if haskey(level_map, replacement_str)
            cat_val = level_map[replacement_str]
            update_counterfactual_replacement!(cf_vec, cat_val)
        else
            error("Level $replacement_str not found in pre-computed mapping for variable $var")
        end
    else
        # Non-categorical variables (boolean, numeric, mixtures) - handle type conversion
        # IMPORTANT: Check BoolCounterfactualVector FIRST before NumericCounterfactualVector
        # to ensure type-stable dispatch (Bool <: Real, so order matters for method resolution)
        if cf_vec isa BoolCounterfactualVector
            # Boolean variables - direct replacement (no conversion needed)
            update_counterfactual_replacement!(cf_vec, replacement)
        elseif cf_vec isa NumericCounterfactualVector{T} where T
            # Numeric variables (Int, Float, etc.) - convert to correct type
            converted_replacement = convert(eltype(cf_vec), replacement)
            update_counterfactual_replacement!(cf_vec, converted_replacement)
        elseif cf_vec isa CategoricalMixtureCounterfactualVector
            # Mixture variables - replacement should be another mixture object
            update_counterfactual_replacement!(cf_vec, replacement)
        else
            # Other types (TypedCounterfactualVector, etc.)
            update_counterfactual_replacement!(cf_vec, replacement)
        end
    end

    return cf_vec
end