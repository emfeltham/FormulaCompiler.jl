# Phase 1: Core Override Integration
# Add required imports and basic integration with existing system

###############################################################################
# REQUIRED IMPORTS (ADD TO TOP OF FormulaCompiler.jl)
###############################################################################

# Add these imports to FormulaCompiler.jl:
# using CategoricalArrays
# using Tables

###############################################################################
# CORE OVERRIDE VECTOR (NO CHANGES NEEDED - WORKS AS-IS)
###############################################################################

"""
    OverrideVector{T} <: AbstractVector{T}

A lazy vector that returns the same override value for all indices.
This avoids allocating full arrays when setting all observations to a representative value.

# Example
```julia
# Instead of: fill(2.5, 1_000_000)  # Allocates 8MB
# Use: OverrideVector(2.5, 1_000_000)  # Allocates ~32 bytes
```
"""
struct OverrideVector{T} <: AbstractVector{T}
    override_value::T
    length::Int
    
    function OverrideVector(value::T, length::Int) where T
        new{T}(value, length)
    end
end

# AbstractVector interface
Base.size(v::OverrideVector) = (v.length,)
Base.length(v::OverrideVector) = v.length
Base.getindex(v::OverrideVector, i::Int) = v.override_value
Base.IndexStyle(::Type{<:OverrideVector}) = IndexLinear()

# Efficient iteration
Base.iterate(v::OverrideVector, state=1) = state > v.length ? nothing : (v.override_value, state + 1)

###############################################################################
# CATEGORICAL OVERRIDE SUPPORT (NO CHANGES NEEDED)
###############################################################################

"""
    create_categorical_override(value, original_column::CategoricalArray)

Create a categorical override vector with proper level information.
"""
function create_categorical_override(value::AbstractString, original_column::CategoricalArray)
    levels_list = levels(original_column)
    
    if value ∉ levels_list
        error("Override value '$value' not in categorical levels: $levels_list")
    end
    
    # Create proper CategoricalValue using CategoricalArrays.jl API
    temp_cat = categorical([value], levels=levels_list)
    categorical_value = temp_cat[1]
    return OverrideVector(categorical_value, length(original_column))
end

function create_categorical_override(value::CategoricalValue, original_column::CategoricalArray)
    return OverrideVector(value, length(original_column))
end

###############################################################################
# DATA SCENARIO INFRASTRUCTURE (NO CHANGES NEEDED)
###############################################################################

"""
    DataScenario

Represents a data scenario with specific variable overrides.
Contains the modified data that can be used directly with compiled formulas.
"""
mutable struct DataScenario
    name::String
    overrides::Dict{Symbol,Any}
    data::NamedTuple
    original_data::NamedTuple
end

"""
    create_scenario(name, original_data; overrides...)

Create a data scenario with specified variable overrides.
"""
function create_scenario(name::String, original_data::NamedTuple; overrides...)
    override_dict = Dict{Symbol,Any}(overrides)
    
    if isempty(override_dict)
        modified_data = original_data
    else
        modified_data = create_override_data(original_data, override_dict)
    end
    
    return DataScenario(name, override_dict, modified_data, original_data)
end

function create_scenario(name::String, original_data::NamedTuple, overrides::Dict{Symbol,<:Any})
    override_dict = Dict{Symbol,Any}(overrides)
    
    if isempty(overrides)
        return DataScenario(name, override_dict, original_data, original_data)
    end
    
    modified_data = create_override_data(original_data, override_dict)
    return DataScenario(name, override_dict, modified_data, original_data)
end

"""
    create_override_data(original_data, overrides)

Create modified data NamedTuple with variable overrides using OverrideVector.
"""
function create_override_data(original_data::NamedTuple, overrides::Dict{Symbol,Any})
    # Start with original data
    modified_dict = Dict{Symbol, Any}()
    
    # Copy all columns, applying overrides where specified
    for (key, original_column) in pairs(original_data)
        if haskey(overrides, key)
            # Create override for this variable
            override_value = overrides[key]
            modified_dict[key] = create_override_vector(override_value, original_column)
        else
            # Keep original column as reference (no copy)
            modified_dict[key] = original_column
        end
    end
    
    # Convert back to NamedTuple for efficiency
    return NamedTuple(modified_dict)
end

"""
    create_override_vector(value, original_column)

Create appropriate OverrideVector based on original column type.
"""
function create_override_vector(value, original_column::AbstractVector)
    if original_column isa CategoricalArray
        return create_categorical_override(value, original_column)
    else
        # Numeric or other types - convert to appropriate type
        converted_value = convert(eltype(original_column), value)
        return OverrideVector(converted_value, length(original_column))
    end
end

###############################################################################
# INTEGRATION WITH EXISTING SYSTEM - NEW METHODS
###############################################################################

"""
    modelrow!(row_vec, compiled::CompiledFormula, scenario::DataScenario, row_idx)

Evaluate model row using a data scenario with CompiledFormula.
"""
function modelrow!(
    row_vec::AbstractVector{Float64}, 
    compiled::CompiledFormula, 
    scenario::DataScenario, 
    row_idx::Int
)
    compiled(row_vec, scenario.data, row_idx)
    return row_vec
end

"""
    modelrow!(row_vec, specialized::SpecializedFormula, scenario::DataScenario, row_idx)

Evaluate model row using a data scenario with SpecializedFormula.
"""
function modelrow!(
    row_vec::AbstractVector{Float64}, 
    specialized::SpecializedFormula, 
    scenario::DataScenario, 
    row_idx::Int
)
    specialized(row_vec, scenario.data, row_idx)
    return row_vec
end

"""
    modelrow(compiled::CompiledFormula, scenario::DataScenario, row_idx) -> Vector{Float64}

Evaluate model row using a data scenario (allocating version).
"""
function modelrow(compiled::CompiledFormula, scenario::DataScenario, row_idx::Int)
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, scenario.data, row_idx)
    return row_vec
end

"""
    modelrow(specialized::SpecializedFormula, scenario::DataScenario, row_idx) -> Vector{Float64}

Evaluate model row using a data scenario with SpecializedFormula (allocating version).
"""
function modelrow(specialized::SpecializedFormula, scenario::DataScenario, row_idx::Int)
    row_vec = Vector{Float64}(undef, length(specialized))
    specialized(row_vec, scenario.data, row_idx)
    return row_vec
end


###############################################################################
# SCENARIO MUTATION OPERATIONS
###############################################################################

"""
    set_override!(scenario::DataScenario, variable::Symbol, value)

Add or update a variable override in the scenario.
Rebuilds the scenario data to reflect the change.

# Example
```julia
scenario = create_scenario("test", data; x = 1.0)
set_override!(scenario, :y, 2.0)      # Add new override
set_override!(scenario, :x, 5.0)      # Update existing override
```
"""
function set_override!(scenario::DataScenario, variable::Symbol, value)
    scenario.overrides[variable] = value
    scenario.data = create_override_data(scenario.original_data, scenario.overrides)  # Use original_data
    return scenario
end

"""
    remove_override!(scenario::DataScenario, variable::Symbol)

Remove a variable override from the scenario.
The variable will revert to its original values.

# Example
```julia
remove_override!(scenario, :x)  # x returns to original values
```
"""
function remove_override!(scenario::DataScenario, variable::Symbol)
    if haskey(scenario.overrides, variable)
        delete!(scenario.overrides, variable)
        scenario.data = create_override_data(scenario.original_data, scenario.overrides)  # Use original_data
    end
    return scenario
end

"""
    update_scenario!(scenario::DataScenario; overrides...)

Update multiple overrides at once.

# Example
```julia
update_scenario!(scenario; x = 2.0, y = 3.0, group = "B")
```
"""
function update_scenario!(scenario::DataScenario; overrides...)
    if !isempty(overrides)
        for (key, value) in overrides
            scenario.overrides[key] = value
        end
        scenario.data = create_override_data(scenario.original_data, scenario.overrides)  # Use original_data
    end
    return scenario
end
