# override.jl
# Core foundation for data scenarios with OverrideVector

###############################################################################
# OVERRIDE VECTOR IMPLEMENTATION
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
# CATEGORICAL OVERRIDE SUPPORT
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
# DATA SCENARIO INFRASTRUCTURE
###############################################################################

"""
    DataScenario

Represents a data scenario with specific variable overrides.
Contains the modified data that can be used directly with compiled formulas.

# Fields
- `name::String`: Descriptive name for the scenario
- `overrides::Dict{Symbol,Any}`: Variable overrides (mutable for iterative development)  
- `data::NamedTuple`: Modified column-table data with OverrideVectors applied
- `original_data::NamedTuple`
"""
mutable struct DataScenario
    name::String
    overrides::Dict{Symbol,Any}  # ~48 bytes + contents
    data::NamedTuple            # ~24 bytes + references  
    original_data::NamedTuple   # ~24 bytes + references
end

"""
    create_scenario(name, original_data; overrides...)

Create a data scenario with specified variable overrides.

# Arguments
- `name::String`: Name for this scenario
- `original_data::NamedTuple`: Original column-table data
- `overrides...`: Keyword arguments for variable overrides

# Example
```julia
data = Tables.columntable(df)

# Override single variable
scenario1 = create_scenario("x_at_mean", data; x = mean(data.x))

# Override multiple variables  
scenario2 = create_scenario("policy", data; x = 2.5, group = "A")

# Use with compiled formula
compiled = compile_formula(model)
row_vec = Vector{Float64}(undef, length(compiled))
compiled(row_vec, scenario1.data, row_idx)
```
"""
function create_scenario(name::String, original_data::NamedTuple; overrides...)
    override_dict = Dict{Symbol,Any}(overrides)
    
    if isempty(override_dict)
        modified_data = original_data
    else
        modified_data = create_override_data(original_data, override_dict)
    end
    
    return DataScenario(name, override_dict, modified_data, original_data)  # Add original_data
end

function create_scenario(name::String, original_data::NamedTuple, overrides::Dict{Symbol,<:Any})
    override_dict = Dict{Symbol,Any}(overrides)
    
    if isempty(overrides)
        return DataScenario(name, override_dict, original_data, original_data)  # Add original_data
    end
    
    modified_data = create_override_data(original_data, override_dict)
    return DataScenario(name, override_dict, modified_data, original_data)  # Add original_data
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

######

###############################################################################
# SCENARIO COLLECTIONS
###############################################################################

"""
    ScenarioCollection

Container for multiple related scenarios.
"""
struct ScenarioCollection
    name::String
    scenarios::Vector{DataScenario}
    original_data::NamedTuple
end

"""
    create_scenario_grid(collection_name, data, variable_grids::Dict{Symbol, <:AbstractVector})

Create a collection of scenarios for all combinations of variable values.

# Arguments
- `collection_name::String`: Name for the scenario collection
- `data::NamedTuple`: Original column-table data
- `variable_grids::Dict{Symbol, <:AbstractVector}`: Variables and their value grids

# Example
```julia
# Single variable grid
collection = create_scenario_grid("x_variation", data, Dict(:x => [1.0, 2.0, 3.0]))

# Multiple variable combinations  
collection = create_scenario_grid("policy_analysis", data, Dict(
    :x => [1.0, 2.0],
    :group => ["A", "B"],
    :treatment => [true, false]
))
# Results in 8 scenarios: all combinations of x × group × treatment
```
"""
function create_scenario_grid(collection_name::String,
                             data::NamedTuple,
                             variable_grids::Dict{Symbol, <:AbstractVector})
    
    if isempty(variable_grids)
        # No grids specified - return collection with single scenario (original data)
        scenario = create_scenario("$(collection_name)_original", data)
        return ScenarioCollection(collection_name, [scenario], data)
    end
    
    # Extract variables and their value grids
    variables = collect(keys(variable_grids))
    value_grids = [collect(variable_grids[var]) for var in variables]
    
    # Generate all combinations using Cartesian product
    scenarios = DataScenario[]
    
    for (i, combination) in enumerate(Iterators.product(value_grids...))
        # Create overrides for this combination
        override_dict = Dict{Symbol, Any}()
        for (j, var) in enumerate(variables)
            override_dict[var] = combination[j]
        end
        
        # Generate descriptive scenario name
        scenario_name = generate_scenario_name(collection_name, variables, combination, i)
        scenario = create_scenario(scenario_name, data, override_dict)
        push!(scenarios, scenario)
    end
    
    return ScenarioCollection(collection_name, scenarios, data)
end

"""
    generate_scenario_name(collection_name, variables, values, index)

Generate a descriptive name for a scenario based on its variable values.
"""
function generate_scenario_name(collection_name::String, variables::Vector{Symbol}, values, index::Int)
    if length(variables) == 1
        # Single variable: "collection_var_value"
        var = variables[1]
        val = values[1]
        return "$(collection_name)_$(var)_$(val)"
    elseif length(variables) <= 3
        # Few variables: "collection_var1_val1_var2_val2"
        parts = String[collection_name]
        for (var, val) in zip(variables, values)
            push!(parts, "$(var)_$(val)")
        end
        return join(parts, "_")
    else
        # Many variables: fall back to index
        return "$(collection_name)_scenario_$(index)"
    end
end

###############################################################################
# SCENARIO UTILITIES
###############################################################################

"""
    Base.length(collection::ScenarioCollection)

Get number of scenarios in collection.
"""
Base.length(collection::ScenarioCollection) = length(collection.scenarios)

"""
    Base.getindex(collection::ScenarioCollection, i::Int)

Get scenario by index.
"""
Base.getindex(collection::ScenarioCollection, i::Int) = collection.scenarios[i]

"""
    Base.iterate(collection::ScenarioCollection, state=1)

Iterate over scenarios in collection.
"""
Base.iterate(collection::ScenarioCollection, state=1) = state > length(collection.scenarios) ? nothing : (collection.scenarios[state], state + 1)

"""
    get_scenario_by_name(collection::ScenarioCollection, name::String)

Find scenario by name in collection.
"""
function get_scenario_by_name(collection::ScenarioCollection, name::String)
    for scenario in collection.scenarios
        if scenario.name == name
            return scenario
        end
    end
    error("Scenario '$name' not found in collection '$(collection.name)'")
end

"""
    list_scenarios(collection::ScenarioCollection)

Print summary of all scenarios in collection.
"""
function list_scenarios(collection::ScenarioCollection)
    println("ScenarioCollection: $(collection.name)")
    println("Number of scenarios: $(length(collection.scenarios))")
    println()
    
    for (i, scenario) in enumerate(collection.scenarios)
        println("  $i. $(scenario.name)")
        if !isempty(scenario.overrides)
            for (var, val) in scenario.overrides
                println("     $var = $val")
            end
        else
            println("     (no overrides - original data)")
        end
    end
end

"""
    list_scenario_overrides(scenario::DataScenario)

Print the overrides for a single scenario.
"""
function list_scenario_overrides(scenario::DataScenario)
    println("Scenario: $(scenario.name)")
    if isempty(scenario.overrides)
        println("  No overrides (original data)")
    else
        println("  Overrides:")
        for (var, val) in scenario.overrides
            println("    $var = $val")
        end
    end
end

###############################################################################
# INTEGRATION WITH modelrow!
###############################################################################

"""
    modelrow!(row_vec, compiled, scenario::DataScenario, row_idx)

Evaluate model row using a data scenario.
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
    modelrow_scenarios!(matrix, compiled, scenarios, row_idx)

Evaluate model row across multiple scenarios.

# Arguments
- `matrix::Matrix{Float64}`: Pre-allocated matrix (size: length(scenarios) × length(compiled))
- `compiled::CompiledFormula`: Compiled formula
- `scenarios::Vector{DataScenario}`: Vector of scenarios to evaluate
- `row_idx::Int`: Row index to evaluate

# Example
```julia
scenarios = [scenario1, scenario2, scenario3]
matrix = Matrix{Float64}(undef, length(scenarios), length(compiled))
modelrow_scenarios!(matrix, compiled, scenarios, 5)
# matrix[1, :] = row 5 under scenario1
# matrix[2, :] = row 5 under scenario2  
# matrix[3, :] = row 5 under scenario3
```
"""
function modelrow_scenarios!(
    matrix::AbstractMatrix{Float64},
    compiled::CompiledFormula,
    scenarios::Vector{DataScenario},
    row_idx::Int
)
    
    @assert size(matrix, 1) >= length(scenarios) "Matrix height insufficient for scenarios"
    @assert size(matrix, 2) == length(compiled) "Matrix width must match compiled formula width"
    
    for (i, scenario) in enumerate(scenarios)
        row_view = view(matrix, i, :)
        modelrow!(row_view, compiled, scenario, row_idx)
    end
    
    return matrix
end
