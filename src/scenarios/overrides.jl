# override_unified.jl
# Complete unified override system with all functionality and fixes integrated

###############################################################################
# CORE OVERRIDE VECTOR IMPLEMENTATION
###############################################################################

# AbstractVector interface
Base.size(v::OverrideVector) = (v.length,)
Base.length(v::OverrideVector) = v.length
Base.getindex(v::OverrideVector, i::Int) = v.override_value
Base.IndexStyle(::Type{<:OverrideVector}) = IndexLinear()

# Efficient iteration
Base.iterate(v::OverrideVector, state=1) = state > v.length ? nothing : (v.override_value, state + 1)

# AbstractVector interface for CategoricalMixtureOverride
Base.size(v::CategoricalMixtureOverride) = (v.length,)
Base.length(v::CategoricalMixtureOverride) = v.length
Base.getindex(v::CategoricalMixtureOverride, i::Int) = v.mixture_obj
Base.IndexStyle(::Type{<:CategoricalMixtureOverride}) = IndexLinear()

# Efficient iteration for CategoricalMixtureOverride  
Base.iterate(v::CategoricalMixtureOverride, state=1) = state > v.length ? nothing : (v.mixture_obj, state + 1)

###############################################################################
# CATEGORICAL OVERRIDE SUPPORT
###############################################################################

"""
    create_categorical_override(value, original_column::CategoricalArray)

Construct constant-value column for categorical variable substitution.

Creates memory-efficient override for categorical variables in counterfactual scenarios,
maintaining categorical properties (levels, ordering) while providing constant value 
across all observations.

# Applications
- Treatment assignment: Fix all subjects to specific treatment condition
- Regional standardization: Evaluate effects holding location constant
- Demographic control: Standardize categorical covariates for comparison

# Arguments
- `value`: Categorical level specification (String, Symbol, Int index, Bool, or CategoricalValue)
- `original_column::CategoricalArray`: Original categorical variable from dataset

# Returns
- `OverrideVector{CategoricalValue}`: Constant-memory categorical override maintaining type properties

# Value Specification Methods
- `String`/`Symbol`: Level name (must exist in original levels)
- `Int`: Level index (1-based indexing into levels array)
- `Bool`: For binary categorical variables
- `CategoricalValue`: Direct categorical value (preserves all properties)

# Example
```julia
# Original categorical variable
treatment = categorical(["Control", "Drug_A", "Control", "Drug_B"], 
                       levels=["Control", "Drug_A", "Drug_B"])

# Treatment counterfactuals
control_override = create_categorical_override("Control", treatment)
drug_a_override = create_categorical_override("Drug_A", treatment)

# Using level indices
control_override = create_categorical_override(1, treatment)  # "Control" is level 1

# Using symbols for convenience
drug_b_override = create_categorical_override(:Drug_B, treatment)
```

# Type Safety
- Validates override value exists in original categorical levels
- Preserves ordering properties (ordered vs unordered categoricals)
- Maintains level structure for statistical compatibility
"""
function create_categorical_override(value::T, original_column::CategoricalArray{T}) where T
    levels_list = levels(original_column)
    
    if value ∉ levels_list
        error("""
        Invalid categorical override: '$value' not found in levels $levels_list
        
        Valid approaches:
        - Use existing level: create_categorical_override("$(levels_list[1])", column)
        - Check for typographical errors in level specification
        - Verify categorical levels with levels(column)
        """)
    end
   
    # Create proper CategoricalValue that preserves ordering and levels
    # (This should work for T = Bool too)
    temp_cat = categorical([value], levels=levels_list, ordered=isordered(original_column))
    categorical_value = temp_cat[1]
    
    return OverrideVector(categorical_value, length(original_column))
end

# Specialized method for String (most common case)
function create_categorical_override(value::AbstractString, original_column::CategoricalArray)

    
    levels_list = levels(original_column)
    
    if value ∉ levels_list
        error("""
        Invalid categorical override: '$value' not found in levels $levels_list
        
        Valid approaches:
        - Use existing level: create_categorical_override("$(levels_list[1])", column)
        - Check for typographical errors in level specification  
        - Verify categorical levels with levels(column)
        """)
    end
    
    # Create proper CategoricalValue using CategoricalArrays.jl API
    temp_cat = categorical([value], levels=levels_list, ordered=isordered(original_column))
    categorical_value = temp_cat[1]
    
    return OverrideVector(categorical_value, length(original_column))
end

# Method for using level index directly
function create_categorical_override(value::Int, original_column::CategoricalArray)
    n_levels = length(levels(original_column))
    if value < 1 || value > n_levels
        error("Level index $value out of range [1, $n_levels] for categorical levels: $(levels(original_column))")
    end
    
    # Get the actual categorical value for this level index
    level_value = levels(original_column)[value]
    temp_cat = categorical([level_value], levels=levels(original_column), ordered=isordered(original_column))
    categorical_value = temp_cat[1]
    
    return OverrideVector(categorical_value, length(original_column))
end

# Method for Symbol (convenience)
function create_categorical_override(value::Symbol, original_column::CategoricalArray)
    return create_categorical_override(string(value), original_column)
end

# Method when already have a CategoricalValue (preserves all properties)
function create_categorical_override(value::CategoricalValue, original_column::CategoricalArray)
    # Verify it's from compatible levels
    if levels(value) != levels(original_column)
        error("CategoricalValue has different levels than target column")
    end
    return OverrideVector(value, length(original_column))
end

# Method for Boolean categorical
function create_categorical_override(value::Bool, original_column::CategoricalArray{Bool})
    levels_list = levels(original_column)
    
    if value ∉ levels_list
        error("""
        Invalid categorical override: '$value' not found in levels $levels_list
        
        Valid approaches:
        - Use existing level: create_categorical_override("$(levels_list[1])", column)
        - Check for typographical errors in level specification  
        - Verify categorical levels with levels(column)
        """)
    end
    
    # Create proper CategoricalValue for Bool
    temp_cat = categorical([value], levels=levels_list, ordered=isordered(original_column))
    categorical_value = temp_cat[1]
    
    return OverrideVector(categorical_value, length(original_column))
end

# Method for categorical mixtures (from Margins.jl)
# This imports the MixtureWithLevels type from Margins.jl
function create_categorical_override(mixture_obj, original_column::CategoricalArray)
    # Check if this is a MixtureWithLevels object by duck typing
    if hasproperty(mixture_obj, :levels) && hasproperty(mixture_obj, :weights) && hasproperty(mixture_obj, :original_levels)
        # This is a categorical mixture - we need special handling
        # For now, we'll create a special OverrideVector type that knows about mixtures
        return CategoricalMixtureOverride(mixture_obj, length(original_column))
    else
        # Fallback to regular categorical override
        return create_categorical_override(string(mixture_obj), original_column)
    end
end

###############################################################################
# DATA SCENARIO INFRASTRUCTURE
###############################################################################

"""
    DataScenario

Counterfactual scenario container with variable substitutions for efficient analysis.

A DataScenario represents a hypothetical version of the data where specified variables
are held constant at chosen values, enabling systematic counterfactual analysis without
data duplication.

# Fields
- `name::String`: Descriptive identifier for the counterfactual scenario
- `overrides::Dict{Symbol,Any}`: Variable substitutions applied in this scenario  
- `data::NamedTuple`: Modified data structure with constant-value columns for overridden variables
- `original_data::NamedTuple`: Reference to original unmodified data

# Usage
DataScenario objects are typically created via `create_scenario()` and used directly 
with `compile_formula()` for zero-allocation counterfactual evaluation.

# Memory Efficiency
Uses OverrideVector for substituted variables, providing O(1) memory overhead
regardless of dataset size compared to O(n) for naive data copying approaches.
"""
mutable struct DataScenario
    name::String
    overrides::Dict{Symbol,Any}
    data::NamedTuple
    original_data::NamedTuple
end

"""
    create_scenario(name, data; variable_values...)
    create_scenario(name, data, overrides::Dict)

Construct counterfactual scenario with specified variable substitutions.

# Applications
- Policy analysis: `create_scenario("minimum_wage_15", data; wage = 15.0)`
- Treatment evaluation: `create_scenario("universal_treatment", data; treatment = true)`
- Sensitivity analysis: `create_scenario("standardized_age", data; age = 40)`
- Standardization: `create_scenario("urban_baseline", data; region = "Urban", education = "College")`

# Arguments
- `name::String`: Descriptive identifier for this counterfactual scenario
- `data::NamedTuple`: Original data in column-table format (from Tables.columntable)
- `variable_values...`: Keyword arguments specifying variable substitutions (or Dict in second method)

# Returns
- `DataScenario`: Counterfactual scenario with variable overrides applied

# Computational Properties
- Memory complexity: O(1) regardless of data size
- Evaluation: Zero-allocation with FormulaCompiler
- Scalability: Constant overhead for arbitrary dataset sizes

# Example
```julia
data = Tables.columntable(df)

# Policy counterfactual: minimum wage at \$15/hour
policy_scenario = create_scenario("min_wage_policy", data; wage = 15.0)

# Treatment counterfactual: universal intervention
treatment_scenario = create_scenario("all_treated", data; 
                                   treatment = true, dose = 100.0)

# Standardization: representative demographics
standard_scenario = create_scenario("representative", data;
                                  age = 35, education = "College", region = "Urban")

# Compile and evaluate (zero-allocation)
compiled = compile_formula(model, policy_scenario.data)
output = Vector{Float64}(undef, length(compiled))
compiled(output, policy_scenario.data, row_idx)
```

Related Functions:
- `compile_formula()`: Compile scenarios for evaluation
- `modelrow!()`: Efficient scenario evaluation
- `create_scenario_grid()`: Multiple scenario construction
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

Construct modified data structure with variable substitutions for counterfactual analysis.

# Applications  
- Internal function supporting `create_scenario()` workflow
- Advanced users requiring direct data modification without scenario wrapper
- Custom counterfactual analysis implementations

# Implementation
Creates new NamedTuple where overridden variables are replaced with OverrideVector 
instances, while non-overridden variables maintain original references.

# Arguments
- `original_data::NamedTuple`: Base data in column-table format
- `overrides::Dict{Symbol,Any}`: Variable-to-value mapping for substitutions

# Returns
- `NamedTuple`: Modified data structure with constant-value columns for overridden variables

# Computational Properties
- Memory: O(1) overhead per override, original data preserved by reference
- Type stability: Maintains NamedTuple structure for FormulaCompiler compatibility
- Variable handling: Automatic type conversion and categorical level validation

# Example
```julia
data = Tables.columntable(df)
overrides = Dict(:treatment => true, :dose => 100.0, :region => "Urban")
modified_data = create_override_data(data, overrides)

# Use directly with FormulaCompiler
compiled = compile_formula(model, modified_data)
```

Related Functions:
- `create_scenario()`: Higher-level interface creating DataScenario objects
- `create_override_vector()`: Creates individual variable overrides
- `compile_formula()`: Compiles modified data for evaluation
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

Construct constant-value vector with automatic type handling for variable substitution.

# Applications
- Internal function supporting `create_override_data()` workflow
- Low-level interface for custom counterfactual implementations
- Type-safe variable substitution with automatic conversion

# Implementation
Creates OverrideVector with appropriate element type based on original column,
handling categorical levels, boolean conversion, and numeric type preservation.

# Arguments
- `value`: Substitution value (any type compatible with original column)
- `original_column::AbstractVector`: Original data column defining target type

# Returns
- `OverrideVector`: Constant-memory vector returning specified value for all indices

# Type Handling
- **Categorical variables**: Validates levels and preserves ordering properties
- **Boolean variables**: Converts to Float64 (0.0/1.0) for statistical compatibility
- **Integer variables**: Handles fractional overrides with automatic Float64 promotion
- **Numeric types**: Attempts exact type conversion with Float64 fallback

# Example
```julia
# Categorical override
treatment_col = categorical(["Control", "Drug_A", "Control"])
override = create_override_vector("Drug_A", treatment_col)

# Numeric override with type conversion
age_col = [25, 30, 35, 40]  # Vector{Int}
override = create_override_vector(35.5, age_col)  # Promotes to Float64

# Boolean override 
enrolled_col = [true, false, true, false]
override = create_override_vector(true, enrolled_col)  # Returns Float64(1.0)
```

Related Functions:
- `create_override_data()`: Uses this function for each overridden variable
- `create_categorical_override()`: Specialized function for categorical variables
- `OverrideVector`: The constant-memory vector type created by this function
"""
function create_override_vector(value, original_column::AbstractVector)
    # Handle categorical mixtures first
    if hasproperty(value, :levels) && hasproperty(value, :weights) && hasproperty(value, :original_levels)
        # This is a MixtureWithLevels object
        if original_column isa CategoricalArray
            return create_categorical_override(value, original_column)
        elseif original_column isa Vector{Bool}
            # Handle Bool mixture for non-categorical Bool column
            # Convert to fractional representation (probability of true)
            mixture = value.mixture
            level_weight_dict = Dict(string.(mixture.levels) .=> mixture.weights)
            false_weight = get(level_weight_dict, "false", 0.0)
            true_weight = get(level_weight_dict, "true", 0.0)
            prob_true = true_weight  # Probability of true
            return OverrideVector(Float64(prob_true), length(original_column))
        else
            error("Categorical mixtures not supported for column type $(typeof(original_column))")
        end
    elseif original_column isa CategoricalArray
        # Categorical handling (including CategoricalArray{Bool})
        return create_categorical_override(value, original_column)
    elseif original_column isa Vector{Bool} && value isa Bool
        # Non-categorical boolean: convert to Float64 (0.0 or 1.0)
        converted_value = Float64(value)
        return OverrideVector(converted_value, length(original_column))
    elseif original_column isa Vector{<:Integer} && value isa AbstractFloat
        # Integer columns with float overrides
        if isinteger(value)
            # Integer-valued float: preserve original integer type
            try
                converted_value = convert(eltype(original_column), value)
                return OverrideVector(converted_value, length(original_column))
            catch InexactError
                # Value too large for integer type, use Float64
                converted_value = Float64(value)
                return OverrideVector(converted_value, length(original_column))
            end
        else
            # Fractional float: convert to Float64 to preserve fractional part
            converted_value = Float64(value)
            return OverrideVector(converted_value, length(original_column))
        end
    else
        # Other numeric or general types - try exact type conversion first
        try
            converted_value = convert(eltype(original_column), value)
            return OverrideVector(converted_value, length(original_column))
        catch MethodError
            # If conversion fails, use Float64 as fallback for numeric types
            if eltype(original_column) <: Number
                converted_value = Float64(value)
                return OverrideVector(converted_value, length(original_column))
            else
                rethrow()
            end
        end
    end
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
    scenario.data = create_override_data(scenario.original_data, scenario.overrides)
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
        scenario.data = create_override_data(scenario.original_data, scenario.overrides)
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
        scenario.data = create_override_data(scenario.original_data, scenario.overrides)
    end
    return scenario
end

"""
    set_categorical_override!(scenario::DataScenario, variable::Symbol, level::Union{String,Symbol,Int})

Set a categorical override with type validation.
"""
function set_categorical_override!(scenario::DataScenario, variable::Symbol, level::Union{String,Symbol,Int})
    original_column = scenario.original_data[variable]
    
    if !(original_column isa CategoricalArray)
        error("Variable $variable is not categorical. Use set_override! for non-categorical variables.")
    end
    
    # Convert and validate the level
    if level isa Symbol
        level = string(level)
    end
    
    if level isa String
        if level ∉ levels(original_column)
            error("Level '$level' not found in categorical levels: $(levels(original_column))")
        end
    elseif level isa Int
        n_levels = length(levels(original_column))
        if level < 1 || level > n_levels
            error("Level index $level out of range [1, $n_levels]")
        end
        level = levels(original_column)[level]  # Convert to string
    end
    
    set_override!(scenario, variable, level)
    return scenario
end

###############################################################################
# SCENARIO COLLECTIONS
###############################################################################

"""
    ScenarioCollection

Systematic collection of counterfactual scenarios for comprehensive analysis.

A ScenarioCollection organizes multiple related DataScenario objects with shared 
metadata, enabling batch operations, systematic evaluation, and organized analysis 
workflows for complex counterfactual studies.

# Applications
- Policy sensitivity analysis: Evaluate effects across parameter ranges
- Treatment dose-response studies: Systematic intervention level testing
- Regional comparison studies: Geographic heterogeneity analysis
- Robustness testing: Model stability across variable ranges

# Fields
- `name::String`: Descriptive identifier for the scenario collection
- `scenarios::Vector{DataScenario}`: Individual counterfactual scenarios
- `original_data::NamedTuple`: Reference to base data structure
- `metadata::Dict{String,Any}`: Collection properties (variables, grid sizes, creation time)

# Collection Interface
Implements standard collection operations:
- `length(collection)`: Number of scenarios in collection
- `collection[i]`: Access individual scenario by index
- `for scenario in collection`: Iterator support for batch processing

# Memory Efficiency
Each scenario maintains O(1) memory overhead regardless of original data size,
making collections viable for systematic analysis with many scenarios.

Related Functions:
- `create_scenario_grid()`: Primary constructor for systematic scenario collections
- `modelrow_scenarios!()`: Batch evaluation across collection scenarios
- `evaluate_scenarios_batch()`: Multi-scenario, multi-row evaluation
"""
struct ScenarioCollection
    name::String
    scenarios::Vector{DataScenario}
    original_data::NamedTuple
    metadata::Dict{String, Any}
    
    function ScenarioCollection(name::String, scenarios::Vector{DataScenario}, original_data::NamedTuple, metadata::Dict{String, Any} = Dict{String, Any}())
        new(name, scenarios, original_data, metadata)
    end
end

# Collection interface
Base.length(collection::ScenarioCollection) = length(collection.scenarios)
Base.getindex(collection::ScenarioCollection, i::Int) = collection.scenarios[i]
Base.iterate(collection::ScenarioCollection, state=1) = state > length(collection.scenarios) ? nothing : (collection.scenarios[state], state + 1)

"""
    create_scenario_grid(collection_name, data, variable_grids::Dict{Symbol, <:AbstractVector})

Construct systematic grid of counterfactual scenarios for comprehensive analysis.

# Applications
- Policy sensitivity analysis: Test outcomes across parameter ranges
- Treatment dose-response: Evaluate effects at multiple intervention levels  
- Regional comparisons: Assess heterogeneity across geographic areas
- Robustness testing: Examine model stability across variable ranges

# Arguments
- `collection_name::String`: Descriptive identifier for the scenario collection
- `data::NamedTuple`: Original data in column-table format
- `variable_grids::Dict{Symbol, <:AbstractVector}`: Variable-to-values mapping for grid construction

# Returns
- `ScenarioCollection`: Systematic collection of counterfactual scenarios

# Computational Properties
- Creates Cartesian product of all variable values
- Memory: O(scenarios) overhead, not O(scenarios × data_size)  
- Evaluation: Each scenario maintains zero-allocation properties

# Example
```julia
data = Tables.columntable(df)

# Create 2×3×2 = 12 scenarios
policy_grid = create_scenario_grid("policy_analysis", data, Dict(
    :treatment => [false, true],
    :dose => [50.0, 100.0, 150.0],
    :region => ["North", "South"]
))

# Evaluate all scenarios
compiled = compile_formula(model)
results = Matrix{Float64}(undef, length(policy_grid), length(compiled))
for (i, scenario) in enumerate(policy_grid)
    compiled(view(results, i, :), scenario.data, row_idx)
end

# Access specific scenario
baseline = get_scenario_by_name(policy_grid, "policy_analysis_treatment_false_dose_50.0_region_North")
```

!!! note
    Creates Cartesian product of all variable values.
    Scenario names are auto-generated based on variable values.
    Use `verbose=true` to print creation progress.
"""
function create_scenario_grid(collection_name::String,
                             data::NamedTuple,
                             variable_grids::Dict{Symbol, <:AbstractVector};
                             verbose::Bool = false)
    
    if isempty(variable_grids)
        scenario = create_scenario("$(collection_name)_original", data)
        return ScenarioCollection(collection_name, [scenario], data)
    end
    
    variables = sort(collect(keys(variable_grids)))
    value_grids = [collect(variable_grids[var]) for var in variables]
    
    scenarios = DataScenario[]
    total_combinations = prod(length.(value_grids))
    
    for (i, combination) in enumerate(Iterators.product(value_grids...))
        override_dict = Dict{Symbol, Any}()
        for (j, var) in enumerate(variables)
            override_dict[var] = combination[j]
        end
        
        scenario_name = generate_scenario_name(collection_name, variables, combination, i)
        scenario = create_scenario(scenario_name, data, override_dict)
        push!(scenarios, scenario)
        
        # if i % max(1, total_combinations ÷ 10) == 0
        #     println("  Progress: $(i)/$(total_combinations) scenarios created")
        # end
    end
    
    verbose && println("Created $(total_combinations) scenario grid combinations")

    metadata = Dict{String, Any}(
        "variables" => variables,
        "grid_sizes" => [length(grid) for grid in value_grids],
        "total_combinations" => total_combinations,
        "created_at" => now()
    )
    
    return ScenarioCollection(collection_name, scenarios, data, metadata)
end

"""
    generate_scenario_name(collection_name, variables, values, index)

Generate descriptive names for scenarios in collections.
"""
function generate_scenario_name(collection_name::String, variables::Vector{Symbol}, values, index::Int)
    if length(variables) == 1
        var = variables[1]
        val = values[1]
        return "$(collection_name)_$(var)_$(val)"
    elseif length(variables) <= 3
        parts = String[collection_name]
        for (var, val) in zip(variables, values)
            push!(parts, "$(var)_$(val)")
        end
        return join(parts, "_")
    else
        return "$(collection_name)_scenario_$(index)"
    end
end

###############################################################################
# SCENARIO COLLECTION UTILITIES
###############################################################################

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
    get_scenarios_by_pattern(collection::ScenarioCollection, pattern::Regex)

Find scenarios matching a name pattern.
"""
function get_scenarios_by_pattern(collection::ScenarioCollection, pattern::Regex)
    matching_scenarios = DataScenario[]
    for scenario in collection.scenarios
        if occursin(pattern, scenario.name)
            push!(matching_scenarios, scenario)
        end
    end
    return matching_scenarios
end

"""
    filter_scenarios(collection::ScenarioCollection, predicate::Function)

Filter scenarios based on a predicate function.
"""
function filter_scenarios(collection::ScenarioCollection, predicate::Function)
    return filter(predicate, collection.scenarios)
end

"""
    list_scenarios(collection::ScenarioCollection; detailed::Bool=false)

List all scenarios in collection with optional details.
"""
function list_scenarios(collection::ScenarioCollection; detailed::Bool=false)
    println("ScenarioCollection: $(collection.name)")
    println("Number of scenarios: $(length(collection.scenarios))")
    
    if haskey(collection.metadata, "total_combinations")
        println("Grid combinations: $(collection.metadata["total_combinations"])")
    end
    
    println()
    
    for (i, scenario) in enumerate(collection.scenarios)
        println("  $i. $(scenario.name)")
        if detailed && !isempty(scenario.overrides)
            for (var, val) in scenario.overrides
                println("     $var = $val")
            end
        elseif !detailed && !isempty(scenario.overrides)
            n_overrides = length(scenario.overrides)
            override_vars = collect(keys(scenario.overrides))
            if n_overrides <= 3
                println("     Overrides: $(join(override_vars, ", "))")
            else
                println("     Overrides: $(join(override_vars[1:2], ", ")), ... ($(n_overrides) total)")
            end
        end
    end
end

"""
    summarize_collection(collection::ScenarioCollection)

Provide statistical summary of collection.
"""
function summarize_collection(collection::ScenarioCollection)
    println("Collection Summary: $(collection.name)")
    println("=" ^ (20 + length(collection.name)))
    
    n_scenarios = length(collection.scenarios)
    println("Total scenarios: $n_scenarios")
    
    # Analyze override patterns
    all_override_vars = Set{Symbol}()
    override_counts = Dict{Symbol, Int}()
    
    for scenario in collection.scenarios
        for var in keys(scenario.overrides)
            push!(all_override_vars, var)
            override_counts[var] = get(override_counts, var, 0) + 1
        end
    end
    
    println("Variables with overrides: $(length(all_override_vars))")
    for var in sort(collect(all_override_vars))
        count = override_counts[var]
        percentage = round(100 * count / n_scenarios, digits=1)
        println("  $var: $count scenarios ($(percentage)%)")
    end
    
    # Memory analysis
    total_override_memory = 0
    for scenario in collection.scenarios
        for (_, col) in pairs(scenario.data)
            if col isa OverrideVector
                total_override_memory += sizeof(col)
            end
        end
    end
    
    println("Total override memory: $(total_override_memory) bytes")
    println("Average per scenario: $(round(total_override_memory / n_scenarios, digits=1)) bytes")
end

"""
    summarize_categorical_scenario(scenario::DataScenario)

Summarize categorical overrides in a scenario.
"""
function summarize_categorical_scenario(scenario::DataScenario)
    println("Scenario: $(scenario.name)")
    
    for (var, override_val) in scenario.overrides
        original_col = scenario.original_data[var]
        
        if original_col isa CategoricalArray
            println("  $var: categorical override = $override_val")
            println("    Available levels: $(levels(original_col))")
            
            # Count original distribution
            orig_counts = Dict()
            for level in levels(original_col)
                orig_counts[level] = count(==(level), string.(original_col))
            end
            println("    Original distribution: $orig_counts")
        else
            println("  $var: override = $override_val ($(typeof(override_val)))")
        end
    end
end

###############################################################################
# BATCH OPERATIONS
###############################################################################

"""
    modelrow_scenarios!(matrix, compiled, scenarios, row_idx)

Evaluate model row across multiple counterfactual scenarios with zero-allocation.

# Applications
- Batch counterfactual evaluation: Compare predictions across scenario set
- Policy comparison: Evaluate single observation under different policy scenarios  
- Sensitivity analysis: Assess prediction stability across parameter variations

# Arguments
- `matrix::AbstractMatrix{Float64}`: Pre-allocated output matrix (scenarios × formula_terms)
- `compiled`: Compiled formula from `compile_formula()`
- `scenarios::Vector{DataScenario}`: Collection of counterfactual scenarios
- `row_idx::Int`: Row index for evaluation across all scenarios

# Implementation
Evaluates the specified row under each scenario, storing results in corresponding 
matrix rows. Maintains zero-allocation properties by reusing pre-allocated matrix.

# Example  
```julia
scenarios = [
    create_scenario("low_dose", data; dose = 50.0),
    create_scenario("high_dose", data; dose = 150.0)
]

compiled = compile_formula(model, data)
results = Matrix{Float64}(undef, length(scenarios), length(compiled))
modelrow_scenarios!(results, compiled, scenarios, 1)  # Evaluate row 1

# Compare predictions
low_dose_pred = dot(coef(model), results[1, :])
high_dose_pred = dot(coef(model), results[2, :])
```

Related Functions:
- `create_scenario()`: Create individual scenarios for evaluation
- `create_scenario_grid()`: Systematic scenario generation
- `modelrow_collection()`: Convenience wrapper for ScenarioCollection
- `evaluate_scenarios_batch()`: Multi-row, multi-scenario evaluation
"""
function modelrow_scenarios!(
    matrix::AbstractMatrix{Float64},
    compiled,  # UnifiedCompiled or compatible callable
    scenarios::Vector{DataScenario},
    row_idx::Int
)
    @assert size(matrix, 1) >= length(scenarios) "Matrix height insufficient for scenarios"
    
    # Determine output width
    output_width = if hasmethod(length, (typeof(compiled),))
        length(compiled)
    else
        # Try to infer from first scenario evaluation
        test_output = try
            if hasmethod(compiled, (Vector{Float64}, typeof(scenarios[1].data), Int))
                temp_vec = Vector{Float64}(undef, 50)  # Guess
                compiled(temp_vec, scenarios[1].data, row_idx)
                temp_vec
            else
                compiled(scenarios[1].data, row_idx)
            end
        catch
            Float64[]
        end
        length(test_output)
    end
    
    @assert size(matrix, 2) >= output_width "Matrix width insufficient for formula output"
    
    # Evaluate each scenario
    for (i, scenario) in enumerate(scenarios)
        row_view = view(matrix, i, 1:output_width)
        try
            compiled(row_view, scenario.data, row_idx)
        catch MethodError
            # Try allocating version
            result = compiled(scenario.data, row_idx)
            row_view .= result
        end
    end
    
    return matrix
end

"""
    modelrow_scenarios!(matrix, compiled, collection::ScenarioCollection, row_idx)

Evaluate model row across all scenarios in a collection.
"""
function modelrow_scenarios!(
    matrix::AbstractMatrix{Float64},
    compiled,
    collection::ScenarioCollection,
    row_idx::Int
)
    return modelrow_scenarios!(matrix, compiled, collection.scenarios, row_idx)
end

"""
    modelrow_collection(compiled, collection::ScenarioCollection, row_idx) -> Matrix{Float64}

Evaluate model row across collection (allocating version).
"""
function modelrow_collection(compiled, collection::ScenarioCollection, row_idx::Int)
    n_scenarios = length(collection.scenarios)
    
    # Determine output width
    output_width = try
        if hasmethod(length, (typeof(compiled),))
            length(compiled)
        else
            test_result = compiled(collection.scenarios[1].data, row_idx)
            length(test_result)
        end
    catch
        error("Could not determine formula output width")
    end
    
    matrix = Matrix{Float64}(undef, n_scenarios, output_width)
    modelrow_scenarios!(matrix, compiled, collection, row_idx)
    return matrix
end

"""
    evaluate_scenarios_batch(compiled, scenarios::Vector{DataScenario}, row_indices::Vector{Int}) -> Array{Float64, 3}

Evaluate multiple scenarios across multiple rows.
Returns 3D array: [scenario_idx, row_idx, formula_output_idx]
"""
function evaluate_scenarios_batch(compiled, scenarios::Vector{DataScenario}, row_indices::Vector{Int})
    n_scenarios = length(scenarios)
    n_rows = length(row_indices)
    
    # Determine output width
    output_width = try
        if hasmethod(length, (typeof(compiled),))
            length(compiled)
        else
            test_result = compiled(scenarios[1].data, row_indices[1])
            length(test_result)
        end
    catch
        error("Could not determine formula output width")
    end
    
    # Pre-allocate 3D result array
    results = Array{Float64, 3}(undef, n_scenarios, n_rows, output_width)
    
    # Evaluate each combination
    for (s_idx, scenario) in enumerate(scenarios)
        for (r_idx, row_idx) in enumerate(row_indices)
            result_view = view(results, s_idx, r_idx, :)
            try
                compiled(result_view, scenario.data, row_idx)
            catch MethodError
                result = compiled(scenario.data, row_idx)
                result_view .= result
            end
        end
    end
    
    return results
end

###############################################################################
# SCENARIO ANALYSIS UTILITIES
###############################################################################

"""
    compare_scenarios(scenarios::Vector{DataScenario}, compiled, row_idx::Int; variable_names::Vector{String}=String[])

Compare formula outputs across scenarios.
"""
function compare_scenarios(scenarios::Vector{DataScenario}, compiled, row_idx::Int; variable_names::Vector{String}=String[])
    n_scenarios = length(scenarios)
    
    # Get outputs for all scenarios
    outputs = []
    for scenario in scenarios
        try
            output = compiled(scenario.data, row_idx)
            push!(outputs, output)
        catch e
            println("Error evaluating scenario '$(scenario.name)': $e")
            push!(outputs, nothing)
        end
    end
    
    # Display comparison
    println("Scenario Comparison (Row $row_idx)")
    println("=" ^ 40)
    
    for (i, scenario) in enumerate(scenarios)
        println("$(i). $(scenario.name)")
        
        # Show overrides
        if !isempty(scenario.overrides)
            override_strs = ["$k=$v" for (k, v) in scenario.overrides]
            println("   Overrides: $(join(override_strs, ", "))")
        end
        
        # Show output
        if outputs[i] !== nothing
            output_vals = outputs[i]
            if length(output_vals) <= 5
                println("   Output: $(round.(output_vals, digits=3))")
            else
                println("   Output: $(round.(output_vals[1:3], digits=3))... ($(length(output_vals)) values)")
            end
        else
            println("   Output: ERROR")
        end
        
        println()
    end
    
    return outputs
end

"""
    find_extreme_scenarios(collection::ScenarioCollection, compiled, row_idx::Int; output_idx::Int=1)

Find scenarios that produce extreme (min/max) values for a given output.
"""
function find_extreme_scenarios(collection::ScenarioCollection, compiled, row_idx::Int; output_idx::Int=1)
    scenario_outputs = Tuple{DataScenario, Float64}[]
    
    for scenario in collection.scenarios
        try
            output = compiled(scenario.data, row_idx)
            if length(output) >= output_idx
                push!(scenario_outputs, (scenario, output[output_idx]))
            end
        catch e
            println("Warning: Error evaluating scenario '$(scenario.name)': $e")
        end
    end
    
    if isempty(scenario_outputs)
        println("No scenarios evaluated successfully")
        return nothing
    end
    
    # Sort by output value
    sort!(scenario_outputs, by=x->x[2])
    
    min_scenario, min_value = scenario_outputs[1]
    max_scenario, max_value = scenario_outputs[end]
    
    println("Extreme Value Analysis (Output $output_idx, Row $row_idx)")
    println("=" ^ 55)
    println("Minimum: $(round(min_value, digits=4))")
    println("  Scenario: $(min_scenario.name)")
    println("  Overrides: $(min_scenario.overrides)")
    println()
    println("Maximum: $(round(max_value, digits=4))")
    println("  Scenario: $(max_scenario.name)")
    println("  Overrides: $(max_scenario.overrides)")
    println()
    println("Range: $(round(max_value - min_value, digits=4))")
    
    return (min_scenario=min_scenario, min_value=min_value, 
            max_scenario=max_scenario, max_value=max_value,
            all_outputs=scenario_outputs)
end

###############################################################################
# COMPATIBILITY FIXES (OVERWRITES)
###############################################################################

"""
    get_or_compile_specialized_formula(model, data)

Get cached specialized formula or compile new one.
FIXED: Cache key now includes data value information to handle scenarios correctly.
"""
function get_or_compile_specialized_formula(model, data)
    # FIXED: Create cache key that includes override information
    # Check if any columns are OverrideVectors and include their values
    override_info = Tuple{Symbol, Any}[]
    for (key, col) in pairs(data)
        if col isa OverrideVector
            push!(override_info, (key, col.override_value))
        end
    end
    
    # Include override info in cache key to prevent incorrect sharing
    cache_key = if isempty(override_info)
        (model, hash(keys(data)))
    else
        (model, hash(keys(data)), override_info)
    end
    
    if haskey(SPECIALIZED_MODEL_CACHE, cache_key)
        return SPECIALIZED_MODEL_CACHE[cache_key]
    else
        specialized = compile_formula(model, data)
        SPECIALIZED_MODEL_CACHE[cache_key] = specialized
        return specialized
    end
end

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

"""
    is_categorical_override(v::OverrideVector)

Check if an OverrideVector contains categorical values.
"""
is_categorical_override(v::OverrideVector) = eltype(v) <: CategoricalValue
is_categorical_override(v) = false

###############################################################################
# TESTING FUNCTIONS
###############################################################################

"""
    test_override_compatibility()

Test that overrides work correctly with the new compilation system.
"""
function test_override_compatibility()
    println("Testing Override Compatibility with New Schema System")
    println("=" ^ 60)
    
    # Test 1: Simple categorical override
    println("\nTest 1: Simple categorical override")
    df = DataFrame(
        x = randn(100), 
        group = categorical(rand(["A", "B", "C"], 100)),
        y = randn(100)
    )
    model = lm(@formula(y ~ x + group), df)
    data = Tables.columntable(df)
    
    scenario = create_scenario("group_B", data; group = "B")
    compiled = compile_formula(model, scenario.data)
    
    # Test execution
    output = Vector{Float64}(undef, length(compiled))
    compiled(output, scenario.data, 1)
    println("  ✅ Basic categorical override works")
    println("  Output: ", round.(output, digits=3))
    
    # Test 2: Mixed continuous-categorical interaction with override
    println("\nTest 2: Mixed interaction with overrides")
    model2 = lm(@formula(y ~ x * group), df)
    scenario2 = create_scenario("fixed_values", data; x = 2.0, group = "A")
    compiled2 = compile_formula(model2, scenario2.data)
    
    output2 = Vector{Float64}(undef, length(compiled2))
    compiled2(output2, scenario2.data, 1)
    println("  ✅ Mixed interaction with overrides works")
    println("  Output: ", round.(output2, digits=3))
    
    # Test 3: Cache key differentiation
    println("\nTest 3: Cache key differentiation")
    scenario3a = create_scenario("test3a", data; x = 1.0)
    scenario3b = create_scenario("test3b", data; x = 2.0)
    
    # These should compile to different cached formulas
    spec3a = get_or_compile_specialized_formula(model, scenario3a.data)
    spec3b = get_or_compile_specialized_formula(model, scenario3b.data)
    
    # Test that they produce different results
    out3a = Vector{Float64}(undef, length(spec3a))
    out3b = Vector{Float64}(undef, length(spec3b))
    spec3a(out3a, scenario3a.data, 1)
    spec3b(out3b, scenario3b.data, 1)
    
    if out3a ≈ out3b
        error("Different scenarios produced identical results - cache key issue!")
    end
    println("  ✅ Cache correctly differentiates scenarios")
    
    # Test 4: All override types
    println("\nTest 4: Various override value types")
    
    # String override (most common)
    s4a = create_scenario("string", data; group = "C")
    
    # Symbol override
    s4b = create_scenario("symbol", data; group = :B)
    
    # Integer index override
    s4c = create_scenario("index", data; group = 2)  # "B" is the 2nd level
    
    for (name, scenario) in [("string", s4a), ("symbol", s4b), ("index", s4c)]
        compiled = compile_formula(model, scenario.data)
        output = Vector{Float64}(undef, length(compiled))
        compiled(output, scenario.data, 1)
        println("  ✅ Override with $name works")
    end
    
    # Test 5: Scenario collections
    println("\nTest 5: Scenario collections")
    collection = create_scenario_grid("test_grid", data, Dict(
        :x => [1.0, 2.0],
        :group => ["A", "B"]
    ))
    
    println("  Created $(length(collection)) scenarios")
    for scenario in collection
        compiled = compile_formula(model, scenario.data)
        output = Vector{Float64}(undef, length(compiled))
        compiled(output, scenario.data, 1)
    end
    println("  ✅ All scenarios in collection work")
    
    # Test 6: Verify constant output for override
    println("\nTest 6: Verify constant output for categorical override")
    scenario6 = create_scenario("constant_B", data; group = "B")
    compiled6 = compile_formula(model, scenario6.data)
    
    # All rows should produce same group effect since override is constant
    outputs = []
    for i in 1:5
        output = Vector{Float64}(undef, length(compiled6))
        compiled6(output, scenario6.data, i)
        push!(outputs, output[3:4])  # Group effect columns
    end
    
    # Check all are identical
    all_same = all(out -> out ≈ outputs[1], outputs)
    if all_same
        println("  ✅ Override produces constant output across rows")
    else
        error("Override should produce constant output but doesn't!")
    end
    
    println(repeat("=", 60))
    println("All override compatibility tests passed! ✅")
    
    return true
end
