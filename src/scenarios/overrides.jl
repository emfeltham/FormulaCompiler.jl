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

###############################################################################
# CATEGORICAL OVERRIDE SUPPORT
###############################################################################

"""
    create_categorical_override(value::T, original_column::CategoricalArray{T}) where T

Generic method for creating categorical overrides with any element type.
Handles Bool, Int, Float64, String, and any other categorical element type.
"""
function create_categorical_override(value::T, original_column::CategoricalArray{T}) where T
    levels_list = levels(original_column)
    
    if value ∉ levels_list
        error("Override value '$value' (type: $(typeof(value))) not in categorical levels: $levels_list")
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
        error("Override value '$value' not in categorical levels: $levels_list")
    end
    
    # Create proper CategoricalValue using CategoricalArrays.jl API
    temp_cat = categorical([value], levels=levels_list, ordered=isordered(original_column))
    categorical_value = temp_cat[1]
    
    # println("DEBUG: Creating override for value='$value'")
    # println("DEBUG: Available levels: $levels_list") 
    # println("DEBUG: Created categorical_value: $categorical_value")
    # println("DEBUG: Level code: $(levelcode(categorical_value))")
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
        error("Override value '$value' not in categorical levels: $levels_list")
    end
    
    # Create proper CategoricalValue for Bool
    temp_cat = categorical([value], levels=levels_list, ordered=isordered(original_column))
    categorical_value = temp_cat[1]
    
    return OverrideVector(categorical_value, length(original_column))
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
- `original_data::NamedTuple`: Original unmodified data for reference
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
compiled = compile_formula(model, scenario.data)
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
        # Categorical handling (including CategoricalArray{Bool})
        return create_categorical_override(value, original_column)
    elseif original_column isa Vector{Bool} && value isa Bool
        # Non-categorical boolean: convert to Float64 (0.0 or 1.0)
        converted_value = Float64(value)
        return OverrideVector(converted_value, length(original_column))
    else
        # Other numeric or general types
        converted_value = convert(eltype(original_column), value)
        return OverrideVector(converted_value, length(original_column))
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

Container for multiple related scenarios with advanced operations.
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

Create a collection of scenarios for all combinations of variable values.
"""
function create_scenario_grid(collection_name::String,
                             data::NamedTuple,
                             variable_grids::Dict{Symbol, <:AbstractVector})
    
    if isempty(variable_grids)
        scenario = create_scenario("$(collection_name)_original", data)
        return ScenarioCollection(collection_name, [scenario], data)
    end
    
    variables = sort(collect(keys(variable_grids)))
    value_grids = [collect(variable_grids[var]) for var in variables]
    
    scenarios = DataScenario[]
    total_combinations = prod(length.(value_grids))
    
    println("Creating scenario grid: $(total_combinations) combinations")
    
    for (i, combination) in enumerate(Iterators.product(value_grids...))
        override_dict = Dict{Symbol, Any}()
        for (j, var) in enumerate(variables)
            override_dict[var] = combination[j]
        end
        
        scenario_name = generate_scenario_name(collection_name, variables, combination, i)
        scenario = create_scenario(scenario_name, data, override_dict)
        push!(scenarios, scenario)
        
        if i % max(1, total_combinations ÷ 10) == 0
            println("  Progress: $(i)/$(total_combinations) scenarios created")
        end
    end
    
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

Evaluate model row across multiple scenarios.
"""
function modelrow_scenarios!(
    matrix::AbstractMatrix{Float64},
    compiled,  # CompiledFormula or SpecializedFormula
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
