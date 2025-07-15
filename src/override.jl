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
"""
mutable struct DataScenario
    name::String
    overrides::Dict{Symbol,Any}
    data::NamedTuple
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
    if isempty(overrides)
        # No overrides - just reference original data
        override_dict = Dict{Symbol,Any}()
        return DataScenario(name, override_dict, original_data)
    end
    
    # Create Dict from keyword arguments
    override_dict = Dict{Symbol,Any}(overrides)
    
    # Create modified data with overrides
    modified_data = create_override_data(original_data, override_dict)
    
    return DataScenario(name, override_dict, modified_data)
end

"""
    create_scenario(name, original_data, overrides::Dict{Symbol})

Create a data scenario with Dict overrides.
"""
function create_scenario(name::String, original_data::NamedTuple, overrides::Dict{Symbol,<:Any})
    if isempty(overrides)
        return DataScenario(name, Dict{Symbol,Any}(), original_data)
    end
    
    # Ensure consistent type
    override_dict = Dict{Symbol,Any}(overrides)
    modified_data = create_override_data(original_data, override_dict)
    
    return DataScenario(name, override_dict, modified_data)
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
    # Update the overrides dict
    scenario.overrides[variable] = value
    
    # Rebuild the data with new overrides
    original_data = extract_original_data(scenario)
    scenario.data = create_override_data(original_data, scenario.overrides)
    
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
        # Remove from overrides
        delete!(scenario.overrides, variable)
        
        # Rebuild the data  
        original_data = extract_original_data(scenario)
        scenario.data = create_override_data(original_data, scenario.overrides)
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
        # Update all overrides
        for (key, value) in overrides
            scenario.overrides[key] = value
        end
        
        # Rebuild data once
        original_data = extract_original_data(scenario)
        scenario.data = create_override_data(original_data, scenario.overrides)
    end
    
    return scenario
end

"""
    extract_original_data(scenario::DataScenario)

Extract the original data from a scenario by finding non-OverrideVector columns.
This is needed for rebuilding scenarios after override changes.
"""
function extract_original_data(scenario::DataScenario)
    original_dict = Dict{Symbol, Any}()
    
    for (key, column) in pairs(scenario.data)
        if column isa OverrideVector
            # This is an override - we need to find the original
            # For now, we'll skip it and let it be rebuilt
            continue
        else
            # This is original data
            original_dict[key] = column
        end
    end
    
    return NamedTuple(original_dict)
end

# Alternative: Store original data reference in DataScenario
"""
    DataScenarioWithOriginal

Enhanced version that keeps reference to original data for efficient rebuilding.
"""
mutable struct DataScenarioWithOriginal
    name::String
    overrides::Dict{Symbol,Any}
    data::NamedTuple
    original_data::NamedTuple  # Keep reference for rebuilding
end

function create_scenario_with_original(name::String, original_data::NamedTuple; overrides...)
    override_dict = Dict{Symbol,Any}(overrides)
    
    if isempty(override_dict)
        modified_data = original_data
    else
        modified_data = create_override_data(original_data, override_dict)
    end
    
    return DataScenarioWithOriginal(name, override_dict, modified_data, original_data)
end

function set_override!(scenario::DataScenarioWithOriginal, variable::Symbol, value)
    scenario.overrides[variable] = value
    scenario.data = create_override_data(scenario.original_data, scenario.overrides)
    return scenario
end

function remove_override!(scenario::DataScenarioWithOriginal, variable::Symbol)
    delete!(scenario.overrides, variable)
    scenario.data = create_override_data(scenario.original_data, scenario.overrides)
    return scenario
end

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

###############################################################################
# TESTING AND EXAMPLES
###############################################################################

"""
    test_scenario_foundation()

Test the scenario foundation for correctness.
"""
function test_scenario_foundation()
    
    println("Testing Dict-based scenario foundation...")
    
    # Create test data
    Random.seed!(42)
    df = DataFrame(
        x = [1.0, 2.0, 3.0, 4.0],
        y = [1.0, 4.0, 9.0, 16.0],
        group = categorical(["A", "B", "A", "B"])
    )
    
    # Setup
    model = lm(@formula(y ~ x * group), df)
    compiled = compile_formula(model)
    data = Tables.columntable(df)
    
    @testset "Dict-Based Scenario Foundation" begin
        
        @testset "OverrideVector" begin
            # Basic functionality
            override_vec = OverrideVector(5.0, 10)
            @test length(override_vec) == 10
            @test override_vec[1] == 5.0
            @test override_vec[5] == 5.0
            @test override_vec[10] == 5.0
            
            # Iteration
            @test collect(override_vec) == fill(5.0, 10)
        end
        
        @testset "Categorical Override" begin
            cat_override = create_categorical_override("B", df.group)
            @test length(cat_override) == length(df.group)
            @test cat_override[1] isa CategoricalValue
            @test string(cat_override[1]) == "B"
        end
        
        @testset "Single Scenario" begin
            # Create scenario with override
            scenario = create_scenario("test", data; x = 5.0)
            
            @test scenario.name == "test"
            @test scenario.overrides[:x] == 5.0
            @test isa(scenario.overrides, Dict)
            
            # Test that override works
            @test scenario.data.x[1] == 5.0
            @test scenario.data.x[4] == 5.0
            
            # Test that other columns are unchanged (references)
            @test scenario.data.y === data.y  # Same object reference
            @test scenario.data.group === data.group
        end
        
        @testset "Scenario Mutation" begin
            scenario = create_scenario("mutable_test", data; x = 1.0)
            
            # Test adding override
            set_override!(scenario, :y, 10.0)
            @test scenario.overrides[:y] == 10.0
            @test scenario.data.y[1] == 10.0
            
            # Test updating override
            set_override!(scenario, :x, 99.0)
            @test scenario.overrides[:x] == 99.0
            @test scenario.data.x[1] == 99.0
            
            # Test removing override
            remove_override!(scenario, :y)
            @test !haskey(scenario.overrides, :y)
            @test scenario.data.y === data.y  # Back to original
            
            # Test bulk update
            update_scenario!(scenario; x = 2.0, group = "A")
            @test scenario.overrides[:x] == 2.0
            @test scenario.overrides[:group] == "A"
        end
        
        @testset "Scenario Grid" begin
            # Create grid of scenarios
            x_values = [0.0, 1.0, 2.0]
            collection = create_scenario_grid("x_test", data, :x, x_values)
            
            @test length(collection) == 3
            @test collection[1].name == "x_test_1"
            @test collection[2].name == "x_test_2"
            @test collection[3].name == "x_test_3"
            
            # Test values
            @test collection[1].overrides[:x] == 0.0
            @test collection[2].overrides[:x] == 1.0
            @test collection[3].overrides[:x] == 2.0
        end
        
        @testset "Integration with modelrow!" begin
            # Test basic evaluation
            scenario = create_scenario("eval_test", data; x = 10.0)
            row_vec = Vector{Float64}(undef, length(compiled))
            
            modelrow!(row_vec, compiled, scenario, 1)
            @test all(isfinite.(row_vec))
            
            # Test multi-scenario evaluation
            scenarios = [
                create_scenario("s1", data; x = 1.0),
                create_scenario("s2", data; x = 2.0)
            ]
            matrix = Matrix{Float64}(undef, 2, length(compiled))
            
            modelrow_scenarios!(matrix, compiled, scenarios, 1)
            @test all(isfinite.(matrix))
            @test matrix[1, :] != matrix[2, :]  # Should be different
        end
    end
    
    println("✅ All Dict-based scenario foundation tests passed!")
    return true
end

"""
    example_scenario_usage()

Demonstrate iterative scenario development with Dict-based overrides and unified grid interface.
"""
function example_scenario_usage()
    
    println("=== Example Dict-Based Scenario Usage ===")
    
    # Setup data and model
    Random.seed!(123)
    df = DataFrame(
        x = randn(100),
        z = abs.(randn(100)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 100)),
        treatment = rand([false, true], 100)
    )
    df.y = 1 .+ 0.5 .* df.x .+ 0.3 .* log.(df.z) .+ 0.2 .* df.treatment .+ randn(100)
    
    model = lm(@formula(y ~ x + log(z) + group + treatment + x*group), df)
    compiled = compile_formula(model)
    data = Tables.columntable(df)
    
    println("Model fitted with $(length(compiled)) coefficients")
    
    # 1. Start with simple scenario
    println("\n1. Creating initial scenario...")
    scenario = create_scenario("policy_development", data; treatment = true)
    list_scenario_overrides(scenario)
    
    # 2. Iteratively add variables
    println("\n2. Adding variables iteratively...")
    set_override!(scenario, :x, mean(data.x))
    println("After adding x override:")
    list_scenario_overrides(scenario)
    
    set_override!(scenario, :group, "A")  
    println("After adding group override:")
    list_scenario_overrides(scenario)
    
    # 3. Modify existing overrides
    println("\n3. Modifying existing overrides...")
    set_override!(scenario, :x, mean(data.x) + std(data.x))
    println("After updating x to mean + std:")
    list_scenario_overrides(scenario)
    
    # 4. Bulk updates
    println("\n4. Bulk scenario updates...")
    update_scenario!(scenario; 
        x = 2.0, 
        treatment = false,
        z = median(data.z)
    )
    println("After bulk update:")
    list_scenario_overrides(scenario)
    
    # 5. Remove some overrides  
    println("\n5. Removing overrides...")
    remove_override!(scenario, :z)
    remove_override!(scenario, :group)
    println("After removing z and group:")
    list_scenario_overrides(scenario)
    
    # 6. Test evaluation at each stage
    println("\n6. Testing final scenario evaluation...")
    row_vec = Vector{Float64}(undef, length(compiled))
    compiled(row_vec, scenario.data, 1)
    prediction = sum(row_vec)  # Simple sum as example
    println("Final prediction: $(round(prediction, digits=3))")
    
    # 7. Demonstrate unified grid interface
    println("\n7. Unified grid interface examples...")
    
    # Single variable grid
    println("Single variable grid:")
    x_collection = create_scenario_grid("x_analysis", data, Dict(:x => [0.0, 1.0, 2.0]))
    list_scenarios(x_collection)
    
    # Multiple variable combinations
    println("\nMultiple variable combinations:")
    combo_collection = create_scenario_grid("policy_grid", data, Dict(
        :treatment => [true, false],
        :group => ["A", "B"]
    ))
    list_scenarios(combo_collection)
    
    # Complex grid with three variables
    println("\nComplex three-variable grid:")
    complex_collection = create_scenario_grid("comprehensive", data, Dict(
        :x => [mean(data.x), mean(data.x) + std(data.x)],
        :treatment => [true, false],
        :group => ["A", "C"]
    ))
    println("Created $(length(complex_collection)) scenarios")
    for (i, scenario) in enumerate(complex_collection.scenarios[1:min(4, end)])
        println("  $i. $(scenario.name): $(scenario.overrides)")
    end
    if length(complex_collection) > 4
        println("  ... ($(length(complex_collection) - 4) more scenarios)")
    end
    
    return (scenario, x_collection, combo_collection, complex_collection)
end