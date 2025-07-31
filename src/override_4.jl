# Phase 4: Advanced Features
# Complete implementation of scenario collections, grids, and batch operations

###############################################################################
# SCENARIO COLLECTIONS - COMPLETE IMPLEMENTATION
###############################################################################

"""
    ScenarioCollection

Container for multiple related scenarios with advanced operations.
"""
struct ScenarioCollection
    name::String
    scenarios::Vector{DataScenario}
    original_data::NamedTuple
    metadata::Dict{String, Any}  # For extensibility
    
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
    
    variables = sort(collect(keys(variable_grids)))  # Sort for consistent ordering
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
        
        # Progress indicator for large grids
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

"""
    create_scenario_combinations(collection_name, data, base_scenarios::Vector{DataScenario}, combinations::Vector{Dict{Symbol, Any}})

Create scenarios by applying combinations to base scenarios.
"""
function create_scenario_combinations(collection_name::String,
                                    data::NamedTuple,
                                    base_scenarios::Vector{DataScenario},
                                    combinations::Vector{Dict{Symbol, Any}})
    scenarios = DataScenario[]
    
    for (i, base_scenario) in enumerate(base_scenarios)
        for (j, combo) in enumerate(combinations)
            # Merge base scenario overrides with combination
            merged_overrides = copy(base_scenario.overrides)
            for (var, val) in combo
                merged_overrides[var] = val
            end
            
            scenario_name = "$(collection_name)_base$(i)_combo$(j)"
            scenario = create_scenario(scenario_name, data, merged_overrides)
            push!(scenarios, scenario)
        end
    end
    
    metadata = Dict{String, Any}(
        "base_scenarios" => length(base_scenarios),
        "combinations" => length(combinations),
        "total_scenarios" => length(scenarios),
        "created_at" => now()
    )
    
    return ScenarioCollection(collection_name, scenarios, data, metadata)
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
# EXPORT AND IMPORT UTILITIES
###############################################################################

"""
    export_scenarios_csv(collection::ScenarioCollection, filename::String)

Export scenario definitions to CSV file.
"""
function export_scenarios_csv(collection::ScenarioCollection, filename::String)
    # Collect all override variables
    all_variables = Set{Symbol}()
    for scenario in collection.scenarios
        union!(all_variables, keys(scenario.overrides))
    end
    all_variables = sort(collect(all_variables))
    
    # Create CSV content
    lines = String[]
    
    # Header
    header = ["scenario_name"; string.(all_variables)]
    push!(lines, join(header, ","))
    
    # Data rows
    for scenario in collection.scenarios
        row = [scenario.name]
        for var in all_variables
            if haskey(scenario.overrides, var)
                push!(row, string(scenario.overrides[var]))
            else
                push!(row, "")
            end
        end
        push!(lines, join(row, ","))
    end
    
    # Write file
    open(filename, "w") do f
        for line in lines
            println(f, line)
        end
    end
    
    println("Exported $(length(collection.scenarios)) scenarios to $filename")
end

"""
    summarize_scenario_impacts(collection::ScenarioCollection, compiled, row_indices::Vector{Int}; baseline_scenario_name::String="")

Analyze the impact of different scenarios relative to a baseline.
"""
function summarize_scenario_impacts(collection::ScenarioCollection, compiled, row_indices::Vector{Int}; baseline_scenario_name::String="")
    # Find baseline scenario
    baseline_scenario = if !isempty(baseline_scenario_name)
        try
            get_scenario_by_name(collection, baseline_scenario_name)
        catch
            println("Warning: Baseline scenario '$baseline_scenario_name' not found, using first scenario")
            collection.scenarios[1]
        end
    else
        collection.scenarios[1]
    end
    
    println("Impact Analysis")
    println("Baseline: $(baseline_scenario.name)")
    println("Row indices: $(length(row_indices)) rows")
    println("=" ^ 50)
    
    # Calculate baseline values
    baseline_outputs = []
    for row_idx in row_indices
        try
            output = compiled(baseline_scenario.data, row_idx)
            push!(baseline_outputs, output)
        catch e
            println("Error with baseline at row $row_idx: $e")
            push!(baseline_outputs, nothing)
        end
    end
    
    # Compare other scenarios to baseline
    for scenario in collection.scenarios
        if scenario === baseline_scenario
            continue
        end
        
        differences = []
        for (i, row_idx) in enumerate(row_indices)
            if baseline_outputs[i] === nothing
                continue
            end
            
            try
                output = compiled(scenario.data, row_idx)
                diff = output .- baseline_outputs[i]
                push!(differences, diff)
            catch e
                println("Error with scenario '$(scenario.name)' at row $row_idx: $e")
            end
        end
        
        if !isempty(differences)
            # Calculate summary statistics
            all_diffs = vcat(differences...)
            mean_diff = mean(all_diffs)
            std_diff = std(all_diffs)
            min_diff = minimum(all_diffs)
            max_diff = maximum(all_diffs)
            
            println("$(scenario.name):")
            println("  Mean difference: $(round(mean_diff, digits=4))")
            println("  Std difference: $(round(std_diff, digits=4))")
            println("  Range: [$(round(min_diff, digits=4)), $(round(max_diff, digits=4))]")
            println("  Overrides: $(scenario.overrides)")
            println()
        end
    end
end
