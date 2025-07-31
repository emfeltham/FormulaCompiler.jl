# Phase 2: Categorical Override Testing
# Comprehensive testing of categorical overrides with Step 2 optimizations

###############################################################################
# CATEGORICAL OVERRIDE EXTENSIONS
###############################################################################

"""
    create_categorical_override(value::Int, original_column::CategoricalArray)

Create categorical override using level index directly.
"""
function create_categorical_override(value::Int, original_column::CategoricalArray)
    n_levels = length(levels(original_column))
    if value < 1 || value > n_levels
        error("Level index $value out of range [1, $n_levels] for categorical levels: $(levels(original_column))")
    end
    
    # Get the actual categorical value for this level index
    level_value = levels(original_column)[value]
    temp_cat = categorical([level_value], levels=levels(original_column))
    categorical_value = temp_cat[1]
    return OverrideVector(categorical_value, length(original_column))
end

"""
    create_categorical_override(value::Symbol, original_column::CategoricalArray)

Create categorical override using Symbol level.
"""
function create_categorical_override(value::Symbol, original_column::CategoricalArray)
    return create_categorical_override(string(value), original_column)
end

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
    
    # Create proper CategoricalValue using the same approach as existing methods
    temp_cat = categorical([value], levels=levels_list)
    categorical_value = temp_cat[1]
    return OverrideVector(categorical_value, length(original_column))
end

###############################################################################
# ENHANCED SCENARIO MUTATION FOR CATEGORICAL
###############################################################################

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
# ADDITIONAL UTILITIES FOR CATEGORICAL WORK
###############################################################################

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
