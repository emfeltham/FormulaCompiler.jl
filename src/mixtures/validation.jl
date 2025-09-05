# validation.jl - Mixture validation functions for FormulaCompiler.jl
# Adapted from Margins.jl src/features/categorical_mixtures.jl

"""
    validate_mixture_against_data(mixture::CategoricalMixture, col, var::Symbol)

Validate that all levels in the mixture exist in the actual data column.
Throws ArgumentError if any mixture levels are not found in the data.

# Arguments
- `mixture::CategoricalMixture`: The mixture specification to validate
- `col`: The data column to validate against
- `var::Symbol`: Variable name for error reporting

# Throws
- `ArgumentError`: If mixture contains levels not found in the data

# Examples
```julia
# Validate mixture against categorical data
data_col = categorical(["A", "B", "C", "A", "B"])
mixture = mix("A" => 0.5, "B" => 0.5)
validate_mixture_against_data(mixture, data_col, :group)  # ✓ Valid

# This would throw an error
bad_mixture = mix("A" => 0.5, "X" => 0.5)  # "X" not in data
validate_mixture_against_data(bad_mixture, data_col, :group)  # ✗ Error
```

This function is used internally by FormulaCompiler's scenario system to ensure
mixture specifications are compatible with the actual data.
"""
function validate_mixture_against_data(mixture::CategoricalMixture, col, var::Symbol)
    # Get actual levels from data
    actual_levels = if isdefined(Main, :CategoricalArrays) && isdefined(CategoricalArrays, :CategoricalArray) && (col isa CategoricalArrays.CategoricalArray)
        string.(CategoricalArrays.levels(col))
    elseif eltype(col) <: Bool
        string.([false, true])
    else
        unique(string.(col))
    end
    
    # Check that all mixture levels exist in data
    mixture_levels_str = string.(mixture.levels)
    missing_levels = setdiff(mixture_levels_str, actual_levels)
    if !isempty(missing_levels)
        throw(ArgumentError("Variable :$var mixture contains levels not found in data: $missing_levels. Available levels: $actual_levels"))
    end
    
    return true
end

"""
    create_balanced_mixture(col) -> CategoricalMixture or Float64

Create a balanced (equal weight) categorical mixture from data column levels.
This provides orthogonal factorial designs for balanced analysis.

For Bool columns, returns 0.5 (50-50 balanced probability).
For categorical columns, assigns equal probability to all levels.

# Arguments
- `col`: Data column (Vector of any categorical type)

# Returns
- `CategoricalMixture`: Mixture with levels and equal weights for categorical data
- `Float64`: For Bool columns, returns 0.5 (balanced probability)

# Examples
```julia
# Boolean column -> 50-50 probability
create_balanced_mixture([true, false, true, false]) # -> 0.5

# Categorical -> Equal weights for all levels
create_balanced_mixture(["A", "B", "C", "A"]) # -> mix("A" => 0.333, "B" => 0.333, "C" => 0.333)

# CategoricalArray -> Uses defined levels, not just observed values
cat_col = categorical(["A", "B"], levels=["A", "B", "C"])
create_balanced_mixture(cat_col) # -> mix("A" => 0.333, "B" => 0.333, "C" => 0.333)
```

This function is useful for creating orthogonal reference grids where you want
to give equal weight to all possible levels of a categorical variable.
"""
function create_balanced_mixture(col)
    if eltype(col) <: Bool
        return 0.5  # Balanced probability for Bool variables
    end
    
    # Get unique levels
    if isdefined(Main, :CategoricalArrays) && isdefined(CategoricalArrays, :CategoricalArray) && (col isa CategoricalArrays.CategoricalArray)
        levels = string.(CategoricalArrays.levels(col))
    else
        levels = sort(unique(string.(col)))
    end
    
    # Create equal weights
    n_levels = length(levels)
    equal_weights = fill(1.0 / n_levels, n_levels)
    
    return CategoricalMixture(levels, equal_weights)
end

"""
    mixture_to_scenario_value(mixture::CategoricalMixture, original_col)

Convert a categorical mixture to a representative value for FormulaCompiler scenario creation.
Uses weighted average encoding to provide a smooth, continuous representation.

# Strategy
- **CategoricalArray**: Weighted average of level indices
- **Bool**: Probability of true (equivalent to current fractional Bool support)  
- **Other**: Weighted average of sorted unique level indices

# Arguments
- `mixture::CategoricalMixture`: The mixture to convert
- `original_col`: The original data column for context

# Returns
- `Float64`: Continuous representation of the mixture

# Examples
```julia
# Boolean mixture -> probability of true
bool_mix = mix(false => 0.3, true => 0.7)
mixture_to_scenario_value(bool_mix, [true, false, true]) # -> 0.7

# Categorical mixture -> weighted average of level indices
cat_mix = mix("A" => 0.6, "B" => 0.4)  
cat_col = categorical(["A", "B", "C"])
mixture_to_scenario_value(cat_mix, cat_col) # -> 1.4 (0.6*1 + 0.4*2)
```

This function is used internally by FormulaCompiler's scenario system to convert
mixture specifications into values that can be used with the existing override system.
"""
function mixture_to_scenario_value(mixture::CategoricalMixture, original_col)
    if isdefined(Main, :CategoricalArrays) && isdefined(CategoricalArrays, :CategoricalArray) && (original_col isa CategoricalArrays.CategoricalArray)
        # Get level mapping
        actual_levels = string.(CategoricalArrays.levels(original_col))
        level_indices = Dict(level => i for (i, level) in enumerate(actual_levels))
        
        # Compute weighted average of indices
        mixture_levels_str = string.(mixture.levels)
        weighted_sum = sum(mixture.weights[i] * level_indices[mixture_levels_str[i]] 
                          for i in 1:length(mixture.levels))
        
        return weighted_sum
    elseif eltype(original_col) <: Bool
        # Handle Bool as special case - return probability of true
        level_weight_dict = Dict(string.(mixture.levels) .=> mixture.weights)
        false_weight = get(level_weight_dict, "false", 0.0)
        true_weight = get(level_weight_dict, "true", 0.0)
        
        # Validate Bool levels
        if !issubset(keys(level_weight_dict), ["false", "true"])
            throw(ArgumentError("Bool variable mixture must use levels 'false' and 'true' or false and true"))
        end
        
        return true_weight  # Probability of true (matches existing fractional Bool support)
    else
        # Generic categorical - weighted average of sorted unique values
        unique_levels = sort(unique(string.(original_col)))
        level_indices = Dict(level => i for (i, level) in enumerate(unique_levels))
        
        mixture_levels_str = string.(mixture.levels)
        weighted_sum = sum(mixture.weights[i] * level_indices[mixture_levels_str[i]] 
                          for i in 1:length(mixture.levels))
        
        return weighted_sum
    end
end

# Method for MixtureWithLevels wrapper
function mixture_to_scenario_value(mixture_with_levels::MixtureWithLevels, original_col)
    return mixture_to_scenario_value(mixture_with_levels.mixture, original_col)
end