# constructors.jl - Mixture constructors for FormulaCompiler.jl
# Adapted from Margins.jl src/features/categorical_mixtures.jl

"""
    mix(pairs...)

Convenient constructor for CategoricalMixture from level => weight pairs.
This is the main user-facing function for creating mixture specifications.

# Arguments
- `pairs...`: Level => weight pairs (e.g., "A" => 0.3, "B" => 0.7)

# Returns
- `CategoricalMixture`: Validated mixture object ready for use with FormulaCompiler

# Examples
```julia
# Basic categorical mixture
group_mix = mix("Control" => 0.4, "Treatment" => 0.6)

# Educational composition
education_mix = mix("high_school" => 0.4, "college" => 0.4, "graduate" => 0.2)

# Regional distribution using symbols
region_mix = mix(:urban => 0.7, :rural => 0.3)

# Boolean mixture (30% false, 70% true)
treated_mix = mix(false => 0.3, true => 0.7)

# Works with any comparable type
age_group_mix = mix("young" => 0.25, "middle" => 0.50, "old" => 0.25)
```

# Validation
The mix() function automatically validates:
- At least one level => weight pair is provided
- All weights are non-negative
- Weights sum to 1.0 (within numerical tolerance)
- All levels are unique

# Integration with FormulaCompiler
```julia
# Use in scenarios for counterfactual analysis
scenario = create_scenario("mixed_population", data; 
    group = mix("A" => 0.3, "B" => 0.7)
)

# Use in reference grids for marginal effects
df = DataFrame(
    x = [1.0, 2.0, 3.0],
    group = [mix("A" => 0.4, "B" => 0.6), 
             mix("A" => 0.4, "B" => 0.6),
             mix("A" => 0.4, "B" => 0.6)]
)
```

# Performance
Mixture creation is lightweight and validation happens at construction time.
The resulting CategoricalMixture objects are compiled into zero-allocation
evaluators by FormulaCompiler's compilation system.
"""
function mix(pairs...)
    isempty(pairs) && throw(ArgumentError("mix() requires at least one level => weight pair"))
    
    levels = [k for (k, v) in pairs]
    weights = [Float64(v) for (k, v) in pairs]  # Ensure Float64 weights
    return CategoricalMixture(levels, weights)
end

"""
    mix_proportional(col) -> CategoricalMixture or Float64

Create a mixture that reflects the observed proportions in the data column.
This provides data-driven mixture specifications for population-representative analysis.

For Bool columns, returns the observed probability of true.
For categorical columns, computes observed frequency proportions.

# Arguments
- `col`: Data column (Vector of any categorical type)

# Returns
- `CategoricalMixture`: Mixture with levels and observed frequency weights for categorical data
- `Float64`: For Bool columns, returns observed probability of true

# Examples
```julia
# Boolean column -> observed probability of true
mix_proportional([true, false, true, true]) # -> 0.75

# Categorical -> Observed frequency proportions
mix_proportional(["A", "B", "A", "A"]) # -> mix("A" => 0.75, "B" => 0.25)

# CategoricalArray -> Uses observed proportions
cat_col = categorical(["Red", "Blue", "Red", "Green"])
mix_proportional(cat_col) # -> mix("Blue" => 0.25, "Green" => 0.25, "Red" => 0.5)
```

This function is ideal for creating reference grids that preserve the observed population
composition, providing more representative counterfactual scenarios.
"""
function mix_proportional(col)
    if eltype(col) <: Bool
        return mean(col)  # Observed probability of true
    end
    
    # Get observed frequency proportions
    level_counts = Dict()
    n_total = length(col)
    
    for value in col
        str_value = string(value)
        level_counts[str_value] = get(level_counts, str_value, 0) + 1
    end
    
    # Convert to proportions
    levels = sort(collect(keys(level_counts)))  # Sort for consistency
    weights = [level_counts[level] / n_total for level in levels]
    
    return CategoricalMixture(levels, weights)
end

# Convenient display for CategoricalMixture
function Base.show(io::IO, m::CategoricalMixture)
    pairs_str = join(["$(repr(l)) => $(w)" for (l, w) in zip(m.levels, m.weights)], ", ")
    print(io, "mix(", pairs_str, ")")
end