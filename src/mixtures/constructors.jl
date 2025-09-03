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

# Convenient display for CategoricalMixture
function Base.show(io::IO, m::CategoricalMixture)
    pairs_str = join(["$(repr(l)) => $(w)" for (l, w) in zip(m.levels, m.weights)], ", ")
    print(io, "mix(", pairs_str, ")")
end