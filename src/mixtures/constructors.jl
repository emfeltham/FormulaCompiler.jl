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

## CounterfactualVector Pattern for Categorical Mixtures

The unified row-wise architecture provides efficient single-row mixture perturbations:

```julia
using FormulaCompiler, DataFrames, Tables

# Prepare data with mixture column
df = DataFrame(
    y = randn(1000),
    x = randn(1000),
    group = fill(mix("A" => 0.4, "B" => 0.6), 1000)  # Baseline mixture
)
data = Tables.columntable(df)

# Compile formula
model = lm(@formula(y ~ x * group), df)
compiled = compile_formula(model, data)

# Pattern 1: Single-row mixture perturbation
# Create counterfactual vector for mixture column
cf_mixture = counterfactualvector(data.group, 1)  # CategoricalMixtureCounterfactualVector

# Apply different mixture to specific row
new_mixture = mix("A" => 0.8, "B" => 0.2)  # Policy counterfactual
update_counterfactual_row!(cf_mixture, 500)  # Target row 500
update_counterfactual_replacement!(cf_mixture, new_mixture)

# Evaluate with counterfactual data
data_cf = (data..., group=cf_mixture)
output = Vector{Float64}(undef, length(compiled))
compiled(output, data_cf, 500)  # Row 500 uses new mixture, others use baseline

# Pattern 2: Population marginal effects with mixture profiles
function mixture_marginal_effects(model, data, base_mixture, alt_mixture)
    compiled = compile_formula(model, data)
    cf_mixture = counterfactualvector(data.group, 1)
    data_cf = (data..., group=cf_mixture)

    n_rows = length(data.x)
    baseline_effects = Vector{Float64}(undef, n_rows)
    alternative_effects = Vector{Float64}(undef, n_rows)

    for row in 1:n_rows
        update_counterfactual_row!(cf_mixture, row)

        # Baseline mixture
        update_counterfactual_replacement!(cf_mixture, base_mixture)
        compiled(view(baseline_effects, row:row), data_cf, row)

        # Alternative mixture
        update_counterfactual_replacement!(cf_mixture, alt_mixture)
        compiled(view(alternative_effects, row:row), data_cf, row)
    end

    return mean(alternative_effects - baseline_effects)
end

# Example: Policy effect of changing group composition
base_mix = mix("A" => 0.4, "B" => 0.6)
policy_mix = mix("A" => 0.7, "B" => 0.3)
effect = mixture_marginal_effects(model, data, base_mix, policy_mix)
```

## Reference Grid Pattern

For systematic marginal effects computation across different mixture profiles:

```julia
# Create reference grid with multiple mixture specifications
mixtures = [
    mix("A" => 1.0, "B" => 0.0),    # Pure A
    mix("A" => 0.5, "B" => 0.5),    # Balanced
    mix("A" => 0.0, "B" => 1.0)     # Pure B
]

# Evaluate effects across all mixture profiles
effects_by_mixture = Vector{Float64}(undef, length(mixtures))
cf_mixture = counterfactualvector(data.group, 1)
data_cf = (data..., group=cf_mixture)

for (i, mixture_spec) in enumerate(mixtures)
    update_counterfactual_replacement!(cf_mixture, mixture_spec)

    # Compute average effect across all rows for this mixture
    row_effects = Vector{Float64}(undef, n_rows)
    for row in 1:n_rows
        update_counterfactual_row!(cf_mixture, row)
        compiled(view(row_effects, row:row), data_cf, row)
    end
    effects_by_mixture[i] = mean(row_effects)
end
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