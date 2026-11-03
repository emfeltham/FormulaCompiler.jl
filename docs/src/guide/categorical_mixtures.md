# Categorical Mixtures in FormulaCompiler.jl

## Overview

FormulaCompiler.jl supports **categorical mixtures** - weighted combinations of categorical levels for profile-based marginal effects computation. Fractional values like `mix("A" => 0.3, "B" => 0.7)` are evaluated by zero-allocation contrast operations that read each row's mixture from the data. For boolean variables, use simple numeric probabilities (e.g., `treated = 0.7` for 70% treatment rate).

**Implementation characteristics:**
- Allocation behavior: 0 bytes allocated during execution (verified in test suite)
- No mixture in the compiled object: levels and weights are read from the data at evaluation time, so one compiled formula serves any mixture
- Statistical integration: Compatible with marginal effects packages (Margins.jl)
- Memory complexity: O(1) memory usage independent of data size

## Quick Start

```julia
using FormulaCompiler, GLM, DataFrames, Tables, CategoricalArrays

# Fit a model on ordinary data
training_data = DataFrame(
    y = randn(200),
    x = randn(200),
    group = categorical(rand(["A", "B"], 200))
)
model = lm(@formula(y ~ x * group), training_data)

# Create data with categorical mixtures
df = DataFrame(
    x = [1.0, 2.0, 3.0],
    group = [mix("A" => 0.3, "B" => 0.7),   # 30% A, 70% B
             mix("A" => 0.3, "B" => 0.7),
             mix("A" => 0.3, "B" => 0.7)]
)

# Compile against the mixture data
data = Tables.columntable(df)
compiled = compile_formula(model, data)

# Zero-allocation evaluation
output = Vector{Float64}(undef, length(compiled))
compiled(output, data, 1)  # [1.0, 1.0, 0.7, 0.7]: group:B is 0.7, x & group:B is 1.0 * 0.7
```

## Mixture Object Interface

`mix` returns a `CategoricalMixture`. Columns of other objects are also detected by **duck typing**: any type with `levels` and `weights` fields is treated as a mixture:

```julia
# Example mixture object structure
struct MixtureExample
    levels::Vector{String}    # ["A", "B"]
    weights::Vector{Float64}  # [0.4, 0.6]
end

# FormulaCompiler detects and evaluates such objects like mix("A" => 0.4, "B" => 0.6)
mixture = MixtureExample(["A", "B"], [0.4, 0.6])
duck_data = (x = [1.0, 2.0], group = [mixture, mixture])
compiled_duck = compile_formula(model, duck_data)
```

A duck-typed object is not validated: its weights are used as given.

## Creating Mixture Data

### Boolean Variables and Population Analysis

**Boolean variables** work seamlessly with FormulaCompiler's continuous interpretation. For population-level analysis and marginal effects, simply use numeric probabilities directly:

```julia
# A model with a Bool predictor
bool_training = DataFrame(y = randn(200), x = randn(200), treated = rand(Bool, 200))
bool_model = lm(@formula(y ~ x + treated), bool_training)

# Population analysis with boolean probabilities
df_prob = DataFrame(
    x = [1.0, 2.0, 3.0],
    treated = fill(0.7, 3)  # 70% treatment probability for population analysis
)

compiled_prob = compile_formula(bool_model, Tables.columntable(df_prob))
output_prob = Vector{Float64}(undef, length(compiled_prob))
compiled_prob(output_prob, Tables.columntable(df_prob), 1)  # [1.0, 1.0, 0.7]: treated column = 0.7
```

**Numeric approach properties:**
- Implementation: Uses standard Float64 values without mixture object construction
- Syntax: `treated = 0.7` represents probability directly
- Performance: Maintains zero-allocation execution
- Compatibility: Works with data modification and counterfactual functions
- Consistency: Matches StatsModels.jl boolean variable semantics

**Application patterns:**
- **Individual scenarios**: `treated = true` or `treated = false`
- **Population analysis**: `treated = 0.6` (60% treatment rate)
- **Marginal effects**: Varying treatment probabilities across reference grids

### Helper Functions

Mixture columns are ordinary vectors of mixtures:

```julia
# Mixture column for a reference grid: one mixture repeated
mixture_spec = mix("A" => 0.3, "B" => 0.7)
column = fill(mixture_spec, 1000)  # 1000 identical rows

# Balanced (equal weight) mixture over a column's levels (unexported)
balanced_mixture = FormulaCompiler.create_balanced_mixture(["A", "B", "C", "A"])
# mix("A" => 0.333..., "B" => 0.333..., "C" => 0.333...); a Bool column gives 0.5

# Add a mixture column to base data
base_data = (x = [1.0, 2.0], y = [0.1, 0.2])
expanded = merge(base_data, (group = fill(mix("A" => 0.5, "B" => 0.5), 2),))
```

### Reference Grid Creation

For marginal effects analysis, create reference grids with mixtures:

```julia
# Method 1: Direct DataFrame creation
reference_grid = DataFrame(
    x = [1.0, 2.0, 3.0],
    continuous_var = [0.0, 0.5, 1.0],
    categorical_mix = fill(mix("A" => 0.5, "B" => 0.5), 3)
)

# Method 2: Adding a mixture column to an existing grid
base_grid = DataFrame(x = [1.0, 2.0, 3.0])
mixture_grid = merge(
    Tables.columntable(base_grid),
    (treatment = fill(mix("Control" => 0.3, "Treated" => 0.7), 3),)
)
```

## Validation and Error Handling

### Automatic Validation

`mix` and the `CategoricalMixture` constructor validate levels and weights:

```julia
# ✓ Valid - the same mixture in every row
valid_data = (x = [1.0, 2.0], group = [mix("A"=>0.3, "B"=>0.7), mix("A"=>0.3, "B"=>0.7)])

# ✓ Valid - a Vector{<:CategoricalMixture} column may hold a different mixture in each row
per_row = (x = [1.0, 2.0], group = [mix("A"=>0.3, "B"=>0.7), mix("A"=>0.5, "B"=>0.5)])

# ✗ Invalid - weights don't sum to 1.0
mix("A"=>0.3, "B"=>0.6)  # Throws ArgumentError
```

### Manual Validation

The validators used by the constructor are exported:

```julia
validate_mixture_weights([0.3, 0.7])        # ✓ Valid
validate_mixture_weights([0.3, 0.6])        # ✗ ArgumentError: sum ≠ 1.0
validate_mixture_levels(["A", "B", "C"])    # ✓ Valid
validate_mixture_levels(["A", "A", "B"])    # ✗ ArgumentError: duplicates
```

`validate_mixture_against_data(mixture, column, name)` checks that a mixture's levels occur in a data column.

## Performance Characteristics

### Compilation
- **No per-mixture specialization**: a new mixture does not create a new operation type or trigger recompilation

### Execution Performance
- **Memory usage**: 0 bytes allocated during execution
- **Time**: the mixture row is a weighted sum of contrast-matrix rows, so its cost grows with the number of levels in the mixture

### Benchmarks

Measured with BenchmarkTools on an Apple M4 Pro (Julia 1.13.3), `y ~ x * group`, median time for one row:

| `group` column | 2 levels | 5 levels |
|----------------|----------|----------|
| `CategoricalArray` | 7 ns | 10 ns |
| Mixture over all levels | 8 ns | 18 ns |

All four allocate 0 bytes. To measure on your system:

```julia
using BenchmarkTools
cat_data = (x = [1.0, 2.0], group = categorical(["A", "B"]))
mix_data = (x = [1.0, 2.0], group = fill(mix("A" => 0.3, "B" => 0.7), 2))
compiled_cat = compile_formula(model, cat_data)
compiled_mix = compile_formula(model, mix_data)
out = Vector{Float64}(undef, length(compiled_cat))
@benchmark $compiled_cat($out, $cat_data, 1)
@benchmark $compiled_mix($out, $mix_data, 1)
```

## Integration with Marginal Effects

### Basic Marginal Effects Workflow

```julia
using FormulaCompiler, GLM

# Create reference grid with mixture (levels of `group` in `model` from the Quick Start)
reference_data = DataFrame(
    x = [0.0, 1.0, 2.0],  # Values to evaluate at
    group = fill(mix("A" => 0.5, "B" => 0.5), 3)  # Population mixture
)

# Compile model
compiled = compile_formula(model, Tables.columntable(reference_data))

# Evaluate marginal effects at each reference point
n_points = nrow(reference_data)
results = Matrix{Float64}(undef, n_points, length(compiled))

for i in 1:n_points
    compiled(view(results, i, :), Tables.columntable(reference_data), i)
end
```

### Integration with Derivatives System

Mixtures work seamlessly with derivative computation:

```julia
using LinearAlgebra: mul!

# Build derivative evaluator with mixture data
vars = [:x]  # Continuous variables for derivatives
de_fd = derivativeevaluator(:fd, compiled, Tables.columntable(reference_data), vars)

# Marginal effects on η with zero allocations (after warmup)
J = Matrix{Float64}(undef, length(compiled), length(vars))
gradient = Vector{Float64}(undef, length(vars))
derivative_modelrow!(J, de_fd, 1)    # 0 bytes
mul!(gradient, J', coef(model))      # gradient[j] = ∂η/∂vars[j]; here β_x + 0.5 β_{x & group:B}
```

## Advanced Usage

### Multiple Mixture Variables

You can have multiple categorical mixture variables in the same model:

```julia
training_tr = DataFrame(
    y = randn(300),
    x = randn(300),
    treatment = categorical(rand(["Control", "Treated"], 300)),
    region = categorical(rand(["North", "South"], 300))
)
model_tr = lm(@formula(y ~ x * treatment * region), training_tr)

df_tr = DataFrame(
    x = [1.0, 2.0, 3.0],
    treatment = [mix("Control" => 0.3, "Treated" => 0.7),
                 mix("Control" => 0.3, "Treated" => 0.7),
                 mix("Control" => 0.3, "Treated" => 0.7)],
    region = [mix("North" => 0.4, "South" => 0.6),
              mix("North" => 0.4, "South" => 0.6),
              mix("North" => 0.4, "South" => 0.6)]
)

compiled_tr = compile_formula(model_tr, Tables.columntable(df_tr))  # Handles multiple mixtures
```

### Complex Mixture Specifications

Support for arbitrary numbers of levels:

```julia
# Multi-level mixture
complex_mixture = mix(
    "Category_A" => 0.25,
    "Category_B" => 0.30, 
    "Category_C" => 0.20,
    "Category_D" => 0.15,
    "Category_E" => 0.10
)

df = DataFrame(
    x = [1.0, 2.0],
    complex_cat = [complex_mixture, complex_mixture]
)
```

### Interaction Terms with Mixtures

Mixtures work with all interaction patterns:

```julia
# Two-way interactions
@formula(y ~ x * mixture_group)

# Three-way interactions  
@formula(y ~ x * z * mixture_group)

# Mixed interactions
@formula(y ~ log(x) * mixture_group * other_categorical)
```

## Implementation Details

### Mixture Operation

A mixture column compiles to a `MixtureContrastOp` that holds only the column name, the
output positions, the model's contrast matrix and its levels. The mixture itself is read from
the data on every row:

```julia
MixtureContrastOp{:group, (1, 2)}(contrast_matrix, LevelIndex(model_levels))
```

Nothing about the compile-time mixture is kept, so a compiled formula (including one cached
by `get_or_compile_formula`) gives the right row for any other mixture: other weights, other
levels, or a different mixture in each row. Before 1.5.0 the levels and weights were embedded
in the operation's type, and a cached formula silently reused the first mixture it was
compiled with. Mixture levels are resolved to contrast-matrix rows once per distinct levels
vector and memoised, so evaluation does not allocate.

### Contrast Matrix Computation

Mixtures are evaluated as weighted combinations of contrast matrices:

```julia
# For dummy coding with mix("A" => 0.3, "B" => 0.7):
# Standard contrast matrix:
#   A: [1, 0]  (A vs reference)
#   B: [0, 1]  (B vs reference)
# 
# Mixture result: 0.3 * [1, 0] + 0.7 * [0, 1] = [0.3, 0.7]
```

### Memory Layout

Memory usage:

- **Compile time**: no mixture-specific state (the operation stores the model's contrast matrix and levels)
- **Execution time**: no allocation (scratch space is part of the compiled formula)
- **Data storage**: No mixture expansion in actual data (O(1) vs O(n))

## Error Messages and Debugging

### Common Error Messages

All are `ArgumentError`s thrown by `mix` (or `CategoricalMixture`) when the mixture is built:

```julia
# Weights don't sum to 1.0: mix("A" => 0.3, "B" => 0.6)
"Mixture weights must sum to 1.0 (±1.0e-10): got [0.3, 0.6] (sum = 0.8999999999999999)"

# Duplicate levels: mix("A" => 0.5, "B" => 0.3, "A" => 0.2)
"Mixture levels must be unique. Duplicates found: [\"A\"]"

# Negative weights: mix("A" => 0.5, "B" => -0.2, "C" => 0.7)
"Mixture weights must be non-negative: [0.5, -0.2, 0.7]"
```

An untyped (`Vector{Any}`) column whose rows hold different mixtures is not detected as a mixture column, and evaluating it fails with an error saying that categorical mixture objects reached a standard `ContrastOp`. Use a `Vector{<:CategoricalMixture}` column (e.g. built with `fill` or a comprehension over `mix`) for per-row mixtures.

### Debugging Tips

1. **Check mixture consistency**: In an untyped (e.g. `Vector{Any}`) column all rows must hold the same mixture; a `Vector{<:CategoricalMixture}` column may vary by row
2. **Validate weights**: Must be non-negative and sum to 1.0 (within 1e-10 tolerance)
3. **Verify levels**: Must be unique strings/symbols
4. **Test detection**: Use `FormulaCompiler.is_mixture_column(column)` (unexported) to check that an untyped column is detected

## Testing and Validation

### Built-in Tests

The mixture tests are `test/test_mixture_evaluation.jl`, `test/test_mixture_api.jl` and `test/test_mixture_modelrows.jl`; they run with the full suite:

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

See `test/README.md` for running a single file.

### Custom Testing

Validate your mixture implementations:

```julia
using Test

# Test mixture detection
@test FormulaCompiler.is_mixture_column([mix("A" => 0.3, "B" => 0.7), mix("A" => 0.3, "B" => 0.7)])
@test !FormulaCompiler.is_mixture_column(["A", "B", "A"])

# Test compilation and execution
data_mix = Tables.columntable(DataFrame(x = [1.0], group = [mix("A" => 0.3, "B" => 0.7)]))
compiled_mix = compile_formula(model, data_mix)
output_mix = Vector{Float64}(undef, length(compiled_mix))

# Should execute without allocation (after a warmup call; build the data outside @allocated)
compiled_mix(output_mix, data_mix, 1)
@test (@allocated compiled_mix(output_mix, data_mix, 1)) == 0
```

## Migration Guide

### From Override System

If you're currently using the override system for categorical mixtures:

```julia
# Pattern 1: Direct Mixture Data (Recommended)
mix_data = DataFrame(x = [1.0, 2.0], group = [mix("A" => 0.3, "B" => 0.7), mix("A" => 0.3, "B" => 0.7)])
compiled = compile_formula(model, Tables.columntable(mix_data))  # → MixtureContrastOp

# Pattern 2: Manual Data Modification (Flexible for dynamic scenarios)
base_data = DataFrame(x = [1.0, 2.0], group = ["A", "A"])
mixture_data = merge(Tables.columntable(base_data), (group = fill(mix("A" => 0.3, "B" => 0.7), 2),))
compiled_cf = compile_formula(model, mixture_data)  # → MixtureContrastOp with data modification

# Pattern 3: Manual Population Analysis
base_data = DataFrame(x = [1.0, 2.0], group = ["A", "A"])
results = []
for (level, weight) in [("A", 0.3), ("B", 0.7)]
    level_data = merge(Tables.columntable(base_data), (group = fill(level, 2),))
    compiled = compile_formula(model, level_data)
    # Evaluate and weight results manually
    push!(results, (compiled, weight))
end
```

Pattern 3 gives the same model row as Patterns 1 and 2 (the weighted sum of the per-level rows), but evaluates the formula once per level instead of once.

## Limitations and Considerations

### Current Limitations

1. **Consistent specifications**: in an untyped column (e.g. `Vector{Any}`) all rows must hold the same mixture; a `Vector{<:CategoricalMixture}` column may hold a different mixture in each row
2. **Duck typing**: objects other than `CategoricalMixture` must have `levels` and `weights` fields, and are not validated

### Design Trade-offs

- **Runtime mixtures**: levels and weights are read from the data on every row (no recompilation for a new mixture); level-to-row resolution is memoised per levels vector, keeping evaluation allocation-free
- **Consistency requirement**: Simplifies implementation but limits some use cases

### Future Enhancements

Potential areas for future development:
- Optimized binary mixture methods
- Integration with more statistical packages
- Support for hierarchical mixture specifications

## References

- **Implementation**: `src/mixtures/` (types, `mix`, validation) and `MixtureContrastOp` in `src/compilation/`
- **Integration**: [Margins.jl](https://github.com/emfeltham/Margins.jl) uses mixtures for profile-based marginal effects
