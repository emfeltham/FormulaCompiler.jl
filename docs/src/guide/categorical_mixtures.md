# Categorical Mixtures in FormulaCompiler.jl

## Overview

FormulaCompiler.jl supports **categorical mixtures** - weighted combinations of categorical levels that enable efficient profile-based marginal effects computation. This feature allows you to specify fractional values like `mix("A" => 0.3, "B" => 0.7)` directly in your data, which are then compiled into zero-allocation evaluators. For boolean variables, use simple numeric probabilities (e.g., `treated = 0.7` for 70% treatment rate).

**Key benefits:**
- **Zero-allocation execution**: tens of nanoseconds per row, 0 bytes allocated (typical; see Benchmark Protocol)
- **Compile-time optimization**: All mixture weights embedded in type parameters
- **Marginal effects ready**: Direct support for statistical packages like Margins.jl
- **Memory efficient**: O(1) memory usage regardless of data size

## Quick Start

```julia
using FormulaCompiler, GLM, DataFrames, Tables

# Create data with categorical mixtures
df = DataFrame(
    x = [1.0, 2.0, 3.0],
    group = [mix("A" => 0.3, "B" => 0.7),   # 30% A, 70% B
             mix("A" => 0.3, "B" => 0.7),
             mix("A" => 0.3, "B" => 0.7)]
)

# Fit and compile model
model = lm(@formula(y ~ x * group), training_data)
compiled = compile_formula(model, Tables.columntable(df))

# Zero-allocation evaluation
output = Vector{Float64}(undef, length(compiled))
compiled(output, Tables.columntable(df), 1)  # Zero allocations; time varies by hardware
```

## Mixture Object Interface

Categorical mixtures are detected via **duck typing** - any object with `levels` and `weights` properties:

```julia
# Example mixture object structure
struct MixtureExample
    levels::Vector{String}    # ["A", "B", "C"]
    weights::Vector{Float64}  # [0.2, 0.3, 0.5]
end

# FormulaCompiler will automatically detect and handle such objects
mixture = MixtureExample(["Control", "Treatment"], [0.4, 0.6])
```

## Creating Mixture Data

### Boolean Variables and Population Analysis

**Boolean variables** work seamlessly with FormulaCompiler's continuous interpretation. For population-level analysis and marginal effects, simply use numeric probabilities directly:

```julia
# Population analysis with boolean probabilities - much simpler!
df = DataFrame(
    x = [1.0, 2.0, 3.0],
    treated = fill(0.7, 3)  # 70% treatment probability for population analysis
)

# Fits naturally with FormulaCompiler's boolean handling
compiled = compile_formula(model, Tables.columntable(df))
output = Vector{Float64}(undef, length(compiled))
compiled(output, Tables.columntable(df), 1)  # treated effect = 0.7
```

**Benefits of Numeric Approach:**
- **Simpler**: No complex mixture objects needed
- **Direct**: `treated = 0.7` is clearer than `mix("false" => 0.3, "true" => 0.7)`
- **Efficient**: Zero-allocation performance maintained
- **Compatible**: Works with all scenario and counterfactual tools
- **StatsModels consistent**: Matches how boolean variables are actually handled

**Use Cases:**
- **Individual scenarios**: `treated = true` or `treated = false`  
- **Population analysis**: `treated = 0.6` (60% treatment rate)
- **Marginal effects**: Varying treatment probabilities across reference grids

### Helper Functions

FormulaCompiler provides several utilities for creating mixture data:

```julia
# Create mixture column for reference grids
mixture_spec = mix("A" => 0.3, "B" => 0.7)  # Your mixture constructor
column = FormulaCompiler.create_mixture_column(mixture_spec, 1000)  # 1000 identical rows

# Create balanced (equal weight) mixtures
balanced_dict = create_balanced_mixture(["A", "B", "C"])
# Returns: Dict("A" => 0.333..., "B" => 0.333..., "C" => 0.333...)
balanced_mixture = mix(balanced_dict...)

# Expand base data with mixture specifications
base_data = (x = [1.0, 2.0], y = [0.1, 0.2])
mixtures = Dict(:group => mix("A" => 0.5, "B" => 0.5))
expanded = FormulaCompiler.expand_mixture_grid(base_data, mixtures)
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

# Method 2: Using helper functions  
base_grid = DataFrame(x = [1.0, 2.0, 3.0])
mixture_grid = FormulaCompiler.expand_mixture_grid(
    Tables.columntable(base_grid), 
    Dict(:treatment => mix("Control" => 0.3, "Treated" => 0.7))
)
```

## Validation and Error Handling

### Automatic Validation

FormulaCompiler automatically validates mixture data during compilation:

```julia
# ✓ Valid - consistent mixtures
valid_data = (x = [1, 2], group = [mix("A"=>0.3, "B"=>0.7), mix("A"=>0.3, "B"=>0.7)])

# ✗ Invalid - inconsistent mixtures
invalid_data = (x = [1, 2], group = [mix("A"=>0.3, "B"=>0.7), mix("A"=>0.5, "B"=>0.5)])
compile_formula(model, invalid_data)  # Throws ArgumentError

# ✗ Invalid - weights don't sum to 1.0  
bad_weights = (x = [1, 2], group = [mix("A"=>0.3, "B"=>0.6), mix("A"=>0.3, "B"=>0.6)])
compile_formula(model, bad_weights)  # Throws ArgumentError
```

### Manual Validation

You can also validate mixture data manually:

```julia
# Validate entire dataset
FormulaCompiler.validate_mixture_consistency!(data)

# Validate individual components
FormulaCompiler.validate_mixture_weights([0.3, 0.7])        # ✓ Valid
FormulaCompiler.validate_mixture_weights([0.3, 0.6])        # ✗ Sum ≠ 1.0
FormulaCompiler.validate_mixture_levels(["A", "B", "C"])    # ✓ Valid  
FormulaCompiler.validate_mixture_levels(["A", "A", "B"])    # ✗ Duplicates
```

## Performance Characteristics

### Compilation Time
- **Mixture detection**: ~1μs per column
- **Type specialization**: ~10μs per unique mixture specification
- **Overall overhead**: <20% increase for mixture-containing formulas

### Execution Performance
- **Simple mixtures**: tens of nanoseconds per row (similar to standard categorical)
- **Complex mixtures**: still on the order of tens to low hundreds of nanoseconds per row
- **Memory usage**: 0 bytes allocated during execution
- **Scaling**: Performance independent of mixture complexity

### Benchmarks

```julia
# Performance comparison (indicative)
@benchmark compiled(output, data, 1)
@benchmark compiled(output, mix_data, 1)
# Overhead should remain modest; measure on your system.
```

## Integration with Marginal Effects

### Basic Marginal Effects Workflow

```julia
using FormulaCompiler, GLM

# Create reference grid with mixture
reference_data = DataFrame(
    x = [0.0, 1.0, 2.0],  # Values to evaluate at
    group = fill(mix("Control" => 0.5, "Treatment" => 0.5), 3)  # Population mixture
)

# Compile model
model = lm(@formula(y ~ x * group), training_data)
compiled = compile_formula(model, Tables.columntable(reference_data))

# Evaluate marginal effects at each reference point
n_points = nrow(reference_data)
results = Matrix{Float64}(undef, n_points, length(compiled))

for i in 1:n_points
    compiled(view(results, i, :), Tables.columntable(reference_data), i)
end
```

### Integration with Derivatives System

Mixtures work seamlessly with FormulaCompiler's derivative system:

```julia
# Build derivative evaluator with mixture data
vars = [:x]  # Continuous variables for derivatives
de_fd = derivativeevaluator_fd(compiled, Tables.columntable(reference_data), vars)

# Compute marginal effects with zero allocations
gradient = Vector{Float64}(undef, length(vars))
marginal_effects_eta!(gradient, de_fd, coef(model), 1)  # 0 bytes
```

## Advanced Usage

### Multiple Mixture Variables

You can have multiple categorical mixture variables in the same model:

```julia
df = DataFrame(
    x = [1.0, 2.0, 3.0],
    treatment = [mix("Control" => 0.3, "Treated" => 0.7),
                 mix("Control" => 0.3, "Treated" => 0.7),
                 mix("Control" => 0.3, "Treated" => 0.7)],
    region = [mix("North" => 0.4, "South" => 0.6),
              mix("North" => 0.4, "South" => 0.6), 
              mix("North" => 0.4, "South" => 0.6)]
)

model = lm(@formula(y ~ x * treatment * region), training_data)
compiled = compile_formula(model, Tables.columntable(df))  # Handles multiple mixtures
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

### Type Specialization

Mixture specifications are embedded in type parameters for maximum performance:

```julia
# Each unique mixture gets its own compiled method
MixtureContrastOp{
    :group,                    # Column name
    (1, 2),                   # Output positions
    (1, 2),                   # Level indices  
    (0.3, 0.7)               # Weights (embedded in type!)
}
```

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

The implementation uses compile-time specialization for optimal memory usage:

- **Compile time**: Mixture specs embedded in types (~0 runtime memory)
- **Execution time**: Only scratch vector allocation (~8 bytes per term)
- **Data storage**: No mixture expansion in actual data (O(1) vs O(n))

## Error Messages and Debugging

### Common Error Messages

```julia
# Inconsistent mixture specifications
"Inconsistent mixture specification in column group at row 2: expected (levels=[\"A\", \"B\"], weights=[0.3, 0.7]), got (levels=[\"A\", \"B\"], weights=[0.5, 0.5])"

# Weights don't sum to 1.0
"Mixture weights in column group do not sum to 1.0: [0.3, 0.6] (sum = 0.9)"

# Duplicate levels
"Mixture in column group contains duplicate levels: [\"A\", \"B\", \"A\"]"

# Negative weights
"Mixture weights in column group must be non-negative: [0.5, -0.2]"
```

### Debugging Tips

1. **Check mixture consistency**: All rows must have identical mixture specifications
2. **Validate weights**: Must be non-negative and sum to 1.0 (within 1e-10 tolerance)
3. **Verify levels**: Must be unique strings/symbols
4. **Test detection**: Use `is_mixture_column(column)` to verify detection

## Testing and Validation

### Built-in Tests

FormulaCompiler includes comprehensive mixture tests:

```bash
# Run mixture-specific tests
julia --project=test -e "include(\"test/test_mixture_detection.jl\")"      # 142 tests
julia --project=test -e "include(\"test/test_categorical_mixtures.jl\")"  # 62 tests

# Run full test suite  
julia --project=. -e "using Pkg; Pkg.test()"  # 237 mixture tests included
```

### Custom Testing

Validate your mixture implementations:

```julia
using Test

# Test mixture detection
@test is_mixture_column([mix("A" => 0.3, "B" => 0.7), mix("A" => 0.3, "B" => 0.7)])
@test !is_mixture_column(["A", "B", "A"])

# Test compilation and execution
df_mix = DataFrame(x = [1.0], group = [mix("A" => 0.3, "B" => 0.7)])
compiled = compile_formula(model, Tables.columntable(df_mix))
output = Vector{Float64}(undef, length(compiled))

# Should execute without allocation
@test (@allocated compiled(output, Tables.columntable(df_mix), 1)) == 0
```

## Migration Guide

### From Override System

If you're currently using the override system for categorical mixtures:

```julia
# Pattern 1: Direct Mixture Data (Recommended - Compile-time Specialization)
mix_data = DataFrame(x = [1.0, 2.0], group = [mix("A" => 0.3, "B" => 0.7), mix("A" => 0.3, "B" => 0.7)])
compiled = compile_formula(model, Tables.columntable(mix_data))  # → MixtureContrastOp (fastest)

# Pattern 2: CounterfactualVector Mixtures (Flexible but slower)
data_cf, cf_vecs = build_counterfactual_data(Tables.columntable(base_data), [:group], 1)
mixture_replacement = mix("A" => 0.3, "B" => 0.7)
update_counterfactual_replacement!(cf_vecs[1], mixture_replacement)
compiled_cf = compile_formula(model, data_cf)  # → ContrastOp with runtime mixture handling

# Pattern 3: Manual Population Analysis (Most flexible)
base_data = DataFrame(x = [1.0, 2.0], group = ["A", "A"])
results = []
for (level, weight) in [("A", 0.3), ("B", 0.7)]
    level_data = merge(Tables.columntable(base_data), (group = fill(level, 2),))
    compiled = compile_formula(model, level_data)
    # Evaluate and weight results manually
    push!(results, (compiled, weight))
end
```

### Performance Benefits

| Approach | Compilation | Memory Usage | Allocations | Relative Speed |
|----------|-------------|--------------|-------------|----------------|
| Override system | Per-scenario | O(scenarios) | 0 bytes | Baseline |
| Compile-time mixtures | Once | O(1) | 0 bytes | ~3-4x faster |

**Note**: Both achieve zero allocations. Absolute timing varies by system; relative speedup is consistent.

## Limitations and Considerations

### Current Limitations

1. **Consistent specifications**: All rows must have identical mixture specifications
2. **Compile-time binding**: Cannot change mixture weights at runtime  
3. **Duck typing dependency**: Mixture objects must have `levels`, `weights`, and `original_levels` properties

### Design Trade-offs

- **Flexibility vs Performance**: Compile-time binding sacrifices runtime flexibility for zero-allocation performance
- **Memory vs Speed**: Type specialization uses more compilation time/memory for faster execution
- **Consistency requirement**: Simplifies implementation but limits some use cases

### Future Enhancements

Potential areas for future development:
- Runtime mixture resolution for varying specifications
- Optimized binary mixture methods
- Integration with more statistical packages
- Support for hierarchical mixture specifications

## References

- **Design Document**: `CATEGORICAL_MIXTURES_DESIGN.md` - Complete technical design
- **Implementation**: Phases 1-5 complete with 237 tests passing
- **Performance Targets**: All targets met (≤110% of standard categorical performance)
- **Integration**: Ready for Margins.jl and other marginal effects packages
