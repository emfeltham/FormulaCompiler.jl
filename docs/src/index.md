# FormulaCompiler.jl

Efficient model matrix evaluation for Julia statistical models. Implements position-mapping compilation to achieve performance improvements across formula types through compile-time specialization.

## Key Features

- Memory efficiency: Per-row evaluation with reduced memory allocation (validated across test cases)
- Computational performance: Improvements over traditional `modelmatrix()` approaches for single-row evaluations  
- Comprehensive compatibility: Supports all valid StatsModels.jl formulas, including complex interactions and mathematical functions
- Categorical mixtures: Compile-time support for weighted categorical specifications for marginal effects
- Scenario analysis: Memory-efficient variable override system for counterfactual analysis
- Unified architecture: Single compilation pipeline accommodates diverse formula structures
- Ecosystem integration: Compatible with GLM.jl, MixedModels.jl, and StandardizedPredictors.jl
- Dual-backend derivatives: Memory-efficient finite differences and ForwardDiff automatic differentiation options

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/emfeltham/FormulaCompiler.jl")
```

## Quick Start

![Workflow](assets/src_getting_started_diagram_8.svg)

*Figure: Basic FormulaCompiler.jl workflow - from statistical model to zero-allocation evaluation*

```julia
using FormulaCompiler, GLM, DataFrames, Tables

# Fit your model normally
df = DataFrame(
    y = randn(1000),
    x = randn(1000),
    z = abs.(randn(1000)) .+ 0.1,
    group = categorical(rand(["A", "B", "C"], 1000))
)

model = lm(@formula(y ~ x * group + log(z)), df)

# Compile once for efficient repeated evaluation  
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# Memory-efficient evaluation suitable for repeated calls
compiled(row_vec, data, 1)  # Zero allocations after warmup
```

## Performance Comparison

Performance results across all tested formula types:

```julia
using BenchmarkTools

# Traditional approach (creates full model matrix)
@benchmark modelmatrix(model)[1, :]
# Traditional approach with allocation overhead

# FormulaCompiler (zero-allocation single row)
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

@benchmark compiled(row_vec, data, 1)
# FormulaCompiler approach with zero allocations

```

**Allocation guarantees** (verified in test suite):
- Core row evaluation (`compiled(row,data,i)`): **0 bytes**
- Scenario evaluation (CounterfactualVector): **0 bytes**
- FD Jacobian (single column): **0 bytes**
- AD Jacobian: **0 bytes**
- Marginal effects η (FD/AD): **0 bytes**
- Marginal effects μ with link functions (FD/AD): **0 bytes**
- Delta method SE: **0 bytes**

Example timings from one environment (Julia 1.11.2, Apple M1):
- Core evaluation: ~10 ns | Scenario: ~10 ns | FD Jacobian: ~31 ns | AD Jacobian: ~43 ns

!!! note "Performance varies by system"
    **Allocation guarantees are universal** (always 0 bytes). Timings vary by hardware, Julia version, and formula complexity. For reproducible benchmarks on your system, see the [Benchmark Protocol](benchmarks.md).

## Allocation Characteristics

FormulaCompiler.jl provides different allocation guarantees depending on the operation:

### Core Model Evaluation
- Zero allocations: `modelrow!()` and direct `compiled()` calls are 0 bytes after warmup
- Performance: Fast per-row evaluation across all formula complexities
- Validated: Test cases confirm zero-allocation performance

### Derivative Operations
FormulaCompiler.jl offers dual concrete type backends for derivatives and marginal effects:

| Backend | Type | Allocations | Performance | Accuracy | Recommendation |
|---------|------|-------------|-------------|----------|----------------|
| Automatic Differentiation | `ADEvaluator` | 0 bytes | Faster | Machine precision | **Strongly preferred - use this** |
| Finite Differences | `FDEvaluator` | 0 bytes | Slower | ~1e-8 | Legacy support only |

```julia
# Use automatic differentiation (strongly recommended)
de = derivativeevaluator_ad(compiled, data, vars)  # Returns ADEvaluator

# Compute derivatives with zero allocations
marginal_effects_eta!(g, de, beta, row)  # 0 bytes, machine precision, faster
```

!!! tip "Backend Selection"
    **Always use `ADEvaluator`** (automatic differentiation). It is:
    - **Faster**: ~22% faster than finite differences
    - **More accurate**: Machine precision vs ~1e-8 for FD
    - **Zero allocations**: Just like FD

    The FD backend (`FDEvaluator`) exists only for legacy compatibility. There are no practical advantages to using it.

## Use Cases

- Monte Carlo simulations: Millions of model evaluations
- Bootstrap resampling: Repeated matrix construction
- Marginal effects: Choose zero-allocation finite differences or faster ForwardDiff ([Margins.jl](https://github.com/emfeltham/Margins.jl) is built on FormulaCompiler.jl)
- Policy analysis: Evaluate many counterfactual scenarios
- Real-time applications: Low-latency prediction serving
- Large-scale inference: Memory-efficient batch processing

## Next Steps

- Read the [Getting Started](getting_started.md) guide for a detailed walkthrough
- Explore [Advanced Features](guide/advanced_features.md) for scenario analysis and memory optimization
- Learn about [Categorical Mixtures](guide/categorical_mixtures.md) for marginal effects computation
- See [StandardizedPredictors Integration](integration/standardized_predictors.md) for comprehensive z-score standardization workflows
- Check out [Examples](examples.md) for real-world use cases
- Review the [Mathematical Foundation](mathematical_foundation.md) for comprehensive theory and implementation details
- Review the [API Reference](api.md) for complete function documentation
- Reproduce results with the [Benchmark Protocol](benchmarks.md)
