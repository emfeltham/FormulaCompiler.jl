# FormulaCompiler.jl

Computationally efficient model matrix evaluation for Julia statistical models. Implements position-mapping compilation to achieve substantial performance improvements across formula types through compile-time specialization.

## Key Features

- **Memory efficiency**: Quick per row evaluation with minimal memory allocation (validated across 2032+ test cases)
- **Computational performance**: Substantial improvements over traditional `modelmatrix()` approaches for single-row evaluations  
- **Comprehensive compatibility**: Supports all valid StatsModels.jl formulas, including complex interactions and mathematical functions
- **Scenario analysis**: Memory-efficient variable override system for counterfactual analysis
- **Unified architecture**: Single compilation pipeline accommodates diverse formula structures
- **Ecosystem integration**: Compatible with GLM.jl, MixedModels.jl, and StandardizedPredictors.jl
- **Dual-backend derivatives**: Memory-efficient finite differences and ForwardDiff automatic differentiation options

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
compiled(row_vec, data, 1)  # ~50ns, minimal allocations
```

## Performance Comparison

Performance results across all tested formula types:

```julia
using BenchmarkTools

# Traditional approach (creates full model matrix)
@benchmark modelmatrix(model)[1, :]
# ~10.2 μs (1 allocation: 896 bytes)

# FormulaCompiler (zero-allocation single row)
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

@benchmark compiled(row_vec, data, 1)
# ~50 ns (0 allocations: 0 bytes)

# Zero allocation across 2032 test cases
```

## Allocation Characteristics

FormulaCompiler.jl provides different allocation guarantees depending on the operation:

### Core Model Evaluation
- **Perfect zero allocations**: `modelrow!()` and direct `compiled()` calls are guaranteed 0 bytes after warmup
- **Performance**: ~50ns per row across all formula complexities
- **Validated**: 2032+ test cases confirm zero-allocation performance

### Derivative Operations
FormulaCompiler.jl offers **dual backends** for derivatives and marginal effects:

| Backend | Allocations | Performance | Use Case |
|---------|-------------|-------------|----------|
| `:fd` (Finite Differences) | **0 bytes** | ~79ns | Strict zero-allocation requirements |
| `:ad` (ForwardDiff) | ~368-400 bytes | ~508ns | Speed and numerical accuracy priority |

```julia
# Choose your backend based on requirements
marginal_effects_eta!(g, de, beta, row; backend=:fd)  # 0 allocations
marginal_effects_eta!(g, de, beta, row; backend=:ad)  # ~368 bytes, faster
```

### When to Use Each Backend
- **Use `:fd`** for: Monte Carlo loops, bootstrap resampling, memory-constrained environments
- **Use `:ad`** for: One-off calculations, interactive analysis, maximum numerical precision

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
- Check out [Examples](examples.md) for real-world use cases
- Review the [Mathematical Foundation](mathematical_foundation.md) for comprehensive theory and implementation details
- Review the [API Reference](api.md) for complete function documentation
