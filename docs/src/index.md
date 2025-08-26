# FormulaCompiler.jl

High-performance, zero-allocation model matrix evaluation for Julia statistical models. Works across all formula types through advanced compile-time specialization.

## Key Features

- **Zero allocations**: ~50ns per row, 0 bytes allocated across all 2032 test cases
- **Significant speedup and efficiency** over `modelmatrix()` for single-row evaluations  
- **Universal compatibility**: Handles any valid StatsModels.jl formula, including complex interactions and functions
- **Advanced scenarios**: Memory-efficient variable overrides for policy analysis
- **Unified architecture**: Single compilation pipeline handles all formula complexities
- **Full ecosystem support**: Works with GLM.jl, MixedModels.jl, StandardizedPredictors.jl
- **Near-zero-allocation derivatives**: ForwardDiff-based automatic differentiation with ~112 bytes per call

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/emfeltham/FormulaCompiler.jl")
```

## Quick Start

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

# Compile once for fast evaluation  
data = Tables.columntable(df)
compiled = compile_formula(model, data)
row_vec = Vector{Float64}(undef, length(compiled))

# Zero-allocation evaluation (call millions of times)
compiled(row_vec, data, 1)  # ~50ns, 0 allocations
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

## Use Cases

- Monte Carlo simulations: Millions of model evaluations
- Bootstrap resampling: Repeated matrix construction
- Marginal effects: Near-zero-allocation automatic differentiation
- Policy analysis: Evaluate many counterfactual scenarios
- Real-time applications: Low-latency prediction serving
- Large-scale inference: Memory-efficient batch processing

## Next Steps

- Read the [Getting Started](getting_started.md) guide for a detailed walkthrough
- Explore [Advanced Features](guide/advanced_features.md) for scenario analysis and memory optimization
- Check out [Examples](examples.md) for real-world use cases
- Review the [API Reference](api.md) for complete function documentation

---

*Built for the Julia statistical ecosystem. Optimized for performance, designed for usability.*