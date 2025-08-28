# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-28

### Added

**Core Engine**
- Zero-allocation statistical formula evaluation (~50ns per row)
- Position mapping compilation system for type-specialized evaluators
- Universal compatibility with StatsModels.jl formula system
- Support for GLM.jl, MixedModels.jl (fixed effects), and CategoricalArrays.jl

**Formula Features**
- Complete formula support: interactions, functions, categoricals, boolean expressions
- All CategoricalArrays contrast types (dummy, effects, helmert, etc.)
- Nested mathematical functions: `log`, `exp`, `sqrt`, `sin`, `cos`, `abs`, powers
- Complex interactions: `x * log(z) * group + sqrt(abs(y))`

**Memory-Efficient Override System**
- `OverrideVector`: Constant vectors using O(1) memory regardless of size
- `DataScenario`: Individual scenario with variable overrides
- `ScenarioCollection`: Batch scenario operations for policy analysis
- `create_scenario_grid()`: Systematic parameter exploration with >99% memory savings

**High-Performance Derivatives**
- Dual-backend system: `:fd` (zero-allocation) and `:ad` (higher accuracy)
- ForwardDiff backend: ≤512 bytes per call with high accuracy
- Finite differences backend: 0 bytes per call (fully optimized)
- Marginal effects for both η (linear predictor) and μ (via link functions)
- Support for GLM link functions: Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt

**Variance Computation Primitives**
- Delta method standard error computation (`delta_method_se`)
- Average marginal effects gradient accumulation (`accumulate_ame_gradient!`)
- Zero-allocation variance primitives for statistical inference
- Cross-validated against analytical solutions (rtol=1e-6, atol=1e-8)

**API Design**
- Clean, intuitive API following Julia conventions
- Comprehensive documentation with examples
- Type-stable operations throughout
- Allocation tracking and performance monitoring tools

### Performance Characteristics

- **Core evaluation**: ~50ns per row, 0 allocations
- **FD derivatives**: 0 allocations after warmup  
- **AD derivatives**: ≤512 bytes per call
- **Memory efficiency**: >99% savings for scenario analysis vs naive approaches
- **Speedup**: 10-100x faster than `modelmatrix()[row, :]`

### Ecosystem Integration

- **GLM.jl**: Linear and generalized linear models
- **MixedModels.jl**: Mixed-effects models (fixed effects extraction)
- **CategoricalArrays.jl**: All contrast types supported
- **Tables.jl**: Universal table format compatibility
- **StatsModels.jl**: Complete formula system support
- **StandardizedPredictors.jl**: Standardized predictor integration

### Documentation

- Complete API reference with examples
- Performance optimization guides
- Mathematical foundation documentation
- Integration examples for major statistical packages
- Architecture overview and design principles

---

*This represents the first significant release of FormulaCompiler.jl, providing a foundation for high-performance statistical computing in Julia.*
