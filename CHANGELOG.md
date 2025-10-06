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
- Dual-backend system: Both `:ad` and `:fd` achieve zero allocations
- `:ad` (ForwardDiff) preferred: Higher accuracy (machine precision) and faster performance
- `:fd` (finite differences): Alternative backend with explicit step size control
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
## [1.1.0] - 2025-09-29

### Added
- **CounterfactualVector System**: Unified row-wise variable substitution architecture
  - `NumericCounterfactualVector{T}`: Type-stable numeric variable substitution
  - `CategoricalCounterfactualVector{T,R}`: Categorical variable substitution with type safety
  - `BoolCounterfactualVector`: Boolean variable substitution
  - `CategoricalMixtureCounterfactualVector{T}`: Categorical mixture support
  - `update_counterfactual_row!`, `update_counterfactual_replacement!`: Efficient update operations
- **Concrete Type API**: Separate evaluator types for maximum performance
  - `derivativeevaluator_fd()`: Returns concrete `FDEvaluator` type (zero allocations)
  - `derivativeevaluator_ad()`: Returns concrete `ADEvaluator` type (bounded allocations)
  - Type-dispatched methods eliminate runtime dispatch overhead entirely
- **Population Analysis Patterns**: Simple loop-based approaches replace complex infrastructure
  - Efficient population marginal effects using existing row-wise functions + averaging
  - CounterfactualVector loop patterns for systematic parameter exploration
  - O(1) memory complexity maintained for all counterfactual operations

### Changed
- **BREAKING**: Complete API migration from backend keywords to concrete types
  - `derivativeevaluator(...; backend=:fd)` → `derivativeevaluator_fd(...)`
  - `marginal_effects_eta!(...; backend=:ad)` → `marginal_effects_eta!(g, de_ad, β, row)`
  - All functions now use compile-time type dispatch instead of runtime keyword dispatch
- **BREAKING**: Population scenario system eliminated for architectural simplicity
  - `create_scenario()`, `DataScenario`, `ScenarioCollection` removed
  - `create_scenario_grid()`, scenario manipulation functions removed
  - Users should use simple loops with CounterfactualVector for population analysis
- **Documentation**: Comprehensive migration to concrete type API patterns
  - All examples updated to show `FDEvaluator`/`ADEvaluator` usage
  - Loop-based population analysis patterns documented
  - Migration guides provided for users transitioning from old API

### Removed
- **Population Override System**: 1200+ lines of infrastructure eliminated
  - `OverrideVector` (population-level): Replaced by individual CounterfactualVector types
  - `create_scenario()` and related functions: Replaced by loop patterns
  - Complex scenario manipulation API: Simplified to direct CounterfactualVector usage
- **Backend Keywords**: All `backend=:fd/:ad` patterns eliminated
  - Runtime dispatch overhead completely removed
  - Keyword argument parsing overhead eliminated
  - Type ambiguity in derivative operations resolved

### Performance Improvements
- **Zero Runtime Dispatch**: Concrete types enable maximum compiler optimization
- **Reduced Memory Footprint**: Evaluators carry only backend-specific infrastructure (30-50% reduction)
- **Faster Construction**: Backend-specific initialization eliminates dual setup overhead
- **Cleaner Type Hierarchy**: 6-9 type parameters vs 16 in previous unified approach
- **Loop Efficiency**: Simple patterns outperform complex population infrastructure

### Migration Guide
Users migrating from v1.0 should:
1. Replace `derivativeevaluator(...; backend=:fd)` with `derivativeevaluator_fd(...)`
2. Replace `derivativeevaluator(...; backend=:ad)` with `derivativeevaluator_ad(...)`
3. Remove backend keywords from all marginal effects function calls
4. Replace `create_scenario()` usage with CounterfactualVector loop patterns
5. Update population analysis to use simple loops + averaging instead of scenario grids

All mathematical correctness and performance characteristics are preserved while eliminating architectural complexity.

## [Unreleased]

### Added
- Benchmark Protocol (docs/src/benchmarks.md) with environment, setup, targets, and reporting template
- Automatic Mermaid regeneration in docs build (docs/make.jl) using `mmdc` if available
- Expanded API docs: added missing low-level derivatives and variance utilities; categorical mixtures utilities

### Changed
- Documentation tone and claims: qualified absolute timings (e.g., "tens of nanoseconds"); emphasized "zero allocations after warmup"
- Clarified derivatives backend trade-offs: both `:ad` and `:fd` achieve zero allocations, `:ad` preferred for higher accuracy
- Corrected examples to pass column-table data to `compile_formula(model, data)` consistently
- Standardized headings and cross-references; added Benchmark Protocol links
