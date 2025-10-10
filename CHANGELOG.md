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

**Memory-Efficient Scenario System**
- `DataScenario`: Individual scenario with variable overrides
- `ScenarioCollection`: Batch scenario operations for policy analysis
- `create_scenario_grid()`: Systematic parameter exploration with >99% memory savings
- Population-level override system for batch counterfactual analysis

**High-Performance Derivatives**
- Dual-backend system with keyword argument selection (`:ad` or `:fd`)
- `:ad` (ForwardDiff) backend: Higher accuracy (machine precision) and faster performance
- `:fd` (finite differences) backend: Alternative with explicit step size control
- Marginal effects for both η (linear predictor) and μ (via link functions)
- Support for GLM link functions: Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt
- `build_derivative_evaluator(...; backend=:ad)` pattern

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
- **Derivatives**: Backend-dependent allocation characteristics
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
- **CounterfactualVector System**: Type-stable row-wise variable substitution architecture
  - `CounterfactualVector{T}`: Abstract supertype for all typed counterfactual vectors
  - `BoolCounterfactualVector`: Type-stable boolean variable substitution
  - `NumericCounterfactualVector{T<:Real}`: Type-stable numeric variable substitution (Int64, Float64, etc.)
  - `StringCounterfactualVector`: Type-stable string variable substitution
  - `CategoricalCounterfactualVector{T,R}`: Type-stable categorical variable substitution with reference type
  - Mutable structs with `row::Int` and `replacement::T` fields for efficient updates
  - O(1) memory complexity for single-row variable substitution without data copying
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
- **Categorical Mixture System**: Fractional categorical specifications for profile-based marginal effects
  - `CategoricalMixture` and `MixtureWithLevels` types for weighted categorical combinations
  - `CategoricalMixtureCounterfactualVector{T}`: Type-stable counterfactual vector for categorical mixtures
  - `mix()` constructor for creating mixture specifications (e.g., `mix("A" => 0.6, "B" => 0.4)`)
  - `create_mixture_column()`, `create_balanced_mixture()`, `expand_mixture_grid()` utilities
  - Zero-allocation evaluation maintaining FormulaCompiler's performance guarantees
  - Full integration with all contrast types and interaction terms
  - Comprehensive validation with clear error messages
- **Compressed Categorical Arrays**: Support for memory-efficient categorical representations
  - UInt8, UInt16, UInt32 reference types in CategoricalArrays
  - Correct handling in counterfactual scenarios and override system
  - Integration with contrast coding system
- **Benchmark Protocol**: Comprehensive benchmarking guidelines (docs/src/benchmarks.md)
  - Environment specification, setup procedures, and performance targets
  - Reporting template for consistent performance tracking
- **Enhanced Test Suite**:
  - `test_categorical_mixtures.jl`: Comprehensive mixture system validation
  - `test_mixture_modelrows.jl`: ModelRow correctness with mixtures
  - `test_compressed_categoricals.jl`: Compressed categorical array support
  - `test_contrast_evaluator.jl`: Zero-allocation discrete contrasts
  - `test_ad_alloc_formula_variants.jl`: Formula pattern allocation profiling
  - `test_formulacompiler_primitives_allocations.jl`: Core primitive performance
  - `test_documentation_examples.jl`: Documentation example validation
  - Debugging utilities: allocation tracing, type stability analysis tools

### Changed
- **Documentation Improvements**:
  - Qualified absolute timings (e.g., "tens of nanoseconds"); emphasized "zero allocations after warmup"
  - Clarified derivatives backend trade-offs: both `:ad` and `:fd` achieve zero allocations, `:ad` preferred for higher accuracy
  - Corrected examples to consistently pass column-table data to `compile_formula(model, data)`
  - Standardized headings and cross-references; added Benchmark Protocol links
  - Expanded API documentation with low-level derivatives, variance utilities, and mixture functions
  - Enhanced mathematical foundation documentation for derivatives and marginal effects
- **Test Organization**:
  - Updated test/README.md with comprehensive test suite documentation
  - Added "Debugging/Diagnostic Utilities" section documenting non-test development tools
  - Improved test descriptions with detailed coverage information
  - Enhanced debugging guidance with code examples for common issues
  - Corrected test file counts and references (24 test files in runtests.jl)
- **Performance Metrics**: Updated with measured values
  - Core evaluation: ~16ns per row (improved from ~50ns)
  - Derivatives (FD): ~65ns Jacobian
  - Derivatives (AD): ~49ns Jacobian
  - Marginal effects η: ~57-82ns (AD/FD)
  - Marginal effects μ: ~83-108ns (AD/FD)

### Fixed
- Categorical counterfactual handling: Improved validation and error messages for categorical structure preservation
- Allocation patterns: Refined buffer management for guaranteed zero allocations in critical paths
- Cross-validation: Enhanced numerical agreement testing between AD and FD backends (rtol=1e-6, atol=1e-8)
