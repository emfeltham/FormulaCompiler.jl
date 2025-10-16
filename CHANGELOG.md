# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - October 2025

### Added

- Categorical mixture system for fractional categorical specifications in profile-based marginal effects
  - `CategoricalMixture` and `MixtureWithLevels` types for weighted categorical combinations
  - `CategoricalMixtureCounterfactualVector{T}` for type-stable mixture counterfactuals
  - `mix()` constructor for mixture specifications (e.g., `mix("A" => 0.6, "B" => 0.4)`)
  - Utility functions: `create_mixture_column()`, `create_balanced_mixture()`, `expand_mixture_grid()`
  - Zero-allocation evaluation maintaining package performance characteristics
  - Integration with all contrast types and interaction terms
  - Validation with error messages for malformed specifications

- Support for compressed categorical arrays
  - UInt8, UInt16, UInt32 reference types in CategoricalArrays
  - Correct handling in counterfactual scenarios and override system
  - Integration with contrast coding system

- Benchmark protocol documentation (`docs/src/benchmarks.md`)
  - Environment specification and setup procedures
  - Performance targets and measurement methodology
  - Reporting template for performance tracking

- Extended test suite
  - `test_categorical_mixtures.jl`: Mixture system validation
  - `test_mixture_modelrows.jl`: ModelRow correctness with mixtures
  - `test_compressed_categoricals.jl`: Compressed categorical array support
  - `test_contrast_evaluator.jl`: Discrete contrast evaluation
  - `test_ad_alloc_formula_variants.jl`: Allocation profiling across formula patterns
  - `test_formulacompiler_primitives_allocations.jl`: Core primitive performance validation
  - `test_documentation_examples.jl`: Documentation example validation
  - Debugging utilities for allocation tracing and type stability analysis

### Changed

- BREAKING: Statistical functions migrated to Margins.jl v2.0
  - `marginal_effects_eta!` moved to Margins.jl for marginal effects on linear predictor
  - `marginal_effects_mu!` moved to Margins.jl for marginal effects on response scale
  - `delta_method_se` moved to Margins.jl for standard error computation
  - `accumulate_ame_gradient!` moved to Margins.jl for AME gradient accumulation
  - `continuous_variables` (for derivatives) moved to Margins.jl
  - FormulaCompiler retains only computational primitives: `derivativeevaluator`, `derivative_modelrow!`, `contrast_modelrow!`
  - Users requiring marginal effects should add Margins.jl: `Pkg.add(url="https://github.com/emfeltham/Margins.jl")`

- Documentation refinements
  - Qualified absolute timings; emphasized zero allocations after warmup
  - Clarified derivatives backend characteristics: both `:ad` and `:fd` achieve zero allocations, with `:ad` preferred for accuracy
  - Corrected examples to consistently use column-table data with `compile_formula(model, data)`
  - Standardized headings and cross-references; added benchmark protocol links
  - Expanded API documentation covering low-level derivatives, variance utilities, and mixture functions
  - Enhanced mathematical foundation documentation for derivatives and marginal effects
  - Standardized derivative evaluator API documentation to use unified dispatcher pattern

- Test organization improvements
  - Updated test/README.md with comprehensive suite documentation
  - Added debugging and diagnostic utilities documentation
  - Improved test descriptions with detailed coverage information
  - Enhanced debugging guidance with code examples

- Performance metrics updated to measured values
  - Core evaluation: approximately 16ns per row (improved from 50ns)
  - Derivatives (FD): approximately 65ns for Jacobian computation
  - Derivatives (AD): approximately 49ns for Jacobian computation
  - Marginal effects η: 57-82ns depending on backend (when using Margins.jl)
  - Marginal effects μ: 83-108ns depending on backend (when using Margins.jl)

### Fixed

- Improved categorical counterfactual handling validation and error messages for categorical structure preservation
- Refined buffer management for guaranteed zero allocations in critical paths
- Enhanced numerical agreement testing between AD and FD backends (rtol=1e-6, atol=1e-8)

### Migration Guide for v1.1.1

Users migrating from v1.1.0 to v1.1.1 should:

1. **Add Margins.jl dependency** if using marginal effects:
   ```julia
   using Pkg
   Pkg.add(url="https://github.com/emfeltham/Margins.jl")
   ```

2. **Update marginal effects code**:
   ```julia
   # OLD (v1.1.0): FormulaCompiler provided these functions
   using FormulaCompiler
   marginal_effects_eta!(g, de, β, row)

   # NEW (v1.1.1): Use Margins.jl
   using FormulaCompiler, Margins
   marginal_effects_eta!(g, de, β, row)  # Now from Margins.jl
   ```

3. **Computational primitives unchanged**: FormulaCompiler continues to provide:
   - `derivativeevaluator(:ad/:fd, ...)` - Evaluator construction
   - `derivative_modelrow!` - Jacobian computation
   - `contrast_modelrow!` - Discrete contrasts
   - Link function derivatives (`_dmu_deta`, `_d2mu_deta2`)

Mathematical correctness and performance characteristics are preserved. The migration enables specialized statistical development in Margins.jl while maintaining FormulaCompiler's computational engine role.

## [1.1.0] - 2025-09-29

### Added

- CounterfactualVector system for type-stable row-wise variable substitution
  - `CounterfactualVector{T}` abstract supertype for typed counterfactual vectors
  - `BoolCounterfactualVector` for boolean variable substitution
  - `NumericCounterfactualVector{T<:Real}` for numeric variables (Int64, Float64, etc.)
  - `StringCounterfactualVector` for string variable substitution
  - `CategoricalCounterfactualVector{T,R}` for categorical variables with reference type
  - Mutable structs with `row::Int` and `replacement::T` fields
  - O(1) memory complexity for single-row variable substitution

- Concrete type API for derivative evaluation
  - `derivativeevaluator_fd()` returns concrete `FDEvaluator` type
  - `derivativeevaluator_ad()` returns concrete `ADEvaluator` type
  - Type-dispatched methods eliminate runtime dispatch overhead

- Population analysis patterns using loop-based approaches
  - Row-wise evaluation with averaging for population-level marginal effects
  - CounterfactualVector loop patterns for parameter exploration
  - O(1) memory complexity maintained for counterfactual operations

### Changed

- BREAKING: API migration from backend keywords to concrete types
  - `derivativeevaluator(...; backend=:fd)` → `derivativeevaluator_fd(...)`
  - `marginal_effects_eta!(...; backend=:ad)` → `marginal_effects_eta!(g, de_ad, β, row)`
  - Functions now use compile-time type dispatch instead of runtime keyword dispatch

- BREAKING: Population scenario system removed
  - `create_scenario()`, `DataScenario`, `ScenarioCollection` removed
  - `create_scenario_grid()` and scenario manipulation functions removed
  - Users should implement analysis using loops with CounterfactualVector

- Documentation updated to reflect concrete type API patterns
  - Examples demonstrate `FDEvaluator`/`ADEvaluator` usage
  - Loop-based population analysis patterns documented
  - Migration guides provided for users transitioning from v1.0

### Removed

- Population override system (approximately 1200 lines)
  - `OverrideVector` (population-level) replaced by individual CounterfactualVector types
  - `create_scenario()` and related functions replaced by loop patterns
  - Scenario manipulation API simplified to direct CounterfactualVector usage

- Backend keyword arguments across all derivative functions
  - Runtime dispatch overhead removed
  - Keyword argument parsing overhead eliminated
  - Type ambiguity in derivative operations resolved

### Performance Improvements

- Concrete types enable compiler optimization without runtime dispatch
- Evaluators reduced to backend-specific infrastructure only (30-50% size reduction)
- Backend-specific initialization eliminates dual setup overhead
- Type parameter count reduced from 16 to 6-9 in evaluator types
- Loop-based patterns show improved performance over complex infrastructure

### Migration Guide for v1.1.0

Users migrating from v1.0 should:

1. **Update derivative evaluator construction** (choose one style):
   ```julia
   # OLD (v1.0): Keyword-based backend selection
   de = derivativeevaluator(compiled, data, vars; backend=:fd)

   # NEW (v1.1.0): Unified dispatcher (recommended for user code)
   de = derivativeevaluator(:fd, compiled, data, vars)

   # Alternative: Direct constructors (also valid)
   de_fd = derivativeevaluator_fd(compiled, data, vars)
   de_ad = derivativeevaluator_ad(compiled, data, vars)
   ```

2. **Remove backend keywords** from function calls:
   ```julia
   # OLD (v1.0): Backend as keyword argument
   marginal_effects_eta!(g, de, β, row; backend=:ad)

   # NEW (v1.1.0): Type dispatch based on evaluator type
   marginal_effects_eta!(g, de_ad, β, row)  # Type dispatch, no keyword
   ```

3. **Replace scenario system** with CounterfactualVector loops:
   ```julia
   # OLD (v1.0): Scenario system
   scenario = create_scenario("policy", data; x = 2.0)
   results = evaluate_scenario(compiled, scenario)

   # NEW (v1.1.0): Loop-based patterns
   n_rows = length(first(data))
   data_policy = merge(data, (x = fill(2.0, n_rows),))
   for row in 1:n_rows
       compiled(output, data_policy, row)
       # Process results
   end
   ```

4. **Update population analysis** to use loops:
   ```julia
   # OLD (v1.0): ScenarioCollection
   grid = create_scenario_grid(data, params)
   results = population_margins(compiled, grid)

   # NEW (v1.1.0): Simple averaging loops
   ame_sum = zeros(length(vars))
   for row in 1:n_rows
       marginal_effects_eta!(g_temp, de, β, row)
       ame_sum .+= g_temp
   end
   ame = ame_sum ./ n_rows
   ```

Mathematical correctness and performance characteristics are preserved while reducing architectural complexity.

## [1.0.0] - 2025-08-28

### Added

- Core compilation engine
  - Zero-allocation statistical formula evaluation (approximately 50ns per row)
  - Position mapping compilation system for type-specialized evaluators
  - Compatibility with StatsModels.jl formula system
  - Support for GLM.jl, MixedModels.jl (fixed effects), and CategoricalArrays.jl

- Formula language support
  - Interactions, functions, categorical variables, and boolean expressions
  - All CategoricalArrays contrast types (dummy, effects, helmert, etc.)
  - Mathematical functions: `log`, `exp`, `sqrt`, `sin`, `cos`, `abs`, powers
  - Complex nested expressions: `x * log(z) * group + sqrt(abs(y))`

- Scenario analysis system
  - `DataScenario` for individual scenarios with variable overrides
  - `ScenarioCollection` for batch scenario operations
  - `create_scenario_grid()` for systematic parameter exploration
  - Memory usage reduction exceeding 99% compared to data duplication approaches

- Derivative computation
  - Dual-backend system with `:ad` (ForwardDiff) and `:fd` (finite differences)
  - ForwardDiff backend: machine-precision accuracy with performance advantages
  - Finite differences backend: explicit step size control
  - Marginal effects for η (linear predictor) and μ (via link functions)
  - GLM link function support: Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt
  - `build_derivative_evaluator(...; backend=:ad)` constructor pattern

- Variance computation primitives
  - Delta method standard error computation (`delta_method_se`)
  - Average marginal effects gradient accumulation (`accumulate_ame_gradient!`)
  - Zero-allocation variance primitives for statistical inference
  - Cross-validation against analytical solutions (rtol=1e-6, atol=1e-8)

- API and documentation
  - Core API following Julia conventions
  - Comprehensive documentation with examples
  - Type-stable operations throughout
  - Allocation tracking and performance monitoring tools

### Performance Characteristics

- Core evaluation: approximately 50ns per row with zero allocations
- Memory efficiency: exceeding 99% savings for scenario analysis compared to data duplication
- Speedup: 10-100x faster than `modelmatrix()[row, :]` for single-row evaluation

### Ecosystem Integration

- GLM.jl: Linear and generalized linear models
- MixedModels.jl: Mixed-effects models with fixed effects extraction
- CategoricalArrays.jl: All contrast types supported
- Tables.jl: Universal table format compatibility
- StatsModels.jl: Complete formula system support
- StandardizedPredictors.jl: Standardized predictor integration (ZScore)

### Documentation

- API reference with usage examples
- Performance optimization guidance
- Mathematical foundation documentation
- Integration examples for major statistical packages
- Architecture overview and design principles

---

This represents the initial release of FormulaCompiler.jl.
