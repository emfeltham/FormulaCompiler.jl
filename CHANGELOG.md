# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - October 2026

### Fixed

- AD derivatives with more than 12 variables (ForwardDiff chunk mode) threw a `MethodError`
  under ForwardDiff ≥ 1.4.5, which the `"0.10, 1.0"` compat bound allows. `fc_jacobian!`
  called ForwardDiff's internal zero-partials `seed!(duals, x)` / `seed!(duals, x, index)`,
  which ForwardDiff 1.4.5 renamed. Seeding now uses local helpers built only on the
  `Dual`/`Partials` constructors, so it no longer depends on ForwardDiff internals.
  Vector mode (≤ 12 variables) was unaffected. Tests added in `test/test_ad_chunk_mode.jl`
  (the suite previously had no chunk-mode coverage).
- Categorical values are now resolved against the fitted model's contrast levels instead of
  the data being evaluated. Before, `ContrastOp` indexed contrast rows by the evaluation
  data's codes, which gave wrong model rows (and wrong predictions, effects and contrasts
  downstream) in several cases:
  - `CategoricalArray` columns: the compiled object baked in a ref-code remap for the
    compile-time pool, and the `get_or_compile_formula` cache key ignored pools. Data with a
    reordered, subset or rebuilt pool, or a reference grid with its own small pool, reused the
    stale remap (e.g. an average prediction of 0.604 instead of 0.496; contrasts with flipped
    signs). The result depended on which data had been compiled first.
  - `String` columns and integer/Bool columns made categorical through `contrasts`: the level
    index was recomputed for every row from `sort(unique(column))` of the evaluation data,
    which was wrong for subsets lacking a level and for custom level orders
    (`DummyCoding(levels = [...])`), and integer values were used directly as row indices
    (wrong unless the codes were exactly `1:k`). It was also O(n) per row with 400 bytes
    allocated per row, so a 20,000-row population pass took seconds.
  - Through `ContrastEvaluator`, Bool and integer categorical contrasts returned all zeros
    because `true`/`false` (and `0`/`1`) both mapped to the first row.
  `ContrastOp` now carries a `LevelIndex` built from `ContrastsMatrix.levels`:
  `CategoricalValue`s resolve through a ref-code → row table keyed by pool identity (rebuilt
  only when a different pool is seen, extended when levels are appended), and other values
  through a hash lookup. Per-row evaluation is O(1) and allocation-free for String columns
  (≈13 ns vs 4.4 µs at n = 1,000 and 98 µs at n = 20,000 for `y ~ x * s`). CategoricalArray
  rows remain allocation-free (≈1–2 ns slower per categorical term). Because compiled objects
  no longer depend on the evaluation data's pool, the cache key needs no pool information and
  same-data cache hits are unchanged. Tests in `test/test_categorical_level_resolution.jl`.
- Missing categorical values (ref code 0) and out-of-range codes were silently clamped to
  the first level. They now throw an `ArgumentError`, as do values that are not levels of the
  fitted model (for example a string never seen in fitting). Tests in
  `test/test_categorical_level_resolution.jl`.
- Link derivative helpers (`_dmu_deta`, `_d2mu_deta2`, used for response-scale effects):
  the logit forms were computed from σ(η)(1 − σ(η)), which is exactly 0 once σ(η) rounds to
  1 (η ≳ 37), so response-scale effects and their gradients vanished for confident
  predictions (true dμ/dη at η = 40 is 4.2e-18). The cloglog second derivative returned NaN
  for η > 709 (0 × −Inf). Both now use stable forms (`exp(-|η|)`, `expm1`, log space).
  `PowerLink` and `NegativeBinomialLink` threw a `MethodError`; they now have analytic
  derivatives (the negative binomial derivative is computed directly because
  `GLM.mueta(::NegativeBinomialLink, η)` returns μ rather than dμ/dη), and any other
  `GLM.Link` falls back to forward-mode AD of `GLM.linkinv`. Tests in
  `test/test_link_derivatives.jl` check every link against closed-form 256-bit BigFloat
  references for η from −800 to 800.

- MixedModels formulas with `zerocorr(...)` random effects failed to compile with
  "Cannot decompose term". `fixed_effects_form` dropped only `RandomEffectsTerm` and the
  unapplied `|` term, so the `ZeroCorr` wrapper fell through to the generic decomposer. Every
  `MixedModels.AbstractReTerm` is now dropped. Tests in `test/test_mixedmodels_integration.jl`
  compare compiled rows with the fitted model's fixed-effects matrix for correlated slopes,
  `zerocorr`, and nested `(1|s/t)` terms.
- A compiled formula given an output vector shorter than `length(compiled)` silently skipped
  the out-of-range writes, leaving stale values in the caller's buffer. It now throws a
  `DimensionMismatch`; longer buffers are still accepted. The check is one integer comparison
  per call and evaluation remains allocation-free.
- Twelve exported names (`marginal_effects_eta!`, `marginal_effects_mu!`, `delta_method_se`,
  `accumulate_ame_gradient!`, ...) had no definitions: their implementations moved to Margins.jl
  in v2.0 but the export list was not updated. They are no longer exported.
  `derivative_modelrow(de, row)`, the allocating counterpart of `derivative_modelrow!` that
  was exported but undefined, is now implemented. `test/test_api_contracts.jl` checks that
  every exported name is defined.

- Contrasts could not target a level the fitted model knows but the data's pool lacks (e.g. a
  subset re-`categorical`-ized after dropping the reference level): categorical counterfactual
  replacements are ref codes into the column's pool. `contrastevaluator` now works on an
  internal copy of such columns with the missing model levels appended; the caller's data is
  not modified or replaced, and columns that already cover the model's levels are used as is.

### Added

- `categorical_levels(compiled)` and `categorical_variables(compiled)`: the variables the fitted
  model treats as categorical and their levels in model order, decided by the model's contrasts
  rather than column element types. An integer, `String`, or `Bool` column declared categorical
  (e.g. `contrasts = Dict(:k => DummyCoding())`) is included. They complement
  `continuous_variables`, so downstream packages need no element-type heuristics.
- `link_derivatives(link, η) -> (μ, dμ/dη, d²μ/dη²)`: public interface to the inverse link and its
  first two derivatives for every `GLM.Link` (the exported `_dmu_deta`/`_d2mu_deta2` remain for
  compatibility).
- `derivative_modelrow!(J, x, de, row)`: also writes the model row Xᵢ into `x`. With the AD
  backend Xᵢ is read from the dual numbers already computed, saving a formula evaluation for
  callers that need both (e.g. response-scale marginal effects and their delta-method gradients).
- `validate_mixture_weights` and `validate_mixture_levels` are now public (the
  `CategoricalMixture` constructor uses them). Errors name the offending weights or duplicate
  levels.

### Changed

- Performance: the AD backend no longer copies every differentiated column into a full-length
  array of dual numbers. `NumericCounterfactualVector` gains a storage type parameter
  (`NumericCounterfactualVector{T,B}`) and stores the original column as Float64, converting on
  read. Storage stays homogeneous, so evaluation remains allocation-free. This saves about
  (chunk + 1) × 8 bytes per row per differentiated variable, about 1 GB for 21 variables at
  n = 500,000. The one-parameter constructor `NumericCounterfactualVector{T}(base, row, value)`
  is unchanged.
- Performance: the response-scale contrast gradient reset its counterfactuals with a
  runtime-indexed loop over a heterogeneous tuple (type-unstable, about 1 KB allocated per
  call). It now uses the unrolled `reset_all_counterfactuals!`.
- `ContrastOp` gains a third type parameter (its `LevelIndex` of model levels).
  `typeof(op).parameters[1]` (the column) and `op isa ContrastOp` are unaffected. The internal
  `extract_level_code` is removed.

### Documentation

- Compiled formulas, derivative evaluators and contrast evaluators are not thread-safe
  (they own mutable scratch buffers). The population-analysis guide showed a `@threads` loop
  sharing one evaluator across threads, which gives wrong results. It now builds one compiled
  formula and evaluator per task, and the performance guide and the `compile_formula` and
  `get_or_compile_formula` docstrings say so explicitly.
- Examples throughout the docs called `marginal_effects_eta!`, `marginal_effects_mu!` and
  `delta_method_se`, which were removed from FormulaCompiler in 1.0 and are not exported by
  Margins.jl. They now use `derivative_modelrow!` with `mul!(g, J', β)` (η scale), `GLM.mueta`
  (μ scale) and `sqrt(dot(∇β, Σ, ∇β))`, or point to Margins.jl's public API.

## [1.4.2] - July 2026

### Fixed

- Contrast-matrix rows are now aligned to the CategoricalArray pool's ref-code order.
  StatsModels builds the contrast matrix from *observed* levels only, but execution indexes
  rows by pool ref code, so a column whose pool contained unused (zero-count) or reordered
  levels selected the wrong contrast row (e.g. off-by-one when an unused level precedes
  observed ones), silently corrupting model rows, predictions and effects.
  `decompose_standard_categorical` now remaps rows by level; unused levels get zero rows,
  which are never selected. Plain `String` columns were unaffected.
  Tests added to `test/test_categorical_correctness.jl`.

## [1.4.1] - June 2026

### Changed

- Widened CategoricalArrays compat to `"0.10, 1"` (supports 1.0/1.1; no source changes).

## [1.4.0] - March 2026

### Added

- Full support for `CenteredTerm` and `ScaledTerm` from StandardizedPredictors.jl
  - `Center()` transformation: centering only (mean=0, preserves original scale; scale=1.0)
  - `Scale()` transformation: scaling only (std=1, preserves original location; center=0.0)
  - `decompose_term!` methods for `CenteredTerm` and `ScaledTerm` (pass-through, same pattern as `ZScoredTerm`)
  - `extract_columns_recursive!` methods for mixed model support
  - All three transformations (`ZScore()`, `Center()`, `Scale()`) now fully supported
  - New test file: `test/test_centered_scaled_predictors.jl`

## [1.3.0] - February 2026

### Fixed

- Zero-allocation AD Jacobian under `--check-bounds=yes` (and therefore `Pkg.test()`)
  - ForwardDiff's `extract_jacobian!` uses `reshape()` which allocates a 48-byte `ReshapedArray` wrapper when bounds checking is enabled — the compiler can't inline the extraction step, so escape analysis fails to elide the heap allocation
  - Replaced `ForwardDiff.jacobian!` with custom `fc_jacobian!` that uses ForwardDiff's seeding and dual evaluation but extracts partials via direct loop indexing, avoiding `reshape` entirely
  - New internal functions: `_extract_jacobian_direct!`, `_extract_jacobian_chunk_direct!`, `fc_jacobian!`
  - Handles both vector mode (single pass) and chunk mode (multiple passes)
  - All 35 previously-failing AD allocation tests now pass

- Type instability in `ContrastEvaluator` counterfactual lookups
  - Added `@generated` specializations for `get_cf_for_var` and `update_counterfactual_for_var!` that unroll variable lookup into compile-time if-chains with literal tuple indices
  - Eliminates 64-byte allocation per call from runtime tuple indexing producing `Union` types

- Integer column handling in AD derivative evaluator
  - `derivativeevaluator_ad` now converts variable columns to `Vector{Float64}` at construction time
  - Handles integer-typed columns (e.g., `age::Vector{Int}`) that require Float64 for ForwardDiff dual arithmetic
  - One-time conversion cost at evaluator construction, zero runtime impact

### Changed

- Julia compat extended to include 1.12

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
