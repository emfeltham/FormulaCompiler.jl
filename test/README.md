# FormulaCompiler.jl Test Suite

`runtests.jl` is the list of what runs; every file below is included there unless noted.
`TEST_PRINCIPLES.md` describes how tests are written (exact checks, allocation measurement).

## Running

```bash
# Full suite, as CI and Pkg.test run it (bounds checking on)
julia --project=. -e 'using Pkg; Pkg.test()'

# One file. Once, point the test environment at this checkout:
julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
# then load what runtests.jl loads, and include the file
julia --check-bounds=yes --project=test -e 'using Test, Random, FormulaCompiler;
    include("test/support/testing_utilities.jl"); include("test/test_models.jl")'
```

Use `--check-bounds=yes` when running files on their own, so that compiled caches and
allocation results match `Pkg.test`.

## Files

### Compilation and model rows
- `test_position_mapping.jl`: the position-mapping compiler (operations, scratch layout).
- `test_models.jl`: compiled rows equal `modelmatrix` across contrasts, interactions,
  transforms and model types.
- `test_allocations.jl`: compiled-row evaluation allocates nothing, over formula shapes from
  `support/testing_utilities.jl`.
- `test_logic.jl`: comparisons and boolean negation in formulas.
- `test_tough_formula.jl`: a large logit formula with many interactions.
- `test_unsupported_functions.jl`: unsupported formula functions fail at compile time.

### Counterfactuals and categoricals
- `test_overrides.jl`: counterfactual vectors and scenarios.
- `test_zero_allocation_overrides.jl`: override evaluation allocates nothing.
- `test_categorical_correctness.jl`: categorical counterfactuals against reference model
  matrices, including interactions.
- `test_compressed_categoricals.jl`: UInt8/UInt16/UInt32 reference types.
- `test_categorical_level_resolution.jl`: data levels resolved against model levels (pools,
  String/Int/Bool, missing, unused and reordered levels).

### Categorical mixtures
- `test_mixture_evaluation.jl`: mixture evaluation primitives against weighted combinations.
- `test_mixture_api.jl`: `mix`, `CategoricalMixture`, `validate_mixture_against_data`,
  `mixture_to_scenario_value`.
- `test_mixture_modelrows.jl`: model rows with mixtures.

### Derivatives and contrasts
- `test_derivative_primitive_allocations.jl`: `derivative_modelrow!` allocates nothing (AD, FD).
- `test_contrast_evaluator.jl`: `contrast_modelrow!`/`contrast_gradient!` against ground
  truth, and their allocations.
- `test_ad_alloc_formula_variants.jl`, `test_formulacompiler_primitives_allocations.jl`:
  allocations for formula variants and batch loops on a ~50-column GLM.
- `test_ad_chunk_mode.jl`: AD Jacobians with more than 12 variables.
- `test_fd_jacobian_column.jl`: `fd_jacobian_column!`, and FD at extreme variable scales.
- `test_derivatives_domain_edge_cases.jl`: derivatives near domain boundaries.
- `test_link_derivatives.jl`: link dμ/dη and d²μ/dη² against BigFloat references.

Marginal effects (`marginal_effects_eta!`/`_mu!`, delta-method SEs) live in Margins.jl and are
tested there.

### API, caches and threads
- `test_api_contracts.jl`: exported names are defined; output buffer sizes are checked.
- `test_public_api.jl`: the public API Margins uses (accessors, hooks, private scratch).
- `test_threaded_level_cache.jl`: runs `threaded_level_cache_checks.jl` (not included
  directly) in a multithreaded subprocess.
- `test_compiled_cache.jl`: the bounded, clearable compiled-formula cache.

### Integrations
- `test_standardized_predictors.jl`, `test_centered_scaled_predictors.jl`:
  StandardizedPredictors.jl (`ZScore`, `Center`, `Scale`).
- `test_glm_integration.jl`: GLM families and links.
- `test_mixedmodels_integration.jl`, `test_mixedmodels_extension.jl`: MixedModels (fixed
  effects), and FormulaCompiler working without it.

### Performance
- `test_large_dataset_performance.jl`: allocations on 100K rows, 60-level factors and
  12-variable evaluators; evaluator memory independent of n.

### Support
- `support/testing_utilities.jl`: formula shapes and test data helpers.
- `support/generate_large_synthetic_data.jl`: the synthetic dataset used by the batch tests.

## Adding tests

Put a test next to the feature it checks and add it to `runtests.jl`. Prefer exact checks
against an independent computation (`modelmatrix`, a hand gradient) over finiteness or `!=`
checks, and measure allocations as described in `TEST_PRINCIPLES.md`.
