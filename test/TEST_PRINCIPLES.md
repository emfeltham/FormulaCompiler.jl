# Testing Principles for FormulaCompiler.jl

This document summarizes project-wide testing practices and patterns. It is intended for contributors writing or updating tests across correctness, allocations, and performance.

## Goals

- Correctness: Mathematical and API correctness across supported features and integrations.
- Allocations: Zero-allocation guarantees for core hot paths; bounded allocations where specified (e.g., AD paths).
- Performance: Validate fast steady-state execution after warm-up, not compilation time.
- Stability: Deterministic tests with reproducible randomness and robust tolerances.

## General Practices

- Use `@testset` and `@test` for all assertions. Avoid `println`/`@info` in CI-visible tests; `@debug` is acceptable for optional diagnostics.
- Seed randomness at the top of each file or `runtests.jl` using `Random.seed!` to ensure determinism.
- Keep test data self-contained; use `Tables.columntable(df)` where appropriate.
- Prefer small, representative datasets in CI. Reserve very large stress tests for opt‑in slow suites (see “Slow/extended tests”).
- Follow repository naming/organization: group tests by subsystem and feature area.

## Correctness Testing

- Compare against reference implementations (e.g., `modelmatrix(model)`) when validating compiled evaluators.
- Use appropriate tolerances:
  - FD vs AD numerical: `rtol=1e-6`, `atol=1e-8` unless a stronger justification exists.
  - Exact structural equality where appropriate (e.g., column counts, shapes).
- Cover edge cases: categorical levels, booleans, ordered categoricals, interactions, and scenario overrides.
- When adding new statistical functionality, include cross-validation between independent computational approaches where possible.

## Allocation and Performance Testing

### BenchmarkTools, not `@allocated`

- Use `BenchmarkTools.@benchmark` with interpolation to measure steady-state memory and time:

  ```julia
  b = @benchmark $compiled($buffer, $data, $row) samples=300 evals=1
  @test minimum(b.memory) == 0
  ```

- Always warm up before measuring (one or more calls) to exclude compilation/transient allocations from the measurement.
- Interpolate all variables (`$compiled`, `$buffer`, `$data`, `$row`) to avoid closure allocations in the benchmark itself.
- For looped patterns, benchmark the loop body directly or create a small helper function to avoid measurement artifacts.

### Targets

- Core evaluator paths (`compiled(output, data, row)` and `modelrow!`) must allocate 0 bytes after warm‑up.
- FD derivatives: 0 bytes after warm‑up. AD derivatives: small bounded allocations (≤ ~512 bytes) from ForwardDiff.
- Categorical mixtures: zero allocations on steady‑state evaluation.

### Bounds checks and `@inbounds`

- Tests validate allocation behavior at the API level; implementation may use `@inbounds` to improve CPU time. Allocation results should not rely on `@inbounds`.

## Categorical and Interaction Coverage

When adding or updating tests for categorical handling, include:

- Single categorical with many levels (≥ 5), verifying zero allocations.
- Continuous × categorical interactions (e.g., `x * group5`).
- Categorical × categorical interactions (e.g., `groupA * groupB`).
- Boolean paths: both `Vector{Bool}` and `CategoricalArray{Bool}`, including interactions (e.g., `x * flag`).
- Mixtures: ensure compilation to `MixtureContrastOp` and zero-allocation evaluation.

## Derivatives and Variance

- Test both backends (`backend=:fd` and `:ad`) where exposed.
- Validate numerical agreement (FD vs AD) on simpler models to the specified tolerances.
- For delta‑method SE and related variance primitives, validate formulas on small, known cases.

## Slow/Extended Tests

- Keep CI fast by using moderate `samples` and small `n`.
- Place large or long‑running tests behind an opt‑in guard, e.g.:

  ```julia
  if get(ENV, "FC_SLOW_TESTS", "0") == "1"
      @testset "Slow" begin
          # extended or large‑n benchmarks
      end
  end
  ```

## Integration Tests

- Validate GLM/MixedModels/StandardizedPredictors integration using real fitted models.
- Extract only necessary model components; keep tests minimal but representative.

## Adding New Tests

- Place new files under `test/` and register them in `test/runtests.jl`.
- Use helper utilities from `test/support/testing_utilities.jl` where reasonable.
- Favor small, focused `@testset`s with clear names describing the scenario under test.

## Anti‑Patterns to Avoid

- Measuring allocations with `@allocated` in hot paths (use BenchmarkTools instead).
- Printing in tests (use `@test`/`@testset`; `@debug` for optional info).
- Nondeterministic tests (unseeded randomness, time‑dependent behavior).
- Overly permissive allocation thresholds that can mask regressions.

Adhering to these principles helps preserve FormulaCompiler’s guarantees: zero‑allocation hot paths, robust numerical behavior, and stable performance across categorical, interaction, and mixture features.

