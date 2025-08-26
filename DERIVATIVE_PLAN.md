# Derivative System Plan

## Objectives

- Build high-performance derivatives for model rows to support Margins.jl.
- Start with robust numerical methods; add ForwardDiff-based Jacobians when practical.
- Handle continuous variables via derivatives; handle discrete variables via contrasts.

## Scope

- Target: derivative of the model matrix row values with respect to inputs (per row), i.e., Jacobian d row_terms / d variables.
- Continuous variables: partial derivatives.
- Discrete variables: contrasts (finite differences between levels or states).
- Out of scope (for v1): analytical derivatives; GLM mean-link derivatives (layer later on top of X and η).

## API Design

- Core
  - `derivative_modelrow!(J, compiled, data, row; vars=:all_continuous, method=:fd, step=:auto)`
    - In-place Jacobian with shape `(n_terms, n_vars)`.
  - `derivative_modelrow(compiled, data, row; ...)`
    - Allocating convenience wrapper.
  - `contrast_modelrow!(Δ, compiled, data, row; var, from, to)`
    - Discrete contrast vector: `Δ = Xrow(to) − Xrow(from)`.

- Variable selection
  - `vars`: `:all_continuous` (discovered from compiled ops), `[:x, :z]`, or prepared variable descriptors (to later support StandardizedPredictors chain rule).

- Targets
  - v1: model row (X) only.
  - v2: helpers for `η = Xβ` via `Δη = J * β` and GLM mean `μ = g⁻¹(η)` with chain rule.

## Backends

- ForwardDiff (FDiff) [Primary]
  - Approach: Use Dual numbers on the row’s selected variables and compute the Jacobian of a closure that writes the model row into a reusable buffer.
  - Zero-allocation path: `ForwardDiff.jacobian!` with a prebuilt and concretely-typed `JacobianConfig`, preallocated `J` and `x`, and a fixed variable set per derivative evaluator.
  - Enablement: make compiled execution generic over element type `T<:Real` (parametric scratch; eliminate hard `Float64` casts). Convert contrast values via `convert(T, val)`.
  - Data injection (no per-call merges): prebuild and store per-eltype override vectors and merged `data_over` NamedTuples once; the AD closure only mutates `row` and `replacement` fields and calls compiled.
  - Typed caches (no Dict lookups on hot path): store `compiled_dual`, `rowvec_dual`, `overrides_dual`, `data_over_dual` as concretely-typed fields initialized once.
  - Typed closure: `g::DerivClosure{<:DerivativeEvaluator}` created post-init; `cfg::ForwardDiff.JacobianConfig{…}` concretely typed and reused.
  - Chunking: use `ForwardDiff.Chunk{N}` where `N = length(vars)`; allow `chunk=:auto`.

- Finite Difference (FD) [Fallback]
  - Central difference per variable: `∂X/∂x ≈ (X(x+h) − X(x−h)) / (2h)`.
  - Step size: `h = c * max(1, |x|)` with `c ≈ eps(Float64)^(1/3)`; allow overrides per variable.
  - Performance: reuse buffers; two evaluations per variable; useful where Duals aren’t supported or for cross-checks.
  - Row-local perturbation: lightweight wrapper that overrides only the queried row’s value (SingleRowOverride) to avoid copying/global overrides.

## Discrete Contrasts

- API: `contrast_modelrow!(Δ, compiled, data, row; var=:group, from="A", to="B")` computes `Xrow(to) − Xrow(from)`.
- Implementation: use the existing scenario/override utilities or a row-local override wrapper (since we evaluate one row, an `OverrideVector` override is acceptable and simple).
- Batch: support returning a small matrix for multiple `(from, to)` pairs if requested.

## Performance Plan

- Reuse buffers: one output vector per evaluation; reuse contrast/derivative scratch where possible.
- Minimize wrapper overhead: pre-create row-local override objects per variable and reuse across steps/h calls.
- Optional parallelism later: parallelize across variables for FD when many continuous vars; ForwardDiff chunking already vectorizes directions.
- No per-call merges: never rebuild the merged `data_over` NamedTuple during evaluation; mutate prebuilt override vectors only.
- Typed fields end-to-end: avoid `Any`/`Dict{DataType,Any}` on hot paths; keep closure and config concretely typed.
- Benchmarks: report allocations (target ~0 per evaluation path) and timings vs model size.

## Integration Points

- Variable discovery: utility to list continuous variables present in compiled ops (exclude intercept, pure categorical encodings, etc.).
- StandardizedPredictors: option `return_space = :transformed | :original` (later). v1: document derivatives are w.r.t. transformed variables; v2: chain rule using stored transform metadata.
- Mixed models: derivatives target the fixed-effects design row (as current compilation does).
- Scenarios: all derivative/contrast APIs accept scenario data and behave identically.

## Testing Strategy

- Correctness
  - Compare FD central differences vs smaller step sizes (sanity convergence).
  - Where feasible, compare ForwardDiff vs FD with tight tolerances.
  - Validate against `modelmatrix(model)` finite differences on simple formulas.
  - Edge cases: non-differentiable points (abs, max); document and test stability.

- Coverage
  - Continuous-only, categorical-only, interactions (`x * group`), nested functions (`log`, `exp`, `sqrt`), standardized predictors, mixed models (fixed part).

- Performance
  - Allocation checks via BenchmarkTools; target zero allocations in steady-state FD path.
  - Scaling tests with increased number of variables and terms.

## Milestones

1. M0: Confirm API (orientation, names, options) and variable selection semantics.
2. M1: Generic-typing pass for compiled execution (parametric `T` scratch; remove `Float64` casts; `convert(T, val)` for contrasts).
3. M2: ForwardDiff-based `DerivativeEvaluator` with row-local wrappers and concretely-typed fields; prebuild per-eltype overrides + merged `data_over`; typed closure + `JacobianConfig`; implement `derivative_modelrow!` (FDiff) without per-call merges.
4. M3: Zero-allocation verification and performance benchmarks; eliminate residual allocations in the AD path.
5. M4: FD fallback backend (central differences) using the same wrappers; implement discrete contrasts API.
6. M5: Batch interfaces (multiple rows) with in-place Jacobian blocks; ensure memory reuse.
7. M6: Docs and examples (marginal effects via `Δη = J * β`; GLM `μ` extension outline) and comprehensive tests.
8. M7: Polish pass — docs (docstrings, guide examples, FAQ), broaden tests (LM/GLM/LMM, standardized predictors), assert zero allocations after warmup across scenarios, API surface review (keep helpers internal), short performance section and MixedModels note.

## Open Questions

- Orientation: `J::Matrix{Float64}(n_terms, n_vars)` acceptable? (Alternative: `n_vars × n_terms`).
- Default backend: prefer `:fd` for now? Add `method=:auto` to select ForwardDiff when `n_vars` is small and ops are Dual-compatible.
- Discrete contrast API: return both `Δ` and the “to” row for convenience?
- Standardized predictors: include chain rule now or defer to v2?

## Current Status

- Executor: Generic over `T<:Real` and Dual-safe; contrasts convert to `T`.
- ForwardDiff path: Prebuilt overrides and merged data; typed closure/config; no per-call merges; steady-state allocations at/near zero.
- FD fallback: Central differences; used for cross-checks and robust baseline.
- Contrasts: Discrete contrasts for categorical variables via row-local overrides (values normalized to CategoricalValue levels).
- Marginal effects: Helpers for η = Xβ and μ via Identity/Log/Logit.

## Polish Plan

- Docs polish
  - Add docstrings for: `build_derivative_evaluator`, `derivative_modelrow!`, `derivative_modelrow`, `derivative_modelrow_fd!`, `derivative_modelrow_fd`, `contrast_modelrow!`, `contrast_modelrow`, `continuous_variables`, `marginal_effects_eta/μ`.
  - Expand guide with GLM(Logit) marginal effects example and a minimal MixedModels fixed-effects example.
  - Add FAQ: choosing vars (`continuous_variables`), chunk sizing (`Chunk{N}` vs `:auto`), derivatives vs contrasts.
- Tests polish
  - LM + GLM(Logit) + MixedModels fixed-effects derivatives/contrasts.
  - Case with standardized predictors (document chain rule note for v2).
  - Multi-variate contrasts, zero-allocation assertions after warmup across scenarios.
- API clarity
  - Keep `SingleRowOverrideVector` internal; ensure exports match intended surface.
  - Import `LinearAlgebra.mul!` where used; keep naming consistent with codebase. (DONE, imported in FormulaCompiler.jl)
- Performance notes
  - Short section: compile once, zero-alloc steady-state; example benchmark snippet.
  - MixedModels note: derivatives target fixed effects only.
