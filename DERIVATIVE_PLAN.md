# Derivative System Plan (Updated)

Updated: August 26, 2025 (incorporates allocation findings from DERIVATIVE_ALLOC_REPORT.md and follow‑up assessment)

## Objectives

- Deliver fast, allocation‑free or near‑free derivatives for per‑row model evaluation to support Margins/marginal effects.
- Prefer designs that preserve the zero‑allocation guarantees of core modelrow evaluation.
- Provide robust finite‑difference fallback and clear guidance for when to use each backend.

## Scope

- Target: Jacobian ∂Xrow/∂vars for a single row; marginal effects on η = Xβ and μ via link chain rule; discrete contrasts.
- Continuous variables: partial derivatives (AD or FD). Discrete: contrasts.
- Out of scope (short term): fully analytical derivative compiler (tracked as long‑term path).

## API Design

- Core
  - `build_derivative_evaluator(compiled, data; vars, chunk=:auto)` → reusable evaluator with preallocated buffers.
  - `derivative_modelrow!(J, evaluator, row)` / `derivative_modelrow(evaluator, row)` → Jacobian.
  - `derivative_modelrow_fd!` / `derivative_modelrow_fd` → finite differences fallback.
  - `contrast_modelrow!` / `contrast_modelrow` → discrete contrasts.
  - `marginal_effects_eta!/η` and `marginal_effects_mu!/μ` → marginal effects.

- Variable selection
  - `vars`: `continuous_variables(compiled, data)` or explicit `[:x, :z]`.

## Backends and Design Choices

- ForwardDiff (primary when full J is needed)
  - Approach: Dual‑typed closure that fills a reusable row buffer; `ForwardDiff.jacobian!` with a concrete `JacobianConfig` and preallocated `J` and `x`.
  - Critical typing requirements to minimize allocations:
    - Concrete evaluator fields for closure and config (no `Any` on hot path).
    - Eagerly materialize concrete `compiled_dual` and `rowvec_dual` and reuse them.
    - Typed, per‑eltype overrides `SingleRowOverrideVector{T}` and per‑eltype merged `data_over` (Float64 and Dual) so `getindex` returns `T`.
    - Column cache stored in a concrete container (no `Vector{Any}`) to avoid Any leaks.
  - Chunking: use `ForwardDiff.Chunk{N}` with `N = length(vars)`; allow `:auto`.

- ForwardDiff gradient for η (preferred for marginal effects)
  - Compute `g = ∂η/∂x` via `ForwardDiff.gradient!` of `h(x) = dot(β, Xrow(x))` with a concrete `GradientConfig`.
  - Easier to achieve true zero allocations than full Jacobian; avoids explicit `J` and post multiply.

- Finite Differences (robust fallback and cross‑check)
  - Central differences with preallocated buffers; generated inner loops; cached column references.
  - Designed to be truly zero‑allocation when types are fully concrete. Retain a “mutate column” internal variant for experiments only (unsafe for public API).

## Discrete Contrasts

- `contrast_modelrow!` computes `Δ = Xrow(to) − Xrow(from)` using a row‑local override; categorical values normalized to column levels.
- Provide small convenience utilities for batching multiple `(from, to)` pairs when needed.

## Performance and Allocation Targets

- Core modelrow: exactly 0 allocations after warmup (already achieved).
- Finite differences Jacobian: 0 allocations after warmup (with concrete types throughout).
- ForwardDiff Jacobian: target 0 allocations after warmup with fully concrete evaluator; accept ≤112–192 bytes on environments where FD internals allocate. Document environment dependence.
- Marginal effects η (gradient): target 0 allocations with concrete gradient config; use this when only η gradients are needed.
- Marginal effects μ: reuse preallocated buffers; expect near‑zero allocations, scaled by derivative method used for η.

## Integration Points

- `continuous_variables(compiled, data)`: discover numeric variables (exclude categorical encodings) from compiled ops.
- StandardizedPredictors: document that derivatives are w.r.t. transformed variables now; plan chain‑rule helpers later.
- Mixed models: derivatives target fixed‑effects design (as compilation does).
- Scenarios: all APIs accept scenario/override data and behave identically; avoid per‑call merges.

## Testing Strategy

- Correctness
  - FD convergence checks; FD vs ForwardDiff agreement on simple models.
  - Compare against `modelmatrix(model)` FD on simple formulas.
  - Edge cases: non‑smooth ops (abs, max); document behavior.

- Coverage
  - Continuous, categorical, interactions, nested functions, standardized predictors, mixed models (fixed part).

- Performance/Allocations
  - Assert: modelrow == 0 allocs; FD Jacobian == 0 allocs; η‑gradient path == 0 allocs (post‑warmup).
  - ForwardDiff Jacobian: assert ≤112–192 bytes unless CI environment achieves 0; flag and record environment details.

## Milestones

1. M0: Confirm API and Jacobian orientation; finalize variable selection utility semantics.
2. M1: Concrete typing pass for derivative evaluator:
   - Concrete fields for closure/config; eager `compiled_dual`/`rowvec_dual`.
   - Typed overrides `SingleRowOverrideVector{T}` and per‑eltype `data_over`.
   - Concrete column cache.
3. M2: Implement η‑gradient path with `ForwardDiff.gradient!`, concrete `GradientConfig`, and tests (aim 0 allocs).
4. M3: Tighten FD Jacobian path to 0 allocations across CI; add tests that enforce it.
5. M4: ForwardDiff Jacobian hardening: pursue 0 allocations on supported Julia/FD combos; document and gate thresholds where internals allocate.
6. M5: Docs and examples: when to use FD vs AD; η‑gradient recommended path; environment note on FD internals.
7. M6: Analytical derivatives roadmap (design doc + small prototype covering a subset of ops).

## Open Questions

- Universal 0‑alloc guarantee for ForwardDiff Jacobian: feasible to promise, or document environment‑dependent floor?
- Public exposure of “mutate column” FD variant (likely no; keep internal/testing only).
- Provide a cache for evaluators keyed by `(compiled, vars, method)` or leave lifecycle explicit.

## Current Status (Reality Check)

- Executor is `T`‑parametric and Dual‑safe; core modelrow is 0‑alloc.
- FD Jacobian path is architected for 0‑alloc; remaining allocs observed in some experiments are attributable to Any‑typed edges and are actionable.
- ForwardDiff Jacobian presently near‑zero (tests enforce ~112–144 bytes); removing Any on the hot path is the next lever to close the gap, though a universal 0 across environments may still be brittle.
- η‑gradient path is the recommended zero‑alloc route for marginal effects where full J is not required.

## Notes from DERIVATIVE_ALLOC_REPORT.md

- Preallocation, cached columns, generated loops, and hardcoded constants reduced FD allocations from 1,872 → 192 bytes.
- The observed “192‑byte floor” is likely due to remaining Any‑typed plumbing (e.g., `Vector{Any}` column cache, Any‑typed overrides/fields) rather than an inherent runtime floor.
- Action: eliminate Any on hot paths before concluding a true floor exists; document environment variance for ForwardDiff internals.

## Implemented (snapshot)

- FD evaluator Jacobian: strict 0‑alloc after warmup via positional hot path and unrolled access; tight-loop benchmark (100k calls) shows 0.
- AD Jacobian and η‑gradient: small, env‑dependent allocations remain (≤256 bytes) with hoisted configs and typed closures.
- Allocation testing consolidated in `test/test_derivative_allocations.jl`; CSV written to `test/derivative_allocations.csv`.
- Variance/SEs workflow documented in `VARIANCE.md` (delta method with single‑column FD J and in‑place gradients).

## Acceptance Criteria

- Core modelrow
  - Allocations: 0 bytes per call after warmup.
  - Timing: ~50ns per simple row (document machine + Julia version).

- Finite differences Jacobian (per row)
  - Allocations: 0 bytes per call after warmup with fully concrete evaluator (typed overrides and column cache).
  - Correctness: ≈ AD Jacobian within rtol=1e-6, atol=1e-8 on smooth cases.
  - Timing: Within 2× of current optimized FD implementation for small n_vars; scales linearly with n_vars.

- ForwardDiff Jacobian (per row)
  - Allocations: 0 bytes per call on supported env (see Environment Matrix) with fully concrete evaluator; otherwise ≤112–256 bytes per call.
  - Correctness: Matches FD central differences within rtol=1e-6, atol=1e-8.
  - Timing: Competitive with or faster than FD central differences for small to moderate n_vars; document chunking impact.

- Marginal effects η (gradient)
  - Allocations: 0 bytes per call with concrete `GradientConfig` and evaluator.
  - Correctness: Matches `transpose(J)*β` within rtol=1e-10 on smooth cases.
  - Timing: Faster than AD Jacobian+multiply for typical n_vars; document crossover.

- Marginal effects μ
  - Allocations: No additional allocations beyond η path; link scaling is allocation‑free.
  - Correctness: Chain rule consistency checks vs analytic link derivatives.

- Documentation and Tests
  - Tests assert the above allocation caps and correctness on CI.
  - README/docs mention environment caveats for ForwardDiff Jacobian.

## Environment Matrix (Expectation, Not Enforcement)

- Julia 1.10 / ForwardDiff X.Y
  - AD Jacobian: expected 0 allocations with fully concrete evaluator.
  - η‑gradient: expected 0 allocations.

- Julia 1.11 / ForwardDiff X.Y
  - AD Jacobian: typically ≤112–192 bytes even when concrete (FD internals); aim for 0 when possible.
  - η‑gradient: expected 0 allocations.

Notes:
- Record actual results in CI logs; do not hard‑fail if AD Jacobian allocates ≤192 bytes on newer runtimes with known FD internals.
- Keep tests strict where stable (FD Jacobian 0‑alloc, η‑gradient 0‑alloc); gate AD Jacobian to caps when needed.

## Typing Checklist (Must‑Have for 0‑Alloc Paths)

- Evaluator fields
  - Closure `g` stored with concrete type parameter; no `Any`.
  - Config `cfg::JacobianConfig` / `GradientConfig` stored concretely.
  - `compiled_dual` and `rowvec_dual` eagerly constructed and concrete.

- Overrides and data
  - `SingleRowOverrideVector{T}` for Float64 and Dual; avoid `AbstractVector{Any}`.
  - Per‑eltype merged `data_over` so `getindex` yields `T` without conversion.

- Column cache
  - Avoid `Vector{Any}`; store as `Vector{<:AbstractVector{T}}` or an `NTuple` for fixed small `nvars`.

- Execution
  - Generated or @inline loops with `@inbounds @fastmath`; no iterators on hot paths.
  - No per‑call merges, no dynamic dispatch, no hidden conversions.

## Profiling Checklist (Per Function)

1. `@code_warntype` on evaluator constructors; ensure no red (Any) on hot fields.
2. `@code_warntype` on `derivative_modelrow!`/`marginal_effects_eta!`/FD paths.
3. `@allocated` after two warm calls; confirm targets (0 or cap).
4. If nonzero allocations persist:
   - Inspect field types and column cache eltypes.
   - Inline small helpers; remove remaining assertions/iterators.
   - Rebuild configs (`JacobianConfig`/`GradientConfig`) to ensure concrete typing.

## η‑Gradient API Guidance

- When to use
  - You need marginal effects on η = Xβ (not the full Jacobian).
  - You care about strict zero allocations and/or speed.

- Usage sketch
  - Build evaluator once with fixed `vars`.
  - Preallocate `g::Vector{Float64}`.
  - Call gradient‑based η marginal effects function (backed by `ForwardDiff.gradient!`).

- Notes
  - This avoids computing/allocating the full Jacobian and the subsequent multiply.
  - For μ, apply link scaling to `g` (already allocation‑free).

## Benchmarks

- Models
  - Simple LM with `x + z + x&group` (2 continuous, one interaction).
  - Moderate LM/GLM with 5–20 continuous predictors.

- Metrics
  - Allocations: per call after warmup for each function.
  - Timing: ns/μs per call; rows/sec for bulk runs.

- Targets
  - Core modelrow: ~50ns, 0 allocs.
  - FD Jacobian: 0 allocs; time scales with n_vars; within 2× current optimized baseline.
- AD Jacobian: ≤ FD time for small n_vars; allocs 0 on supported envs or ≤112–192 bytes.
- η‑gradient: < AD Jacobian+multiply; 0 allocs.

## Implementation Sequence (Low-Risk Work Order)

Phase 1: Typed Overrides (foundational)
- Goal: Remove `Any` in overrides and merged data for Float64 paths first.
- Changes:
  - Add `SingleRowOverrideVector{T} <: AbstractVector{T}` with fields `base::AbstractVector{T}`, `row::Int`, `replacement::T`.
  - Add `build_row_override_data_typed(base, vars, row, ::Type{T}) -> (data_over, overrides)` to produce typed NamedTuples and override vectors.
  - Use typed overrides + merged data for Float64 FD paths immediately; defer Dual path to Phase 3.
- Expected outcome: FD Jacobian easier to keep 0‑alloc; no API changes.

Phase 2: η‑Gradient Path (deliver zero‑alloc marginal effects)
- Goal: Provide a fast, 0‑alloc path for marginal effects on η without building full J.
- Changes:
  - Implement `marginal_effects_eta_grad!`/`marginal_effects_eta_grad` using `ForwardDiff.gradient!` on `h(x) = dot(β, Xrow(x))`.
  - Reuse evaluator buffers; pass `β` each call (closure may hold `Ref` if needed without realloc).
  - Tests: assert 0 allocations after warmup; compare to `transpose(J) * β`.
- Expected outcome: Immediate zero‑alloc marginal effects path for η.

Phase 3: Dual‑Typed Paths (typed data_over + overrides + rowvec/compiled)
- Goal: Remove `Any` on AD path.
- Changes:
  - Lazily build Dual‑typed `overrides_dual::Vector{SingleRowOverrideVector{Tx}}` and `data_over_dual` keyed by `Tx`, alongside `compiled_dual` and `rowvec_dual`.
  - Ensure AD closure updates the correct (Float64/Dual) sets.
- Expected outcome: ForwardDiff Jacobian becomes fully concrete except FD internals; reduces residual allocations on many envs.

Phase 4: Column Cache Typing (reduce `Any` in FD and AD)
- Goal: Eliminate `Vector{Any}` column cache.
- Changes:
  - Option A: `Vector{AbstractVector{<:Real}}` for simplicity and concrete field type.
  - Option B: `NTuple{N,AbstractVector{Tᵢ}}` when `nvars` is small and fixed for stronger specialization.
- Expected outcome: More consistent inference in hot loops; no per‑call overhead.

Phase 4.5: FD evaluator hot-path (strict 0‑alloc)
- Goal: Guarantee 0 allocations per call for FD evaluator.
- Changes:
  - Add positional hot-path (`derivative_modelrow_fd_pos!`) that forwards to internal functions without keyword overhead.
  - Use fully concrete FD overrides and unrolled NTuple column access in generated FD evaluator.
- Expected outcome: FD evaluator 0‑alloc after warmup; confirmed via BenchmarkTools and tight-loop tests.

Phase 5: AD g/cfg Concrete Storage (careful about type cycles)
- Goal: Make `g` and `cfg` fields concrete without introducing type cycles.
- Approach:
  - Keep `DerivClosure` around `RefValue{DE}`.
  - Parameterize evaluator with extra type params `G`/`C` or use an internal constructor to set concrete `g::G` and `cfg::C` after building other fields.
  - Alternatively, two‑step builder: construct a shell evaluator, build `g`/`cfg`, then finalize the evaluator with those concrete fields.
- Expected outcome: AD Jacobian hits 0‑alloc on more envs; otherwise remains ≤112–192 bytes.

Tests and Policy Alignment
- Update tests to:
  - Assert FD Jacobian 0‑alloc after warmup.
  - Assert η‑gradient 0‑alloc after warmup.
  - Cap AD Jacobian at ≤112–192 bytes unless env shows 0 allocations (see Environment Matrix).
- Keep μ thresholds tied to η method.
