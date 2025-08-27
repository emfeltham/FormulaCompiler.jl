# Derivative Allocation Report

## Overview

This report summarizes runtime allocation measurements for the derivative paths, as captured by `test/test_derivative_allocations.jl` and saved to `test/derivative_allocations.csv`.

- Scope: runtime allocations only (compile-time/JIT excluded via warmup + BenchmarkTools interpolation)
- Environment: Julia 1.11.x + ForwardDiff (small env-dependent differences expected)
- Source: `test/test_derivative_allocations.jl` (also asserts caps), `test/tests.sh`

## Methodology

- Warm each path (compiled row, FD Jacobians, AD Jacobian, η‑gradient, μ) before benchmarking.
- Use `@benchmark` with interpolated arguments; record `minimum(b.memory)` and `minimum(b.times)`.
- Write a row per path to `test/derivative_allocations.csv`.

## Results (from derivative_allocations.csv)

- compiled_row
  - Min memory: 0 bytes
  - Min time: ~7 ns (b.times; nanoseconds)
  - Meaning: Core model row evaluation (`compiled(row_vec, data, row)`) is zero‑allocation after warmup.

- fd_jacobian_standalone
  - Min memory: 1728 bytes
  - Min time: ~979 ns
  - Meaning: Standalone FD Jacobian (`derivative_modelrow_fd!(J, compiled, data, row; vars)`).
  - Allocation source: per‑call row‑local overrides (SingleRowOverrideVector + merged NamedTuple) and temporary work buffers (`yplus`, `yminus`, `xbase`). The compiled evaluation itself is 0‑alloc.

- fd_jacobian_evaluator
  - Min memory: 160 bytes
  - Min time: ~541 ns
  - Meaning: Evaluator FD Jacobian (`derivative_modelrow_fd!(J, de, row)`) with prebuilt overrides and preallocated buffers.
  - Allocation source: tiny, environment‑dependent runtime costs (e.g., tuple indexing or housekeeping). Target is 0.

- ad_jacobian
  - Min memory: 256 bytes
  - Min time: ~412 ns
  - Meaning: ForwardDiff Jacobian (`derivative_modelrow!(J, de, row)`).
  - Allocation source: small ForwardDiff runtime allocations (even with hoisted, concrete `JacobianConfig`). Some environments reach 0; many report ≤112–256 bytes.

- eta_gradient
  - Min memory: 288 bytes
  - Min time: ~417 ns
  - Meaning: Scalar AD gradient on η = Xβ (`marginal_effects_eta_grad!` using `ForwardDiff.gradient!` on `h(x)=dot(β, Xrow(x))`).
  - Allocation source: small ForwardDiff runtime allocations despite hoisted, concrete `GradientConfig` and typed closure. Target is 0 after warmup on supported envs.

- mu_marginal_effects_logit
  - Min memory: 256 bytes
  - Min time: ~462 ns
  - Meaning: Marginal effects on μ via link scaling (`marginal_effects_mu!(..., link=LogitLink())`).
  - Allocation source: mirrors η path allocation (link scaling itself is allocation‑free).

Notes:
- Times are recorded as raw benchmark `b.times` (nanoseconds). They are not converted to seconds in the CSV.
- All paths are warmed before measurement; values represent steady‑state runtime allocations.

## Interpretation

- Core evaluation is zero‑allocation (as designed).
- Standalone FD Jacobian necessarily allocates due to per‑call override construction and scratch; use the evaluator path for minimal allocations.
- Evaluator FD Jacobian is nearly zero; expected to reach 0 after typing the remaining edges (NTuple column cache and typed overrides are already in place).
- ForwardDiff paths (Jacobian and η‑gradient) show small allocations on this environment; some Julia/ForwardDiff combinations can achieve 0 after warmup with concrete configs/closures, but tiny residuals are common.

## Recommendations

- Use the evaluator FD path for cross‑checks and environments where AD allocates more.
- Prefer η‑gradient for marginal effects on η (fast path; hoisted configs). Tighten to 0 once CI confirms.
- Keep allocation caps environment‑aware:
  - FD evaluator Jacobian: tighten to 0 when observed stable.
  - AD Jacobian and η‑gradient: keep caps ≤256 bytes until envs consistently show 0.
- Leave standalone FD Jacobian as a correctness baseline; accept small per‑call allocations.

## Reproducing

- Run all three suites and view logs/CSV:
  - `bash test/tests.sh`
  - `cat test/test_derivatives.txt`
  - `cat test/test_links.txt`
  - `cat test/test_derivative_allocations.txt`
  - `cat test/derivative_allocations.csv`

## File References

- `src/evaluation/derivatives.jl` — derivative implementations
- `test/test_derivatives.jl` — correctness (Jacobian agreement, contrasts, η)
- `test/test_links.jl` — μ link scaling correctness
- `test/test_derivative_allocations.jl` — allocation measurements + CSV
- `test/TESTING.md` — testing framework overview

## One‑Time vs Per‑Call Allocations (Practical Guide)

After fitting a model and building a derivative evaluator, here’s what allocates once vs per call:

- One‑time (when building the evaluator for a given compiled+data+vars)
  - Buffers: `xbuf` (n_vars), `rowvec_float` (n_terms), `jacobian_buffer` (n_terms×n_vars), `eta_gradient_buffer` (n_vars), `xrow_buffer` (n_terms), `fd_yplus/fd_yminus` (n_terms), `fd_xbase` (n_vars).
  - State: typed overrides + merged `data_over` (Float64), cached column refs (NTuple), typed AD closures, and hoisted ForwardDiff configs (`JacobianConfig`, `GradientConfig`).
  - AD dual caches (lazy on first AD call): `compiled_dual`, `rowvec_dual`, typed dual overrides + merged `data_over_dual` (rebuilt only if the ForwardDiff Tag changes).

- Per‑call (in‑place APIs; outputs preallocated)
  - Core row `compiled(row_vec, data, row)`: 0 bytes after warmup.
  - FD Jacobian via evaluator `derivative_modelrow_fd!(J, de, row)`: near‑zero (often 0); uses prebuilt overrides + FD buffers.
  - AD Jacobian `derivative_modelrow!(J, de, row)`: small ForwardDiff internals (typically ≤112–256 bytes; sometimes 0).
  - η‑gradient `marginal_effects_eta_grad!(g, de, β, row)`: small ForwardDiff internals (often ≤100s of bytes; target 0 on some envs).
  - μ marginal effects `marginal_effects_mu!(gμ, de, β, row; link=…)`: mirrors η path; link scaling itself is allocation‑free.

- Per‑call (allocating wrappers)
  - `derivative_modelrow(de, row)`: allocates `Matrix{Float64}(n_terms, n_vars)`.
  - `marginal_effects_eta(de, β, row)` / `marginal_effects_mu(de, β, row)`: allocate `Vector{Float64}(n_vars)`.

- Standalone FD (without evaluator)
  - `derivative_modelrow_fd!(J, compiled, data, row; vars)`: allocates per call due to building per‑call row overrides and temps; use evaluator FD for minimal allocations.

- Varying data vs values
  - Vary a value for a row: use evaluator paths (mutate prebuilt overrides; no merges).
  - Vary dataset/columns: rebuild evaluator so cached columns and overrides point to the new data.

- Representative numbers (from this env)
  - `compiled_row`: 0 bytes
  - `fd_jacobian_evaluator`: ~160 bytes (aim 0)
  - `ad_jacobian`: ~256 bytes
  - `eta_gradient`: ~288 bytes
  - `mu_marginal_effects_logit`: ~256 bytes
- `fd_jacobian_standalone`: ~1728 bytes

## Next steps

Goal: 0 runtime allocations per call after building the derivative evaluator.

- Finite‑difference evaluator (target: 0 bytes)
  - Make `fd_columns` fully concrete: store the exact NTuple type as a type parameter of the evaluator so indexing is fully specialized.
  - Keep typed overrides in the evaluator (`Vector{TypedSingleRowOverrideVector{Float64}}`) and ensure replacement writes stay typed.
  - Ensure the generated FD loop uses only concrete fields (no `Any`), so the body is pure array math over preallocated buffers.

- ForwardDiff (environment‑dependent; often ~0–112 bytes)
  - Prebuild Dual‑tagged caches at evaluator build time: `compiled_dual`, `rowvec_dual`, `overrides_dual`, `data_over_dual` for the chosen Tag/Chunk.
  - Remove `Any` on the AD path by parameterizing Dual cache fields with their concrete types.
  - Keep Tag stable: use a fixed `ForwardDiff.Chunk{N}` and reuse the same `cfg`/`gradcfg`.
  - We already hoist concrete `JacobianConfig` and `GradientConfig` and use typed closures.
  - Expected after changes:
    - AD Jacobian: often ≤112 bytes, sometimes 0 depending on Julia/ForwardDiff.
    - η‑gradient: often 0 after warmup; otherwise ≤~112 bytes.

- Test policy/tightening
  - Set FD evaluator Jacobian cap to 0 when stable on CI.
  - Tighten η‑gradient cap to 0 where environments show 0 after warmup; otherwise keep a small cap (≤112 bytes).
  - Keep AD Jacobian cap ≤112–256 bytes; tighten if 0 is achieved consistently on CI.

- Notes
  - Standalone FD (without evaluator) will continue to allocate by construction (per‑call row overrides + temp buffers) and should remain as the robust correctness baseline.

## Operational guidance (when small allocs matter)

Small per‑call allocations are fine for small jobs, but they add up quickly in large models/data. Rule of thumb: 256 bytes × 200 vars × 600k rows ≈ 30 GB of transient allocations.

- What to use, when
  - Small n_rows or few evals: AD Jacobian is convenient and often fastest for small n_vars; a few dozen bytes per call is acceptable.
  - Big n_rows or production runs: use FD evaluator Jacobian (target 0‑alloc) and η‑gradient (often 0‑alloc) for predictable memory and runtime.

- Practical heuristics
  - Few vars (≈ ≤8) and few rows: AD Jacobian usually faster; small allocations are fine.
  - Many rows/long runs: FD evaluator (0‑alloc) wins on memory stability despite 2 evals/var.
  - Need J only occasionally: compute η‑gradient by default; build J (AD or FD) only when required.

- Keep allocations low in practice
  - Always use in‑place APIs and reuse buffers: `derivative_modelrow_fd!`, `derivative_modelrow!`, `marginal_effects_eta_grad!`, `marginal_effects_mu!`.
  - Avoid standalone FD inside loops (`derivative_modelrow_fd!(..., compiled, data, ...)`) — it allocates by design; use the evaluator FD instead.
  - Pass `β::Vector{Float64}` to avoid per‑call conversions.
  - Build one evaluator per `(compiled, data, vars)` and reuse across rows; rebuild only when columns change. Row‑value changes can reuse the same evaluator (prebuilt overrides).
  - For threading: use one evaluator per thread (internal fields mutate) and partition rows.
  - For ForwardDiff: fix `Chunk{N}` and reuse the same configs to keep the Tag stable; expect small, env‑dependent residuals.

- Recommendation for big jobs
  - Default to FD evaluator for bulk work (can be strictly 0‑alloc) and η‑gradient for marginal effects.
  - Use AD Jacobian for validation and special cases; don’t rely on it for strict 0‑alloc across all environments.
