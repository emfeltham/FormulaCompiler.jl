# Testing Framework

This repository separates correctness tests from allocation/performance tests to keep results clear and actionable.

## How To Run

- All tests
  - `julia --project=. -e 'using Pkg; Pkg.test()' > test/test.txt 2>&1`
  - or `julia --project=. test/runtests.jl` > test/runtests.txt 2>&1`
- Individual suites (examples)
  - Root project + bootstrap
    - `julia --project=. -e 'include("test/bootstrap.jl"); _fc_load_testdeps(); include("test/test_derivatives.jl")'`
    - `julia --project=. -e 'include("test/bootstrap.jl"); _fc_load_testdeps(); include("test/test_links.jl")'`
    - `julia --project=. -e 'include("test/bootstrap.jl"); _fc_load_testdeps(csv=true); include("test/test_derivative_allocations.jl")'`
  - Test project
    - `julia --project=test -e 'include("test/bootstrap.jl"); _fc_load_testdeps(); include("test/test_derivatives.jl")'`

## Suite Organization

- Correctness (feature‑focused)
  - `test_derivatives.jl`: Jacobians (AD vs FD), discrete contrasts, marginal effects on η (gη = J'β).
  - `test_derivatives_extended.jl`: GLM(Logit) and MixedModels variants; AD vs FD agreement.
  - `test_links.jl`: Marginal effects on μ; validates link scaling (Identity/Log/Logit/…)
  - Other suites cover position mapping, scenarios/overrides, models.
- Allocations (runtime only)
  - `test_allocations.jl`: Sole home for allocation/performance checks; uses BenchmarkTools to measure runtime allocations only (not compile‑time).

## Allocation Testing Methodology

- Framework: `BenchmarkTools` with variable interpolation and warmup to isolate runtime.
  - Warm all paths first (compiled(row), FD/AD Jacobians, η‑gradient, μ).
  - Measure with `@benchmark ...` and assert on `minimum(b.memory)` for steady‑state allocations.
  - Interpolation (`$var`) avoids benchmarking global variable access.
- What we measure
  - `compiled(row_vec, data, row)`: Core model row evaluation.
  - Finite‑difference Jacobians: standalone (compiled+data) and evaluator path (generated FD with prebuilt state).
  - ForwardDiff Jacobian (vector AD).
  - η‑gradient (scalar AD on h(x)=dot(β, Xrow(x))).
  - μ marginal effects (η path + link scaling).
- What we do NOT measure
  - First‑run compilation/JIT — warmup occurs before benchmarking.
  - Build‑time or one‑time allocations during evaluator construction.

## Current Acceptance Targets (runtime)

- Core evaluation
  - `compiled(row_vec, data, row)`: 0 bytes after warmup.
- Finite differences
  - Standalone Jacobian: small, bounded allocations (per‑call row override build). Current cap ≤ 2048 bytes.
  - Evaluator Jacobian (generated FD): trending to 0; current cap ≤ 256 bytes (tighten as typing completes).
- ForwardDiff
  - Jacobian (vector AD): environment‑dependent small allocations; current cap ≤ 512 bytes.
  - η‑gradient (scalar AD): trending to 0 with hoisted configs; current cap ≤ 512 bytes (tighten as we confirm 0 on CI).
- μ marginal effects
  - Follows the η path + link scaling; current cap ≤ 512 bytes.

Notes:
- ForwardDiff internal behavior can vary across Julia/ForwardDiff versions; caps are set to be strict but realistic.
- When lower numbers are observed consistently in CI, caps will be tightened accordingly.

## Correctness Checks (high level)

- Derivatives
  - AD vs FD Jacobian agreement: `isapprox(J, J_fd; rtol=1e-6, atol=1e-8)` on representative models.
  - Discrete contrasts `Δ = X(to) − X(from)` validated against explicit override evaluation.
  - Marginal effects on η: `gη ≈ J'β` for selected rows.
- Links
  - μ link scaling for several links: Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt.
  - Spot‑checked analytic scale relationships (e.g., Log: `exp(η)`, Logit: `σ(η)(1−σ(η))`).

## Conventions & Tips

- Group by feature: keep correctness in feature suites; reserve allocations for `test_allocations.jl`.
- Seed RNG for stability: `Random.seed!` used in top‑level runners.
- Keep tests minimal and targeted — prefer small models and few rows for speed.
- Prefer `@testset` nesting to make failures easy to locate.
- For allocation tests, prefer BenchmarkTools over `@allocated` to avoid compile‑time noise.

## Adding New Tests

- Correctness
  - Place derivative/link correctness next to related suites (`test_derivatives*.jl`, `test_links.jl`).
  - Use small, representative models and assert numerical relationships.
- Allocation
  - Add new runtime allocation checks to `test_allocations.jl` only.
  - Warm paths, use `@benchmark`, assert on `minimum(b.memory)` and (optionally) timings.
  - Document environment caveats if ForwardDiff or Julia version specifics apply.

---

This structure keeps correctness easy to reason about and isolates performance/allocations in one place for reproducible, runtime‑only measurements.
