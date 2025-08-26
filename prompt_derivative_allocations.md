Title: Eliminate remaining allocations in ForwardDiff-based modelrow derivatives

Context

- Repo: FormulaCompiler.jl (unified position-mapping compiler, zero-alloc core).
- Core rows: modelrow! and compiled(row_vec, data, row) are exactly zero allocation after warmup.
- Derivatives: ForwardDiff path is near-zero but still shows tiny steady-state allocations on some
environments (e.g., 112–576 bytes per call). FD fallback is fine.

Goal

- Make ForwardDiff-based derivatives exactly zero allocation after warmup across environments:
    - marginal_effects_eta!/mu! using the same ForwardDiff evaluator
- Keep current architecture: prebuilt overrides + merged data, typed closure/config, no per-call
merges or Dict lookups.

Relevant files

- src/compilation/execution.jl
- src/evaluation/derivatives.jl
- src/evaluation/modelrow.jl
- test/test_derivatives.jl
- test/test_derivatives_extended.jl
- test/test_links.jl

Current implementation (high level)

- UnifiedCompiled{T, Ops, S, O} is Dual-safe (no Float64 hard-coding).
- DerivativeEvaluator prebuilds:
    - overrides::Vector{SingleRowOverrideVector} and a merged data_over NamedTuple (reused for
Float64 and Dual).
    - rowvec_float, and lazily rowvec_dual + compiled_dual.
- DerivClosure holds a Ref to the evaluator to keep closure type concrete; JacobianConfig is built
once with that closure.
- No per-call NamedTuple merges; closure only mutates overrides and calls compiled.

Observed issues

- On some environments, derivative_modelrow! still allocates ~112–256 bytes per call;
marginal_effects_mu! with links up to ~768 bytes.

Tasks

1. Diagnose allocations

- Identify exact sources using @allocated, @code_warntype, and @profile where useful.
- Confirm if allocations originate in ForwardDiff.jacobian! internals vs. our closure or data path.

2. Remove remaining allocations

- Ensure absolutely no Any or abstract fields on hot path (g, cfg, compiled_dual, rowvec_dual,
data_over).
- Make closure and config fully concrete without relying on Ref if it affects inference; or keep Ref
but verify no type instability.
- Verify all merged data and overrides are prebuilt and not reconstructed or re-tagged per call.
- Consider tagging strategies that prevent ForwardDiff from rebuilding caches between calls.
- Keep current API; don’t change external behavior.

3. Verify and tighten tests

- Make derivative allocation assertions strict (== 0) in:
    - test/test_derivatives.jl
    - test/test_derivatives_extended.jl
    - test/test_links.jl (if feasible; otherwise justify any remaining internal link-specific
allocations and tighten thresholds as much as possible).

Acceptance criteria

- derivative_modelrow!(J, de, row) → 0 allocations after warmup on the user’s environment.
- marginal_effects_eta!/mu! → 0 allocations after warmup (if strictly possible across links;
otherwise documented, with thresholds tightened and rationale).
- All tests pass under strict zero-allocation assertions where claimed.
- README/CLAUDE.md/AGENTS.md reflect final state succinctly.
