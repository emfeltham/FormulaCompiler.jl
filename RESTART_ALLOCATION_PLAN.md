# Zero-Allocation Restoration Plan (branch: restart)

## Summary

The plan is to restore zero-allocation, per-row evaluation by compiling formulas into fully typed data and operation structures that Julia can specialize away at runtime. Strategy: remove symbol-based access in hot paths, make continuous and function inputs type-stable via Val{Column} and parametric evaluators, and use tuple-encoded operations for interactions. Categorical-only already passes; the work focuses on continuous, functions, and interactions.

Note that “compile to fully typed structs” is the right mental model: we’re producing parametric, tuple-backed structures so execution is monomorphic and allocation-free (without metaprogramming).

## Goal
- Restore zero-allocation per-row evaluation across LM/GLM/LMM/GLMM, including functions and interactions, without metaprogramming.

## Baseline (restart/test/allocation_results.csv)
- Categorical-only: 0 bytes (passes).
- Any continuous term: ~16 bytes.
- Functions: ~32 bytes.
- Interactions: allocations scale up (112–2416 bytes) as continuous reads multiply.
- Root cause: symbol-based data access for continuous values in execution paths.

## Phase 1 — Continuous (fast win)
- Make continuous column access type-stable and allocation-free.
- Changes:
  - `src/evaluators.jl`: parametrize `ContinuousEvaluator` → `ContinuousEvaluator{Column}` (keep `position`; drop `column::Symbol`). Parametrize `PrecomputedContinuousOp{Column}`.
  - `src/compile_term.jl`: construct `ContinuousEvaluator{term.sym}(start_position)`; push `PrecomputedContinuousOp{column_symbol}(pos)`.
  - `src/step1_specialized_core.jl`:
    - Represent columns as `NTuple{N, Val{Column}}` (or derive `Column` from type param).
    - In executor, use `get_data_value_type_stable(input, Val{col}(), row_idx)`.
  - `src/step4_interactions.jl`:
    - In `execute_operation!(CompleteFormulaData, ...)`, call step1 executor: `execute_operation!(data.continuous, op.continuous, ...)` instead of `execute_complete_continuous_operations!`.
- Acceptance: simple/multiple continuous, mixed (no functions), and non-function interactions drop to 0 bytes.

## Phase 2 — Functions (typed inputs)
- Eliminate symbol-based reads in function argument evaluation.
- Changes:
  - `src/step3_functions.jl`: ensure `get_input_value_zero_alloc` prefers `Val{Column}` and `ScratchPosition{P}`; decomposition emits typed sources (avoid `Symbol`).
  - `src/step4_interactions.jl`: when `arg_eval isa ContinuousEvaluator{Column}`, use `get_data_value_type_stable(input_data, Val{Column}(), row_idx)` in function value retrieval.
- Acceptance: function rows (unary/binary on continuous) drop from ~32 bytes to 0.

## Phase 3 — Interactions (end-to-end)
- Ensure all continuous reads inside interactions are Val-based.
- Changes:
  - `src/step4_interactions.jl`: in `get_value_from_source` and helpers, derive `Column` from `ContinuousEvaluator{Column}` and use Val-based access; avoid raw `Symbol` reads.
  - Keep tuple-based index patterns and pre-evals (already zero-alloc friendly).
- Acceptance: interaction rows (cont×cat, cat×cat, func×cat; three-/four-way) reach 0 bytes.

## Phase 4 — Hardening (optional)
- Categoricals already pass; optionally make more robust later:
  - `src/step2_categorical_support.jl`: consider `SpecializedCategoricalData{N,Positions,Val{Column}}` and use `getproperty(input, Val{Column}())`. Defer unless mixed/function cases show regressions.
- Remove `generated_function_interactions.jl` only after the typed path achieves 0 bytes across the survey.

## Validation
- Quick checks:
  - `julia --project test/quick_test.jl`
  - Use `@allocated compiled(buf, data, 1) == 0` for representative LM/GLM/LMM/GLMM cases.
- Survey:
  - `julia --project=test test/allocation_survey.jl`
  - Inspect `test/allocation_results.csv` — expect 0 bytes across all rows after each phase is implemented.
- If any row > 0: locate the call site still using symbol-based access and replace with Val-based.

## Work Method
- Implement Phase 1; re-run survey.
- Implement Phase 2; re-run survey.
- Implement Phase 3; re-run survey.
- Optional Phase 4 hardening/cleanup after all rows are 0.
