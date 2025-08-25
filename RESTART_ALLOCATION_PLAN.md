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

## Phase 1 — Continuous (architecture-first)
- Make continuous column access type-stable and allocation-free by carrying column identity in types (no runtime wrapping).
- Changes:
  - `src/evaluators.jl`: parametrize `ContinuousEvaluator` → `ContinuousEvaluator{Column}` (keep `position`; drop `column::Symbol`). Parametrize `PrecomputedContinuousOp{Column}`.
  - `src/compile_term.jl`: construct `ContinuousEvaluator{term.sym}(start_position)`; push `PrecomputedContinuousOp{column_symbol}(pos)`.
  - `src/step1_specialized_core.jl`:
    - Store columns as `NTuple{N, (Val{:x}(), Val{:y}(), …)}` — i.e., actual `Val` instances in the tuple, so `Cols` encodes the columns at the type level.
    - In the executor, use the stored `Val` directly:
      - `colval = data.columns[i]`
      - `val = get_data_value_type_stable(input, colval, row_idx)`
      - Do not construct `Val{col}()` in the loop.
  - `src/step4_interactions.jl`:
    - In `execute_operation!(CompleteFormulaData, ...)`, call step1 executor: `execute_operation!(data.continuous, op.continuous, ...)` instead of `execute_complete_continuous_operations!`.
- Acceptance: simple/multiple continuous, mixed (no functions), and non-function interactions drop to 0 bytes.

## Phase 2 — Functions (typed inputs)
Eliminate symbol-based reads in function argument evaluation using the reorganized layout.

Alignment with reorganization
- Files to edit (new structure):
  - `src/compilation/pipeline/step3/types.jl` and `src/compilation/pipeline/step3/main.jl`
  - `src/evaluation/data_access.jl` (or migrate accessors there)

Changes
- Compile-time inputs only: `UnaryFunctionData`/`IntermediateBinaryFunctionData`/`FinalBinaryFunctionData` must carry inputs as `Val{Column}` or `ScratchPosition{P}` or `Float64` — never `Symbol`.
- Decomposition: In `step3/main.jl`, when encountering `ContinuousEvaluator{Column}`, emit `Val{Column}` inputs (not `Symbol`). If nested functions, emit `ScratchPosition{P}`.
- Accessor: Centralize `get_data_value_type_stable(data, ::Val{Column}, row)` in `evaluation/data_access.jl`. Ensure it resolves field index from types (constant-foldable) and is `@inline`.
- Execution: In all `execute_operation!` for step3 types, call `get_input_value_zero_alloc` methods that dispatch on `Val{Column}`, `ScratchPosition{P}`, `Int`, `Float64` only.

Acceptance
- Function rows (unary/binary on continuous) drop from ~32 bytes to 0 across LM/GLM/LMM/GLMM.

## Phase 3 — Interactions (end-to-end)
Ensure all continuous reads inside interactions are Val-based and integrate with typed functions.

Alignment with reorganization
- Files to edit (new structure):
  - `src/compilation/pipeline/step4/types.jl` and `src/compilation/pipeline/step4/main.jl`
  - `src/compilation/pipeline/step4_function_interactions.jl` (if still used)

Changes
- Component typing: Where interaction components include continuous inputs, ensure the component or its pre-evaluated source carries `Val{Column}` or `InteractionScratchPosition{P}` in type parameters or stored tuples.
- Value retrieval: Update `get_value_from_source`/`get_component_interaction_value` to dispatch on typed sources only; for `ContinuousEvaluator{Column}`, route through `get_data_value_type_stable(data, Val{Column}(), row)`.
- Pre-evals: Keep pre-eval tuples; ensure any function pre-eval inputs are typed (`Val{Column}`/`ScratchPosition{P}`) after Phase 2.
- Keep tuple patterns and bounds validations; avoid Any/Vector-based intermediates.

Acceptance
- Interaction rows (cont×cat, cat×cat, func×cat; 3- and 4-way) show 0 bytes allocation in the survey.

## Reorganization Alignment (overview)
- APIs: Use the clarified two-phase API:
  - `compile_formula_complete(model, data) → CompiledFormula`
  - `compile_formula(compiled::CompiledFormula) → SpecializedFormula`
  - `compile_formula(model, data) → SpecializedFormula`
- Locations: Apply changes in the reorganized pipeline files:
  - Step 1/2 are already 0 alloc; Step 3 fixes live in `compilation/pipeline/step3/*`, Step 4 fixes in `compilation/pipeline/step4/*`.
- Accessors: Unify column access under `evaluation/data_access.jl` (or migrate), using `Val{Column}` only.


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

## Constraints (no runtime wrapping)
- All column identity must be present in types at compile time (parametric evaluators, stored `Val{Column}` in tuples).
- `get_data_value_type_stable(data, ::Val{Column}, i)` must leverage types to resolve the field index (constant-foldable), with no dynamic `Symbol` lookup or `Val{col}` construction inside hot loops.

## Work Method
- Implement Phase 1; re-run survey.
- Implement Phase 2; re-run survey.
- Implement Phase 3; re-run survey.
- Optional Phase 4 hardening/cleanup after all rows are 0.
