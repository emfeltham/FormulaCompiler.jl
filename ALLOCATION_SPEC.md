# Zero-Allocation Restoration — Engineering Spec

This spec details precise, mechanical edits to restore zero-allocation row evaluation for model matrices without metaprogramming. It is written for an automated coding agent to implement safely and incrementally.

## Scope
- Eliminate per-row allocations in execution of compiled formulas across LM/GLM/LMM/GLMM, including functions and interactions.
- Keep existing public APIs and file/module layout.
- Prefer compile-time typing (Val dispatch, tuple ops) over metaprogramming.

Non-goals: New features, refactors beyond what is necessary, dependency changes.

## Definitions
- “Typed column access”: `get_data_value_type_stable(input, Val{col}(), i)` or `getproperty(input, Val{col}())` for NamedTuple tables — no symbol-based lookup at runtime.
- “Step1 executor”: `execute_operation!(data::ContinuousData{N,...}, op::ContinuousOp{N,...}, ...)` in `src/step1_specialized_core.jl` using Val dispatch and unrolled paths.

## High-Level Plan (Phases)
1) Continuous: Replace CompleteFormula continuous stage with step1 executor (Val-based).
2) Categoricals: Encode column at type level in `SpecializedCategoricalData` and access via `Val`.
3) Functions: Ensure function input sources are typed (Val{Column} or ScratchPosition) and remove symbol-based access.
4) Interactions: Ensure value retrieval for all component types avoids symbol-based lookup; keep tuple patterns.
5) Tests/Survey: Validate with `test/allocation_survey.jl`, add micro allocation asserts.

## Detailed Tasks

### T1 — Continuous execution via Val (fast win)
File: `src/step4_interactions.jl`

- In `execute_operation!(data::CompleteFormulaData, op::CompleteFormulaOp, output, input_data, row_idx)`:
  - CURRENT: calls `execute_complete_continuous_operations!(data.continuous, ...)`.
  - CHANGE: replace with step1 executor
    - `execute_operation!(data.continuous, op.continuous, output, input_data, row_idx)`
  - Leave the constants/categoricals/functions/interactions phases as-is.

- Optional (cleanup): If no other code uses `execute_complete_continuous_operations!`, mark it as deprecated or remove in a later pass. Do not break compilation now.

Acceptance:
- Re-run survey; rows with continuous terms drop from 16 bytes to 0 (LM/GLM/LMM/GLMM), excluding function cases for now.

### T2 — Typed categoricals

IGNORE: already works

File: `src/step2_categorical_support.jl`

- Redefine data type to carry column as a type parameter while keeping positions as an NTuple:
  - BEFORE:
    - `struct SpecializedCategoricalData{N, Positions}` with fields: `contrast_matrix`, `positions`, `n_levels`, `n_contrasts`, `column::Symbol`.
  - AFTER:
    - `struct SpecializedCategoricalData{N, Positions, ColumnVal}` where `ColumnVal` is a type like `Val{Col}`
    - Fields: `contrast_matrix::Matrix{Float64}`, `positions::Positions`, `n_levels::Int`, `n_contrasts::Int`
    - Remove `column::Symbol` field.
    - Constructor: `SpecializedCategoricalData(contrast_matrix, positions::NTuple{N,Int}, n_levels, ::Val{Column}) where {N, Column}` computes `n_contrasts` and returns `new{N, typeof(positions), Val{Column}}(...)`.

- Update creator in `analyze_categorical_operations(evaluator::CombinedEvaluator)`:
  - Build `positions` as `NTuple` as today.
  - Get column symbol from `cat_eval.column` and pass `Val{cat_eval.column}` to the constructor.

- Update execution path (`execute_categorical_recursive!`):
  - Replace `column_data = getproperty(input_data, cat_data.column)` with typed access:
    - Define helper: `@inline get_column_val_type(::SpecializedCategoricalData{N,Pos,Val{C}}) where {N,Pos,C} = Val{C}`
    - Use: `colval = get_column_val_type(cat_data); column_data = getproperty(input_data, colval())`
    - Then `level = extract_level_code_zero_alloc(column_data, row_idx)` (as today).

Acceptance:
- Categorical-only and mixed (cat+cont) cases remain/become 0 bytes.

### T3 — Typed function inputs
Files: `src/step3_functions.jl`, `src/step4_interactions.jl`

- In `src/step3_functions.jl`:
  - Ensure `get_input_value_zero_alloc` supports and prefers:
    - `::Val{Column}` for column sources → call `get_data_value_type_stable(input_data, ::Val{Column}, row_idx)`.
    - `::ScratchPosition{P}` (already present).
    - Remove or avoid `::Symbol` overloads in new emission paths (keep fallback for backward compat but do not use it).
  - In function decomposition/`UnaryFunctionData` and friends, ensure `input_source` is either `Val{Column}` or `ScratchPosition{P}`, not a `Symbol`.

- In `src/step4_interactions.jl`, function value retrieval in interactions:
  - In `get_value_from_source(source::Int, component::FunctionEvaluator, ...)`: when reading `ContinuousEvaluator{Column}` argument, replace
    - `get_data_value_specialized(input_data, get_column_symbol(arg_eval), row_idx)`
    - WITH `get_data_value_type_stable(input_data, Val{get_column_symbol(arg_eval)}(), row_idx)`
  - Keep existing constant/categorical handling.

Acceptance:
- Rows involving unary/binary functions on continuous inputs drop from ~32 bytes to 0.

### T4 — Interactions value retrieval
File: `src/step4_interactions.jl`

- Ensure `get_value_from_source` avoids raw symbol-based column access:
  - For `source::Symbol`, current design routes to `get_component_interaction_value(component, ...)`; confirm that for `ContinuousEvaluator{Column}`/function components with typed inputs, this path does not perform symbol-based data access.
  - Where reading continuous values directly, use `get_data_value_type_stable(input_data, Val{Column}(), row_idx)`; retrieve `Column` via `get_column_symbol(::ContinuousEvaluator{Column})`.
  - For categorical components inside interactions, rely on `SpecializedCategoricalData{...,ColumnVal}` or existing `CategoricalEvaluator` lookup path which is already zero-alloc.

Acceptance:
- Mixed interactions (cont×cat, func×cat, cat×cat) evaluate per row with 0 allocations; four-way interaction rows drop to 0 allocations once functions are typed (T3).

### T5 — Survey + tests
Files: `test/quick_test.jl`, `test/allocation_survey.jl`

- Add micro allocation assertions in quick test (optional but recommended):
  - After compiling a simple model with continuous terms, assert `@allocated compiled(buf, data, 1) == 0` (guard behind `try`/`catch` to avoid breaking dev if evolving).

- Use the survey to validate:
  - Run: `julia --project=test test/allocation_survey.jl`
  - Confirm `memory_bytes == 0` for: Intercept only, Simple continuous, Multiple continuous, Mixed, Simple interaction, Function, Three-/Four-way interaction (allow temporal exceptions only if listed).

## Notes & Constraints
- NamedTuple field access with `getproperty(nt, ::Val{sym})` is type-stable and allocation-free; prefer it everywhere we read input data.
- Do not remove `generated_function_interactions.jl` until all survey rows are 0 bytes without it. Then delete the file and any references in a separate PR/commit.

## Acceptance Criteria (Overall)
- All LM/GLM/LMM/GLMM rows in `test/allocation_results.csv` show `memory_bytes == 0` after warmup (as produced by the provided survey script).
- No public API changes; existing tests pass.
- The continuous path no longer uses symbol-based data access at runtime.
- OVERWRITE EXISTING BAD FUNCTIONS
- DON'T MAINTAIN BACKWARDS COMPATIBILITY
