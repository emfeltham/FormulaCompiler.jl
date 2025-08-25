# Allocation Status

## Summary
Zero-allocation execution is achieved across the board except for function–interaction cases. Constants, continuous, categorical, functions (outside interactions), and standard interactions (without functions) all run at 0 allocations per row. Remaining allocations are isolated to multiway interactions that include functions.

## Current Results (from test/allocation_results.csv)
- Zero allocations (0 bytes):
  - LM/GLM/LMM/GLMM: intercept-only, continuous-only, categorical-only, mixed (no functions), standard interactions (2–4 way) without functions, and function-only cases.
- Remaining allocations:
  - LM: Four-way with function: 1680 bytes
  - LM: Complex interaction (includes function): 544 bytes

Timing remains competitive across all rows; the goal is to remove the last remaining allocations without regressing performance.

## Diagnosis (why these bytes remain)
- Function-in-interaction paths still build and execute function pre-evaluations within Step 4 for some multiway cases. While all column access is fully typed (Val{Column}) and function outputs are referenced via typed scratch positions, the local pre-eval construction in interactions for multiway cases introduces residual allocations.

## Plan to Close the Gap
1. Promote function pre-evaluations used by interactions into the global Step 3 pass:
   - Compile these functions into Step 3’s `SpecializedFunctionData` with dedicated scratch slots.
   - Ensure inputs are typed: `Val{Column}`, `ScratchPosition{P}`, or `Float64` — never `Symbol` at runtime.
2. In Step 4, remove per-interaction pre-evals; interactions only multiply typed sources:
   - Continuous: `Val{Column}` via `get_data_value_type_stable`.
   - Function outputs: `ScratchPosition{P}`.
   - Categoricals: contrast lookups (symbol stays local to categorical path only).

## Validation
- Re-run `julia --project=test test/allocation_survey.jl` and confirm the two LM rows drop to 0 bytes.
- Spot-check representative function×categorical formulas; assert `@allocated compiled(buf, data, 1) == 0`.

This keeps the fully typed architecture end-to-end and eliminates the last per-row allocations.

