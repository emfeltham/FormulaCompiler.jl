# Function×Interaction Zero‑Allocation Plan (Option A)

## Goal
- Eliminate remaining allocations in function–interaction rows by making Step 4 read function outputs from the global function scratch, not by constructing pre‑evals per interaction.
- Preserve fully typed, monomorphic execution with no per‑row wrapping or symbol lookups.

## Design Overview
- Single source of truth for function outputs: Step 3 computes all function results for a row into `function_scratch`.
- Step 4 interactions consume function results via typed scratch references (no recomputation, no copies).
- Keep intermediate Kronecker products in `interaction_scratch` as today.

## Types & Accessors
- Add `FunctionScratchPosition{P}` (mirror of `InteractionScratchPosition{P}`) in Step 3 or a shared `types.jl`.
- Add `get_value_from_source(::FunctionScratchPosition{P}, ...) = function_scratch[P]` in `evaluation/data_access.jl` (or Step 4 main if preferred).
- Optional: `ScratchSpaces` struct bundling both `function_scratch` and `interaction_scratch` to reduce argument clutter.

## Pipeline Changes
- Step 3 (src/compilation/pipeline/step3):
  - Ensure every function node has a stable scratch slot `P` (already true).
  - Expose mapping from function nodes to `FunctionScratchPosition{P}` for downstream use.
- Step 4 Analysis (src/compilation/pipeline/step4/main.jl):
  - When encountering function components inside interactions, emit `FunctionScratchPosition{P}` as the input source instead of building pre‑evals.
  - Remove any per‑interaction function pre‑eval construction.
- Step 4 Execution:
  - Update interaction executors to accept both scratch arrays (or a `ScratchSpaces`) and route function reads through `FunctionScratchPosition{P}` from `function_scratch`.
  - Intermediate Kronecker writes remain in `interaction_scratch`.

## Signatures to Touch
- `execute_operation!(data::CompleteFormulaData, ...)`: pass both scratches (already present as separate fields).
- `execute_interaction_operations!` and `get_value_from_source`: add/overload for `FunctionScratchPosition{P}` and ensure access to `function_scratch`.
- Interaction data structs do not change except for typed input sources that may now include `FunctionScratchPosition{P}`.

## Migration Steps
1) Implement `FunctionScratchPosition{P}` and accessor.
2) Replace function pre‑evals in Step 4 analysis with `FunctionScratchPosition{P}` sources.
3) Remove now‑unused pre‑eval tuples/logic from interaction data creation.
4) Thread `function_scratch` into interaction execution (or introduce `ScratchSpaces`).
5) Verify “Function in interaction” row → 0 bytes; then LM four‑way/complex rows → 0 bytes.

## Validation
- Run `test/allocation_survey.jl`; confirm all rows 0 bytes.
- Microchecks: `@allocated compiled(buf, data, 1) == 0` for function×categorical and multiway function interactions.

## Risks & Mitigations
- Signature changes: confine to Step 4 execution helpers; keep public API stable.
- Cross‑step coupling: document the scratch contract in code comments; add small unit tests around accessor behavior.
