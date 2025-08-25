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

---

## Feasibility Assessment (Updated)

Status: Highly feasible and already partially implemented.

- Existing infrastructure:
  - Scratch types exist: `ScratchPosition{P}` (Step 3) and `InteractionScratchPosition{P}` (Step 4).
  - Typed access exists: Val{Column} dispatch in both Step 3 and Step 4.
  - Separate scratch spaces already present on `CompleteFormulaData`.
- Partially aligned today:
  - Function pre‑evals for interactions are being promoted to the global function phase.
  - Interaction data creation now empties pre‑eval tuples (per‑interaction pre‑evals removed).

What remains minimal:
1) Add `FunctionScratchPosition{P}` (mirror of `InteractionScratchPosition{P}`).
2) Add accessor overload to read from `function_scratch` for `FunctionScratchPosition{P}`.
3) Update interaction analysis to emit `FunctionScratchPosition{P}` instead of any lingering pre‑eval use.
4) Clean up any remaining pre‑eval references and ensure Step 4 execution reads function outputs from `function_scratch`.

Effort: 1–2 hours. Risk: low. Performance: improves (removes remaining allocations) while keeping execution fully typed.

## Actionable Checklist (Files)
- Types & accessors:
  - Add `FunctionScratchPosition{P}` in `src/compilation/pipeline/step3/types.jl` (or shared types).
  - Add `get_value_from_source(::FunctionScratchPosition{P}, ...)` in `src/compilation/pipeline/step4/main.jl` (or `evaluation/data_access.jl`) to read `function_scratch[P]`.
- Analysis:
  - `src/compilation/pipeline/step4/main.jl`: in interaction analysis, emit `FunctionScratchPosition{P}` for any function‑derived input (P comes from Step 3’s mapping for that function node).
- Execution:
  - Ensure `execute_interaction_operations!` (and callees) receive access to `function_scratch` (already present via `CompleteFormulaData`).
  - Interactions must not create/execute per‑interaction pre‑evals; they must only multiply typed sources (`Val{Column}`, `FunctionScratchPosition{P}`, constants) and write to `interaction_scratch`/output.

## Quick Verification Path
1) Add the type + accessor (compile & run smoke test).
2) Switch just the new “Function in interaction” survey case to use `FunctionScratchPosition{P}` → verify 0 bytes.
3) Apply the same mapping in n‑way interactions → re‑run full survey and confirm LM four‑way/complex rows drop to 0 bytes.
