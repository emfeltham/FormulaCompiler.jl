# ALLOCATION_NOTES.md

user
before we do phase 3. is there going to be a problem solving this for the case
of functions within interactions? e.g., (`...+ log(x) * a`)

codex
Short answer: it’s solvable, and we’ve already set up the pieces.

- Approach: pre-evaluate the function part (e.g., log(x)) in Step 3 into
a typed scratch slot (ScratchPosition{P}) and feed that into Step 4. The
categorical a remains as before.
- Execution order: Step 4’s pre-evals run first (tuple, zero-alloc), then the
interaction multiplies the scalar function result by each contrast column.
- Types: interaction inputs are either Val{Column} (for raw continuous),
ScratchPosition{P} (for function outputs), or constants — no symbols at
runtime.
- Width: function outputs are scalar, so width = categorical width; pattern
tuple stays zero-alloc.

Pitfalls to avoid: don’t try to read columns for functions inside Step 4;
always pre-eval via Step 3 to scratch and then use the typed scratch position
in interactions.