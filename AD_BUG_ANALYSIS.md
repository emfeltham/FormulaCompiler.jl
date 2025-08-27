# AD Bug Analysis: Interaction Terms Always Return Derivative=1

## Problem Summary
When using ForwardDiff (AD) to compute derivatives of interaction terms like `x & group3`, the derivative with respect to `x` is always 1.0 regardless of the categorical value. This is incorrect - the derivative should be 0 when the categorical level doesn't match.

## Root Cause
The bug occurs in the interaction between:
1. **TypedSingleRowOverrideVector{Dual}** - overrides continuous variables with Dual numbers
2. **ContrastOp** - handles categorical variables and their contrast matrices
3. **BinaryOp{:*}** - multiplies the continuous and categorical components

## What's Happening

### Correct Behavior (FD)
When using finite differences:
1. Override `:x` with perturbed Float64 values
2. ContrastOp correctly extracts categorical level (e.g., "A" → level 1)
3. Contrast matrix returns 0 for non-matching levels (e.g., group3:B when group="A")
4. Interaction: `x * 0 = 0` → derivative is 0 ✓

### Incorrect Behavior (AD)
When using ForwardDiff:
1. Override `:x` with Dual numbers (value + partials)
2. ContrastOp extracts categorical level correctly
3. BUT: The contrast matrix value gets converted to Dual type
4. The conversion `convert(Dual, 0.0)` creates a Dual with value=0 but **partials=0**
5. When this multiplies with the Dual `:x`, something goes wrong
6. Result: derivative is always 1.0 ✗

## The Bug Location

In `execution.jl` line 362:
```julia
scratch[pos] = convert(eltype(scratch), op.contrast_matrix[level, i])
```

When `eltype(scratch)` is a Dual type and `op.contrast_matrix[level, i]` is 0.0 (for non-matching categorical level), the conversion creates a "constant" Dual with zero partials. But the interaction multiplication doesn't properly handle this.

## Why It's Complex

The fix is complex because:

1. **Type System Interactions**: The position-mapping system assumes uniform element types in scratch space. Having mixed Float64/Dual types would break this assumption.

2. **Categorical Variables Are Constants**: Categorical variables don't have derivatives - they're discrete. Their contrast matrix values should act as constants in AD.

3. **Interaction Semantics**: The interaction `x & group3:B` should have:
   - derivative = 1 when group="B" (active)
   - derivative = 0 when group≠"B" (inactive)

4. **Current Architecture**: The system treats all scratch positions uniformly with `eltype(scratch)`. Categorical contrasts get converted to this type, losing their "constant" nature.

## Potential Solutions

### Solution 1: Proper Dual Construction (Simplest)
Modify ContrastOp to create proper constant Duals:
```julia
# In execute_op for ContrastOp
val = op.contrast_matrix[level, i]
if eltype(scratch) <: ForwardDiff.Dual
    # Create a constant Dual (value with zero partials)
    scratch[pos] = ForwardDiff.Dual{typeof(ForwardDiff.Tag(zero(eltype(scratch))))}(val, zero(ForwardDiff.partials(zero(eltype(scratch)))))
else
    scratch[pos] = convert(eltype(scratch), val)
end
```

### Solution 2: Separate Categorical Handling
Keep categorical values as Float64 even when other values are Dual:
- Requires architectural changes to support mixed types
- More complex but cleaner separation of concerns

### Solution 3: Tag-Aware Contrast Matrix
Store contrast matrices per Dual tag:
- Build Dual-typed contrast matrices on demand
- Ensures proper partial propagation

## Complexity Assessment

**Difficulty: Medium**

The simplest fix (Solution 1) requires:
1. Detecting when we're in Dual mode in ContrastOp
2. Creating proper constant Duals with zero partials
3. Ensuring this works with ForwardDiff's tag system

The fix is localized to `execute_op` for `ContrastOp` but requires careful handling of ForwardDiff's Dual type construction.

## Testing Requirements

After fix, verify:
1. `x & group3:A` has derivative=1 when group="A", 0 otherwise
2. `x & group3:B` has derivative=1 when group="B", 0 otherwise  
3. All marginal effects tests pass with both AD and FD backends
4. No performance regression