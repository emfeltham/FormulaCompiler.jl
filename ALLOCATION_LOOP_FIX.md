# FormulaCompiler Loop Allocation Fix

## Problem Summary

**Issue**: FormulaCompiler claims "zero-allocation" but actually allocates ~200 bytes per call when used in loops, the primary usage pattern for statistical computing.

**Test Results**:
- Single `compiled(output, data, 1)` after warmup: **0 bytes** ✅
- Loop `compiled(output, data, i)` for 1000 iterations: **202.688 bytes/call** ❌
- Loop `modelrow!(output, compiled, data, i)`: **426.816 bytes/call** ❌

**Impact**: Margins.jl's analysis is correct - FormulaCompiler is NOT zero-allocation for practical usage patterns.

## Root Cause Analysis

### Primary Cause: `fill!(scratch, zero(T))` in Hot Path

**Location**: `src/compilation/execution.jl:91`
```julia
function (f::UnifiedCompiled{T, Ops, S, O})(output::AbstractVector{T}, data::NamedTuple, row_idx::Int) where {T, Ops, S, O}
    scratch = f.scratch
    fill!(scratch, zero(T))  # ← ALLOCATION SOURCE: zero(T) creates new object each call
    execute_ops(f.ops, scratch, data, row_idx)
    copy_outputs_from_ops!(f.ops, output, scratch)
    return nothing
end
```

**Why This Allocates**:
1. **`zero(T)`** constructs a new zero value on each call
2. **Type instability** in loop contexts prevents optimization
3. **GC pressure** from repeated object creation

### Secondary Cause: `modelrow!` Wrapper Overhead

**Additional Allocation**: `modelrow!` adds ~224 bytes/call on top of core compilation
- `@assert` statements with string interpolation
- Cache dictionary operations (`get_or_compile_formula`)
- Function call overhead in loops

## Proposed Fixes

### 🟢 **Fix 1: Replace `zero(T)` with Literal** (Easy - 5 minutes)

**Change**:
```julia
# Current (allocating)
fill!(scratch, zero(T))  

# Fixed (zero-allocation)
fill!(scratch, 0.0)
```

**Rationale**: Literal `0.0` is a compile-time constant, `zero(T)` constructs runtime objects.

### 🟡 **Fix 2: Optimize Scratch Clearing** (Moderate - 30 minutes)

**Current**: Clears entire scratch buffer every call
**Better**: 
```julia
# Option A: Broadcasting (may be faster)
scratch .= 0.0

# Option B: Only clear used positions (best)
@inbounds for i in 1:used_scratch_positions
    scratch[i] = 0.0
end
```

### 🔴 **Fix 3: Lazy Scratch Management** (Complex - 2-4 hours)

**Concept**: Track which scratch positions are "dirty" and only clear as needed
```julia
# Track dirty positions in compiled formula
struct UnifiedCompiled{T, Ops, S, O}
    ops::Ops
    scratch::Vector{T}
    dirty_positions::Vector{Int}  # New field
end

# Clear only dirty positions from previous execution
function clear_dirty_scratch!(scratch, dirty_positions)
    @inbounds for pos in dirty_positions
        scratch[pos] = 0.0
    end
    empty!(dirty_positions)  # Reset for next round
end
```

### 🟡 **Fix 4: Remove modelrow! Overhead** (Moderate - 1 hour)

**Issues**:
1. Remove `@assert` statements or make them compile-time only
2. Cache optimization: pre-hash cache keys 
3. Streamline function signatures

## Implementation Priority

### Phase 1: Quick Win (5 minutes)
```julia
# src/compilation/execution.jl:91
- fill!(scratch, zero(T))
+ fill!(scratch, 0.0)
```
**Expected Impact**: Reduce core allocation from ~200 bytes to ~50 bytes per call

### Phase 2: Broadcasting Optimization (30 minutes)
```julia
# src/compilation/execution.jl:91
- fill!(scratch, 0.0)
+ scratch .= 0.0
```
**Expected Impact**: Further reduce allocation, potentially to 0 bytes

### Phase 3: Validation
Run comprehensive tests to ensure zero-allocation is achieved:
```julia
# Test loop allocation
allocs = @allocated for i in 1:1000; compiled(output, data, i); end
@assert allocs == 0 "Loop allocation not fixed: $allocs bytes"
```

## Testing Strategy

### Automated Test Suite
A comprehensive allocation test has been created:

**File**: `test/test_loop_allocations.jl`

**Usage**:
```bash
# Run allocation validation test
julia --project=. test/test_loop_allocations.jl

# Expected output (before fix):
# ❌ FAILURE: Loop allocations detected - fix needed
# Exit code: 1

# Expected output (after fix):
# ✅ SUCCESS: Core loop performance verified!
# Exit code: 0
```

**Test Coverage**:
- Single `compiled()` calls (warmup validation)
- Loop of `compiled()` calls (core allocation test)
- Loop of `modelrow!()` calls (wrapper overhead)
- Loop of cached `modelrow!()` calls (cache overhead)
- Loop of `marginal_effects_eta!()` calls (derivative functions)

### Before Fix
```julia
# Expected: ~200 bytes/call
julia --project=. test/test_loop_allocations.jl
# Shows: ❌ CORE ISSUE: Direct compiled() calls allocate 202.688 bytes/call in loops
```

### After Fix
```julia
# Expected: 0 bytes/call
julia --project=. test/test_loop_allocations.jl
# Should show: ✅ FormulaCompiler achieves zero-allocation loop performance
```

### Regression Prevention
```julia
# Add to CI/testing workflow
julia --project=. test/test_loop_allocations.jl || exit 1

# Run existing test suite to ensure correctness maintained
julia --project=. -e "using Pkg; Pkg.test()"
```

## Expected Outcomes

### Performance Improvement
- **Core evaluation**: ~202 bytes/call → **0 bytes/call**
- **modelrow!**: ~427 bytes/call → **~50 bytes/call** (after wrapper cleanup)
- **Statistical workflows**: Enable true zero-allocation marginal effects, AME calculations

### Documentation Accuracy
- FormulaCompiler's "zero-allocation" claims become accurate
- Margins.jl's concerns resolved
- Performance matches documentation promises

## Risk Assessment

### Low Risk Fixes
- **Fix 1 & 2**: Simple literal substitutions, minimal behavior change
- **Backwards compatibility**: Maintained (all APIs unchanged)
- **Type system**: Compatible with existing Float64 specialization

### Validation Required
- **Numerical accuracy**: Ensure scratch clearing doesn't affect results
- **Performance regressions**: Verify no slowdown in single-call patterns
- **Memory leaks**: Confirm no retention of scratch state between calls

## Conclusion

This is a **high-impact, low-risk fix** that addresses the fundamental discrepancy between FormulaCompiler's claims and reality. The primary allocation source (`zero(T)`) is easily fixable, and secondary optimizations can achieve true zero-allocation performance for the loops that statistical applications require.

**Estimated Total Effort**: 2-6 hours for complete fix + testing
**Business Value**: Removes main blocker for adoption in performance-critical statistical computing workflows