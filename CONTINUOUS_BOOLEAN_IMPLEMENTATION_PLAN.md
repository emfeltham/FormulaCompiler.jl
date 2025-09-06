# Boolean Variable Allocation Fix Plan

## Problem Summary

**Memory allocation should be O(1) - independent of dataset size.** Currently both boolean and continuous variables show problematic scaling patterns.

**Primary Issue (Boolean)**: `_gradient_with_scenario()` calls `copy(row_buf)` creating O(n) allocations  
**Secondary Issue (Continuous)**: Array access in hot loop creates 3.6x scaling despite perfect FormulaCompiler calls

**Root Cause Analysis**: 
- ✅ **FormulaCompiler is perfect**: Individual `marginal_effects_eta!` calls show 0 bytes allocation
- ✅ **Gradient accumulation is perfect**: `_accumulate_unweighted_ame_gradient!` shows 32 bytes regardless of size
- ❌ **Array indexing is the culprit**: `ame_sum += g_buf[var_idx]` in loop creates O(n) allocations due to bounds checking

**Goal**: Achieve ~1x scaling (O(1) memory) for all variable types, matching categorical performance.

## Current Performance Status

| Variable Type | Current Scaling | Status | Target | Feasibility | Notes |
|---------------|-----------------|--------|---------|-------------|-------|
| **Boolean** | 18.2x | ❌ Poor | **~1x** | ✅ **High** | Clear fix: eliminate `copy(row_buf)` |
| **Continuous** | 3.6x | ⚠️ Scaling | **~1x** | ✅ **High** | Clear fix: `@inbounds` array access in loop |
| **Categorical** | ~1x | ✅ Optimal | **~1x** | ✅ **Done** | Already O(1) memory |

**Key Discovery**: Both issues have **clear, simple fixes**! 
- FormulaCompiler functions: 0 bytes ✅  
- Gradient accumulation: 32 bytes ✅
- Problem source: Julia array access bounds checking in hot loops

## Problem Analysis & Solutions

### Boolean Variables: Clear Path Forward ✅

**Problem Location**: `src/engine/utilities.jl` in `_gradient_with_scenario()`:
```julia
function _gradient_with_scenario(compiled, scenario, row, scale, β, link, row_buf)
    # ... compute gradient ...
    return copy(row_buf)  # ← PROBLEM: Creates new vector every call
end
```

**Impact**: Boolean processing calls this twice per row (true + false scenarios):
- 1000 rows × 2 scenarios = 2000 vector allocations  
- Each vector ~ length(β) Float64s

**Confidence**: **High** - this is straightforward buffer management

### Continuous Variables: Inefficient Loop vs Built-in Batch Operation ✅

**Root Cause Found**: Manual loop with array indexing instead of using FormulaCompiler's optimized batch function

**Problem Location**: `src/engine/utilities.jl` in `_compute_continuous_ame()`:
```julia
# CURRENT (inefficient manual loop):
ame_sum = 0.0
for row in rows
    FormulaCompiler.marginal_effects_eta!(g_buf, de, β, row; backend=backend)
    ame_sum += g_buf[var_idx]  # ← Array access triggers allocations
end
ame_val = ame_sum / length(rows)

# Plus separate gradient computation:
_accumulate_unweighted_ame_gradient!(gβ_accumulator, de, β, rows, var; ...)
```

**Evidence**:
- Individual FormulaCompiler calls: 0 bytes ✅
- FormulaCompiler batch functions: 0 bytes ✅ (confirmed in tests)
- Our manual loop: 40K bytes for 1000 rows ❌

**Optimal Solution**: Use FormulaCompiler's built-in `accumulate_ame_gradient!` which does both AME computation and gradient accumulation in a single, optimized batch operation

**Confidence**: **Very High** - leverages existing tested 0-byte FormulaCompiler function

## Solution Implementations

### Boolean Variables: Pre-allocated Gradient Buffers

**Step 1**: Add gradient buffers to `MarginsEngine`:
```julia
# Add to MarginsEngine constructor  
grad_buf_true::Vector{Float64}   # Pre-allocated for true scenario 
grad_buf_false::Vector{Float64}  # Pre-allocated for false scenario
```

**Step 2**: Modify `_gradient_with_scenario` to use in-place operations:
```julia
function _gradient_with_scenario!(grad_buf, compiled, scenario, row, scale, β, link, row_buf)
    FormulaCompiler.modelrow!(row_buf, compiled, scenario.data, row)
    if scale === :response
        η = dot(row_buf, β)
        link_deriv = GLM.mueta(link, η)
        grad_buf .= link_deriv .* row_buf  # In-place assignment
    else
        grad_buf .= row_buf  # In-place copy
    end
    return grad_buf
end
```

### Continuous Variables: Replace Manual Loop with Batch Operation

**Architectural Solution**: Replace the entire inefficient loop with FormulaCompiler's optimized batch function:

```julia
# CURRENT (inefficient - two separate operations):
# 1. Manual loop for AME value computation
ame_sum = 0.0
for row in rows
    FormulaCompiler.marginal_effects_eta!(g_buf, de, β, row; backend=backend)
    ame_sum += g_buf[var_idx]  # ← Triggers allocations
end  
ame_val = ame_sum / length(rows)

# 2. Separate gradient computation
_accumulate_unweighted_ame_gradient!(gβ_accumulator, de, β, rows, var; ...)

# OPTIMAL (efficient - single batch operation):
# Use FormulaCompiler's built-in batch function that does both in one optimized call
FormulaCompiler.accumulate_ame_gradient!(
    engine.gβ_accumulator, engine.de, engine.β, rows, var;
    link=(scale === :response ? engine.link : GLM.IdentityLink()),
    backend=backend
)

# Compute AME value from the accumulated gradient
# (Implementation detail: extract AME from gradient result or modify FC to return both)
ame_val = compute_ame_value_from_gradient(engine.gβ_accumulator, rows, var)
```

**Key Benefits**:
- **Eliminates problematic array indexing entirely** - no bounds checking issues
- **Uses tested 0-byte FormulaCompiler function** - proven performance  
- **Single operation** instead of loop + separate gradient computation
- **Architectural elegance** - leverages FormulaCompiler's design intent

## Implementation Plan

### Phase 1: Boolean Variable Fix (60 minutes) ✅ High Confidence

#### Step 1.1: Update MarginsEngine Type (15 minutes)
- Add `grad_buf_true` and `grad_buf_false` fields
- Update constructor to initialize buffers
- File: `src/engine/core.jl`

#### Step 1.2: Modify Gradient Functions (30 minutes)
- Replace `_gradient_with_scenario` with in-place `_gradient_with_scenario!`
- Update all call sites to use pre-allocated buffers
- File: `src/engine/utilities.jl`

#### Step 1.3: Test and Validate (15 minutes)
- Run allocation scaling test
- Verify mathematical correctness unchanged
- **Expected: Boolean scaling 18.2x → ~1x (>95% improvement)**

### Phase 2: Continuous Variable Fix (45 minutes) ✅ Very High Confidence

#### Step 2.1: Replace Manual Loop with FormulaCompiler Batch Function (30 minutes)
**Goal**: Eliminate the entire inefficient loop by using FormulaCompiler's optimized batch operation

**File**: `src/engine/utilities.jl` in `_compute_continuous_ame()` function  
**Change**: Replace the entire loop + gradient computation with a single FormulaCompiler call:

```julia
# REMOVE this entire section:
ame_sum = 0.0
for row in rows
    if scale === :response
        FormulaCompiler.marginal_effects_mu!(engine.g_buf, engine.de, engine.β, row; link=engine.link, backend=backend)
    else  # scale === :link
        FormulaCompiler.marginal_effects_eta!(engine.g_buf, engine.de, engine.β, row; backend=backend)
    end
    ame_sum += engine.g_buf[var_idx]
end
ame_val = ame_sum / length(rows)

_accumulate_unweighted_ame_gradient!(
    engine.gβ_accumulator, engine.de, engine.β, rows, var;
    link=(scale === :response ? engine.link : GLM.IdentityLink()), 
    backend=backend
)

# REPLACE with this single optimized call:
FormulaCompiler.accumulate_ame_gradient!(
    engine.gβ_accumulator, engine.de, engine.β, rows, var;
    link=(scale === :response ? engine.link : GLM.IdentityLink()),
    backend=backend
)

# Compute AME value (need to implement this helper)
ame_val = _extract_ame_from_gradient(engine.gβ_accumulator, engine.de, engine.β, rows, var, scale, backend)
```

#### Step 2.2: Implement AME Value Extraction Helper (10 minutes)
**Goal**: Extract AME value from the gradient computation or modify FormulaCompiler to return both

**Options**:
1. **Compute AME separately** using a single batch call to `marginal_effects_eta!/mu!` over all rows
2. **Modify FormulaCompiler** to return both gradient and AME value  
3. **Extract from gradient mathematical relationship**

#### Step 2.3: Test and Validate (5 minutes)
- Run allocation scaling test
- Verify mathematical correctness unchanged  
- **Expected: Continuous scaling 3.6x → ~1x (>75% improvement)**

### Phase 3: Validation and Testing (30 minutes)

#### Step 3.1: Allocation Scaling Test
```julia
# Expected results after both fixes:
# Boolean scaling: 18.2x → ~1x (>95% improvement) 
# Continuous scaling: 3.6x → ~1x (>75% improvement)
# Categorical scaling: ~1x → ~1x (maintain optimal)
# Mathematical results: Identical (rtol < 1e-12)
```

#### Step 2.2: Correctness Validation
- Compare results before/after fix
- Ensure gradients are identical
- Test with various boolean interactions

## Success Criteria

### Performance Targets

| Variable Type | Current | Target | Improvement | Confidence | Fix Type |
|---------------|---------|--------|-------------|------------|----------|
| **Boolean** | 18.2x | ~1x | >95% | ✅ **High** | Buffer pre-allocation |
| **Continuous** | 3.6x | ~1x | >75% | ✅ **Very High** | FormulaCompiler batch operation |
| **Categorical** | ~1x | ~1x | Maintain | ✅ **High** | Already optimal |

**Success Criteria**: All variable types achieve O(1) memory usage regardless of dataset size  
**Total Timeline**: 105 minutes (60 + 45) - both fixes have high confidence  
**Key Advantage**: Continuous fix uses proven 0-byte FormulaCompiler function instead of patching our own code

### Correctness Requirements
- **Mathematical accuracy**: Identical results (rtol < 1e-12)
- **Gradient correctness**: Parameter gradients unchanged
- **Standard errors**: SE calculations identical

## Risk Assessment

### Boolean Variables ✅ Low Risk
- **Nature**: Pure memory management optimization (buffer pre-allocation)
- **Algorithm**: No changes - identical mathematical results guaranteed  
- **Rollback**: Simple revert to `copy(row_buf)` if needed
- **Isolation**: Changes confined to gradient computation functions

### Continuous Variables ⚠️ Medium Risk  
- **Investigation Risk**: May require deeper changes than anticipated
- **Unknown Factors**: Root cause of 6.3x scaling not yet identified
- **Complexity**: Could involve FormulaCompiler integration or Julia compiler issues
- **Mitigation**: Thorough profiling before implementing changes

## Expected Impact

### Phase 1 Impact (Boolean Fix) ✅ High Confidence
- **Fix**: Pre-allocated gradient buffers eliminate `copy(row_buf)` allocations
- **Performance**: 18.2x → ~1x scaling (>95% improvement)  
- **Risk**: Low - straightforward buffer management, no algorithmic changes
- **Timeline**: 60 minutes implementation

### Phase 2 Impact (Continuous Fix) ✅ Very High Confidence  
- **Fix**: Replace manual loop with FormulaCompiler's optimized `accumulate_ame_gradient!` batch function
- **Performance**: 3.6x → ~1x scaling (>75% improvement)
- **Risk**: Very Low - uses existing tested 0-byte FormulaCompiler function
- **Timeline**: 45 minutes implementation

### Overall System Impact
**Before**: Only categorical variables achieve O(1) memory scaling
**After Both Phases**: All variable types achieve O(1) memory scaling - **goal achieved!**

**Key Architectural Insight**: The continuous variable problem wasn't a Julia performance bug to patch - it was **using the wrong API**. FormulaCompiler already provides the optimal batch operation we needed (`accumulate_ame_gradient!`), we just weren't using it. This is more elegant than patching array indexing with `@inbounds`.