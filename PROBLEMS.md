# Unified Compilation System - Major Problems & Solutions

## Current Status: Systematic Failures in Interaction Terms

**Test Results Summary:**
- ✅ Simple formulas: 100% pass (intercept, continuous, categorical, functions)
- ❌ Interaction formulas: ~90% fail with dimension mismatches
- **Root Issue**: Compiled output has fewer columns than `modelmatrix()` expects

---

## PROBLEM 1: Categorical Interaction Expansion (CRITICAL)

### Issue
**Location**: `src/compilation/decomposition.jl:268-269`

```julia
# For multi-output terms (categorical), use first position for now
# TODO: Handle categorical×continuous properly
push!(positions, pos[1])  # ← BUG: Only uses first contrast level!
```

### Impact
- `x * group3` where `group3` has contrasts `[level1, level2]`
- **Expected**: 2 interaction columns (`x*level1`, `x*level2`)
- **Actual**: 1 interaction column (only `x*level1`)
- **Result**: Output size (5) ≠ ModelMatrix size (6)

### Solution Plan
1. **Implement Kronecker Product Expansion**:
   ```julia
   # x * group3 → [x] × [group3_contrast1, group3_contrast2]
   # Result: [x*contrast1, x*contrast2]
   ```

2. **Multi-output Position Handling**:
   - Change `InteractionTerm` to return `Vector{Int}` for multi-column results
   - Update `decompose_formula` to handle vector position results
   - Generate correct number of `CopyOp` operations

3. **Cascade Multi-way Interactions**:
   ```julia
   # x * y * group3 → [x] × [y] × [contrast1, contrast2]
   # Result: [x*y*contrast1, x*y*contrast2]
   ```

---

## PROBLEM 2: Output Position Mapping Inconsistency

### Issue
**Location**: `src/compilation/decomposition.jl:130-136`

```julia
# Add copy operations for final output
for (out_idx, scratch_pos) in enumerate(ctx.output_positions)
    push!(ctx.operations, CopyOp{scratch_pos, out_idx}())
end
```

### Impact
- `ctx.output_positions` doesn't account for interaction expansion
- Copy operations create wrong number of output columns
- Mismatch between internal computation and final output size

### Solution Plan
1. **Fix Position Collection**:
   - Properly flatten multi-output terms when building `ctx.output_positions`
   - Handle `Vector{Int}` returns from interaction terms

2. **Validate Output Size**:
   ```julia
   @assert length(ctx.output_positions) == expected_modelmatrix_width
   ```

---

## PROBLEM 3: Position Caching with Multi-output Terms

### Issue
**Location**: `src/compilation/types.jl:298`

```julia
position_map::Dict{Any, Union{Int, Vector{Int}}}  # Term → scratch position(s)
```

### Impact
- Caching logic assumes single positions for most operations
- Multi-output interaction results break caching assumptions
- Inconsistent position retrieval

### Solution Plan
1. **Standardize Position Types**:
   - Always return `Vector{Int}` (even for single positions: `[pos]`)
   - Or create proper union handling throughout

2. **Cache Validation**:
   - Ensure cached multi-output positions work correctly
   - Handle interaction sub-components properly

---

## PROBLEM 4: Type System Inconsistency

### Issue
**Location**: Throughout execution system

### Impact
- Operations expect single positions in type parameters
- Multi-output interactions don't fit current operation types
- No type for "multiple related operations"

### Solution Plan
1. **Extend Operation Types**:
   ```julia
   # New operation for multiple simultaneous multiplications
   struct MultiInteractionOp{InputPositions, OutputPositions} <: AbstractOp end
   ```

2. **Or Flatten Approach**:
   - Generate separate `BinaryOp{:*, pos1, pos2, out}` for each interaction
   - Keep current type system but generate more operations

---

## PROBLEM 5: Formula Schema Integration

### Issue
**Location**: `src/compilation/compilation.jl:100`

### Impact
- May not be properly extracting schema-applied formula structure
- StatsModels formula expansion might not match our decomposition

### Solution Plan
1. **Debug Schema Application**:
   ```julia
   # Compare our decomposition against StatsModels expansion
   our_terms = decompose_formula_debug(formula, data)
   expected_terms = StatsModels.terms(formula)
   ```

2. **Validate Against ModelMatrix**:
   - Check column names and ordering
   - Ensure our expansion matches StatsModels exactly

---

## IMPLEMENTATION PLAN

### Phase 1: Fix Categorical Interactions (PRIORITY 1)
**Target**: Get `x * group3` working
1. ✅ Identify root cause (line 268-269)
2. 🔄 Implement proper Kronecker expansion
3. 🔄 Fix output position collection
4. 🔄 Test simple 2-way categorical interactions

### Phase 2: Multi-way Interactions (PRIORITY 2)  
**Target**: Get `x * y * group3` working
1. 🔄 Extend Kronecker expansion to n-way
2. 🔄 Handle cascaded multiplication operations
3. 🔄 Test complex interaction formulas

### Phase 3: Function Interactions (PRIORITY 3)
**Target**: Get `log(x) * group3` working
1. 🔄 Ensure function outputs work with interactions
2. 🔄 Test function × categorical combinations

### Phase 4: Edge Cases & Validation (PRIORITY 4)
1. 🔄 Mixed model integration
2. 🔄 GLM compatibility
3. 🔄 Performance regression testing

---

## SUCCESS CRITERIA

### Correctness Tests
- [ ] All `test_formulas.lm` pass (currently: 6/15 pass)
- [ ] All `test_formulas.glm` pass (currently: 7/10 pass) 
- [ ] All `test_formulas.lmm` pass (currently: 5/6 pass)
- [ ] Complex interactions work across all model types

### Performance Tests  
- [ ] Zero allocations maintained for all formula types
- [ ] Performance competitive with old multi-step system

### Integration Tests
- [ ] ModelRow interface works with all formulas
- [ ] Scenario system works with interactions
- [ ] Edge cases handled properly

---

## DEBUG STRATEGY

### Immediate Next Steps
1. **Create minimal reproduction**:
   ```julia
   df = DataFrame(x = [1.0, 2.0], group = ["A", "B"])
   model = lm(@formula(y ~ x * group), df)
   # Debug: why does this produce 3 columns instead of 4?
   ```

2. **Compare against working system**:
   - Use old multi-step as reference
   - Identify exact differences in column expansion

3. **Step-by-step debugging**:
   - Print `ctx.output_positions` before/after each term
   - Verify operation generation
   - Check final output size calculation

---

## RISK ASSESSMENT

### High Risk
- **Complexity of Kronecker expansion**: Easy to get combinatorics wrong
- **Performance impact**: More operations per interaction term
- **Type system compatibility**: May need operation type changes

### Medium Risk  
- **Schema integration**: StatsModels compatibility
- **Edge cases**: Unusual formula patterns
- **Mixed model handling**: Different formula structures

### Low Risk
- **Simple formula regression**: These already work
- **Documentation**: Position mapping concepts still valid
- **Test infrastructure**: Well-established test cases

---

**Bottom Line**: The core position mapping architecture is sound. We have a specific, well-defined bug in categorical interaction expansion that's causing systematic dimension mismatches. Fix the Kronecker expansion, and the system should work correctly.