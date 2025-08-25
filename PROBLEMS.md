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

### Solution Plan (Based on Working Restart Branch Implementation)

The restart branch already solved this correctly. We need to port the proven solution:

1. **Port Kronecker Product Expansion Logic**:
   ```julia
   # From restart branch: compute_interaction_pattern_tuple()
   function compute_interaction_pattern_tuple(width1::Int, width2::Int)
       pattern_tuple = ntuple(width1 * width2) do idx
           # StatsModels convention: kron(b, a) means a varies fast, b varies slow
           j = ((idx - 1) ÷ width1) + 1  # Slow index (second component)  
           i = ((idx - 1) % width1) + 1  # Fast index (first component)
           (i, j)
       end
   end
   
   # For x * group3 where group3 has 2 contrasts:
   # Pattern: [(1,1), (1,2)] → [x*contrast1, x*contrast2]
   ```

2. **Port Multi-Component Interaction Decomposition**:
   ```julia
   # From restart: decompose_interaction_tree_zero_alloc()
   # - Handle multi-output categorical terms properly
   # - Generate separate BinaryOp for each interaction combination
   # - Use scratch space for intermediate results in n-way interactions
   ```

3. **Port Dynamic Categorical Level Extraction**:
   ```julia
   # From restart: extract_level_code_zero_alloc()
   @inline function extract_level_code_zero_alloc(column_data::CategoricalVector, row_idx::Int)
       return Int(levelcode(column_data[row_idx]))
   end
   
   # Key: Extract levels dynamically during execution, not pre-computed
   ```

4. **Adapt ContrastOp to Use Dynamic Level Extraction**:
   ```julia
   # Current ContrastOp stores contrast matrix but needs column symbol
   struct ContrastOp{Column, OutPositions} <: AbstractOp 
       contrast_matrix::Matrix{Float64}
       # Column is already in type parameter - good!
   end
   
   # Update execution to extract level dynamically:
   execute_op(op::ContrastOp{Col, Positions}, scratch, data, row_idx) = 
       level = extract_level_code_zero_alloc(getproperty(data, Col), row_idx)
   ```

5. **Multi-output Position Handling Algorithm**:
   ```julia
   # In decompose_term!(ctx, term::InteractionTerm, data_example):
   
   # Step 1: Get component positions (some may be Vector{Int})
   component_positions = []
   for t in term.terms
       pos = decompose_term!(ctx, t, data_example)
       push!(component_positions, isa(pos, Int) ? [pos] : pos)
   end
   
   # Step 2: Compute Kronecker expansion using restart branch logic
   interaction_combinations = compute_all_interaction_combinations(component_positions)
   
   # Step 3: Generate BinaryOp for each combination
   output_positions = Int[]
   for position_combo in interaction_combinations
       out_pos = allocate_position!(ctx)
       # Create multiplication chain for this combination
       push!(ctx.operations, create_interaction_ops(position_combo, out_pos))
       push!(output_positions, out_pos)
   end
   
   return output_positions  # Vector{Int} for multi-output
   ```

**Key Implementation Files to Reference**:
- `git show restart:src/compilation/pipeline/step4/main.jl` (Interaction logic)
- `git show restart:src/compilation/pipeline/step2_categorical.jl` (Categorical handling)
- `git show restart:src/compilation/pipeline/step4/types.jl` (Type definitions)

---

## PROBLEM 2: Output Position Mapping Inconsistency ✅ **RESOLVED BY RESTART SOLUTION**

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

### ✅ Resolved by Restart Branch Solution
The restart branch handles this correctly through its interaction decomposition system:

1. **Multi-output Position Collection**: The restart branch properly handles `Vector{Int}` returns from interaction terms and flattens them into the correct output positions.

2. **Structured Position Management**: Uses `IntermediateInteractionData` and `FinalInteractionData` to track all positions correctly.

3. **Proper Copy Operation Generation**: Each interaction output position gets proper copy operations to the final output.

**No additional work needed** - this is automatically fixed by porting Problem 1's solution.

---

## PROBLEM 3: Position Caching with Multi-output Terms ✅ **RESOLVED BY RESTART SOLUTION**

### Issue
**Location**: `src/compilation/types.jl:298`

```julia
position_map::Dict{Any, Union{Int, Vector{Int}}}  # Term → scratch position(s)
```

### Impact
- Caching logic assumes single positions for most operations
- Multi-output interaction results break caching assumptions
- Inconsistent position retrieval

### ✅ Resolved by Restart Branch Solution
The restart branch handles this through sophisticated type-based position management:

1. **Compile-time Position Embedding**: Uses types like `IntermediateInteractionData{..., PatternTuple, ...}` where positions are embedded in type parameters, eliminating runtime caching issues.

2. **Structured Position Types**: 
   ```julia
   struct InteractionScratchPosition{P}  # Position P embedded at compile-time
   struct CompleteInteractionData{IntermediateTuple, FinalTuple}  # All positions in tuples
   ```

3. **No Runtime Position Lookup**: The restart branch avoids the caching problem entirely by using compile-time position specialization.

**No additional work needed** - the restart branch architecture eliminates this problem through design.

---

## PROBLEM 4: Type System Inconsistency ✅ **PARTIALLY RESOLVED - NEEDS ADAPTATION**

### Issue
**Location**: Throughout execution system

### Impact
- Operations expect single positions in type parameters
- Multi-output interactions don't fit current operation types
- No type for "multiple related operations"

### ✅ Restart Branch Has Solution, Needs Adaptation
The restart branch solved this with a sophisticated multi-layered type system:

1. **Specialized Operation Types**:
   ```julia
   struct IntermediateInteractionData{C1, C2, Input1Type, Input2Type, PatternTuple, PreEvalTuple}
   struct FinalInteractionData{C1, C2, Input1Type, Input2Type, PatternTuple, PreEvalTuple}
   struct InteractionOp{I, F}  # I = intermediate count, F = final count
   ```

2. **Compile-time Pattern Tuples**: Instead of runtime vectors, uses tuples like `PatternTuple` for zero-allocation execution.

3. **Hierarchical Type Architecture**: Multi-step system with constants → categorical → functions → interactions.

**⚠️ Adaptation Required**: We need to either:
- **Option A**: Port the full restart branch type hierarchy to unified system
- **Option B**: Use "Flatten Approach" - generate separate `BinaryOp{:*, pos1, pos2, out}` for each interaction combination (simpler integration with current unified system)

---

## PROBLEM 5: Formula Schema Integration ✅ **RESOLVED BY RESTART SOLUTION**

### Issue
**Location**: `src/compilation/compilation.jl:100`

### Impact
- May not be properly extracting schema-applied formula structure
- StatsModels formula expansion might not match our decomposition

### ✅ Resolved by Restart Branch Solution
The restart branch has extensive schema integration:

1. **Comprehensive Schema Extraction**:
   ```julia
   # From restart: term_compiler.jl
   struct CategoricalSchemaInfo
       dummy_contrasts::Matrix{Float64}         # DummyCoding (k-1 columns)
       full_dummy_contrasts::Matrix{Float64}    # FullDummyCoding (k columns)  
       main_effect_contrasts::Union{Matrix{Float64}, Nothing}
       n_levels::Int
       levels::Vector{String}
       level_codes::Vector{Int}
       column::Symbol
   end
   ```

2. **Model-Specific Schema Handling**:
   - `extract_categorical_schema_from_glm_model(model::TableRegressionModel)`
   - `extract_categorical_schema_from_mixed_model(model::Union{LinearMixedModel, GeneralizedLinearMixedModel})`
   - Proper contrast matrix extraction from fitted models

3. **StatsModels Compatibility**: The restart branch was extensively tested against all model types and achieves exact `modelmatrix()` matching.

**No additional work needed** - the restart branch already solved schema integration comprehensively.

---

## REVISED IMPLEMENTATION PLAN

Based on analysis, **most problems are resolved by porting the restart branch solution**. The implementation is significantly simpler than originally estimated.

### Phase 1: Port Restart Branch Core Logic ⭐ **SINGLE PHASE NEEDED**
**Target**: Port proven working solution

#### 1A: Port Interaction Expansion Logic
- ✅ Root cause identified (line 268-269)
- 🔄 Port `compute_interaction_pattern_tuple()` from restart branch
- 🔄 Port `decompose_interaction_tree_zero_alloc()` logic
- 🔄 Use **Flatten Approach**: Generate separate `BinaryOp{:*, pos1, pos2, out}` for each interaction combination

#### 1B: Port Dynamic Categorical Extraction
- 🔄 Port `extract_level_code_zero_alloc()` function  
- 🔄 Update `ContrastOp` execution to use dynamic level extraction
- 🔄 Port `CategoricalSchemaInfo` extraction system

#### 1C: Fix Output Position Collection
- 🔄 Update `decompose_formula()` to properly flatten multi-output interaction results
- 🔄 Ensure `ctx.output_positions` accounts for all interaction expansions

#### 1D: Integration Testing
- 🔄 Test all interaction types: `x * group3`, `x * y * group3`, `log(x) * group3`
- 🔄 Verify against working restart branch results
- 🔄 Ensure zero allocations maintained

### ~~Phase 2-4: Eliminated~~ 
**Rationale**: Problems 2, 3, 5 are automatically resolved by Phase 1. Problem 4 uses simple "Flatten Approach" integration.

### Success Criteria Unchanged
- [ ] All `test_formulas` pass correctness tests
- [ ] Zero allocations maintained  
- [ ] ModelRow and Scenario integration work

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