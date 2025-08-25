# FormulaCompiler Architecture Removal Plan

This document outlines the systematic removal of the old step-based architecture in favor of the unified compilation system.

## Executive Summary

The UnifiedCompiler has successfully achieved 100% zero allocations across all 35 test formulas, making the old step-based architecture obsolete. This removal plan transitions FormulaCompiler to use only the unified system while preserving integration components for future development.

## Current State

- ✅ **UnifiedCompiler**: Complete and tested (100% zero allocations)
- ✅ **New test suite**: `test_allocation_survey.jl` passes all 35 formulas
- ❌ **Old test suite**: `test_models.jl` uses obsolete `compile_formula()` and `SpecializedFormula`
- ❌ **Mixed architecture**: FormulaCompiler.jl includes both old and new systems

## Removal Strategy

### Phase 1: Update Test System ✅ Ready
Update the active test suite to use the unified system before removing old code.

**File to update:**
- `test/test_models.jl`

**Changes needed:**
```julia
# Replace old API calls:
compile_formula(model, data) → compile_formula_unified(model, data)
SpecializedFormula → UnifiedCompiled

# Update test assertions accordingly
```

### Phase 2: Update Main Module ✅ Ready  
Update FormulaCompiler.jl to remove old includes and exports.

**File to update:**
- `src/FormulaCompiler.jl`

**Remove these includes:**
```julia
# Step-based compilation pipeline
include("compilation/pipeline/step1_constants.jl")
include("compilation/pipeline/step2_categorical.jl") 
include("compilation/pipeline/step3_functions.jl")
include("compilation/pipeline/step4_interactions.jl")
include("compilation/pipeline/step4_function_interactions.jl")

# Old compilation system
include("compilation/term_compiler.jl")
include("compilation/legacy_compiled.jl")

# Old evaluator system
include("evaluation/evaluators.jl")
include("evaluation/data_access.jl")
include("evaluation/function_ops.jl")
```

**Add unified system includes:**
```julia
# Unified compilation system
include("unified/compilation.jl")
# This automatically includes: types.jl, execution.jl, scratch.jl, decomposition.jl
```

**Update exports:**
```julia
# Remove old exports:
export compile_formula, compile_formula_complete, test_new_interaction_system

# Add unified exports:
export compile_formula_unified, compile_unified, UnifiedCompiled
```

### Phase 3: Remove Obsolete Files ⚠️ Destructive
Delete files that are no longer used by any part of the system.

**Files to delete:**
```bash
# Step-based pipeline (9 files)
rm src/compilation/pipeline/step1_constants.jl
rm src/compilation/pipeline/step2_categorical.jl  
rm src/compilation/pipeline/step3_functions.jl
rm src/compilation/pipeline/step4_interactions.jl
rm src/compilation/pipeline/step4_function_interactions.jl

# Old compilation system (2 files)
rm src/compilation/term_compiler.jl
rm src/compilation/legacy_compiled.jl

# Old evaluator system (3 files)
rm src/evaluation/evaluators.jl
rm src/evaluation/data_access.jl
rm src/evaluation/function_ops.jl

# Remove empty directories
rmdir src/compilation/pipeline/
rmdir src/compilation/
rmdir src/evaluation/
```

## Files to Keep

### Core Unified System
- `src/unified/` (all files) - The new compilation system
- `src/integration/mixed_models.jl` - Required for `fixed_effects_form()`
- `src/dev/testing_utilities.jl` - Required for test data and formulas

### Integration Components (Future Development)
- `src/core/utilities.jl` - `not()` and `OverrideVector` utilities
- `src/scenarios/overrides.jl` - Scenario analysis system  
- `src/evaluation/modelrow.jl` - High-level interface

These will be integrated with the unified system in future development.

## Dependencies Analysis

### What the Unified System Requires
1. **StatsModels integration**: Uses `StatsModels.formula()`, `FormulaTerm`, `MatrixTerm`, etc.
2. **Mixed models support**: Uses `fixed_effects_form()` from `integration/mixed_models.jl`  
3. **Test utilities**: Uses `make_test_data()` and `test_formulas` from `dev/testing_utilities.jl`
4. **External packages**: StatsModels, GLM, MixedModels, CategoricalArrays, Tables, BenchmarkTools

### What Gets Removed
1. **Step-based architecture**: All step1-4 files and coordination logic
2. **Evaluator hierarchy**: Abstract evaluator types and dispatch system
3. **Legacy compilation**: Old `compile_formula()` and `SpecializedFormula`
4. **Term compiler**: Complex term processing system

## Risk Assessment

### Low Risk ✅
- **Phase 1 & 2**: Only updates existing functionality to use new system
- **Unified system**: Thoroughly tested (100% zero allocations on 35 formulas)
- **Integration components**: Kept for future use

### Medium Risk ⚠️
- **Phase 3**: File deletion is irreversible
- **Unused test files**: Many `test_override_*.jl` files reference old system but aren't in active test suite

### Mitigations
1. **Git safety**: All changes in version control
2. **Test-driven**: Update tests first, ensure they pass before file removal
3. **Staged approach**: Complete each phase before proceeding
4. **Backup plan**: Can revert commits if issues discovered

## Testing Strategy

### Phase 1 Testing
```bash
# After updating test_models.jl:
julia --project=. test/test_models.jl
julia --project=. test/test_allocation_survey.jl
```

### Phase 2 Testing  
```bash
# After updating FormulaCompiler.jl:
julia --project=. -e "using FormulaCompiler"  # Should load without error
julia --project=. test/runtests.jl           # Should pass all active tests
```

### Phase 3 Testing
```bash
# After file removal:
julia --project=. -e "using Pkg; Pkg.test()"  # Full test suite
```

## Success Criteria

- [ ] `test_models.jl` passes with unified system
- [ ] `test_allocation_survey.jl` continues to pass (100% zero allocations)
- [ ] `FormulaCompiler.jl` loads without errors
- [ ] All active tests pass after file removal
- [ ] Package size reduced by removing ~3000+ lines of obsolete code
- [ ] Clean architecture with only unified system active

## Post-Removal Benefits

1. **Simplified codebase**: Single compilation system instead of two
2. **Reduced maintenance**: No coordination between old and new systems  
3. **Better performance**: Only the optimized unified system active
4. **Cleaner API**: Remove confusing dual `compile_formula()` functions
5. **Future-ready**: Integration components ready for unified system integration

## Timeline

- **Phase 1**: 15 minutes (update test_models.jl)
- **Phase 2**: 30 minutes (update FormulaCompiler.jl, test loading)  
- **Phase 3**: 15 minutes (remove files, final testing)
- **Total**: ~1 hour for complete transition

## Next Steps

1. Execute Phase 1: Update `test_models.jl` 
2. Execute Phase 2: Update `FormulaCompiler.jl`
3. Execute Phase 3: Remove obsolete files
4. Plan integration of override/scenario system with unified architecture
5. Update documentation to reflect new architecture

---

*This removal plan transitions FormulaCompiler from a dual-architecture system to a unified, high-performance compilation system while preserving components needed for future integration work.*