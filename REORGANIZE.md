# FormulaCompiler.jl Reorganization Plan

## Executive Summary

This plan reorganizes the FormulaCompiler.jl codebase from a flat 16-file structure into a logical, hierarchical organization that clarifies the system architecture and prepares for fixing allocation issues in Steps 3 & 4.

## Current Problems

### Structural Issues
- **Flat directory**: All 16 files in `/src` with no logical grouping
- **Confusing names**: `step1_specialized_core.jl` vs `CompiledFormula.jl` naming inconsistency  
- **Mixed concerns**: Debug code (`debug_step2.jl`) mixed with production code
- **Massive files**: `step4_interactions.jl` (1,911 lines), `compile_term.jl` (885 lines)
- **Unclear relationships**: Two compilation systems with unclear boundaries

### Development Impact
- Hard to understand system architecture and execution flow
- Difficult to identify where allocation issues occur (Steps 3 & 4)
- New contributors cannot easily navigate the codebase
- Allocation fixes require understanding system-wide type propagation patterns

## Target Structure

```
src/
├── FormulaCompiler.jl              # Main module (imports only)
├── core/                           # Core types and interfaces
│   ├── types.jl                    # Central type definitions
│   ├── interfaces.jl               # AbstractEvaluator hierarchy  
│   └── utilities.jl                # Helper functions (not(), etc.)
├── compilation/                    # Multi-level compilation system
│   ├── legacy_compiled.jl          # Level 1: CompiledFormula (backward compatibility)
│   ├── term_compiler.jl            # Term compilation engine
│   └── pipeline/                   # Level 2: 4-step specialization pipeline
│       ├── step1_constants.jl      # Constants + continuous (ZERO ALLOCATION ✅)
│       ├── step2_categorical.jl    # Categorical variables (ZERO ALLOCATION ✅)
│       ├── step3_functions.jl      # Functions (~32 bytes allocation ⚠️)
│       └── step4_interactions.jl   # Interactions (96-864+ bytes ⚠️)
├── evaluation/                     # Runtime execution system
│   ├── evaluators.jl              # Evaluator implementations
│   ├── data_access.jl             # Column access patterns
│   ├── function_ops.jl            # Function application
│   └── modelrow.jl                # High-level modelrow interface
├── scenarios/                      # Override and scenario system  
│   ├── overrides.jl               # OverrideVector and basic overrides
│   ├── scenarios.jl               # DataScenario and collections
│   └── scenario_integration.jl    # Integration with compilation pipeline
├── integration/                    # External package support
│   ├── mixed_models.jl            # MixedModels.jl support
│   ├── glm_support.jl             # GLM.jl integration
│   └── tables_support.jl          # Tables.jl integration  
└── dev/                           # Development and debugging
    ├── testing_utilities.jl       # Testing helper functions
    ├── benchmarks.jl              # Performance benchmarking
    └── debug_tools.jl             # Debug utilities
```

## File Mapping Plan

### Phase 1: Core Infrastructure

| Current File | New Location | Rationale |
|--------------|-------------|-----------|
| `FormulaCompiler.jl` | `FormulaCompiler.jl` | Simplified to imports + exports only |
| `evaluators.jl` | `core/interfaces.jl` + `evaluation/evaluators.jl` | Split abstract types from implementations |
| `fixed_helpers.jl` | `integration/mixed_models.jl` + `core/utilities.jl` | Separate MixedModels support from core utilities |

### Phase 2: Compilation System  

| Current File | New Location | Rationale |
|--------------|-------------|-----------|
| `CompiledFormula.jl` | `compilation/legacy_compiled.jl` | Clear this is Level 1 (legacy) system |
| `compile_term.jl` | `compilation/term_compiler.jl` | More descriptive name |
| `step1_specialized_core.jl` | `compilation/pipeline/step1_constants.jl` | Clear what this step handles |
| `step2_categorical_support.jl` | `compilation/pipeline/step2_categorical.jl` | Consistent naming |
| `step3_functions.jl` | `compilation/pipeline/step3_functions.jl` | Move to pipeline directory |
| `step4_interactions.jl` | `compilation/pipeline/step4_interactions.jl` | Move to pipeline directory |

### Phase 3: Execution & Support

| Current File | New Location | Rationale |
|--------------|-------------|-----------|
| `apply_function.jl` | `evaluation/function_ops.jl` | More descriptive name |
| `get_data_value_specialized.jl` | `evaluation/data_access.jl` | Clear purpose |
| `modelrow.jl` | `evaluation/modelrow.jl` | High-level interface |
| `override_unified.jl` | `scenarios/overrides.jl` + `scenarios/scenarios.jl` | Split by functionality |
| `generated_function_interactions.jl` | `compilation/pipeline/step4_interactions.jl` | Merge with interactions |

### Phase 4: Development Support

| Current File | New Location | Rationale |
|--------------|-------------|-----------|
| `testing.jl` | `dev/testing_utilities.jl` | Clear it's dev support |
| `debug_step2.jl` | `dev/debug_tools.jl` | Move debug code out of production |

## Implementation Steps

### Step 1: Create Directory Structure
```bash
mkdir -p src/{core,compilation/pipeline,evaluation,scenarios,integration,dev}
```

### Step 2: Move Files with Git History Preservation
```bash
# Core infrastructure
git mv src/evaluators.jl src/evaluation/evaluators.jl
git mv src/fixed_helpers.jl src/integration/mixed_models.jl

# Compilation system  
git mv src/CompiledFormula.jl src/compilation/legacy_compiled.jl
git mv src/compile_term.jl src/compilation/term_compiler.jl
git mv src/step1_specialized_core.jl src/compilation/pipeline/step1_constants.jl
git mv src/step2_categorical_support.jl src/compilation/pipeline/step2_categorical.jl
git mv src/step3_functions.jl src/compilation/pipeline/step3_functions.jl  
git mv src/step4_interactions.jl src/compilation/pipeline/step4_interactions.jl

# Evaluation system
git mv src/apply_function.jl src/evaluation/function_ops.jl
git mv src/get_data_value_specialized.jl src/evaluation/data_access.jl
git mv src/modelrow.jl src/evaluation/modelrow.jl

# Scenarios
git mv src/override_unified.jl src/scenarios/overrides.jl

# Development
git mv src/testing.jl src/dev/testing_utilities.jl
git mv src/debug_step2.jl src/dev/debug_tools.jl
```

### Step 3: Update Include Statements in FormulaCompiler.jl
```julia
# Core types and interfaces
include("core/utilities.jl")
include("core/interfaces.jl")

# Compilation system
include("compilation/term_compiler.jl")
include("compilation/legacy_compiled.jl")
include("compilation/pipeline/step1_constants.jl")
include("compilation/pipeline/step2_categorical.jl")
include("compilation/pipeline/step3_functions.jl")
include("compilation/pipeline/step4_interactions.jl")

# Evaluation system
include("evaluation/evaluators.jl")
include("evaluation/data_access.jl")
include("evaluation/function_ops.jl")
include("evaluation/modelrow.jl")

# Scenarios and overrides
include("scenarios/overrides.jl")

# External integration
include("integration/mixed_models.jl")

# Development utilities (only in dev builds)
include("dev/testing_utilities.jl")
```

### Step 4: File Content Reorganization

#### Split Large Files
- **`step4_interactions.jl` (1,911 lines)**: Consider splitting into:
  - `step4_interactions/types.jl` - Type definitions
  - `step4_interactions/operations.jl` - Operation implementations  
  - `step4_interactions/compilation.jl` - Compilation logic
  - `step4_interactions/execution.jl` - Runtime execution

#### Extract Common Types  
Create `src/core/types.jl` with:
- All `struct` definitions used across multiple files
- Type aliases and constants
- Core abstract types

#### Clean Up Dependencies
- Remove development dependencies from production imports
- Clarify which functions are exported vs internal
- Document inter-file dependencies

### Step 5: Update Documentation

#### Update CLAUDE.md
- Reflect new file structure in development commands
- Update architecture description with new organization
- Add file navigation guide for new structure

#### Update README.md  
- No changes needed (public API unchanged)

#### Create Architecture Guide
New file: `docs/ARCHITECTURE.md` explaining:
- Directory structure rationale
- Data flow between components
- Where to make common changes

## Validation Plan

### Step 1: Compilation Check
```bash
julia --project=. -e "using FormulaCompiler"
```

### Step 2: Test Suite Validation  
```bash
julia --project=. -e "include(\"test/runtests.jl\")"
```
- **Success criteria**: All 234 tests still pass
- **Expected result**: Same 25.9s runtime

### Step 3: Performance Regression Check
```bash  
julia --project=. -e "include(\"test/test_performance.jl\")"
```
- **Success criteria**: No performance regression
- **Key metrics**: Still ~50ns per row for simple formulas

### Step 4: Allocation Status Verification
- **Zero allocation**: Constants, continuous variables (Steps 1 & 2)
- **Known allocations**: Functions (~32 bytes), Interactions (96-864+ bytes)
- **Success criteria**: Allocation patterns unchanged

## Rollback Plan

### Git-Based Rollback
```bash
# If issues found after reorganization
git revert <reorganization-commit-sha>

# Or reset to pre-reorganization state  
git reset --hard <pre-reorganization-commit-sha>
```

### Incremental Rollback
- Each phase can be rolled back independently
- File moves are reversible with git
- Include statement changes are easily reverted

## Risk Mitigation

### Low Risk Items ✅
- **File moves**: Preserve git history, no logic changes
- **Directory creation**: No impact on existing code
- **Include statement updates**: Mechanical changes

### Medium Risk Items ⚠️
- **Large file splits**: Could introduce subtle bugs
- **Dependency reorganization**: Might affect initialization order
- **Type extraction**: Could break type inference

### High Risk Items ❌  
- **None identified**: This is purely structural reorganization

## Benefits After Reorganization

### For Allocation Fixes
- **Clear reference implementation**: `pipeline/step1_constants.jl` shows working `Val{Column}` pattern
- **Problem isolation**: Issues clearly in `pipeline/step3_functions.jl` and `pipeline/step4_interactions.jl`
- **Type flow visibility**: Easy to trace type propagation through pipeline stages

### For Development
- **Logical navigation**: Developers can find code by purpose
- **Clear boundaries**: Each directory has single responsibility
- **Scalable structure**: Easy to add new features without cluttering

### For Maintenance
- **Easier debugging**: Problems localized to specific directories
- **Cleaner testing**: Test what matters without touching debug code
- **Documentation alignment**: Structure matches conceptual model

## Timeline Estimate

- **Phase 1-2 (File moves)**: 2-3 hours
- **Phase 3 (Include updates)**: 1 hour  
- **Phase 4 (Content reorganization)**: 4-6 hours
- **Phase 5 (Documentation)**: 1-2 hours
- **Validation & testing**: 1-2 hours

**Total estimate**: 9-14 hours of focused work

## Success Metrics

1. **✅ All tests pass**: 234 tests, <30s runtime
2. **✅ No performance regression**: Allocation patterns unchanged  
3. **✅ Clear structure**: New developers can navigate easily
4. **✅ Ready for allocation fixes**: Clear path to propagate `Val{Column}` pattern
5. **✅ Git history preserved**: `git log --follow` works for moved files

This reorganization sets up FormulaCompiler.jl for successful resolution of the remaining allocation issues by providing clear architectural boundaries and making the working zero-allocation patterns (Steps 1 & 2) obvious reference implementations for fixing the allocation issues in Steps 3 & 4.