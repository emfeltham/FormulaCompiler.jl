# FormulaCompiler.jl Architecture Guide

## Overview

FormulaCompiler.jl is a high-performance statistical model evaluation system that achieves **zero-allocation performance** (~50ns per row) through **compile-time type specialization** and **tuple-based execution**. This guide explains the system architecture after the major reorganization completed in August 2024.

## Core Philosophy

**Trade implementation complexity for runtime speed**: Move all expensive computations to compile time, leaving only simple, type-stable operations for runtime evaluation.

## System Architecture

### Two-Level Compilation System

#### Level 1: Legacy System (Backward Compatibility)
- **Location**: `src/compilation/legacy_compiled.jl`
- **Type**: `CompiledFormula`
- **Performance**: ~100ns per row (good but not optimal)
- **Purpose**: Maintain backward compatibility with existing code
- **Structure**: Evaluator trees with some runtime dispatch

#### Level 2: Specialized System (High Performance)
- **Location**: `src/compilation/pipeline/`
- **Type**: `SpecializedFormula`  
- **Performance**: ~50ns per row, zero allocations
- **Purpose**: Maximum performance through complete compile-time specialization
- **Structure**: Four-step compilation pipeline with tuple-based execution

## Four-Step Compilation Pipeline

### Step 1: Constants and Continuous Variables ✅
- **File**: `compilation/pipeline/step1_constants.jl` (270 lines)
- **Status**: **ZERO ALLOCATION** - Reference implementation
- **Key Pattern**: `Val{Column}` compile-time dispatch
- **Types**: `ConstantData{N}`, `ContinuousData{N, Cols}`
- **Operations**: `ConstantOp{N}`, `ContinuousOp{N, Cols}`

**This is the gold standard** - the pattern that Steps 3&4 need to emulate.

### Step 2: Categorical Variables ✅  
- **File**: `compilation/pipeline/step2_categorical.jl` (341 lines)
- **Status**: **ZERO ALLOCATION** - Working perfectly
- **Key Feature**: Pre-computed contrast matrices with efficient lookups
- **Types**: `SpecializedCategoricalData{N, Positions}`

### Step 3: Functions ⚠️
- **Files**: 
  - `compilation/pipeline/step3_functions.jl` (entry point)
  - `compilation/pipeline/step3/types.jl` (function types)
  - `compilation/pipeline/step3/main.jl` (implementation)
- **Status**: **~32 bytes allocation** - Needs fixing
- **Issue**: Function arguments use symbol-based column lookup instead of `Val{Column}`
- **Types**: `SpecializedFunctionData{UnaryTuple, IntermediateTuple, FinalTuple}`

### Step 4: Interactions ⚠️
- **Files**:
  - `compilation/pipeline/step4_interactions.jl` (entry point)  
  - `compilation/pipeline/step4/types.jl` (interaction types)
  - `compilation/pipeline/step4/main.jl` (implementation)
- **Status**: **96-864+ bytes allocation** - Needs fixing  
- **Issue**: Inconsistent column access patterns, function arguments within interactions allocate
- **Types**: `CompleteInteractionData{IntermediateTuple, FinalTuple}`

## Runtime Execution System

### Core Components

#### Evaluator Hierarchy
- **Location**: `src/evaluation/evaluators.jl`
- **Purpose**: Abstract base types and concrete evaluator implementations
- **Key Types**: `AbstractEvaluator`, `ConstantEvaluator`, `ContinuousEvaluator`, etc.
- **Pattern**: Self-contained evaluators with their own position and scratch space requirements

#### Data Access Layer
- **Location**: `src/evaluation/data_access.jl`
- **Purpose**: Column access abstraction layer
- **Critical Issue**: This is where symbol-based vs `Val{Column}` access patterns are implemented
- **Fix Target**: Must propagate `Val{Column}` pattern from Step 1

#### ModelRow Interface
- **Location**: `src/evaluation/modelrow.jl` 
- **Purpose**: High-level API for model matrix evaluation
- **Key Functions**: `modelrow!(output, model, data, row_idx)`, `modelrow(model, data, row_idx)`

## Scenario and Override System

### Memory-Efficient "What-If" Analysis
- **Location**: `src/scenarios/overrides.jl`
- **Key Innovation**: `OverrideVector{T}` - lazy constant vector (~32 bytes vs MBs for full arrays)
- **Use Cases**: Policy analysis, counterfactual scenarios
- **Types**: `DataScenario`, `ScenarioCollection`

## External Integration

### Package Support
- **Location**: `src/integration/`
- **MixedModels**: `mixed_models.jl` - Fixed effects extraction from mixed-effects models
- **Future**: GLM, Tables.jl specific optimizations can be added here

### Supported Ecosystems
- **GLM.jl**: Linear and generalized linear models
- **MixedModels.jl**: Mixed-effects models
- **StandardizedPredictors.jl**: ZScore standardization
- **CategoricalArrays.jl**: All categorical types and contrasts
- **Tables.jl**: Various table formats (recommend `Tables.columntable`)

## Development and Debugging

### Development Tools
- **Location**: `src/dev/`
- **Testing**: `testing_utilities.jl` - Helper functions for testing and benchmarking
- **Debugging**: `debug_tools.jl` - Debug utilities (moved out of production src/)

### Testing Strategy
- **Location**: `test/`
- **Current Status**: 234 tests passing (25.6s runtime)
- **Key Tests**: `test_models.jl` (core functionality), allocation regression tests
- **Seed**: Fixed at `06515` for reproducible tests

## Data Flow Architecture

```
Input Data (Tables.jl format)
    ↓
Term Compilation (compilation/term_compiler.jl)
    ↓
Legacy OR Specialized Compilation
    ↓
┌─────────────────────────────────────────────────────────────┐
│ SPECIALIZED PIPELINE (4 steps)                             │
├─────────────────────────────────────────────────────────────┤
│ Step 1: Constants/Continuous (Val{Column} ✅)               │
│    ↓                                                        │
│ Step 2: Categorical (Contrast matrices ✅)                  │
│    ↓                                                        │
│ Step 3: Functions (Symbol-based ❌ → needs Val{Column})     │
│    ↓                                                        │
│ Step 4: Interactions (Mixed patterns ❌ → needs Val{Column})│
└─────────────────────────────────────────────────────────────┘
    ↓
Runtime Execution (evaluation/)
    ↓
Output Vector (zero allocations for Steps 1&2)
```

## Critical Insights for Allocation Fixes

### Root Cause Analysis

**The allocation problem is architectural inconsistency**:

1. **Steps 1 & 2**: Use `Val{Column}` compile-time dispatch → **Zero allocation**
2. **Steps 3 & 4**: Fall back to runtime symbol-based lookup → **Allocations**

### Solution Path

**Propagate the `Val{Column}` pattern from Step 1 through Steps 3 & 4**:

1. **Reference**: Study `step1_constants.jl` - how `ContinuousData{N, Cols}` uses `Val{Column}`
2. **Target**: Make function arguments in Step 3 use the same pattern
3. **Integration**: Ensure Step 4 interactions inherit the zero-allocation column access
4. **Abstraction**: Update `evaluation/data_access.jl` to support both patterns during transition

### Why Step 4 is 1,911 Lines

**Complex type propagation**: Step 4 tries to maintain zero-allocation guarantees while integrating with the function system that breaks the architectural pattern. The complexity arises from attempting to bridge two incompatible column access systems.

## Performance Characteristics

### Current State (Post-Reorganization)
- **Zero Allocation**: Constants, continuous variables, categorical variables
- **~32 bytes**: Function evaluation (symbol-based column access)
- **96-864+ bytes**: Interactions (scales with interaction width)

### Target State (After Allocation Fixes)
- **Zero Allocation**: All operations
- **Performance**: ~50ns per row for complex formulas
- **Memory**: ~32 bytes for override vectors vs MBs for full arrays

## Directory Rationale

### Why This Structure?

1. **Clear Problem Isolation**: Issues are contained to specific directories (`step3/`, `step4/`)
2. **Reference Implementation Obvious**: `step1_constants.jl` stands out as the pattern to follow
3. **Logical Grouping**: Related functionality lives together (`compilation/`, `evaluation/`, `scenarios/`)
4. **Scalable**: Easy to add new features without cluttering existing directories
5. **Development Friendly**: Debug/test code separated from production code

### Navigation Mental Model

- **Working on allocation fixes?** → Focus on `compilation/pipeline/`
- **Adding new model support?** → Look at `integration/`
- **Performance optimization?** → Start with `evaluation/`
- **Testing/debugging?** → Use `dev/` directory tools

## Future Architecture Evolution

### Short Term (Allocation Fixes)
1. Unify column access patterns around `Val{Column}`
2. Eliminate symbol-based runtime lookups in function arguments
3. Achieve zero allocation for all operations

### Medium Term (Performance)
1. Further optimize scratch space management
2. Explore @generated functions for complex interactions
3. Add more specialized evaluator types

### Long Term (Features)
1. Analytical derivatives system (partially implemented in `/future`)  
2. GPU acceleration for large datasets
3. Additional model ecosystem integration

This architecture provides a solid foundation for both fixing the remaining allocation issues and scaling the system for future development.