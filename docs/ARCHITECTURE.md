# FormulaCompiler.jl Architecture Guide

## Overview

FormulaCompiler.jl achieves **100% zero-allocation** performance across all formula types through a **unified compilation pipeline** that transforms statistical formulas into specialized, type-stable execution paths. This guide explains the current architecture after achieving complete zero-allocation performance.

## Core Philosophy

**Unified position mapping**: Transform all formula operations into compile-time position-specialized operations that execute with zero runtime allocation through careful Julia compiler cooperation.

## System Architecture

### Unified Compilation Pipeline

The package uses a single, unified compilation system that replaced the previous multi-step approach:

- **Location**: `src/compilation/`
- **Entry Point**: `compile_formula(model, data)`
- **Result**: `UnifiedCompiled` - tuple-based specialized execution
- **Performance**: ~50ns per row, **zero allocations** across all formula types
- **Achievement**: 100% success rate across 105 test cases

## Core Components

### 1. Formula Decomposition
- **File**: `src/compilation/decomposition.jl`
- **Purpose**: Converts StatsModels formulas into sequences of primitive operations
- **Key Function**: `decompose_formula(formula, data_example)`
- **Output**: Vector of operations, scratch size requirements, output size

### 2. Operation Types
- **File**: `src/compilation/types.jl`
- **Core Types**:
  - `LoadOp{Col, Out}` - Load column data with compile-time positions
  - `ConstantOp{Val, Out}` - Constant values
  - `UnaryOp{Func, In, Out}` - Mathematical functions
  - `BinaryOp{Func, In1, In2, Out}` - Binary operations
  - `ContrastOp{Col, Positions}` - Categorical contrast matrices
  - `CopyOp{InPos, OutIdx}` - Copy operations to output

**Key Innovation**: All operations embed positions as type parameters, enabling complete compile-time specialization.

### 3. Execution Engine
- **File**: `src/compilation/execution.jl`
- **Core Type**: `UnifiedCompiled{Ops, ScratchSize, OutputSize}`
- **Key Features**:
  - **Hybrid dispatch**: Empirically tuned threshold (≤25 ops: recursive, >25 ops: @generated)
  - **Position mapping**: All array accesses use compile-time indices
  - **Zero allocation**: Achieved through type specialization and pre-allocated scratch space

#### Execution Strategy

```julia
function (f::UnifiedCompiled)(output, data, row_idx)
    scratch = f.scratch  # Pre-allocated, reused
    fill!(scratch, 0.0)  # Clear for this row
    execute_ops(f.ops, scratch, data, row_idx)  # Process operations
    copy_outputs_from_ops!(f.ops, output, scratch)  # Transfer results
end
```

**Critical Insight**: The `RECURSION_LIMIT = 25` threshold was empirically determined to handle Julia's heuristic tuple specialization behavior reliably.

### 4. Operation Execution
- **Individual operation execution**: Each operation type has specialized `execute_op` methods
- **Compile-time dispatch**: All positions embedded in type parameters
- **Example**:
```julia
@inline function execute_op(::LoadOp{Col, Out}, scratch, data, row_idx) where {Col, Out}
    scratch[Out] = Float64(getproperty(data, Col)[row_idx])
end
```

## Runtime Execution System

### High-Level Interface
- **File**: `src/evaluation/modelrow.jl`
- **Key Functions**:
  - `modelrow!(output, compiled, data, row_idx)` - In-place evaluation
  - `modelrow(compiled, data, row_idx)` - Allocating version
  - `ModelRowEvaluator` - Object-based interface

### Scratch Space Management
- **File**: `src/compilation/scratch.jl`
- **Innovation**: Each compiled formula gets precisely sized scratch buffer
- **Efficiency**: Reused across all evaluations, no runtime allocation
- **Safety**: Size determined at compile time based on operation requirements

## Scenario and Override System

### Memory-Efficient "What-If" Analysis
- **File**: `src/scenarios/overrides.jl`
- **Key Innovation**: `OverrideVector{T}` - lazy constant vector
- **Memory**: ~32 bytes vs MBs for full arrays
- **Use Cases**: Policy analysis, counterfactual scenarios
- **Types**: `DataScenario`, `ScenarioCollection`

**Example**:
```julia
# Traditional: 8MB allocation
traditional = fill(42.0, 1_000_000)

# FormulaCompiler: ~32 bytes
efficient = OverrideVector(42.0, 1_000_000)
```

## External Integration

### Package Support
- **File**: `src/integration/mixed_models.jl`
- **Purpose**: Extract fixed effects from mixed-effects models
- **Function**: `get_fixed_effects_formula(mixed_model)`

### Supported Ecosystems
- **GLM.jl**: Linear and generalized linear models
- **MixedModels.jl**: Mixed-effects models (fixed effects extraction)
- **StandardizedPredictors.jl**: ZScore standardization
- **CategoricalArrays.jl**: All categorical types and contrasts
- **Tables.jl**: Various table formats (optimized for `Tables.columntable`)

## Development Infrastructure

### Core Utilities
- **File**: `src/core/utilities.jl`
- **Key Types**: `OverrideVector`, utility functions

### Testing Framework
- **File**: `src/dev/testing_utilities.jl`
- **Key Functions**: 
  - `make_test_data()` - Generate test datasets
  - `test_zero_allocation(model, data)` - Allocation testing
  - `test_formulas` - Comprehensive formula test suite

## Data Flow Architecture

```
Statistical Model (GLM, LMM, etc.)
    ↓
Schema-Applied Formula Extraction
    ↓
Formula Decomposition
    ↓ 
Operation Vector Generation
    ↓
Tuple Conversion & Type Specialization
    ↓
UnifiedCompiled Creation
    ↓
Runtime Execution (Zero Allocation)
    ↓
Output Vector
```

## Performance Architecture

### Compilation Strategy Selection

The system uses **hybrid dispatch** to handle Julia's heuristic compilation behavior:

```julia
@inline function execute_ops(ops::Tuple, scratch, data, row_idx)
    if length(ops) <= RECURSION_LIMIT  # 25
        execute_ops_recursive(ops, scratch, data, row_idx)
    else
        execute_ops_generated(ops, scratch, data, row_idx)  
    end
end
```

**Why this works**:
- **≤25 operations**: Julia reliably specializes recursive tuple execution
- **>25 operations**: @generated functions force complete specialization
- **No gray zone**: Avoids unpredictable 26-40 operation range

### Zero-Allocation Guarantees

1. **Position Mapping**: All array accesses use compile-time indices
2. **Pre-allocation**: Scratch space allocated once, reused
3. **Type Specialization**: Operations dispatched on type parameters
4. **Column Access**: `getproperty(data, :column)` compiled to direct access

## Critical Technical Insights

### The Final Fix: RECURSION_LIMIT Tuning

The last allocation issue was solved by recognizing that **Julia's tuple specialization is heuristic-based**:

- **Problem**: Complex formulas hit Julia's unpredictable specialization zone
- **Solution**: Lowered threshold from 35 → 25 for reliable specialization
- **Result**: 100% zero allocation across all 105 test cases
- **Learning**: Conservative empirical tuning beats theoretical limits

### Position Mapping System

**Core Innovation**: Embed all position information in type parameters:

```julia
# Traditional (runtime): scratch[runtime_position] = data[runtime_column][row]
# FormulaCompiler (compile-time): scratch[3] = getproperty(data, :x)[row]

LoadOp{:x, 3}()  # Column :x → scratch position 3 (known at compile time)
```

## Testing and Validation

### Comprehensive Test Suite
- **105 test cases** across all model types (LM, GLM, LMM, GLMM)
- **Zero allocation verification** for every formula type
- **Correctness testing** against `modelmatrix()` results
- **Performance benchmarking** with allocation monitoring

### Test Categories
1. **Simple formulas**: Basic terms and continuous variables
2. **Categorical variables**: All contrast matrix types
3. **Functions**: Mathematical operations (`log`, `exp`, `sqrt`, etc.)
4. **Interactions**: Including complex multi-way interactions
5. **Mixed formulas**: Combinations of all above

## Directory Structure Rationale

```
src/
├── FormulaCompiler.jl          # Main module, exports
├── core/utilities.jl           # Basic utilities
├── compilation/                # Unified compilation system
│   ├── compilation.jl          # Main entry points
│   ├── decomposition.jl        # Formula → operations
│   ├── types.jl                # Operation type definitions
│   ├── execution.jl            # Runtime execution engine
│   └── scratch.jl              # Scratch space management
├── evaluation/modelrow.jl      # High-level API
├── scenarios/overrides.jl      # Override system
├── integration/mixed_models.jl # External packages
└── dev/testing_utilities.jl    # Development tools
```

**Design Principles**:
1. **Logical grouping**: Related functionality together
2. **Clear separation**: Compilation vs runtime vs integration
3. **Development support**: Debug/test tools separate from core
4. **Scalable**: Easy to extend without restructuring

## Future Directions

### Completed Achievements ✅
- **100% zero allocation** across all formula types
- **Universal formula support** for StatsModels.jl
- **Robust performance** handling Julia's compilation heuristics
- **Complete ecosystem integration**

### Potential Extensions
1. **Analytical derivatives**: Foundation exists in `future/` directory
2. **GPU acceleration**: Position mapping enables GPU kernels
3. **Additional optimizations**: Cache-friendly execution patterns
4. **Extended ecosystem**: More specialized model types

## Summary

The unified architecture successfully achieves the primary goal: **100% zero-allocation, high-performance formula evaluation** across all statistical model types. The key innovations—position mapping, hybrid dispatch, and Julia-aware compilation—create a robust foundation that handles real-world formula complexity while maintaining exceptional performance.