# Plan to Restore Zero-Allocation Concrete Types

## Overview

The FormulaCompiler system was designed to achieve zero allocations through Julia's natural type system, but the concrete type flow has been broken. Currently getting `ContinuousData{1, Tuple{Symbol}}` (abstract) instead of `ContinuousData{1, (:x,)}` (concrete), causing 16-byte allocations in basic cases.

## Phase 1: Fix Symbol Type Flow at Source

### 1. Trace the compilation pipeline
- Find where `ContinuousEvaluator` objects are created (likely in `compile_term`)
- Ensure symbols are captured as concrete types from the very beginning
- Fix the type flow so concrete symbols propagate naturally through the system

### 2. Fix `analyze_continuous_operations`
- The `ntuple` should receive concrete symbols, not abstract `Symbol` types
- Ensure natural type inference creates `Tuple{:x}` instead of `Tuple{Symbol}`
- No metaprogramming needed - just proper type flow

## Phase 2: Fix Data Access

### 3. Update execution to use concrete symbols
- Modify `execute_operation!` to use the existing `get_data_value_type_stable` with `Val{:x}`
- Or enable direct `data.x[row_idx]` access if types are truly concrete
- Remove usage of the "stupid" `get_data_value_specialized`

## Phase 3: Apply Same Fix to Functions and Interactions

### 4. Fix function and interaction type flow
- Apply the same concrete type flow fixes to function compilation
- Ensure interaction components inherit concrete types naturally
- No special metaprogramming - just consistent type propagation

## Phase 4: Verification

### 5. Test zero allocations are restored
- Verify `ContinuousData{1, (:x,)}` instead of `ContinuousData{1, Tuple{Symbol}}`
- Confirm allocation survey shows zero allocations for all basic cases

## Core Principle

Fix the **upstream type flow** so that concrete symbol types flow naturally through the entire system without needing any metaprogramming tricks. The zero-allocation system should work through Julia's natural type inference and specialization.

## Current Status

- **Constants**: ✅ Zero allocations (working correctly)
- **Categoricals**: ✅ Zero allocations (working correctly) 
- **Continuous variables**: ❌ 16-byte allocations (broken type flow)
- **Functions**: ❌ 32-byte allocations (broken type flow)
- **Interactions**: ❌ 100s of bytes allocations (broken type flow)

## Target Types

**Current (broken)**:
- `ContinuousData{1, Tuple{Symbol}}` - abstract type, requires runtime dispatch
- `UnaryFunctionData{typeof(log), Symbol}` - abstract symbol type

**Target (working)**:
- `ContinuousData{1, (:x,)}` - concrete symbol in type
- `UnaryFunctionData{typeof(log), :z}` - concrete symbol in type

The key is ensuring symbols become compile-time type parameters, not runtime values.