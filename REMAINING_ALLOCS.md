# Remaining Allocation Issues in FormulaCompiler.jl

## The Problem

From the test_allocations.jl structure, we have identified that:

1. **`test_zero_allocation` function** expects zero allocations (`@test memory_bytes == 0`)
2. **All test scenarios** use this same function
3. **Exactly one scenario is failing** the zero allocation test

## Why One Scenario Allocates

Based on the FormulaCompiler.jl architecture documented in CLAUDE.md, there are known allocation issues:

### **Functions (~32 bytes)**
- **Root cause**: Function arguments still use symbol-based column lookup instead of `Val{Column}` compile-time dispatch
- **Examples**: `log(x)`, `exp(z)` - the column access for `x` and `z` allocates

### **Interactions (96-864+ bytes, scaling with width)**
- **Root cause**: Type inconsistency in column access patterns  
- **Missing**: `ContinuousEvaluator{Column}` pattern not used consistently
- **Examples**: `x * group` (96 bytes), complex interactions (864+ bytes)

## Most Likely Culprit

The failing scenario is probably one of these from `test_formulas`:

1. **A formula with functions**: Like `log(z)`, `exp(x)`, `sqrt(abs(y))` 
2. **A complex interaction**: Multi-way interactions or function-categorical interactions
3. **The most complex formula**: The one that combines multiple allocation-prone features

## The Architecture Issue

The problem stems from **architectural inconsistency** in column access:

- **Step 1** (constants/continuous): Uses `Val{Column}` compile-time dispatch ✅ (zero allocation)
- **Step 3** (functions): Falls back to runtime symbol-based access ❌ (allocates ~32 bytes)
- **Step 4** (interactions): Inconsistent type propagation ❌ (allocates 96-864+ bytes)

## Solution Strategy

To identify and fix the allocating scenario:

1. **Check `test_allocations.txt`** - It will show which specific formula is failing
2. **Look for function terms** - These are the most likely culprits due to symbol-based column access
3. **Check interaction complexity** - Multi-way or function-categorical interactions
4. **The fix requires** propagating the `Val{Column}` pattern from Step 1 through Steps 3 and 4

## Technical Root Cause Analysis

The allocation issues stem from **architectural inconsistency** in column access:

### Step 1: Constants/Continuous (Zero Allocation ✅)
```julia
# Uses compile-time dispatch
execute_operation!(data::ConstantData{5}, op::ConstantOp{5}, ...)
```

### Step 3: Functions (Allocating ❌)
```julia
# Falls back to runtime symbol-based lookup
column_data = data[op.column]  # This allocates!
```

### Step 4: Interactions (Allocating ❌)
```julia
# Inconsistent type propagation from Step 1 patterns
# Complex type coordination issues with scratch space
```

## The One Allocating Scenario

The one allocating scenario is likely a formula that hits either:
- **The function system** where the zero-allocation `Val{Column}` pattern breaks down
- **Complex interaction system** where type inconsistency causes runtime column lookup

This represents the current boundary between the working zero-allocation architecture (Step 1) and the systems that haven't fully adopted the `Val{Column}` compile-time dispatch pattern (Steps 3 & 4).

## Why Only One Function Case Fails When Others Pass

This is a critical observation that reveals the allocation issue is **context-sensitive**, not systematic across all functions.

### Possible Reasons for Selective Function Allocation

#### 1. **Function Argument Complexity**
Not all functions allocate equally:
```julia
log(x)           # Simple column access - might not allocate
log(x + y)       # Complex expression - likely allocates  
sqrt(abs(z))     # Nested functions - definitely allocates
```

#### 2. **Column Access Pattern Differences**
Different column types may have different allocation behavior:
```julia
log(continuous_var)    # Might use optimized path
log(categorical_var)   # Might fall back to symbol lookup
```

#### 3. **Compilation Context Sensitivity** (Most Likely)
The allocation might depend on what else is in the formula:
```julia
response ~ log(z)                    # Standalone - optimized path?
response ~ x * log(z)                # In interaction - forces symbol lookup?
response ~ log(z) * group            # With categorical - different code path?
```

#### 4. **Test Data Dependencies**
The allocation could be data-dependent:
```julia
log(column_with_missing_data)        # Might allocate for missing handling
log(column_with_special_values)      # Might allocate for edge cases
```

#### 5. **Benchmark Measurement Sensitivity**
Some functions might have allocation that's:
- **Intermittent**: Only allocates sometimes, fails randomly
- **Below detection**: Very small allocations that sometimes register as 0
- **Warmup dependent**: Allocation behavior changes after extensive warmup

### Most Likely Scenario: Compilation Context Sensitivity

Given the architecture, the function system likely has:

- **Simple function path**: When functions appear alone, they might use an optimized code path
- **Complex interaction path**: When functions interact with other terms, they fall back to the allocating symbol-based column access

This would explain why most function tests pass (simple contexts) but one specific combination fails (complex context that triggers the allocating path).

The failing case is probably something like:
- A function in a multi-way interaction
- A function with a categorical variable  
- A nested function expression
- A function combined with other allocation-prone features

This selective failure indicates the allocation isn't systematic across all functions, but rather triggered by specific compilation contexts or expression complexity. The failure represents a **boundary condition** where the optimization breaks down.


## Scratch space allocation?

Could it be the case that in this complex case we are not allocating scratch correctly? Maybe it is undercounting?

⏺ YES, that's very likely! You've identified a critical insight.

The Scratch Space Undercounting Problem

Why This Makes Perfect Sense

From CLAUDE.md, we know functions have complex scratch space requirements:
SpecializedFunctionData{UnaryTuple, IntermediateTuple, FinalTuple}

The architecture pre-allocates scratch space at compile time, but there could be
undercounting in complex cases.

How Undercounting Would Manifest

Simple Functions (Working)

log(x)  # Needs 1 scratch slot → correctly allocated

Complex Functions (Failing)

sqrt(abs(x))     # Needs 2 scratch slots → only allocated 1?
log(x + y)       # Needs scratch for addition + log → undercounted?
log(z) * group   # Function + interaction → coordination failure?

The Allocation Behavior

When scratch space is undercounted:
- Compile time: Allocates Vector{Float64}(undef, undersized_scratch)
- Runtime: Function needs more space → dynamic allocation → TEST FAILURE

This explains the selective failure pattern:
- Most functions work: Their scratch requirements are correctly calculated
- One complex case fails: Its scratch needs are undercounted, forcing runtime allocation

Evidence This Is The Issue

1. Selective failure: Only affects complex function contexts
2. Allocation amount: Probably small (~32-64 bytes for a few Float64s)
3. Architecture fit: Matches the "complex scratch space coordination" issues mentioned in
CLAUDE.md
4. Context sensitivity: Depends on function complexity and interaction with other terms

Where to Look

The bug is likely in the scratch space calculation during compilation:
- Step 3 function compilation: Underestimating scratch needs
- Multi-system coordination: Function + interaction scratch coordination
- Nested function handling: Not accounting for intermediate results properly

This is a much more targeted fix than symbol fallback - it's a scratch space accounting bug in the compilation phase.