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

## Targeted Fix Plan - Actual Root Cause Identified

### Investigation Results

Through systematic testing, we've identified the exact allocation issue:

#### **Test Case Analysis**
- **Failing formula**: `logistic_response ~ x * y * group3 + log(abs(z)) + group4` → **96 bytes**
- **Working variant**: `logistic_response ~ x * y * group4 + log(abs(z)) + group3` → **0 bytes**

#### **Key Discoveries**
1. **NOT scratch space undercounting** - scratch allocation is correct (20 positions)
2. **NOT systematic function/interaction issues** - individual components work fine:
   - 3-way interactions alone: 0 bytes ✅
   - Functions alone: 0 bytes ✅  
   - All combinations except the exact failing case: 0 bytes ✅
3. **Specific categorical dimension issue**:
   - `group3`: 3 levels → 2 contrast columns
   - `group4`: 4 levels → 3 contrast columns
   - **Swapping them fixes the allocation!**

#### **Operation Count Anomaly**
- **Failing case**: 33 operations, 96 bytes ❌
- **Working case**: 40 operations, 0 bytes ✅

The working version has **more operations**, suggesting the failing case takes a "shortcut" optimization that causes allocation.

### Root Cause Analysis

The 96-byte allocation is caused by a **type inference failure** or **memory allocation in ContrastOp execution** for specific categorical dimension combinations during complex formula compilation.

#### **Evidence Points To:**
1. **ContrastOp execution path** - categorical contrast operations
2. **Dimensional interaction** - specific combination of 2-column + 3-column categoricals  
3. **Compilation optimization bug** - certain patterns trigger fallback allocation path
4. **Type instability** - 96 bytes suggests vector allocation during contrast computation

### Targeted Fix Strategy

#### **Step 1: Locate the Allocation Source**
1. **Profile the exact failing case** with allocation tracking
2. **Examine ContrastOp execution** in `src/compilation/execution.jl:336-350`
3. **Check categorical dimension handling** in interaction compilation
4. **Look for type instability** in contrast matrix operations

#### **Step 2: Identify the Compilation Path Difference**  
```julia
# Why does this fail?
x * y * group3 + log(abs(z)) + group4  # 33 ops, 96 bytes

# But this works?  
x * y * group4 + log(abs(z)) + group3  # 40 ops, 0 bytes
```

**Hypothesis**: The compilation system has different optimization paths based on categorical ordering/dimensions that cause one path to allocate.

#### **Step 3: Fix the Type Instability**
Likely locations for the fix:

1. **ContrastOp execution** (`src/compilation/execution.jl:336`):
   ```julia
   # Current (potentially allocating):
   column_data = getproperty(data, Col)
   level = extract_level_code_zero_alloc(column_data, row_idx)
   
   # May need type stabilization for specific dimension combinations
   ```

2. **Categorical interaction compilation** (`src/compilation/decomposition.jl`):
   - Check interaction pattern generation for dimensional edge cases
   - Ensure consistent `Val{Column}` dispatch patterns

3. **Operation sequence generation**:
   - Fix the "shortcut" optimization that reduces operations but causes allocation
   - Ensure the full zero-allocation path is always taken

#### **Step 4: Implementation Plan**
1. **Add type annotations** to ensure stable ContrastOp execution
2. **Fix categorical dimension handling** in interaction compilation
3. **Remove problematic optimization** that causes the operation count reduction
4. **Add regression test** for the specific failing categorical dimension combination

#### **Step 5: Validation**
1. **Failing test passes**: `x * y * group3 + log(abs(z)) + group4` → 0 bytes
2. **No regressions**: All other allocation tests remain at 0 bytes
3. **Operation consistency**: Both variants should have similar operation counts
4. **Add stress tests**: Various categorical dimension combinations

### Expected Outcome
- **True zero-allocation achievement**: All 105 allocation tests pass
- **Robust categorical handling**: No dimension-specific allocation issues  
- **Consistent compilation**: Similar optimization paths regardless of categorical ordering
- **Clear understanding**: Document the categorical dimension interaction patterns that caused the issue

This represents the final allocation issue blocking true zero-allocation performance across all formula types.


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

## Plan to Resolve Scratch Space Allocation Issue

### Step 1: Identify the Failing Test Case
1. **Run the allocation test suite** and capture which specific formula fails
2. **Examine the failing formula** to understand its complexity:
   - Is it a nested function? (`sqrt(abs(x))`)
   - Function with arithmetic? (`log(x + y)`)
   - Function in interaction? (`log(z) * group`)
3. **Measure the actual allocation amount** to confirm it's scratch-space-sized (~32-64 bytes)

### Step 2: Trace Scratch Space Calculation
1. **Find scratch space calculation code** in the compilation pipeline:
   - Look in `src/compilation/scratch.jl`
   - Check Step 3 function compilation in `src/compilation/`
   - Review `SpecializedFunctionData` type definitions
2. **Add debug output** to see scratch space requirements vs. allocated:
   ```julia
   println("Function: $func, Calculated scratch: $calc_scratch, Actual needed: $actual")
   ```
3. **Compare simple vs complex cases** to see where calculation diverges

### Step 3: Fix the Scratch Space Accounting
Likely fixes based on common undercounting scenarios:

#### **Nested Function Fix**
```julia
# Current (wrong): Only counts final result
scratch_needed = 1

# Fixed: Counts all intermediate results  
scratch_needed = count_intermediate_steps(nested_function)
```

#### **Function + Interaction Coordination Fix**
```julia
# Current (wrong): Function and interaction calculated separately
function_scratch = calc_function_scratch(func)
interaction_scratch = calc_interaction_scratch(terms)

# Fixed: Account for coordination overhead
total_scratch = function_scratch + interaction_scratch + coordination_overhead
```

#### **Complex Expression Fix**
```julia
# Current (wrong): Assumes simple column access
scratch_for_log_x = 1

# Fixed: Accounts for expression evaluation
scratch_for_log_x_plus_y = 2  # One for (x+y), one for log result
```

### Step 4: Implement and Test
1. **Apply the fix** to the scratch space calculation
2. **Run allocation tests** to confirm the failing case now passes
3. **Verify no regressions** - ensure other tests still pass
4. **Add specific test** for the complex case that was failing

### Step 5: Validate the Architecture
1. **Confirm zero allocations** across all test formulas
2. **Add stress tests** for other complex function patterns:
   - `sqrt(abs(log(x)))`  (deeply nested)
   - `log(x * y) * group` (expression + interaction)
   - `exp(x + y + z)`     (multi-argument expression)
3. **Document the fix** and update scratch space calculation guidelines

### Expected Outcome
- **All allocation tests pass** (achieve true zero-allocation goal)
- **Targeted fix** without architectural changes
- **Robust scratch space accounting** for complex function scenarios
- **Clear path forward** for any future scratch space coordination issues

This approach treats it as a **compilation-time accounting bug** rather than a fundamental architectural problem, which aligns with the selective failure pattern observed.
