# Function Access Plan: Global Function Pre-Evaluation

## Problem Statement

The current architecture evaluates functions differently depending on context:
- **Standalone functions** (e.g., `log(x)`) are evaluated in Step 3
- **Functions in interactions** (e.g., `exp(x) * y`) are evaluated within Step 4

This split causes allocations because Step 4 can't leverage Step 3's zero-allocation function infrastructure. The FUNCTION_INTERACTION_PLAN failed because it assumed all functions were pre-evaluated in Step 3, but they aren't.

## Current Architecture Issues

1. **Step 3 only sees standalone FunctionEvaluators** - it doesn't know about functions embedded in InteractionEvaluators
2. **Step 4 creates new scratch positions** for functions it encounters, not referencing Step 3's positions
3. **No mapping exists** between FunctionEvaluators in interactions and their Step 3 scratch positions
4. **Functions are evaluated multiple times** if they appear in multiple interaction terms

## Proposed Solution: Global Function Registry

### Core Concept
Create a global function registry during the analysis phase that:
1. Identifies ALL functions across the entire formula (standalone and within interactions)
2. Assigns each unique function a single function_scratch position
3. Evaluates ALL functions once in Step 3
4. Provides Step 4 with a mapping from FunctionEvaluator → FunctionScratchPosition

### Architecture Changes

#### Phase 1: Function Discovery & Registration

**New file**: `src/compilation/pipeline/function_registry.jl`

```julia
struct FunctionRegistry
    evaluators::Vector{FunctionEvaluator}           # All unique functions
    positions::Dict{FunctionEvaluator, Int}         # Evaluator → scratch position
    scratch_size::Int                               # Total scratch needed
end

function build_function_registry(evaluator::CombinedEvaluator) -> FunctionRegistry
    # 1. Traverse entire evaluator tree
    # 2. Extract all FunctionEvaluators (including from interactions)
    # 3. Deduplicate based on function identity
    # 4. Assign sequential scratch positions
    # 5. Return registry
end
```

#### Phase 2: Step 3 Enhancement

**Modify**: `src/compilation/pipeline/step3/main.jl`

```julia
function analyze_evaluator(evaluator::CombinedEvaluator, registry::FunctionRegistry)
    # Current: Only processes evaluator.function_evaluators
    # New: Process ALL functions from registry
    
    all_functions = registry.evaluators  # Instead of evaluator.function_evaluators
    
    # Decompose and create operations for ALL functions
    # Each function writes to its assigned registry position
end

function execute_operation!(data, op, scratch, output, input_data, row_idx)
    # Execute ALL functions from registry
    # Results go to their registry-assigned positions in function_scratch
end
```

#### Phase 3: Step 4 Integration

**Modify**: `src/compilation/pipeline/step4/main.jl`

```julia
function analyze_evaluator(evaluator::CombinedEvaluator, registry::FunctionRegistry)
    # Pass registry through analysis
    interaction_data = analyze_interactions(evaluator, registry)
    # ...
end

function decompose_interaction_tree_zero_alloc(interaction_eval, temp_allocator, registry)
    # When encountering FunctionEvaluator:
    if comp isa FunctionEvaluator
        # Look up pre-assigned position from registry
        func_position = registry.positions[comp]
        push!(processed_input_sources, FunctionScratchPosition(func_position))
        # No allocation, no pre-eval - just reference the position
    end
end
```

#### Phase 4: Top-Level Coordination

**Modify**: `src/compilation/pipeline/step4/main.jl` (or wherever the main compile_formula is)

```julia
function compile_formula(model, data)
    evaluator = build_evaluator_from_model(model)
    
    # NEW: Build global function registry first
    registry = build_function_registry(evaluator)
    
    # Pass registry to both Step 3 and Step 4
    function_data, function_op = analyze_evaluator_step3(evaluator, registry)
    interaction_data, interaction_op = analyze_evaluator_step4(evaluator, registry)
    
    # Create formula with registry-sized function_scratch
    function_scratch = Vector{Float64}(undef, registry.scratch_size)
    # ...
end
```

## Implementation Steps

### Step 1: Create Function Registry Infrastructure
1. Create `function_registry.jl` with `FunctionRegistry` struct
2. Implement `build_function_registry` that traverses evaluator tree
3. Extract functions from both standalone and interaction contexts
4. Implement function deduplication (same function + args = same position)

### Step 2: Update Step 3 to Use Registry
1. Modify `analyze_evaluator` to accept registry parameter
2. Process ALL functions from registry instead of just `evaluator.function_evaluators`
3. Ensure each function writes to its registry-assigned position
4. No changes to execution - it already handles any function configuration

### Step 3: Update Step 4 to Use Registry
1. Modify `analyze_evaluator` to accept registry parameter
2. Pass registry to `decompose_interaction_tree_zero_alloc`
3. When FunctionEvaluator encountered, look up position from registry
4. Create `FunctionScratchPosition` with the registry position
5. Remove function pre-eval creation logic

### Step 4: Wire Registry Through Pipeline
1. Update top-level `compile_formula` to create registry
2. Pass registry to both Step 3 and Step 4 analysis
3. Size function_scratch based on registry.scratch_size
4. Ensure both steps reference the same registry

## Benefits

1. **Zero allocations**: Functions evaluated once in Step 3's zero-allocation system
2. **No redundant computation**: Each function evaluated exactly once
3. **Clean separation**: Step 3 handles ALL function evaluation, Step 4 only references results
4. **Simpler Step 4**: No function pre-eval logic needed in interactions
5. **Better performance**: Shared function results across multiple interaction terms

## Validation

### Test Cases
1. `exp(x) * y` - Function in simple interaction
2. `exp(x) * y * group3` - Function in multi-way interaction  
3. `exp(x) * y + exp(x) * z` - Same function in multiple interactions
4. `log(exp(x)) * y` - Nested function in interaction
5. `exp(x) * log(y) * group3` - Multiple functions in single interaction

### Expected Results
- All function×interaction cases: 0 bytes allocation
- Function evaluation happens exactly once per row
- Correct numerical results maintained

## Risk Mitigation

### Deduplication Complexity
- Risk: Determining if two FunctionEvaluators are "the same" 
- Mitigation: Initially, treat each FunctionEvaluator as unique (no dedup)
- Future optimization: Add function equality comparison

### Backward Compatibility
- Risk: Breaking existing working code
- Mitigation: Registry only affects function handling, other paths unchanged
- Add feature flag if needed: `use_global_function_registry`

### Memory Usage
- Risk: Larger function_scratch for formulas with many functions
- Mitigation: Already have scratch space, just organizing it better
- Benefit: Deduplication actually reduces memory for repeated functions

## Success Metrics

1. Allocation survey shows 0 bytes for:
   - "Function in interaction": `exp(x) * y`
   - "Four-way w/ function": `exp(x) * y * group3 * group4`
   - "Complex interaction": Multiple functions and interactions

2. Performance improvement:
   - Reduced computation for formulas with repeated functions
   - Maintained ~50ns per row for simple formulas

3. Code clarity:
   - Clear separation of concerns (Step 3 = functions, Step 4 = interactions)
   - Simplified Step 4 interaction logic

## Alternative Approaches Considered

1. **Local function pre-eval in Step 4**: Already tried, causes allocations
2. **Inline function evaluation**: Would duplicate Step 3's logic in Step 4
3. **Metaprogramming**: Against project constraints
4. **Accept allocations**: Not acceptable for performance goals

The global registry approach is the cleanest solution that maintains separation of concerns while achieving zero allocations.