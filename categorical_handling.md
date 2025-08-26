# Categorical Handling in FormulaCompiler.jl (Restart Branch)

## Overview

The restart branch successfully handles all categorical variables and their interactions through a sophisticated multi-layered approach that preserves exact compatibility with StatsModels.jl while achieving zero-allocation performance.

## Key Components

### 1. CategoricalSchemaInfo Structure

The restart branch uses a comprehensive schema extraction system that captures all necessary categorical information from fitted models:

```julia
struct CategoricalSchemaInfo
    dummy_contrasts::Matrix{Float64}         # DummyCoding (k-1 columns) - always available
    full_dummy_contrasts::Matrix{Float64}    # FullDummyCoding (k columns) - always available  
    main_effect_contrasts::Union{Matrix{Float64}, Nothing}  # What's used for main effects (if any)
    n_levels::Int
    levels::Vector{String}
    level_codes::Vector{Int}
    column::Symbol
end
```

This structure stores both dummy coding and full dummy coding matrices because different contrast types are needed depending on whether the categorical appears as:
- Main effect → Uses fitted contrast matrix (usually DummyCoding)
- Interaction-only → May use FullDummyCoding for non-redundant interactions

### 2. Contrast Matrix Extraction

The restart branch extracts contrast information directly from the fitted model's terms:

```julia
# From CategoricalTerm in the formula
contrast_matrix = term.contrasts.matrix
levels = term.contrasts.levels
n_levels = length(levels)
```

This ensures we use the **exact** contrast matrix that StatsModels used during model fitting, preserving perfect compatibility.

### 3. Dynamic Level Extraction

Instead of pre-computing level codes, the restart branch extracts them dynamically at runtime:

```julia
@inline function extract_level_code_zero_alloc(column_data::CategoricalVector, row_idx::Int)
    return Int(levelcode(column_data[row_idx]))
end
```

This approach:
- Maintains zero allocations through type-stable dispatch
- Supports OverrideVector for scenarios
- Works correctly with any categorical encoding

### 4. Categorical Evaluator

The CategoricalEvaluator stores the contrast matrix and applies it during evaluation:

```julia
struct CategoricalEvaluator <: AbstractEvaluator
    column::Symbol
    contrast_matrix::Matrix{Float64}
    n_levels::Int
    positions::Vector{Int}
    level_codes::Vector{Int}  # Empty - extracted dynamically
end
```

During execution:
1. Extract level code for current row
2. Apply contrast matrix row corresponding to that level
3. Store results in designated positions

## Interaction Handling

### The Kronecker Product Pattern

The restart branch correctly implements the Kronecker product expansion for interactions through the InteractionEvaluator.

**Critical: The Kronecker Ordering Convention**

The restart branch follows the standard mathematical convention for Kronecker products where `kron(B, A)` means:
- A varies **fast** (inner loop) 
- B varies **slow** (outer loop)

```julia
function compute_interaction_pattern_tuple(width1::Int, width2::Int)
    pattern_tuple = ntuple(n_patterns) do idx
        # Match StatsModels: kron(b, a) means a varies fast, b varies slow
        j = ((idx - 1) ÷ width1) + 1  # Slow index (second component)
        i = ((idx - 1) % width1) + 1  # Fast index (first component)
        (i, j)
    end
end
```

This produces the ordering:
- For widths [2, 3]: [(1,1), (2,1), (1,2), (2,2), (1,3), (2,3)]
- First component cycles through all values before second component advances

This is **opposite** to what the unified branch currently does, which has the first component in the outer loop!

```julia
struct InteractionEvaluator{N, ComponentsTuple, WidthsTuple} <: AbstractEvaluator
    components::ComponentsTuple  # Tuple of component evaluators
    widths::WidthsTuple          # Width of each component's output
    positions::Vector{Int}       # Output positions for interaction results
    start_position::Int
    total_width::Int
end
```

### Interaction Evaluation Process

For an interaction like `group3 * group4`:

1. **Component Evaluation**: Each component evaluator produces its contrast values
   - `group3` (levels A,B,C) → 2 contrast columns
   - `group4` (levels W,X,Y,Z) → 3 contrast columns

2. **Kronecker Product Expansion**: The interaction produces 2×3 = 6 columns
   - The ordering follows StatsModels convention
   - Components are multiplied in the correct Kronecker order

3. **Runtime Computation**:
   ```julia
   # Pseudo-code for interaction evaluation
   for each output position in Kronecker pattern:
       value = 1.0
       for each component:
           component_value = evaluate_component_at_position(component, pattern_position)
           value *= component_value
       output[position] = value
   ```

### Contrast Type Selection

The restart branch has sophisticated logic for choosing contrast types in interactions:

```julia
function determine_interaction_contrast_type(
    comp::CategoricalTerm,
    all_components::Union{Vector, Tuple},
    main_effect_vars::Set{Symbol},
    categorical_schema::Dict{Symbol, CategoricalSchemaInfo}
)
    # Rule 1: Has main effect → always use DummyCoding
    if comp.sym in main_effect_vars
        return :dummy_coding
    end
    
    # Rule 2: Pure non-redundant categorical interaction → FullDummyCoding
    # Rule 3: Mixed interaction with continuous → DummyCoding
end
```

This ensures:
- Main effects use their fitted contrasts
- Interaction-only categoricals can use full dummy coding when appropriate
- Mixed continuous-categorical interactions use appropriate contrasts

## Four-Way Interaction Handling

The restart branch handles complex multi-way interactions correctly through:

1. **Recursive Kronecker Expansion**: 
   - For `x * y * group3 * group4`, it recursively builds the full interaction
   - Each level maintains the correct ordering

2. **Component Width Tracking**:
   - Each component's width is tracked in tuples
   - The total width is the product of all component widths

3. **Pattern-Based Evaluation**:
   - The Kronecker pattern determines which component values to multiply
   - This ensures correct value placement in the output

## Key Differences from Unified Branch

### What the Unified Branch is Missing:

1. **Incorrect Kronecker Ordering**: The unified branch's `compute_all_interaction_combinations` doesn't follow StatsModels' ordering convention

2. **No Contrast Matrix Storage**: The unified branch doesn't extract and store the actual contrast matrices from the fitted model terms

3. **Simplified Interaction Handling**: The unified branch generates simple BinaryOp multiplications without understanding the Kronecker structure

4. **Missing Context Awareness**: The unified branch doesn't distinguish between main effects and interaction-only categoricals

### Solutions from Restart Branch:

1. **Use term.contrasts directly**: Extract the contrast matrix from the CategoricalTerm
2. **Implement proper Kronecker ordering**: Follow StatsModels' convention exactly
3. **Store contrast matrices in operations**: Include the contrast matrix as part of the ContrastOp
4. **Context-aware compilation**: Track which variables have main effects

## Implementation in Unified Branch

To fix the unified branch, we need to:

1. **Fix Kronecker Ordering**: Update `compute_all_interaction_combinations` to match StatsModels' convention

2. **Extract Contrast Matrices**: Use `term.contrasts.matrix` from CategoricalTerm

3. **Store Contrasts in Operations**: The ContrastOp already stores the matrix - this is correct

4. **Fix Interaction Generation**: Ensure the interaction combinations are generated in the right order

The key insight is that the restart branch already solved all these problems comprehensively. The unified branch should adopt the same categorical handling approach, particularly:
- Extracting contrasts directly from terms
- Using proper Kronecker product ordering
- Maintaining the distinction between main effects and interaction-only terms