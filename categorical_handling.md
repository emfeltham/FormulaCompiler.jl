# Categorical Handling in FormulaCompiler.jl

## Overview

FormulaCompiler.jl handles all categorical variables and their interactions through a sophisticated position-mapping system that preserves exact compatibility with StatsModels.jl while achieving zero-allocation performance.

## Architecture

### Position-Based Categorical Evaluation

The current system uses a **unified position-mapping approach** where categorical variables are handled through:

1. **Compile-time contrast extraction**: Extract contrast matrices directly from fitted model terms
2. **Position specialization**: Map categorical outputs to fixed positions in the output vector
3. **Runtime level resolution**: Dynamically extract categorical level codes with zero allocations
4. **Contrast application**: Apply pre-extracted contrast matrices to compute final values

### Core Components

#### Contrast Matrix Extraction

The system extracts contrast information directly from fitted model terms during compilation:

```julia
# Extract from CategoricalTerm in the model formula
contrast_matrix = term.contrasts.matrix
levels = term.contrasts.levels
n_levels = length(levels)
```

This ensures we use the **exact** contrast matrix that StatsModels used during model fitting, maintaining perfect compatibility with:
- DummyCoding (k-1 columns)
- EffectsCoding 
- HelmertCoding
- Custom contrast schemes

#### Dynamic Level Code Extraction

Level codes are extracted dynamically at runtime with zero allocations:

```julia
@inline function extract_level_code_zero_alloc(column_data::CategoricalVector, row_idx::Int)
    return Int(levelcode(column_data[row_idx]))
end
```

This approach:
- Maintains zero-allocation performance through type-stable dispatch  
- Supports OverrideVector for scenario analysis
- Works correctly with any categorical encoding
- Handles missing values appropriately

#### Categorical Operation Structure

Categorical variables are compiled into specialized operations that store:

```julia
struct CategoricalOperation
    column_symbol::Symbol
    contrast_matrix::Matrix{Float64}  # Pre-extracted from model
    output_positions::Vector{Int}     # Where to store results
    n_levels::Int
    scratch_positions::Vector{Int}    # Intermediate calculations if needed
end
```

## Interaction Handling

### Kronecker Product Implementation

Categorical interactions follow the mathematical Kronecker product convention where for `kron(B, A)`:
- A varies **fast** (inner loop)
- B varies **slow** (outer loop)

```julia
function compute_interaction_positions(width1::Int, width2::Int)
    positions = Vector{Tuple{Int,Int}}(undef, width1 * width2)
    idx = 1
    for j in 1:width2      # Slow-varying (second component)
        for i in 1:width1  # Fast-varying (first component)
            positions[idx] = (i, j)
            idx += 1
        end
    end
    return positions
end
```

This produces the StatsModels-compatible ordering:
- For interaction widths [2, 3]: [(1,1), (2,1), (1,2), (2,2), (1,3), (2,3)]
- First component cycles through all values before second component advances

### Interaction Compilation Process

For an interaction like `group1 * group2`:

1. **Component Analysis**: 
   - Extract contrast matrices for each categorical component
   - Determine output widths: `group1` (3 levels) → 2 columns, `group2` (4 levels) → 3 columns
   - Calculate total interaction width: 2 × 3 = 6 columns

2. **Position Mapping**:
   - Allocate 6 consecutive positions in the output vector
   - Map each Kronecker product position to specific output locations
   - Store position mappings in the interaction operation

3. **Runtime Evaluation**:
   ```julia
   # Evaluate interaction at runtime
   level1 = extract_level_code_zero_alloc(data.group1, row_idx)
   level2 = extract_level_code_zero_alloc(data.group2, row_idx)
   
   # Apply contrast matrices
   contrast1_values = contrast_matrix1[level1, :]
   contrast2_values = contrast_matrix2[level2, :]
   
   # Compute Kronecker product and store in output positions
   for (i, pos) in enumerate(interaction_positions)
       output[pos] = contrast1_values[pos[1]] * contrast2_values[pos[2]]
   end
   ```

### Multi-Way Interactions

Complex interactions like `x * group1 * group2 * group3` are handled through recursive Kronecker expansion:

1. **Hierarchical Width Calculation**: Each component's width is computed and the total width is the product
2. **Nested Position Mapping**: Positions are mapped following the mathematical Kronecker convention at each level  
3. **Efficient Runtime Evaluation**: All position mappings are pre-computed, runtime only does simple multiplications

## Contrast Type Selection

### Main Effects vs Interaction-Only

The system distinguishes between:

- **Main Effects**: Use the fitted model's contrast matrix exactly as specified
- **Interaction-Only**: May use different contrasts to avoid redundancy

```julia
function determine_contrast_matrix(
    categorical_term::CategoricalTerm,
    has_main_effect::Bool,
    interaction_context::InteractionContext
)
    if has_main_effect
        # Use fitted model's contrast matrix
        return categorical_term.contrasts.matrix
    else
        # For interaction-only terms, may use FullDummyCoding
        # to avoid rank deficiency issues
        return create_appropriate_contrast_matrix(categorical_term, interaction_context)
    end
end
```

### Context-Aware Compilation

The compilation process tracks:
- Which categorical variables appear as main effects
- Which appear only in interactions
- The specific contrast types used in the original model fit

This ensures that the compiled evaluator produces identical results to `modelmatrix(model)`.

## Scenario System Integration

### Categorical Overrides

The override system seamlessly handles categorical variables:

```julia
# Create scenario with categorical override
scenario = create_scenario("treatment_group", data; group = "Treatment")

# The override system automatically:
# 1. Validates the level exists in the original categorical
# 2. Creates an OverrideVector that returns the specified level
# 3. Maintains compatibility with the contrast matrix system
```

### Memory Efficiency

Categorical overrides are extremely memory efficient:
- **OverrideVector**: O(1) memory regardless of data size
- **Level storage**: Only stores the target level, not the full categorical array
- **Contrast compatibility**: Works seamlessly with all contrast types

## Performance Characteristics

### Zero-Allocation Execution

The categorical system achieves zero allocations through:

1. **Pre-computed contrast matrices**: Stored at compile time
2. **Fixed position mappings**: All output locations determined during compilation
3. **Type-stable level extraction**: `levelcode()` calls are inlined and type-stable
4. **Direct memory access**: No intermediate allocations for categorical lookups

### Benchmarks

- **Simple categorical**: ~50ns per row, 0 allocations
- **Complex interactions**: <500ns per row, 0 allocations  
- **Scenario overrides**: Same performance as original data
- **Memory usage**: >99% reduction vs naive scenario copying

## Supported Features

### Contrast Types
- **DummyCoding**: Reference level omitted (k-1 columns)
- **EffectsCoding**: Sum-to-zero constraints  
- **HelmertCoding**: Sequential differences
- **Custom contrasts**: Any user-defined contrast matrix

### Data Types
- **Standard categoricals**: `CategoricalArray` with any level type
- **Boolean categoricals**: `true`/`false` handling
- **Ordered categoricals**: Respects ordering in contrasts
- **String/Symbol levels**: Full Unicode support

### Integration
- **StatsModels.jl**: Perfect compatibility with all formula features
- **CategoricalArrays.jl**: Full support for all categorical operations
- **GLM.jl/MixedModels.jl**: Seamless integration with fitted models
- **Tables.jl**: Works with any table format via `Tables.columntable`

## Implementation Details

### Error Handling

The system provides comprehensive error checking:
- **Invalid levels**: Clear error messages when override levels don't exist
- **Mismatched contrasts**: Validation that contrast matrices match categorical structure
- **Type safety**: Compile-time detection of type mismatches

### Edge Cases

Special handling for:
- **Single-level categoricals**: Degenerate case with identity contrast
- **Empty categoricals**: Proper handling of zero-length categorical arrays  
- **Missing values**: Integration with CategoricalArrays.jl missing value handling
- **Level reordering**: Robust to changes in level ordering between model fitting and evaluation

The categorical handling system represents a sophisticated balance between performance, correctness, and usability, providing exact StatsModels.jl compatibility while achieving the zero-allocation performance goals of FormulaCompiler.jl.