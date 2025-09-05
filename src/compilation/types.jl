"""
# Position Mapping Type System

The FormulaCompiler uses a **position-indexed type system** where all scratch buffer 
positions and output indices are embedded as compile-time type parameters. This enables
zero-allocation execution through complete type specialization.

## Core Position Mapping Concepts

### Scratch Positions
- **Purpose**: Intermediate computation workspace  
- **Allocation**: Linear, starting from 1
- **Lifetime**: Single row evaluation
- **Usage**: `scratch[pos] = computed_value`

### Output Positions  
- **Purpose**: Final model matrix columns
- **Mapping**: 1:1 with `modelmatrix(model)` columns
- **Usage**: `output[idx] = scratch[pos]`

### Type Parameter Embedding
All positions are embedded in operation types at compile time:
- `LoadOp{:x, 3}`: Load column `:x` into scratch position 3  
- `BinaryOp{:*, 1, 2, 4}`: Multiply scratch[1] × scratch[2] → scratch[4]
- `CopyOp{5, 2}`: Copy scratch[5] to output[2]
"""

# Abstract base for all operations
abstract type AbstractOp end

"""
    LoadOp{Column, OutPos} <: AbstractOp

**Data Loading Operation**: Loads column data into scratch position.

## Position Mapping Role
- **Input**: Column from data NamedTuple  
- **Output**: Single scratch position
- **Type Specialization**: Column name and position embedded at compile time

## Examples
```julia
LoadOp{:x, 1}()      # data.x[row] → scratch[1]  
LoadOp{:group, 5}()  # data.group[row] → scratch[5]
```

## Execution
Zero-allocation data access via compile-time column dispatch:
```julia
execute_op(::LoadOp{:x, 3}, scratch, data, row_idx) = 
    scratch[3] = Float64(data.x[row_idx])
```
"""
struct LoadOp{Column, OutPos} <: AbstractOp end

"""  
    ConstantOp{Value, OutPos} <: AbstractOp

**Constant Value Operation**: Places compile-time constants into scratch positions.

## Position Mapping Role
- **Input**: Compile-time constant (type parameter)
- **Output**: Single scratch position  
- **Usage**: Intercepts, fixed coefficients, literal values

## Examples
```julia
ConstantOp{1.0, 1}()    # Intercept: 1.0 → scratch[1]
ConstantOp{2.5, 7}()    # Literal: 2.5 → scratch[7] 
```

## Type Embedding Benefits
- No runtime constant storage
- Aggressive compiler optimization  
- Zero memory overhead per constant
"""
struct ConstantOp{Value, OutPos} <: AbstractOp end

"""
    UnaryOp{Func, InPos, OutPos} <: AbstractOp

**Unary Function Operation**: Applies functions to scratch positions.

## Position Mapping Role  
- **Input**: Single scratch position containing argument
- **Output**: Single scratch position for result
- **Functions**: :exp, :log, :log1p, :sqrt, :abs, :sin, :cos, :- (negation)

## Examples
```julia
UnaryOp{:log, 2, 3}()   # log(scratch[2]) → scratch[3]
UnaryOp{:exp, 1, 4}()   # exp(scratch[1]) → scratch[4] 
UnaryOp{:-, 5, 6}()     # -scratch[5] → scratch[6]
```

## Position Dependencies
Input position must be computed before this operation executes.
The decomposition system ensures proper ordering.
"""  
struct UnaryOp{Func, InPos, OutPos} <: AbstractOp end

"""
    BinaryOp{Func, InPos1, InPos2, OutPos} <: AbstractOp

**Binary Operation**: Combines two scratch positions with mathematical operations.

## Position Mapping Role
- **Inputs**: Two scratch positions (operands)
- **Output**: Single scratch position (result)
- **Functions**: :+, :-, :*, :/, :^ (arithmetic operations)

## Examples  
```julia
BinaryOp{:*, 1, 2, 3}()   # scratch[1] * scratch[2] → scratch[3] (interaction)
BinaryOp{:+, 4, 5, 6}()   # scratch[4] + scratch[5] → scratch[6] (addition)
BinaryOp{:^, 2, 7, 8}()   # scratch[2] ^ scratch[7] → scratch[8] (power)
```

## Position Dependencies  
Both input positions must be computed before this operation executes.
Critical for interaction terms and compound expressions.
"""
struct BinaryOp{Func, InPos1, InPos2, OutPos} <: AbstractOp end

"""
    ComparisonOp{Op, InPos, Constant, OutPos} <: AbstractOp

**Comparison Operation**: Compares scratch position value against literal constant.

## Position Mapping Role
- **Input**: Single scratch position containing value to compare
- **Constant**: Compile-time literal constant (embedded in type)
- **Output**: Single scratch position containing comparison result (0.0 or 1.0)
- **Operations**: :(<=), :(>=), :(<), :(>), :(==), :(!=)

## Examples
```julia
ComparisonOp{:(<=), 2, 5.0, 3}()   # (scratch[2] <= 5.0) → scratch[3]
ComparisonOp{:(>), 1, 0.0, 4}()    # (scratch[1] > 0.0) → scratch[4]
ComparisonOp{:(==), 5, 1.0, 6}()   # (scratch[5] == 1.0) → scratch[6]
```

## Boolean Output
Comparison results are converted to Float64 for model matrix compatibility:
- `true` → `1.0`
- `false` → `0.0`

## Type Parameter Benefits
- **Zero runtime storage**: Constants embedded at compile time
- **Type specialization**: Each comparison operator gets optimized method
- **Aggressive optimization**: Compiler can inline constant comparisons
"""
struct ComparisonOp{Op, InPos, Constant, OutPos} <: AbstractOp end

"""
    ComparisonBinaryOp{Op, LHSPos, RHSPos, OutPos} <: AbstractOp

**Binary Comparison Operation**: Compares two scratch position values.

## Position Mapping Role
- **LHS Input**: Left-hand side scratch position (e.g., variable value)
- **RHS Input**: Right-hand side scratch position (e.g., function result)  
- **Output**: Single scratch position containing comparison result (0.0 or 1.0)
- **Operations**: :(<=), :(>=), :(<), :(>), :(==), :(!=)

## Examples
```julia
ComparisonBinaryOp{:(<=), 2, 3, 4}()   # (scratch[2] <= scratch[3]) → scratch[4]
ComparisonBinaryOp{:(>), 1, 5, 6}()    # (scratch[1] > scratch[5]) → scratch[6]
```

## Use Cases
Enables comparisons with function calls on right side:
- `(x <= inv(2))` - compare variable against function result
- `(y > log(10))` - compare against logarithm
- `(z == sqrt(4))` - compare against square root

## Performance
Slight overhead vs ComparisonOp (constant RHS) due to additional scratch access,
but maintains zero-allocation execution through position mapping.
"""
struct ComparisonBinaryOp{Op, LHSPos, RHSPos, OutPos} <: AbstractOp end

"""
    NegationOp{InPos, OutPos} <: AbstractOp

**Boolean Negation Operation**: Applies logical negation to scratch position value.

## Position Mapping Role  
- **Input**: Single scratch position containing boolean-like value
- **Output**: Single scratch position containing negated result (0.0 or 1.0)
- **Function**: Logical NOT operation (!)

## Examples
```julia
NegationOp{3, 4}()   # !scratch[3] → scratch[4]
NegationOp{1, 5}()   # !scratch[1] → scratch[5]
```

## Boolean Conversion
Input values are converted to boolean then negated:
- Non-zero values treated as `true` → negated to `0.0`  
- Zero values treated as `false` → negated to `1.0`
- Result always Float64 for model matrix compatibility

## Usage in Formulas
Enables direct support for `!flag` syntax in formulas without preprocessing.
"""
struct NegationOp{InPos, OutPos} <: AbstractOp end

"""
    ContrastOp{Column, OutPositions} <: AbstractOp

**Categorical Contrast Operation**: Expands categorical variables using contrast matrices.

## Position Mapping Role
- **Input**: Single categorical column
- **Output**: Multiple consecutive scratch positions (one per contrast)
- **Matrix Storage**: Contrast matrix stored as field (not type parameter)

## Position Allocation
```julia  
# CategoricalTerm with 3 levels, 2 contrasts → positions [4, 5]
ContrastOp{:group, (4, 5)}(contrast_matrix)

# Execution fills multiple positions:
# scratch[4] = contrast_matrix[level, 1] 
# scratch[5] = contrast_matrix[level, 2]
```

## Multi-Output Handling
Unlike other operations, ContrastOp produces multiple outputs simultaneously.
The position tuple `(4, 5)` represents all allocated positions.

## Type vs Field Storage
- **Type parameter**: Output positions (compile-time)
- **Field storage**: Contrast matrix (runtime data, but pre-computed)
"""
struct ContrastOp{Column, OutPositions} <: AbstractOp 
    contrast_matrix::Matrix{Float64}  # Pre-computed contrast matrix
end

"""
    MixtureContrastOp{Column, OutPositions, LevelIndices, Weights} <: AbstractOp

**Categorical Mixture Contrast Operation**: Applies pre-computed weighted contrast combinations
for categorical mixture specifications.

## Position Mapping Role
- **Input**: Mixture specification (levels and weights embedded in type)
- **Output**: Multiple consecutive scratch positions (one per contrast)  
- **Matrix Storage**: Contrast matrix for the categorical levels
- **Specialization**: Level indices and weights embedded as type parameters

## Type Parameter Embedding
All mixture specifications are embedded at compile time for maximum performance:
- **LevelIndices**: Tuple of indices into contrast matrix rows
- **Weights**: Tuple of mixture weights (corresponding to levels)
- **OutPositions**: Tuple of scratch positions to fill

## Examples
```julia
# Binary mixture: 30% "A", 70% "B" with positions [4, 5]
MixtureContrastOp{
    :group,           # Column name
    (4, 5),          # Output positions  
    (1, 2),          # Level indices (A=1, B=2 in contrast matrix)
    (0.3, 0.7)       # Mixture weights
}(contrast_matrix)

# Execution computes weighted combination:
# scratch[4] = 0.3 * contrast_matrix[1, 1] + 0.7 * contrast_matrix[2, 1]
# scratch[5] = 0.3 * contrast_matrix[1, 2] + 0.7 * contrast_matrix[2, 2]
```

## Performance Benefits  
- **Zero-allocation execution**: All computations unrolled at compile time
- **Type specialization**: Each mixture specification gets its own compiled method
- **Memory efficiency**: No runtime storage of mixture specifications
- **Cache friendly**: Pre-computed weighted contrasts avoid repeated calculations

## Compile-Time Optimization
The type parameter embedding enables aggressive compiler optimization:
- Loop unrolling for small mixtures
- Constant folding for weights
- Inlined matrix access patterns
"""
struct MixtureContrastOp{Column, OutPositions, LevelIndices, Weights} <: AbstractOp
    contrast_matrix::Matrix{Float64}  # Pre-computed contrast matrix for categorical levels
end

"""
    CopyOp{InPos, OutIdx} <: AbstractOp

**Output Copy Operation**: Transfers scratch values to final output vector.

## Position Mapping Role
- **Input**: Scratch position (intermediate result)  
- **Output**: Output vector index (final model matrix column)
- **Purpose**: Maps internal scratch space to user-visible output

## Examples
```julia
CopyOp{1, 1}()  # scratch[1] → output[1] (intercept)  
CopyOp{3, 2}()  # scratch[3] → output[2] (transformed variable)
CopyOp{7, 5}()  # scratch[7] → output[5] (interaction term)
```

## Execution Phase
CopyOp operations execute after all computational operations complete.
This separates internal computation from output formatting.

## Model Matrix Correspondence  
Output indices directly correspond to `modelmatrix(model)` columns:
- `output[1]` = first model matrix column
- `output[2]` = second model matrix column  
- etc.
"""
struct CopyOp{InPos, OutIdx} <: AbstractOp end

"""
    UnifiedCompiled{OpsTuple, ScratchSize, OutputSize}

**Core Position-Mapped Formula Evaluator**: The final compiled representation that embeds 
complete position mappings in the type system.

## Position Mapping Architecture

This type represents the culmination of the position mapping system:

### Type Parameter Embedding
- **OpsTuple**: Complete operation tuple with all positions embedded in types
- **ScratchSize**: Maximum scratch position needed (compile-time known)  
- **OutputSize**: Number of output columns (matches `modelmatrix` width)

### Runtime Components  
- **ops**: Tuple of typed operations (e.g., `(LoadOp{:x,1}(), BinaryOp{:*,1,2,3}(), ...)`)
- **scratch**: Pre-allocated workspace of exact size `ScratchSize`

## Position Mapping Benefits

### Memory Efficiency
```julia
# Fixed memory allocation (no per-row allocation):
# - scratch: Vector{Float64}(undef, ScratchSize) → O(scratch_size)  
# - output: Provided by caller → O(output_size)
# - Total: O(max_positions) regardless of formula complexity
```

### Type Stability & Performance
```julia
# All positions compile-time specialized:
UnifiedCompiled{
    (LoadOp{:x, 1}, BinaryOp{:*, 1, 2, 3}, CopyOp{3, 1}),  # OpsTuple
    3,  # ScratchSize (positions 1,2,3 used)
    1   # OutputSize (single output column)
}
```

### Zero-Allocation Execution
The position mapping enables zero-allocation evaluation:
1. Reuse same scratch buffer for all rows
2. All array accesses use compile-time indices
3. No dynamic dispatch or memory allocation during execution

## Usage Pattern

```julia
compiled = compile_formula(model, data)
output = Vector{Float64}(undef, length(compiled))

for row_idx in 1:nrows
    compiled(output, data, row_idx)  # Zero allocations!
    # Process output...
end
```
"""
struct UnifiedCompiled{T, OpsTuple, ScratchSize, OutputSize}
    ops::OpsTuple              # NTuple of operations
    scratch::Vector{T}         # Pre-allocated scratch buffer of exact size
    
    function UnifiedCompiled{T, OpsTuple, ScratchSize, OutputSize}(ops::OpsTuple) where {T, OpsTuple, ScratchSize, OutputSize}
        scratch = Vector{T}(undef, ScratchSize)
        new{T, OpsTuple, ScratchSize, OutputSize}(ops, scratch)
    end
end

# Get the output size of the compiled formula
Base.length(::UnifiedCompiled{T, OpsTuple, ScratchSize, OutputSize}) where {T, OpsTuple, ScratchSize, OutputSize} = OutputSize

"""
    CompilationContext

**Position Mapping State Manager**: Tracks position allocation and term caching during compilation.

## Core Position Mapping Functions

### Position Allocation
- **`next_position`**: Next available scratch position (monotonically increasing)
- **`output_positions`**: Final mapping from scratch positions to output columns

### Term Caching & Reuse  
- **`position_map`**: Cache mapping `StatsModel terms → scratch positions`
- **Benefits**: Identical terms reuse same positions (e.g., `x` in `x + x*z`)

### Operation Building
- **`operations`**: Accumulates typed operations with embedded positions
- **Conversion**: Vector → Tuple for type stability in final `UnifiedCompiled`

## Position Mapping Workflow

```julia
ctx = CompilationContext()

# Process term: allocate position, create operation, cache mapping
pos = allocate_position!(ctx)                    # pos = 1  
push!(ctx.operations, LoadOp{:x, pos}())        # Store operation
ctx.position_map[:x] = pos                      # Cache for reuse

# Later encounter of same term: reuse cached position  
cached_pos = ctx.position_map[:x]               # cached_pos = 1 (no new allocation)
```

## Invariants Maintained

1. **Monotonic allocation**: `next_position` only increases
2. **No double allocation**: Each term cached after first allocation  
3. **Position consistency**: Same term always maps to same position
4. **Operation order**: Dependencies computed before dependents
"""
mutable struct CompilationContext
    operations::Vector{AbstractOp}  # Will become tuple
    position_map::Dict{Any, Union{Int, Vector{Int}}}    # Term → scratch position(s)
    next_position::Int
    output_positions::Vector{Int}
    
    CompilationContext() = new(AbstractOp[], Dict{Any, Union{Int, Vector{Int}}}(), 1, Int[])
end

# Helper to allocate scratch positions
function allocate_position!(ctx::CompilationContext)
    pos = ctx.next_position
    ctx.next_position += 1
    return pos
end

# Helper to allocate multiple positions
function allocate_positions!(ctx::CompilationContext, n::Int)
    positions = collect(ctx.next_position:(ctx.next_position + n - 1))
    ctx.next_position += n
    return positions
end
