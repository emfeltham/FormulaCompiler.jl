# UnifiedCompiler Execution Engine
# Zero-allocation execution through compile-time dispatch

"""
    RECURSION_LIMIT = 10

Threshold for switching between recursive and @generated execution strategies.

Julia's tuple specialization is heuristic-based with no guaranteed cutoff. Lowered from 25
to 10 to reduce compilation overhead for complex models with many parameters.

- **≤10 ops**: Recursive execution (reduced compilation)
- **>10 ops**: @generated execution (forced specialization)

Note: The original value of 25 was optimal for zero-allocation with simple models, but
causes excessive compilation (90M+ allocations) for complex models with 100+ parameters.
"""
const RECURSION_LIMIT = 10

"""
    (compiled::UnifiedCompiled)(output, data, row_idx) -> nothing

**Zero-Allocation Position-Mapped Execution**: The main execution engine that realizes
the position mapping system's performance benefits.

## Position Mapping Execution Model

This function demonstrates how position mappings enable zero-allocation formula evaluation:

### Phase 1: Scratch Space Preparation
- **Reuse**: Uses formula's pre-allocated scratch buffer (no allocation)
- **Reset**: Clears scratch space for current row (`fill!(scratch, 0.0)`)
- **Fixed Size**: Scratch buffer size determined at compile time

### Phase 2: Operation Execution  
- **Type-Specialized Dispatch**: Each operation uses compile-time position parameters
- **Sequential Processing**: Operations execute in dependency order
- **Zero Allocation**: All positions known at compile time → pure array indexing

### Phase 3: Output Transfer
- **Position Mapping**: Transfers scratch results to output using `CopyOp` mappings
- **Model Matrix Compatibility**: Output ordering matches `modelmatrix(model)`

## Position Mapping Examples

```julia
# Compiled operations with embedded positions:
ops = (
    LoadOp{:x, 1}(),           # data.x[row] → scratch[1]
    ConstantOp{1.0, 2}(),      # 1.0 → scratch[2]  
    BinaryOp{:*, 1, 2, 3}(),   # scratch[1] * scratch[2] → scratch[3]
    CopyOp{2, 1}(),            # scratch[2] → output[1] (intercept)
    CopyOp{1, 2}(),            # scratch[1] → output[2] (x) 
    CopyOp{3, 3}()             # scratch[3] → output[3] (interaction)
)

# Execution trace for row_idx=5:
# scratch[1] = data.x[5]              # LoadOp{:x, 1}
# scratch[2] = 1.0                    # ConstantOp{1.0, 2}  
# scratch[3] = scratch[1] * scratch[2] # BinaryOp{:*, 1, 2, 3}
# output[1] = scratch[2]              # CopyOp{2, 1}
# output[2] = scratch[1]              # CopyOp{1, 2}
# output[3] = scratch[3]              # CopyOp{3, 3}
```

## Performance Characteristics

### Memory Usage
- **Scratch space**: Fixed allocation, reused across all rows
- **No allocations**: All array accesses use compile-time indices
- **Cache efficiency**: Sequential scratch access pattern

### Type Specialization Benefits
- **Compile-time positions**: No runtime position lookup or calculation
- **Method specialization**: Separate native code for each operation type
- **Aggressive optimization**: Compiler can optimize entire execution pipeline

## Arguments
- `output`: Pre-allocated output vector (size = `OutputSize`)
- `data`: Column data as NamedTuple (e.g., `(x=[1,2,3], group=["A","B","C"])`)
- `row_idx`: Row index to evaluate (1-based)

## Returns
`nothing` (results written to `output` argument)
"""
function (f::UnifiedCompiled{T, Ops, S, O})(
    output::AbstractVector{T}, 
    data::NamedTuple, 
    row_idx::Int
) where {T, Ops, S, O}
    # Use the formula's own pre-allocated scratch
    scratch = f.scratch
    fill!(scratch, zero(T))  # Clear scratch
    execute_ops(f.ops, scratch, data, row_idx)  # Execute main operations
    copy_outputs_from_ops!(f.ops, output, scratch)  # Execute copy operations
    return nothing
end

# ============================================================================
# Generated functions for large tuples (force complete specialization)
# ============================================================================

# Generated execution for operations - forces complete unrolling
@generated function execute_ops_generated(
    ops::Tuple{Vararg{Any,N}}, 
    scratch::AbstractVector{T}, 
    data::NamedTuple, 
    row_idx::Int
) where {N, T}
    # Build expressions for each operation
    exprs = Expr[]
    for i in 1:N
        push!(exprs, :(execute_op(ops[$i], scratch, data, row_idx)))
    end
    
    # Return block with all operations unrolled
    return quote
        $(exprs...)
        nothing
    end
end

# ============================================================================
# Hybrid dispatch strategy
# ============================================================================

"""
    execute_ops(ops::Tuple, scratch, data, row_idx) -> nothing

**Position-Mapped Operation Dispatcher**: Smart execution strategy that preserves 
position mapping benefits across different formula sizes.

## Position Mapping Execution Strategies

The function uses **hybrid dispatch** to maintain zero-allocation execution while 
handling both small and large operation tuples:

### Small Tuples (≤ 35 operations): Recursive Execution
- **Method**: `execute_ops_recursive` with compile-time tuple unrolling
- **Benefits**: Natural Julia tuple specialization
- **Position Handling**: All positions embedded in recursive call chain
- **Performance**: Optimal for most statistical formulas

### Large Tuples (> 35 operations): Generated Execution  
- **Method**: `@generated execute_ops_generated` with forced unrolling
- **Benefits**: Bypasses Julia's tuple specialization limits  
- **Position Handling**: All positions embedded in generated code
- **Performance**: Maintains zero-allocation for complex formulas

## Position Mapping Preservation

Both strategies preserve the core position mapping invariants:

```julia
# All operations maintain compile-time position specialization:
execute_op(LoadOp{:x, 1}(), scratch, data, row_idx)    # scratch[1] = data.x[row_idx]
execute_op(BinaryOp{:*, 1, 2, 3}(), scratch, data, row_idx) # scratch[3] = scratch[1] * scratch[2]
```

## Performance Characteristics

- **Zero allocations**: Regardless of dispatch strategy chosen
- **Type stability**: All positions remain compile-time constants
- **Optimal execution**: Compiler specializes entire operation sequence
"""
@inline function execute_ops(ops::Tuple, scratch, data, row_idx)
    if length(ops) <= RECURSION_LIMIT
        # Use recursion for small tuples (better for compiler)
        execute_ops_recursive(ops, scratch, data, row_idx)
    else
        # Use generated for large tuples (force specialization)
        execute_ops_generated(ops, scratch, data, row_idx)
    end
end

# ============================================================================
# Original recursive implementation
# ============================================================================

# Recursive tuple execution (compile-time unrolled for small tuples)
@inline execute_ops_recursive(::Tuple{}, scratch, data, row_idx) = nothing
@inline function execute_ops_recursive(ops::Tuple, scratch, data, row_idx)
    execute_op(first(ops), scratch, data, row_idx)
    execute_ops_recursive(Base.tail(ops), scratch, data, row_idx)
end

"""
## Individual Operation Execution with Position Mapping

Each operation type demonstrates how position mappings enable zero-allocation execution
through compile-time position specialization. All functions are `@inline` for zero overhead.

The position mapping system allows the compiler to:
1. **Eliminate bounds checks**: All positions known at compile time  
2. **Optimize array access**: Direct indexing without indirection
3. **Specialize entire pipeline**: Separate native code per position combination
"""

"""
    execute_op(::LoadOp{Col, Out}, scratch, data, row_idx)

**Column Loading with Position Mapping**: Demonstrates how position embedding enables
zero-allocation data access.

## Position Mapping Implementation

```julia
# Type specialization embeds both column name and output position:
LoadOp{:x, 3}()  →  scratch[3] = Float64(data.x[row_idx])
LoadOp{:group, 7}()  →  scratch[7] = Float64(data.group[row_idx])
```

## Compile-Time Benefits
- **Col parameter**: Column access becomes `getproperty(data, :x)` (no symbol lookup)
- **Out parameter**: Array access becomes `scratch[3]` (no position calculation)  
- **Type conversion**: Consistent `Float64` conversion for all data types
- **Bounds safety**: Position `Out` guaranteed valid by compilation system

## Zero-Allocation Guarantee
- No dynamic dispatch on column name
- No runtime position calculation or lookup
- No temporary allocations for type conversion
- Direct memory access with compile-time indices
"""
@inline function execute_op(::LoadOp{Col, Out}, scratch, data, row_idx) where {Col, Out}
    scratch[Out] = convert(eltype(scratch), getproperty(data, Col)[row_idx])
end

# Constant value
@inline function execute_op(::ConstantOp{Val, Out}, scratch, data, row_idx) where {Val, Out}
    scratch[Out] = convert(eltype(scratch), Val)
end

# Unary operations
@inline function execute_op(::UnaryOp{:exp, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = exp(scratch[In])
end

@inline function execute_op(::UnaryOp{:log, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = log(scratch[In])
end

@inline function execute_op(::UnaryOp{:log1p, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = log1p(scratch[In])
end

@inline function execute_op(::UnaryOp{:sqrt, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = sqrt(scratch[In])
end

@inline function execute_op(::UnaryOp{:abs, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = abs(scratch[In])
end

@inline function execute_op(::UnaryOp{:sin, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = sin(scratch[In])
end

@inline function execute_op(::UnaryOp{:cos, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = cos(scratch[In])
end

@inline function execute_op(::UnaryOp{:inv, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = inv(scratch[In])
end

@inline function execute_op(::UnaryOp{:-, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = -scratch[In]
end

"""
    execute_op(::BinaryOp{Func, In1, In2, Out}, scratch, data, row_idx)

**Binary Operations with Position Mapping**: Shows how multiple position parameters
enable zero-allocation arithmetic operations.

## Position Mapping for Multiple Inputs/Outputs

```julia
# Interaction term: x * group_level1  
BinaryOp{:*, 2, 5, 8}()  →  scratch[8] = scratch[2] * scratch[5]

# Where:
# scratch[2] = data.x[row_idx]        (from LoadOp{:x, 2})
# scratch[5] = contrast_matrix[level, 1] (from ContrastOp{:group, (5,6)})  
# scratch[8] = interaction result     (output position)
```

## Position Dependency Management
- **In1, In2**: Input positions must be computed before this operation
- **Out**: Output position must not conflict with any input positions  
- **Ordering**: Compilation system ensures dependency satisfaction

## Zero-Allocation Implementation  
All positions embedded at compile time → pure array arithmetic:
"""

# Binary operations
@inline function execute_op(::BinaryOp{:+, In1, In2, Out}, scratch, data, row_idx) where {In1, In2, Out}
    scratch[Out] = scratch[In1] + scratch[In2]
end

@inline function execute_op(::BinaryOp{:-, In1, In2, Out}, scratch, data, row_idx) where {In1, In2, Out}
    scratch[Out] = scratch[In1] - scratch[In2]
end

@inline function execute_op(::BinaryOp{:*, In1, In2, Out}, scratch, data, row_idx) where {In1, In2, Out}
    scratch[Out] = scratch[In1] * scratch[In2]
end

@inline function execute_op(::BinaryOp{:/, In1, In2, Out}, scratch, data, row_idx) where {In1, In2, Out}
    scratch[Out] = scratch[In1] / scratch[In2]
end

@inline function execute_op(::BinaryOp{:^, In1, In2, Out}, scratch, data, row_idx) where {In1, In2, Out}
    scratch[Out] = scratch[In1] ^ scratch[In2]
end

# Standardization operation
@inline function execute_op(::StandardizeOp{InPos, OutPos, Center, Scale}, scratch, data, row_idx) where {InPos, OutPos, Center, Scale}
    scratch[OutPos] = (scratch[InPos] - Center) / Scale
end

# Comparison operations
@inline function execute_op(::ComparisonOp{:(<=), InPos, Constant, OutPos}, scratch, data, row_idx) where {InPos, Constant, OutPos}
    scratch[OutPos] = Float64(scratch[InPos] <= Constant)
end

@inline function execute_op(::ComparisonOp{:(>=), InPos, Constant, OutPos}, scratch, data, row_idx) where {InPos, Constant, OutPos}
    scratch[OutPos] = Float64(scratch[InPos] >= Constant)
end

@inline function execute_op(::ComparisonOp{:(<), InPos, Constant, OutPos}, scratch, data, row_idx) where {InPos, Constant, OutPos}
    scratch[OutPos] = Float64(scratch[InPos] < Constant)
end

@inline function execute_op(::ComparisonOp{:(>), InPos, Constant, OutPos}, scratch, data, row_idx) where {InPos, Constant, OutPos}
    scratch[OutPos] = Float64(scratch[InPos] > Constant)
end

@inline function execute_op(::ComparisonOp{:(==), InPos, Constant, OutPos}, scratch, data, row_idx) where {InPos, Constant, OutPos}
    scratch[OutPos] = Float64(scratch[InPos] == Constant)
end

@inline function execute_op(::ComparisonOp{:(!=), InPos, Constant, OutPos}, scratch, data, row_idx) where {InPos, Constant, OutPos}
    scratch[OutPos] = Float64(scratch[InPos] != Constant)
end

# Boolean negation operation
@inline function execute_op(::NegationOp{InPos, OutPos}, scratch, data, row_idx) where {InPos, OutPos}
    scratch[OutPos] = Float64(!(scratch[InPos] != 0.0))
end

# Binary comparison operations
@inline function execute_op(::ComparisonBinaryOp{:(<=), LHSPos, RHSPos, OutPos}, scratch, data, row_idx) where {LHSPos, RHSPos, OutPos}
    scratch[OutPos] = Float64(scratch[LHSPos] <= scratch[RHSPos])
end

@inline function execute_op(::ComparisonBinaryOp{:(>=), LHSPos, RHSPos, OutPos}, scratch, data, row_idx) where {LHSPos, RHSPos, OutPos}
    scratch[OutPos] = Float64(scratch[LHSPos] >= scratch[RHSPos])
end

@inline function execute_op(::ComparisonBinaryOp{:(<), LHSPos, RHSPos, OutPos}, scratch, data, row_idx) where {LHSPos, RHSPos, OutPos}
    scratch[OutPos] = Float64(scratch[LHSPos] < scratch[RHSPos])
end

@inline function execute_op(::ComparisonBinaryOp{:(>), LHSPos, RHSPos, OutPos}, scratch, data, row_idx) where {LHSPos, RHSPos, OutPos}
    scratch[OutPos] = Float64(scratch[LHSPos] > scratch[RHSPos])
end

@inline function execute_op(::ComparisonBinaryOp{:(==), LHSPos, RHSPos, OutPos}, scratch, data, row_idx) where {LHSPos, RHSPos, OutPos}
    scratch[OutPos] = Float64(scratch[LHSPos] == scratch[RHSPos])
end

@inline function execute_op(::ComparisonBinaryOp{:(!=), LHSPos, RHSPos, OutPos}, scratch, data, row_idx) where {LHSPos, RHSPos, OutPos}
    scratch[OutPos] = Float64(scratch[LHSPos] != scratch[RHSPos])
end

###############################################################################
# DYNAMIC CATEGORICAL LEVEL EXTRACTION (From restart branch)
###############################################################################

"""
    extract_level_code(column_data, row_idx::Int) -> Int

Extract level code with zero allocations using type-stable dispatch.
Handles both regular CategoricalVector and OverrideVector for scenarios.
"""
@inline function extract_level_code(column_data::CategoricalVector, row_idx::Int)
    return Int(levelcode(column_data[row_idx]))
end

@inline function extract_level_code(column_data::OverrideVector{<:CategoricalValue}, row_idx::Int)
    # For OverrideVector, all rows have the same value - extract once, no allocation
    return Int(levelcode(column_data.override_value))
end

@inline function extract_level_code(column_data::AbstractVector, row_idx::Int)
    # Fallback for other vector types that contain categorical values
    cat_value = column_data[row_idx]
    if isa(cat_value, CategoricalValue)
        return Int(levelcode(cat_value))
    elseif isa(cat_value, Integer)
        return Int(cat_value)
    elseif isa(cat_value, String)
        # Handle String values that may come from ForwardDiff conversion
        # This can happen when categorical values get processed through dual number contexts
        # We need to look up the level index in the original categorical structure
        if isa(column_data, OverrideVector) && isa(column_data.override_value, CategoricalValue)
            # For override vectors, find the level code from the override value's pool
            override_val = column_data.override_value
            pool = override_val.pool
            level_idx = findfirst(==(cat_value), pool.levels)
            if level_idx === nothing
                error("String value '$cat_value' not found in categorical levels $(pool.levels)")
            end
            return level_idx
        else
            error("Cannot extract level code from String '$cat_value' without categorical context")
        end
    elseif isa(cat_value, Bool)
        # Handle boolean values: false = level 1, true = level 2
        return cat_value ? 2 : 1
    elseif hasproperty(cat_value, :level)
        return Int(cat_value.level)
    elseif hasproperty(cat_value, :levels) && hasproperty(cat_value, :weights)
        # Handle mixture objects that incorrectly reach this path
        # This is a defensive fix - mixtures should normally be handled by MixtureContrastOp
        error("Categorical mixture objects should not reach extract_level_code(). " *
              "This indicates a compilation routing issue. " *
              "Mixture objects should be compiled to MixtureContrastOp, not ContrastOp. " *
              "Check that is_mixture_column() detection is working during compilation. " *
              "Mixture type: $(typeof(cat_value))")
    else
        error("Cannot extract level code from $(typeof(cat_value))")
    end
end


# Specialized method for Vector{Bool} for better performance with boolean auto-promotion
@inline function extract_level_code(column_data::Vector{Bool}, row_idx::Int)
    # Handle boolean values directly: false = level 1, true = level 2
    # This matches the boolean contrast matrix: [false, true] → [0.0, 1.0]
    return column_data[row_idx] ? 2 : 1
end

# Categorical contrast operation (multi-output) with dynamic level extraction
@inline function execute_op(
    op::ContrastOp{Col, Positions}, 
    scratch, 
    data, 
    row_idx
) where {Col, Positions}
    # Get categorical column data
    column_data = getproperty(data, Col)
    
    # Check if this is a categorical mixture override
    if column_data isa CategoricalMixtureOverride
        # Handle categorical mixture: weighted combination of contrast rows
        mixture_obj = column_data.mixture_obj
        
        # Initialize positions to zero
        for (i, pos) in enumerate(Positions)
            scratch[pos] = 0.0
        end
        
        # Map mixture level names to contrast matrix row indices
        original_levels = mixture_obj.original_levels
        
        # Compute weighted combination
        for (level_name, weight) in zip(mixture_obj.levels, mixture_obj.weights)
            level_name_str = string(level_name)
            level_idx = findfirst(==(level_name_str), original_levels)
            if level_idx !== nothing
                for (i, pos) in enumerate(Positions)
                    scratch[pos] += weight * op.contrast_matrix[level_idx, i]
                end
            end
        end
    else
        # Standard categorical handling (unchanged)
        # Extract level code dynamically with zero allocations
        level = extract_level_code(column_data, row_idx)
        
        # Clamp to valid range (safety check)
        n_levels = size(op.contrast_matrix, 1)
        level = clamp(level, 1, n_levels)
        
        # Apply contrast matrix (stored as field)
        for (i, pos) in enumerate(Positions)
            scratch[pos] = convert(eltype(scratch), op.contrast_matrix[level, i])
        end
    end
end

"""
    execute_op(::MixtureContrastOp{Col, Positions, LevelIndices, Weights}, scratch, data, row_idx)

**Categorical Mixture Execution with Position Mapping**: Implements zero-allocation 
execution for categorical mixture specifications using compile-time type embedding.

## Position Mapping for Mixtures

The mixture execution demonstrates advanced position mapping where:
- **Positions**: Output scratch positions (compile-time tuple)
- **LevelIndices**: Contrast matrix row indices (compile-time tuple)
- **Weights**: Mixture weights (compile-time tuple)

## Execution Algorithm

```julia
# For binary mixture: 30% "A", 70% "B" with positions [4, 5]
MixtureContrastOp{:group, (4, 5), (1, 2), (0.3, 0.7)}

# Execution computes weighted combination:
# scratch[4] = 0.3 * contrast_matrix[1, 1] + 0.7 * contrast_matrix[2, 1]
# scratch[5] = 0.3 * contrast_matrix[1, 2] + 0.7 * contrast_matrix[2, 2]
```

## Zero-Allocation Implementation

All computations unrolled at compile time:
- **Loop unrolling**: Each (level_idx, weight) pair becomes separate computation
- **Type specialization**: Separate method for each mixture specification
- **Constant folding**: Weights become compile-time constants
- **Direct indexing**: All positions and indices known at compile time

## Performance Benefits

- **No runtime mixture resolution**: All specifications embedded in type
- **No dynamic dispatch**: Each mixture gets specialized execution method
- **Cache friendly**: Sequential scratch position access
- **Compiler optimization**: Full pipeline optimization possible
"""
@inline function execute_op(
    op::MixtureContrastOp{Col, Positions, LevelIndices, Weights}, 
    scratch, 
    data, 
    row_idx
) where {Col, Positions, LevelIndices, Weights}
    # Zero out positions first
    for pos in Positions
        scratch[pos] = 0.0
    end
    
    # Compute weighted combination (fully unrolled at compile time)
    # This loop gets unrolled by the compiler since LevelIndices and Weights are compile-time tuples
    for (level_idx, weight) in zip(LevelIndices, Weights)
        for (contrast_col, pos) in enumerate(Positions)
            scratch[pos] += weight * op.contrast_matrix[level_idx, contrast_col]
        end
    end
end

# Note: The general method above handles all cases efficiently, including binary mixtures.
# Julia's compiler will optimize the tuple iteration for small, compile-time tuples.

# Copy from scratch to output (CopyOp doesn't need output as separate arg)
@inline function execute_op(::CopyOp{InPos, OutIdx}, scratch, data, row_idx) where {InPos, OutIdx}
    # CopyOp is handled separately in copy_outputs!
    # This is a no-op during main execution
    return nothing
end

# ============================================================================
# Generated copy function for large tuples
# ============================================================================

# Generated function for copy operations - forces complete unrolling
@generated function copy_outputs_generated!(
    ops::Tuple{Vararg{Any,N}}, 
    output::AbstractVector{T}, 
    scratch::AbstractVector{T}
) where {N, T}
    exprs = Expr[]
    for i in 1:N
        push!(exprs, :(copy_single_output!(ops[$i], output, scratch)))
    end
    
    return quote
        $(exprs...)
        nothing
    end
end

# ============================================================================
# Hybrid copy dispatch
# ============================================================================

# Smart dispatch for copy operations based on tuple size
@inline function copy_outputs_from_ops!(ops::Tuple, output, scratch)
    if length(ops) <= RECURSION_LIMIT
        # Use recursion for small tuples
        copy_outputs_recursive!(ops, output, scratch)
    else
        # Use generated for large tuples
        copy_outputs_generated!(ops, output, scratch)
    end
end

# ============================================================================
# Original recursive copy implementation
# ============================================================================

# Recursive copy operations
@inline copy_outputs_recursive!(::Tuple{}, output, scratch) = nothing
@inline function copy_outputs_recursive!(ops::Tuple, output, scratch)
    copy_single_output!(first(ops), output, scratch)
    copy_outputs_recursive!(Base.tail(ops), output, scratch)
end

# Handle CopyOp operations
@inline function copy_single_output!(::CopyOp{InPos, OutIdx}, output, scratch) where {InPos, OutIdx}
    if OutIdx <= length(output) && InPos <= length(scratch)
        output[OutIdx] = scratch[InPos]
    end
end

# Skip non-copy operations
@inline copy_single_output!(op, output, scratch) = nothing
