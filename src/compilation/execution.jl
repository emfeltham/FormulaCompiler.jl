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

### Scratch Space Preparation
- **Reuse**: Uses formula's pre-allocated scratch buffer (no allocation)
- **Reset**: Clears scratch space for current row (`fill!(scratch, 0.0)`)
- **Fixed Size**: Scratch buffer size determined at compile time

### Operation Execution  
- **Type-Specialized Dispatch**: Each operation uses compile-time position parameters
- **Sequential Processing**: Operations execute in dependency order
- **Zero Allocation**: All positions known at compile time → pure array indexing

### Output Transfer
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
    # A short buffer used to make copy operations skip silently, leaving stale values
    length(output) >= O || _throw_output_too_short(length(output), O)
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

### Small Tuples (≤ 10 operations): Recursive Execution
- **Method**: `execute_ops_recursive` with compile-time tuple unrolling
- **Benefits**: Natural Julia tuple specialization
- **Position Handling**: All positions embedded in recursive call chain
- **Performance**: Optimal for most statistical formulas

### Large Tuples (> 10 operations): Generated Execution  
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
# CATEGORICAL LEVEL RESOLUTION
###############################################################################
#
# A ContrastOp's matrix rows follow the model's contrast levels. Each row's value is
# resolved to a matrix row with `contrast_row(op.index, column, row_idx)`:
#
# - CategoricalValue (CategoricalArray columns, views, Vector{CategoricalValue}):
#   pool ref code → row via a table built once per pool (`PoolRowMap`). The table is keyed
#   by pool identity (CategoricalPools only ever grow by appending levels, which extends
#   the table), so data with a reordered, subset or rebuilt pool is resolved correctly.
# - Other values (String, Bool, integer codes, ...): hash lookup in the model's levels.
#
# Missing values and values that are not model levels throw; nothing is clamped.

@noinline function _throw_missing_level(::LevelIndex)
    throw(ArgumentError("missing value in a categorical column; FormulaCompiler cannot " *
                        "evaluate model rows for observations with missing categorical values"))
end

@noinline function _throw_unknown_level(idx::LevelIndex, value)
    if hasproperty(value, :levels) && hasproperty(value, :weights)
        error("Categorical mixture objects should not reach a standard ContrastOp. " *
              "This indicates a compilation routing issue: mixture columns compile to " *
              "MixtureContrastOp. Mixture type: $(typeof(value))")
    end
    throw(ArgumentError("value $(repr(value)) is not a level of the fitted model's " *
                        "categorical variable (model levels: $(join(repr.(idx.levels), ", ")))"))
end

@noinline function _throw_bad_code(idx::LevelIndex, code)
    throw(ArgumentError("categorical code $code is out of range 1:$(length(idx.levels))"))
end

# Build (and publish) the ref code → row table for `pool`.
@noinline function _rebuild_pool_map!(idx::LevelIndex, pool)
    pool_levels = CategoricalArrays.levels(pool)  # in ref-code order
    rows = zeros(Int, length(pool_levels))
    for (r, lev) in enumerate(pool_levels)
        rows[r] = _lookup_level(idx, lev isa CategoricalValue ? CategoricalArrays.unwrap(lev) : lev)
    end
    m = PoolRowMap(pool, rows)
    idx.cache.map = m
    return m
end

@inline _lookup_level(idx::LevelIndex, value) = get(idx.lookup, value, 0)
# Positional index: integer values are rows
@inline _lookup_level(idx::LevelIndex{Nothing}, value::Integer) =
    (1 <= value <= length(idx.levels) ? Int(value) : 0)
@inline _lookup_level(idx::LevelIndex{Nothing}, value) = 0

"""
    pool_ref_row(idx::LevelIndex, pool, ref::Integer) -> Int

Contrast-matrix row for ref code `ref` of `pool` (0 = missing → error).
Allocation-free unless `pool` differs from the last pool seen.
"""
@inline function pool_ref_row(idx::LevelIndex, pool, ref::Integer)
    ref == 0 && _throw_missing_level(idx)
    m = idx.cache.map
    # Pools only grow by appending levels, so a table built for this pool stays valid for
    # every ref code it covers; a larger ref code means levels were added since. Only `===`
    # and a length check on the hot path: `pool` may be abstractly typed here (e.g. the
    # base array of a CategoricalCounterfactualVector), and this must not dispatch on it.
    if m.pool !== pool || ref > length(m.rows)
        m = _rebuild_pool_map!(idx, pool)::PoolRowMap
    end
    (1 <= ref <= length(m.rows)) || _throw_bad_code(idx, ref)
    r = @inbounds m.rows[ref]
    r == 0 && _throw_unknown_pool_level(idx, pool, ref)
    return r
end

@noinline _throw_unknown_pool_level(idx::LevelIndex, pool, ref) =
    _throw_unknown_level(idx, CategoricalArrays.unwrap(CategoricalArrays.levels(pool)[ref]))

# Positional index: the ref code is the row
@inline function pool_ref_row(idx::LevelIndex{Nothing}, pool, ref::Integer)
    ref == 0 && _throw_missing_level(idx)
    (1 <= ref <= length(idx.levels)) || _throw_bad_code(idx, ref)
    return Int(ref)
end

"""
    value_row(idx::LevelIndex, value) -> Int

Contrast-matrix row for one categorical value.
"""
@inline value_row(idx::LevelIndex, v::CategoricalValue) =
    pool_ref_row(idx, CategoricalArrays.pool(v), CategoricalArrays.refcode(v))
@inline value_row(idx::LevelIndex, ::Missing) = _throw_missing_level(idx)
@inline function value_row(idx::LevelIndex, v)
    r = _lookup_level(idx, v)
    r == 0 && _throw_unknown_level(idx, v)
    return r
end
@inline function value_row(idx::LevelIndex{Nothing}, v::Integer)
    r = _lookup_level(idx, v)
    r == 0 && _throw_bad_code(idx, v)
    return r
end

"""
    contrast_row(idx::LevelIndex, column, row_idx) -> Int

Contrast-matrix row for `column[row_idx]`, resolved against the model's levels.
Specialised for `CategoricalCounterfactualVector` (which yields raw ref codes) in
`counterfactual_vectors.jl`.
"""
@inline contrast_row(idx::LevelIndex, column::AbstractVector, row_idx::Int) =
    value_row(idx, column[row_idx])

# Categorical contrast operation (multi-output) with dynamic level extraction
@inline function execute_op(
    op::ContrastOp{Col, Positions},
    scratch,
    data,
    row_idx
) where {Col, Positions}
    # Get categorical column data
    column_data = getproperty(data, Col)

    # Check if this is a categorical mixture counterfactual vector
    if column_data isa CategoricalMixtureCounterfactualVector
        _execute_mixture_row!(op, scratch, column_data[row_idx])
    else
        # Resolve the value to a contrast-matrix row via the model's levels
        level = contrast_row(op.index, column_data, row_idx)

        # Apply contrast matrix (stored as field)
        for (i, pos) in enumerate(Positions)
            scratch[pos] = convert(eltype(scratch), @inbounds op.contrast_matrix[level, i])
        end
    end
end

# Categorical mixture value in a standard ContrastOp column: weighted combination of rows
function _execute_mixture_row!(op::ContrastOp{Col, Positions}, scratch, mixture_obj) where {Col, Positions}
    for pos in Positions
        scratch[pos] = 0.0
    end
    for (level_name, weight) in zip(mixture_obj.levels, mixture_obj.weights)
        level_idx = _mixture_level_row(op.index, level_name)
        for (i, pos) in enumerate(Positions)
            scratch[pos] += weight * op.contrast_matrix[level_idx, i]
        end
    end
    return nothing
end

function _mixture_level_row(idx::LevelIndex, level_name)
    r = _lookup_level(idx, level_name)
    r != 0 && return r
    name = string(level_name)
    for (i, l) in enumerate(idx.levels)
        string(l) == name && return i
    end
    _throw_unknown_level(idx, level_name)
end
_mixture_level_row(idx::LevelIndex{Nothing}, level_name) =
    error("mixture level $(repr(level_name)) cannot be resolved: ContrastOp has no level information")

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

@noinline _throw_output_too_short(n, O) = throw(DimensionMismatch(
    "output vector has length $n, but the compiled formula writes $O values; use length(compiled)"))

# Handle CopyOp operations (output length checked by the caller; scratch sized at compile time)
@inline function copy_single_output!(::CopyOp{InPos, OutIdx}, output, scratch) where {InPos, OutIdx}
    @inbounds output[OutIdx] = scratch[InPos]
    return nothing
end

# Skip non-copy operations
@inline copy_single_output!(op, output, scratch) = nothing
