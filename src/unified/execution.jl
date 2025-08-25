# UnifiedCompiler Execution Engine
# Zero-allocation execution through compile-time dispatch

using CategoricalArrays

# Threshold for switching from recursion to @generated
const RECURSION_LIMIT = 35  # Below Julia's ~40 element specialization limit

# Main execution function - zero allocation
function (f::UnifiedCompiled{Ops, S, O})(
    output::AbstractVector{Float64}, 
    data::NamedTuple, 
    row_idx::Int
) where {Ops, S, O}
    # Use the formula's own pre-allocated scratch
    scratch = f.scratch
    fill!(scratch, 0.0)  # Clear scratch
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
    scratch::AbstractVector{Float64}, 
    data::NamedTuple, 
    row_idx::Int
) where N
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

# Smart dispatch based on tuple size
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

# Individual operation execution (all inlined for zero overhead)

# Load column value
@inline function execute_op(::LoadOp{Col, Out}, scratch, data, row_idx) where {Col, Out}
    scratch[Out] = Float64(getproperty(data, Col)[row_idx])
end

# Constant value
@inline function execute_op(::ConstantOp{Val, Out}, scratch, data, row_idx) where {Val, Out}
    scratch[Out] = Val
end

# Unary operations
@inline function execute_op(::UnaryOp{:exp, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = exp(scratch[In])
end

@inline function execute_op(::UnaryOp{:log, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = log(scratch[In])
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

@inline function execute_op(::UnaryOp{:-, In, Out}, scratch, data, row_idx) where {In, Out}
    scratch[Out] = -scratch[In]
end

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

# Categorical contrast operation (multi-output)
@inline function execute_op(
    op::ContrastOp{Col, Positions}, 
    scratch, 
    data, 
    row_idx
) where {Col, Positions}
    # Get categorical level
    cat_value = getproperty(data, Col)[row_idx]
    
    # Extract level code properly
    level = if isa(cat_value, Integer)
        cat_value
    elseif hasproperty(cat_value, :level)
        Int(cat_value.level)
    else
        # For CategoricalValue from CategoricalArrays
        Int(CategoricalArrays.levelcode(cat_value))
    end
    
    # Apply contrast matrix (stored as field)
    for (i, pos) in enumerate(Positions)
        scratch[pos] = op.contrast_matrix[level, i]
    end
end

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
    output::AbstractVector{Float64}, 
    scratch::AbstractVector{Float64}
) where N
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