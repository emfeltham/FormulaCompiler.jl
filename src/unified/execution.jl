# UnifiedCompiler Execution Engine
# Zero-allocation execution through compile-time dispatch

using CategoricalArrays

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

# Recursive tuple execution (compile-time unrolled)
@inline execute_ops(::Tuple{}, scratch, data, row_idx) = nothing
@inline function execute_ops(ops::Tuple, scratch, data, row_idx)
    execute_op(first(ops), scratch, data, row_idx)
    execute_ops(Base.tail(ops), scratch, data, row_idx)
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

# Copy outputs from operations
@inline copy_outputs_from_ops!(::Tuple{}, output, scratch) = nothing
@inline function copy_outputs_from_ops!(ops::Tuple, output, scratch)
    copy_single_output!(first(ops), output, scratch)
    copy_outputs_from_ops!(Base.tail(ops), output, scratch)
end

# Handle CopyOp operations
@inline function copy_single_output!(::CopyOp{InPos, OutIdx}, output, scratch) where {InPos, OutIdx}
    if OutIdx <= length(output) && InPos <= length(scratch)
        output[OutIdx] = scratch[InPos]
    end
end

# Skip non-copy operations
@inline copy_single_output!(op, output, scratch) = nothing