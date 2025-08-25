# UnifiedCompiler Type Definitions
# Zero-allocation formula compilation through typed operations

# Abstract base for all operations
abstract type AbstractOp end

# Load a column value from data
struct LoadOp{Column, OutPos} <: AbstractOp end

# Constant value
struct ConstantOp{Value, OutPos} <: AbstractOp end

# Unary function application
struct UnaryOp{Func, InPos, OutPos} <: AbstractOp end

# Binary operation
struct BinaryOp{Func, InPos1, InPos2, OutPos} <: AbstractOp end

# Categorical contrast expansion (multi-output)
# Store contrast matrix as struct field, not type parameter
struct ContrastOp{Column, OutPositions} <: AbstractOp 
    contrast_matrix::Matrix{Float64}  # Pre-computed contrast matrix
end

# Copy from scratch to output
struct CopyOp{InPos, OutIdx} <: AbstractOp end

# Main compiled formula type
struct UnifiedCompiled{OpsTuple, ScratchSize, OutputSize}
    ops::OpsTuple  # NTuple of operations
    scratch::Vector{Float64}  # Pre-allocated scratch buffer of exact size
    
    function UnifiedCompiled{OpsTuple, ScratchSize, OutputSize}(ops::OpsTuple) where {OpsTuple, ScratchSize, OutputSize}
        scratch = Vector{Float64}(undef, ScratchSize)
        new{OpsTuple, ScratchSize, OutputSize}(ops, scratch)
    end
end

# Get the output size of the compiled formula
Base.length(::UnifiedCompiled{OpsTuple, ScratchSize, OutputSize}) where {OpsTuple, ScratchSize, OutputSize} = OutputSize

# Compilation context for building operations
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