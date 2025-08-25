# step3/types.jl
# Type definitions for the function system (Step 3 of compilation pipeline)
# Universal function system using only unary and binary operations
# Phase A Fix: Zero-allocation function position management

###############################################################################
# ZERO-ALLOCATION FUNCTION DATA TYPES (PHASE A FIX)
###############################################################################

"""
    ScratchPosition{P}

Compile-time wrapper for scratch positions to enable zero-allocation dispatch.
"""
struct ScratchPosition{P}
    position::Int
    
    ScratchPosition(pos::Int) = new{pos}(pos)
end

"""
    UnaryFunctionData{F, InputType}

Compile-time specialized unary function with known function type and input source.
InputType is Symbol (column), Int (output position), ScratchPosition (scratch), or Float64 (constant).
"""
struct UnaryFunctionData{F, InputType}
    func::F
    input_source::InputType
    position::Int
    
    function UnaryFunctionData(func::F, input_source::T, position::Int) where {F, T}
        new{F, T}(func, input_source, position)
    end
end

"""
    IntermediateUnaryFunctionData{F, InputType}

NEW: Specialized for intermediate unary results that write to scratch space.
"""
struct IntermediateUnaryFunctionData{F, InputType}
    func::F
    input_source::InputType
    scratch_position::Int  # Always writes to scratch
    
    function IntermediateUnaryFunctionData(func::F, input_source::T, scratch_pos::Int) where {F, T}
        new{F, T}(func, input_source, scratch_pos)
    end
end

struct IntermediateBinaryFunctionData{F, Input1Type, Input2Type}
    func::F
    input1::Input1Type
    input2::Input2Type
    scratch_position::Int  # Always concrete Int, never Nothing
    
    function IntermediateBinaryFunctionData(func::F, input1::T1, input2::T2, scratch_pos::Int) where {F, T1, T2}
        new{F, T1, T2}(func, input1, input2, scratch_pos)
    end
end

"""
    FinalBinaryFunctionData{F, Input1Type, Input2Type}

Specialized for final results that write to output array.
No Union types - always writes to output at known position.
"""
struct FinalBinaryFunctionData{F, Input1Type, Input2Type}
    func::F
    input1::Input1Type
    input2::Input2Type
    output_position::Int  # Always concrete Int, never Nothing
    
    function FinalBinaryFunctionData(func::F, input1::T1, input2::T2, output_pos::Int) where {F, T1, T2}
        new{F, T1, T2}(func, input1, input2, output_pos)
    end
end

"""
    SpecializedFunctionData{UnaryTuple, IntermediateTuple, FinalTuple}

Updated function data with separate intermediate and final binary operations.
No Union types anywhere - complete compile-time specialization.
"""
struct SpecializedFunctionData{UnaryTuple, IntermediateTuple, FinalTuple}
    unary_functions::UnaryTuple           # NTuple{N, UnaryFunctionData{...}}
    intermediate_binaries::IntermediateTuple  # NTuple{M, IntermediateBinaryFunctionData{...}}
    final_binaries::FinalTuple           # NTuple{K, FinalBinaryFunctionData{...}}
end

"""
    FunctionOp{N, M, K}

Updated operation encoding with separate counts for each operation type.
"""
struct FunctionOp{N, M, K}
    function FunctionOp(n_unary::Int, n_intermediate::Int, n_final::Int)
        new{n_unary, n_intermediate, n_final}()
    end
end

###############################################################################
# LINEARIZED OPERATION TYPES (UPDATED)
###############################################################################

"""
    LinearizedOperation

Intermediate representation for function decomposition.
Updated to support intermediate vs final distinction.
"""
struct LinearizedOperation
    operation_type::Symbol  # :unary, :intermediate_binary, or :final_binary
    func::Function
    inputs::Vector{Union{Val, Int, Float64, ScratchPosition}}  # Typed inputs: Val{Column}, positions, constants, scratch
    output_position::Int
    scratch_position::Union{Int, Nothing}  # For intermediate operations only
end

"""
    TempAllocator

Manages temporary position allocation during decomposition.
"""
mutable struct TempAllocator
    next_temp::Int
    temp_base::Int
    
    function TempAllocator(temp_start::Int)
        new(temp_start, temp_start)
    end
end

function allocate_temp!(allocator::TempAllocator)
    temp_pos = allocator.next_temp
    allocator.next_temp += 1
    return temp_pos
end
