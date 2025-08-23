# step4/types.jl
# Type definitions for the interaction system
# Part of the 4-step compilation pipeline - handles all interaction terms

"""
    FunctionPreEvalOperation

Operation to pre-evaluate a function to scratch before using in interaction.
Stores compiled function data from functions step.
"""
struct FunctionPreEvalOperation
    function_data::SpecializedFunctionData  # The compiled function from step3
    function_op::FunctionOp                 # The operation descriptor from step3
    target_scratch_position::Int            # Where to start writing in scratch
end

###############################################################################
# ZERO-ALLOCATION INTERACTION-BASED DATA STRUCTURE
###############################################################################

"""
    CompleteInteractionData{IntermediateTuple, FinalTuple}

Groups operations using compile-time tuples instead of Vector{Any}.
Mirrors the function system's tuple-based approach exactly.
"""
struct CompleteInteractionData{IntermediateTuple, FinalTuple}
    intermediate_operations::IntermediateTuple  # NTuple{N, IntermediateInteractionData{...}}
    final_operations::FinalTuple               # NTuple{M, FinalInteractionData{...}}
    interaction_index::Int
    
    function CompleteInteractionData(
        intermediate_tuple::IT, 
        final_tuple::FT, 
        index::Int
    ) where {IT, FT}
        new{IT, FT}(intermediate_tuple, final_tuple, index)
    end
end

# OVERWRITE: Replace SpecializedInteractionData entirely with zero-allocation version
"""
    SpecializedInteractionData{CompleteInteractionTuple}

Uses compile-time tuple of CompleteInteractionData.
"""
struct SpecializedInteractionData{CompleteInteractionTuple, I, F}
    complete_interactions::CompleteInteractionTuple  # NTuple{N, CompleteInteractionData{...}}
    
    function SpecializedInteractionData(complete_tuple::T) where T
        # Pre-compute operation counts at compile time using zero-allocation tuple iteration
        total_intermediate, total_final = count_operations_zero_alloc(complete_tuple)
        new{T, total_intermediate, total_final}(complete_tuple)
    end
end

###############################################################################
# CORE INTERACTION POSITION TYPES (MIRROR FUNCTION SYSTEM)
###############################################################################

"""
    InteractionScratchPosition{P}

Compile-time wrapper for interaction scratch positions.
Mirrors ScratchPosition{P} from functions exactly.
"""
struct InteractionScratchPosition{P}
    position::Int
    
    InteractionScratchPosition(pos::Int) = new{pos}(pos)
end

"""
    IntermediateInteractionData{C1, C2, Input1Type, Input2Type, PatternTuple, PreEvalTuple}

Uses compile-time tuples for patterns and pre-evals.
"""
struct IntermediateInteractionData{C1, C2, Input1Type, Input2Type, PatternTuple, PreEvalTuple}
    component1::C1
    component2::C2
    input1_source::Input1Type
    input2_source::Input2Type
    width1::Int
    width2::Int
    index_pattern::PatternTuple          # FIXED: Now a tuple, not Vector
    scratch_position::Int
    function_pre_evals::PreEvalTuple     # FIXED: Now a tuple, not Vector
    
    function IntermediateInteractionData(
        comp1::C1, comp2::C2, 
        input1::T1, input2::T2,
        w1::Int, w2::Int,
        pattern::PT,
        scratch_pos::Int,
        pre_evals::PET
    ) where {C1, C2, T1, T2, PT, PET}
        new{C1, C2, T1, T2, PT, PET}(comp1, comp2, input1, input2, w1, w2, pattern, scratch_pos, pre_evals)
    end
end

"""
    FinalInteractionData{C1, C2, Input1Type, Input2Type, PatternTuple, PreEvalTuple}

Uses compile-time tuples for patterns and pre-evals.
"""
struct FinalInteractionData{C1, C2, Input1Type, Input2Type, PatternTuple, PreEvalTuple}
    component1::C1
    component2::C2
    input1_source::Input1Type
    input2_source::Input2Type
    width1::Int
    width2::Int
    index_pattern::PatternTuple          # FIXED: Now a tuple, not Vector
    output_position::Int
    function_pre_evals::PreEvalTuple     # FIXED: Now a tuple, not Vector
    
    function FinalInteractionData(
        comp1::C1, comp2::C2,
        input1::T1, input2::T2,
        w1::Int, w2::Int,
        pattern::PT,
        output_pos::Int,
        pre_evals::PET
    ) where {C1, C2, T1, T2, PT, PET}
        new{C1, C2, T1, T2, PT, PET}(comp1, comp2, input1, input2, w1, w2, pattern, output_pos, pre_evals)
    end
end

"""
    InteractionOp{I, F}

Mirrors FunctionOp{N, M, K} structure exactly.
I = intermediate interaction count, F = final interaction count.
"""
struct InteractionOp{I, F}
    function InteractionOp(n_intermediate::Int, n_final::Int)
        new{n_intermediate, n_final}()
    end
end