# evaluators.jl
# Complete recursive implementation that handles all cases

"""
    AbstractEvaluator

Base type for all self-contained evaluators with positions and scratch space.
"""
abstract type AbstractEvaluator end

"""
    ConstantEvaluator

Self-contained constant evaluator.
"""
struct ConstantEvaluator <: AbstractEvaluator
    value::Float64
    position::Int # Where output goes in model matrix
    # No scratch space needed for constants
end

"""
    ContinuousEvaluator

Self-contained continuous variable evaluator.
"""
struct ContinuousEvaluator <: AbstractEvaluator
    column::Symbol
    position::Int # Where output goes in model matrix
    # No scratch space needed for direct data access
end

"""
    CategoricalEvaluator

Self-contained categorical evaluator with pre-computed lookup tables.
"""
struct CategoricalEvaluator <: AbstractEvaluator
    column::Symbol
    contrast_matrix::Matrix{Float64}
    n_levels::Int
    positions::Vector{Int}
    level_codes::Vector{Int}  # Pre-extracted level codes
end

"""
    FunctionEvaluator

Self-contained function evaluator with argument scratch space.
"""
struct FunctionEvaluator <: AbstractEvaluator
    func::Function
    arg_evaluators::Vector{AbstractEvaluator}
    position::Int # Where output goes in model matrix
    scratch_positions::Vector{Int} # Scratch space for argument evaluation
    arg_scratch_map::Vector{UnitRange{Int}} # Where each argument's result goes in scratch
end

"""
    InteractionEvaluator{N, ComponentTuple, WidthTuple}

Fully typed interaction evaluator with recursive tuple-based execution.
Achieves zero allocations through compile-time type specialization.
"""
struct InteractionEvaluator{N, ComponentTuple, WidthTuple} <: AbstractEvaluator where {
    ComponentTuple <: NTuple{N, AbstractEvaluator},
    WidthTuple <: NTuple{N, Int}
}
    components::ComponentTuple           # NTuple{N, AbstractEvaluator} - compile-time
    component_widths::WidthTuple         # NTuple{N, Int} - compile-time
    positions::Vector{Int}               # Output positions
    start_position::Int                  # Base output position
    total_width::Int                     # Total output width
end

"""
    ZScoreEvaluator

Self-contained Z-score evaluator.
"""
struct ZScoreEvaluator <: AbstractEvaluator
    underlying::AbstractEvaluator
    center::Float64
    scale::Float64
    positions::Vector{Int} # Where outputs go in model matrix
    scratch_positions::Vector{Int} # Scratch space for underlying evaluation
    underlying_scratch_map::UnitRange{Int} # Where underlying result goes in scratch
end

# Precomputed operations to eliminate field access during execution
struct PrecomputedConstantOp
    value::Float64 # Ensure this is exactly Float64
    position::Int64 # Ensure this is exactly Int64 (not Int)
end

struct PrecomputedContinuousOp
    column::Symbol # This should be fine
    position::Int64 # Ensure this is exactly Int64
end

"""
    CombinedEvaluator

Updated container with enhanced interaction support.
"""
struct CombinedEvaluator <: AbstractEvaluator
    # Pre-computed operations to eliminate field access
    constant_ops::Vector{PrecomputedConstantOp}
    continuous_ops::Vector{PrecomputedContinuousOp}
    
    # Keep evaluator objects for complex operations that need field access
    categorical_evaluators::Vector{CategoricalEvaluator}
    function_evaluators::Vector{FunctionEvaluator}
    interaction_evaluators::Vector{InteractionEvaluator}  # Now enhanced with mixed-type support
    
    total_width::Int
    max_scratch_needed::Int
end

"""
    ScaledEvaluator

Self-contained scaled evaluator.
"""
struct ScaledEvaluator <: AbstractEvaluator
    evaluator::AbstractEvaluator
    scale_factor::Float64
    positions::Vector{Int}           # Where outputs go in model matrix
    scratch_positions::Vector{Int}   # Scratch space for underlying evaluation
    underlying_scratch_map::UnitRange{Int}  # Where underlying result goes in scratch
end

"""
    ProductEvaluator

Self-contained product evaluator.
"""
struct ProductEvaluator <: AbstractEvaluator
    components::Vector{AbstractEvaluator}
    position::Int                    # Where product goes in model matrix (always scalar)
    scratch_positions::Vector{Int}   # Scratch space for component evaluation
    component_scratch_map::Vector{UnitRange{Int}}  # Where each component goes in scratch
end

###############################################################################
# COMPONENT TYPE ANALYSIS
###############################################################################

"""
    get_component_output_width(evaluator::AbstractEvaluator) -> Int

Get the output width for interaction planning (may differ from structural width).
"""
function get_component_output_width(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator || evaluator isa ContinuousEvaluator
        return 1  # Scalar output
    elseif evaluator isa CategoricalEvaluator
        return size(evaluator.contrast_matrix, 2)  # Number of contrasts
    elseif evaluator isa FunctionEvaluator
        return 1  # Functions produce scalar output
    elseif evaluator isa InteractionEvaluator
        return length(evaluator.positions)
    else
        return output_width_structural(evaluator)  # Fallback
    end
end

"""
    requires_scratch_space(evaluator::AbstractEvaluator) -> Bool

Determine if component needs scratch space for intermediate computation.
"""
function requires_scratch_space(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator || evaluator isa ContinuousEvaluator
        return false  # Direct values, no intermediate computation
    elseif evaluator isa CategoricalEvaluator
        return false  # Uses pre-computed contrast lookup
    elseif evaluator isa FunctionEvaluator
        # Functions in interactions need scratch (can't write to final position)
        return true
    else
        return true  # Conservative default for complex evaluators
    end
end

###############################################################################
# EVALUATOR ANALYSIS FUNCTIONS
###############################################################################

"""
    output_width(evaluator::AbstractEvaluator) -> Int

Get output width from positions.
"""
output_width(eval::ConstantEvaluator) = 1
output_width(eval::ContinuousEvaluator) = 1
output_width(eval::CategoricalEvaluator) = length(eval.positions)
output_width(eval::FunctionEvaluator) = 1
output_width(eval::ZScoreEvaluator) = length(eval.positions)
output_width(eval::CombinedEvaluator) = eval.total_width
output_width(eval::ScaledEvaluator) = length(eval.positions)
output_width(eval::ProductEvaluator) = 1

"""
    get_positions(evaluator::AbstractEvaluator) -> Vector{Int}

Get model matrix positions for any evaluator.
"""
get_positions(eval::ConstantEvaluator) = [eval.position]
get_positions(eval::ContinuousEvaluator) = [eval.position]
get_positions(eval::CategoricalEvaluator) = eval.positions
get_positions(eval::FunctionEvaluator) = [eval.position]
get_positions(eval::ZScoreEvaluator) = eval.positions
get_positions(eval::ScaledEvaluator) = eval.positions
get_positions(eval::ProductEvaluator) = [eval.position]

function get_positions(eval::CombinedEvaluator)
    positions = Int[]
    
    # Collect from precomputed operations
    for op in eval.constant_ops
        push!(positions, op.position)
    end
    for op in eval.continuous_ops
        push!(positions, op.position)
    end
    
    # Collect from complex evaluators (these still have get_positions methods)
    for sub_eval in eval.categorical_evaluators
        append!(positions, get_positions(sub_eval))
    end
    for sub_eval in eval.function_evaluators
        append!(positions, get_positions(sub_eval))
    end
    for sub_eval in eval.interaction_evaluators
        append!(positions, get_positions(sub_eval))
    end
    
    return positions
end

"""
    get_scratch_positions(evaluator::AbstractEvaluator) -> Vector{Int}

Get scratch space positions for any evaluator.
"""
get_scratch_positions(eval::ConstantEvaluator) = Int[]  # No scratch needed
get_scratch_positions(eval::ContinuousEvaluator) = Int[]  # No scratch needed
get_scratch_positions(eval::CategoricalEvaluator) = Int[]  # No scratch needed
# get_scratch_positions(eval::FunctionEvaluator) = eval.scratch_positions
function get_scratch_positions(eval::FunctionEvaluator)
    # start with this function’s own scratch slots
    positions = collect(eval.scratch_positions)
    # then recurse into each argument evaluator
    for arg_eval in eval.arg_evaluators
        append!(positions, get_scratch_positions(arg_eval))
    end
    return positions
end

get_scratch_positions(eval::ZScoreEvaluator) = eval.scratch_positions
get_scratch_positions(eval::ScaledEvaluator) = eval.scratch_positions
get_scratch_positions(eval::ProductEvaluator) = eval.scratch_positions

function get_scratch_positions(eval::CombinedEvaluator)
    positions = Int[]
    
    # Precomputed operations (constants and continuous) don't need scratch space
    # No scratch positions to collect from them
    
    # Collect scratch positions from complex evaluators only
    for sub_eval in eval.categorical_evaluators
        append!(positions, get_scratch_positions(sub_eval))
    end
    for sub_eval in eval.function_evaluators
        append!(positions, get_scratch_positions(sub_eval))
    end
    for sub_eval in eval.interaction_evaluators
        append!(positions, get_scratch_positions(sub_eval))
    end
    
    return positions
end

"""
    max_scratch_needed(evaluator::AbstractEvaluator) -> Int

Get maximum scratch position needed by evaluator tree.
"""
function max_scratch_needed(evaluator::AbstractEvaluator)
    scratch_positions = get_scratch_positions(evaluator)
    return isempty(scratch_positions) ? 0 : maximum(scratch_positions)
end

"""
    get_scratch_positions(eval::InteractionEvaluator) -> Vector{Int}

Interactions require zero scratch positions.
"""
get_scratch_positions(eval::InteractionEvaluator) = Int[]

"""
    max_scratch_needed(evaluator::InteractionEvaluator) -> Int

Interactions require zero scratch space.
"""
function max_scratch_needed(evaluator::InteractionEvaluator)
    return 0
end

"""
    output_width(eval::InteractionEvaluator) -> Int

Get output width from total_width field.
"""
output_width(eval::InteractionEvaluator) = eval.total_width

"""
    get_positions(eval::InteractionEvaluator) -> Vector{Int}

Get model matrix positions.
"""
get_positions(eval::InteractionEvaluator) = eval.positions


###############################################################################
# SELF-CONTAINED COMPILATION SYSTEM
###############################################################################

"""
    ScratchAllocator

Tracks scratch space allocation during compilation.
"""
mutable struct ScratchAllocator
    next_position::Int
    
    ScratchAllocator() = new(1)
end

"""
    allocate_scratch!(allocator::ScratchAllocator, size::Int) -> UnitRange{Int}

Allocate a block of scratch space.
"""
function allocate_scratch!(allocator::ScratchAllocator, size::Int)
    if size == 0
        return 1:0  # Empty range
    end
    
    start_pos = allocator.next_position
    end_pos = start_pos + size - 1
    allocator.next_position = end_pos + 1
    
    return start_pos:end_pos
end

################################################################################

"""
    output_width_structural(evaluator::AbstractEvaluator) -> Int

Get output width based on structure (for evaluators without positions assigned yet).
"""
function output_width_structural(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator || evaluator isa ContinuousEvaluator
        return 1
    elseif evaluator isa CategoricalEvaluator
        return size(evaluator.contrast_matrix, 2)
    elseif evaluator isa FunctionEvaluator || evaluator isa ProductEvaluator
        return 1
    elseif evaluator isa InteractionEvaluator || evaluator isa ZScoreEvaluator || evaluator isa ScaledEvaluator
        return length(evaluator.positions)
    elseif evaluator isa CombinedEvaluator
        return evaluator.total_width
    else
        return 1
    end
end

###############################################################################
# 8. UTILITY FUNCTIONS
###############################################################################

function extract_all_columns(term::AbstractTerm)
    columns = Symbol[]
    extract_columns_recursive!(columns, term)
    return unique(columns)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::Union{ContinuousTerm, Term})
    push!(columns, term.sym)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::CategoricalTerm)
    push!(columns, term.sym)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::FunctionTerm)
    for arg in term.args
        extract_columns_recursive!(columns, arg)
    end
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::InteractionTerm)
    for comp in term.terms
        extract_columns_recursive!(columns, comp)
    end
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::ZScoredTerm)
    extract_columns_recursive!(columns, term.term)
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::MatrixTerm)
    for sub_term in term.terms
        extract_columns_recursive!(columns, sub_term)
    end
end

function extract_columns_recursive!(columns::Vector{Symbol}, term::Union{InterceptTerm, ConstantTerm})
    # No columns
end
