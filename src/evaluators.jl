# evaluators.jl
# Complete recursive implementation that handles all cases

# Global context for categorical levels during compilation
const CATEGORICAL_LEVELS_CONTEXT = Ref{Union{Dict{Symbol, Vector{Int}}, Nothing}}(nothing)

"""
    set_categorical_context!(levels::Dict{Symbol, Vector{Int}})
    
Set global categorical levels context for compilation.
"""
function set_categorical_context!(levels::Dict{Symbol, Vector{Int}})
    CATEGORICAL_LEVELS_CONTEXT[] = levels
end

"""
    clear_categorical_context!()
    
Clear global categorical levels context after compilation.
"""
function clear_categorical_context!()
    CATEGORICAL_LEVELS_CONTEXT[] = nothing
end

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

struct ParametricFunctionEvaluator{F,N} <: AbstractEvaluator
    func::F
    arg_evaluators::NTuple{N,AbstractEvaluator}
    arg_scratch_map::NTuple{N,UnitRange{Int}}
end

"""
    InteractionEvaluator{N}

Self-contained interaction evaluator with component scratch space.
"""
struct InteractionEvaluator{N} <: AbstractEvaluator
    components::Vector{AbstractEvaluator}
    total_width::Int
    positions::Vector{Int} # Where interaction terms go in model matrix
    scratch_positions::Vector{Int} # Scratch space for component evaluation
    component_scratch_map::Vector{UnitRange{Int}} # Where each component goes in scratch
    kronecker_pattern::Vector{NTuple{N,Int}} # Pre-computed interaction pattern; N known at compile time!
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

Container with precomputed operations for execution.
"""
struct CombinedEvaluator <: AbstractEvaluator
    # Pre-computed operations to eliminate field access
    constant_ops::Vector{PrecomputedConstantOp}
    continuous_ops::Vector{PrecomputedContinuousOp}
    
    # Keep evaluator objects for complex operations that need field access
    categorical_evaluators::Vector{CategoricalEvaluator}
    function_evaluators::Vector{FunctionEvaluator}
    interaction_evaluators::Vector{InteractionEvaluator}
    
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
output_width(eval::InteractionEvaluator) = length(eval.positions)
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
get_positions(eval::InteractionEvaluator) = eval.positions
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
get_scratch_positions(eval::InteractionEvaluator) = eval.scratch_positions
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

"""
    Helper function to allocate scratch space for nested terms.
"""
function allocate_scratch_for_nested_term(term::AbstractTerm, scratch_allocator::ScratchAllocator)
    term_width = width(term)
    if term_width > 0
        return allocate_scratch!(scratch_allocator, term_width)
    else
        return 1:0  # Empty range for zero-width terms
    end
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

"""
    compute_kronecker_pattern(component_widths::Vector{Int}) -> Vector{Tuple{Vararg{Int}}}

Compute the Kronecker‑product index pattern for an interaction term of arbitrary arity.

# Arguments

- `component_widths::Vector{Int}`: A vector of positive integers, where each entry `w_i` is the number of columns contributed by the i‑th component (e.g., the number of basis functions, dummy columns, or features for that variable or transform).

# Returns

- `pattern::Vector{Tuple{Vararg{Int}}}`: A vector of tuples, each of length `n = length(component_widths)`. Each tuple `(i₁, i₂, …, iₙ)` represents one combination of column indices: take column `i₁` from component 1, `i₂` from component 2, …, `iₙ` from component n, and multiply them to form one column of the full interaction design.

# Details

- Preallocates a vector of length `prod(component_widths)` to hold all index combinations.
- Uses `Iterators.product` to efficiently generate the Cartesian product of the ranges `1:w_i`.
- Converts each `NTuple{n,Int}` returned by `product` into a plain `Tuple{Vararg{Int}}`.
- Complexity: Time and memory are O(∏₁ⁿ w_i). Suitable for moderate‑sized interactions.
- Throws an `ArgumentError` if any width is less than 1.

# Examples

```julia
julia> compute_kronecker_pattern([2, 3])
6-element Vector{Tuple{Vararg{Int}}}:
 (1, 1)
 (1, 2)
 (1, 3)
 (2, 1)
 (2, 2)
 (2, 3)

julia> compute_kronecker_pattern([2, 3, 2])
12-element Vector{Tuple{Vararg{Int}}}:
 (1, 1, 1)
 (1, 1, 2)
 (1, 2, 1)
 (1, 2, 2)
 (1, 3, 1)
 (1, 3, 2)
 (2, 1, 1)
 (2, 1, 2)
 (2, 2, 1)
 (2, 2, 2)
 (2, 3, 1)
 (2, 3, 2)
```
"""
function compute_kronecker_pattern(component_widths::Vector{Int})
    # Validate input
    if any(w -> w < 1, component_widths)
        throw(ArgumentError("All component widths must be positive integers. Received: $(component_widths)"))
    end

    N = length(component_widths)
    
    # Prepare ranges 1:w for each component - use ntuple for type stability
    ranges = ntuple(i -> 1:component_widths[i], N)

    # Preallocate output vector with parametric type
    total = prod(component_widths)
    pattern = Vector{NTuple{N,Int}}(undef, total)

    # Fill with index tuples from Cartesian product
    idx = 1
    for combo in Iterators.product(ranges...)
        pattern[idx] = combo  # combo is already NTuple{N,Int}
        idx += 1
    end

    return pattern
end

###############################################################################
# TESTING UTILITIES
###############################################################################

"""
    apply_function_safe(func::Function, args...)

Safe function application with domain checking.
"""
function apply_function_safe(func::Function, args...)
    if length(args) == 1
        val = args[1]
        if func === log
            return val > 0.0 ? log(val) : (val == 0.0 ? -Inf : NaN)
        elseif func === exp
            return exp(clamp(val, -700.0, 700.0))
        elseif func === sqrt
            return val ≥ 0.0 ? sqrt(val) : NaN
        elseif func === abs
            return abs(val)
        elseif func === sin
            return sin(val)
        elseif func === cos
            return cos(val)
        elseif func === tan  # ← Add this if you need it
            return tan(val)
        else
            return Float64(func(val))
        end
    elseif length(args) == 2
        val1, val2 = args[1], args[2]
        if func === (+)
            return val1 + val2
        elseif func === (-)
            return val1 - val2
        elseif func === (*)
            return val1 * val2
        elseif func === (/)
            return val2 == 0.0 ? (val1 == 0.0 ? NaN : (val1 > 0.0 ? Inf : -Inf)) : val1 / val2
        elseif func === (^)
            if val1 == 0.0 && val2 < 0.0
                return Inf
            elseif val1 < 0.0 && !isinteger(val2)
                return NaN
            else
                return val1^val2
            end
        else
            return Float64(func(val1, val2))
        end
    else
        return Float64(func(args...))
    end
end

"""
    test_function_safety()

Test that apply_function_safe handles edge cases correctly.
"""
function test_function_safety()
    println("Testing apply_function_safe improvements...")
    
    # Test log domain errors
    @assert isnan(apply_function_safe(log, -1.0)) "log(-1) should return NaN"
    @assert apply_function_safe(log, 0.0) == -Inf "log(0) should return -Inf" 
    @assert apply_function_safe(log, 1.0) == 0.0 "log(1) should return 0"
    println("✓ log domain handling works")
    
    # Test sqrt domain errors
    @assert isnan(apply_function_safe(sqrt, -1.0)) "sqrt(-1) should return NaN"
    @assert apply_function_safe(sqrt, 4.0) == 2.0 "sqrt(4) should return 2"
    println("✓ sqrt domain handling works")
    
    # Test division by zero
    @assert apply_function_safe(/, 1.0, 0.0) == Inf "1/0 should return Inf"
    @assert apply_function_safe(/, -1.0, 0.0) == -Inf "-1/0 should return -Inf"
    @assert isnan(apply_function_safe(/, 0.0, 0.0)) "0/0 should return NaN"
    println("✓ division by zero handling works")
    
    # Test power function edge cases
    @assert apply_function_safe(^, 0.0, -1.0) == Inf "0^(-1) should return Inf"
    @assert isnan(apply_function_safe(^, -1.0, 0.5)) "(-1)^0.5 should return NaN"
    println("✓ power function edge cases work")
    
    println("All function safety tests passed!")
    return true
end

###############################################################################
# OPTIMIZED KRONECKER PATTERN APPLICATION
###############################################################################

"""
    apply_kronecker_pattern_to_positions!(pattern::Vector{NTuple{N,Int}},
                                         component_scratch_map::Vector{UnitRange{Int}},
                                         scratch::Vector{Float64},
                                         output::AbstractVector{Float64},
                                         output_positions::Vector{Int}) where N

UPDATED: Apply Kronecker pattern to specific positions without enumerate().
Overwrites old method.
"""
function apply_kronecker_pattern_to_positions!(
    pattern::Vector{NTuple{N,Int}},
    component_scratch_map::Vector{UnitRange{Int}},
    scratch::Vector{Float64},
    output::AbstractVector{Float64},
    output_positions::Vector{Int}
) where N
    
    pattern_length = length(pattern)
    
    @inbounds for idx in 1:pattern_length
        if idx <= length(output_positions)
            indices = pattern[idx]
            
            # Type-stable computation with compile-time known N
            product = 1.0
            for i in 1:N
                scratch_pos = first(component_scratch_map[i]) + indices[i] - 1
                product *= scratch[scratch_pos]
            end
            
            output[output_positions[idx]] = product
        end
    end
    
    return nothing
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

############

function debug_categorical_evaluator(eval::CategoricalEvaluator)
    println("CategoricalEvaluator Debug:")
    println("  Column: $(eval.column)")
    println("  Level codes: $(eval.level_codes)")
    println("  Level codes length: $(length(eval.level_codes))")
    println("  Is empty: $(isempty(eval.level_codes))")
end
