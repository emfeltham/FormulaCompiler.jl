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
    get_categorical_levels_for_column(column::Symbol) -> Vector{Int}
    
Get pre-extracted categorical levels for a column from global context.
"""
function get_categorical_levels_for_column(column::Symbol)
    context = CATEGORICAL_LEVELS_CONTEXT[]
    if context !== nothing && haskey(context, column)
        return context[column]
    else
        return Int[]  # Empty vector for non-categorical or missing columns
    end
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
    position::Int                    # Where output goes in model matrix
    # No scratch space needed for constants
end

"""
    ContinuousEvaluator

Self-contained continuous variable evaluator.
"""
struct ContinuousEvaluator <: AbstractEvaluator
    column::Symbol
    position::Int                    # Where output goes in model matrix
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
    position::Int                    # Where output goes in model matrix
    scratch_positions::Vector{Int}   # Scratch space for argument evaluation
    arg_scratch_map::Vector{UnitRange{Int}}  # Where each argument's result goes in scratch
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
    positions::Vector{Int}           # Where outputs go in model matrix
    scratch_positions::Vector{Int}   # Scratch space for underlying evaluation
    underlying_scratch_map::UnitRange{Int}  # Where underlying result goes in scratch
end

# Precomputed operations to eliminate field access during execution
struct PrecomputedConstantOp
    value::Float64  # Ensure this is exactly Float64
    position::Int64 # Ensure this is exactly Int64 (not Int)
end

struct PrecomputedContinuousOp
    column::Symbol   # This should be fine
    position::Int64  # Ensure this is exactly Int64
end

"""
    CombinedEvaluator

Container with precomputed operations for zero-allocation execution.
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
    compile_term(
        term::AbstractTerm, start_position::Int = 1, 
        scratch_allocator::ScratchAllocator = ScratchAllocator()
    ) -> (AbstractEvaluator, Int)

Compile term into self-contained evaluator with positions and scratch space.
"""
function compile_term(
    term::AbstractTerm, 
    start_position::Int = 1, 
    scratch_allocator::ScratchAllocator = ScratchAllocator(),
    categorical_levels::Union{Dict{Symbol, Vector{Int}}, Nothing} = nothing
)
    # Use explicit levels if provided, fall back to global context
    levels = categorical_levels !== nothing ? categorical_levels : CATEGORICAL_LEVELS_CONTEXT[]
    
    if term isa InterceptTerm
        evaluator = ConstantEvaluator(hasintercept(term) ? 1.0 : 0.0, start_position)
        return evaluator
        
    elseif term isa ConstantTerm
        evaluator = ConstantEvaluator(Float64(term.n), start_position)
        return evaluator
        
    elseif term isa Union{ContinuousTerm, Term}
        evaluator = ContinuousEvaluator(term.sym, start_position)
        return evaluator
        
    elseif term isa CategoricalTerm
        contrast_matrix = Matrix{Float64}(term.contrasts.matrix)
        n_contrasts = size(contrast_matrix, 2)
        positions = collect(start_position:(start_position + n_contrasts - 1))
        
        # Debug: Check what levels we have
        # println("Compiling CategoricalTerm for $(term.sym)")
        # println("  levels dict: $(levels)")
        # println("  levels is nothing: $(levels === nothing)")
        # if levels !== nothing
        #     println("  has key $(term.sym): $(haskey(levels, term.sym))")
        #     if haskey(levels, term.sym)
        #         println("  level codes: $(levels[term.sym])")
        #     end
        # end
        
        # Use explicit levels if available
        level_codes = if levels !== nothing && haskey(levels, term.sym)
            levels[term.sym]
        else
            println("  WARNING: Using empty level codes for $(term.sym)!")
            Int[]  # Empty fallback
        end
        
        evaluator = CategoricalEvaluator(
            term.sym,
            contrast_matrix,
            size(contrast_matrix, 1),
            positions,
            level_codes
        )
        
        # debug_categorical_evaluator(evaluator)
        return evaluator
        
    elseif term isa FunctionTerm
        arg_evaluators = AbstractEvaluator[]
        arg_scratch_map = UnitRange{Int}[]
        
        for arg in term.args
            arg_width = 1
            arg_scratch = allocate_scratch!(scratch_allocator, arg_width)
            arg_start_pos = first(arg_scratch)
            
            # Pass categorical_levels through recursive call
            arg_eval = compile_term(arg, arg_start_pos, scratch_allocator, categorical_levels)
            
            push!(arg_evaluators, arg_eval)
            push!(arg_scratch_map, arg_scratch)
        end
        
        all_scratch = Int[]
        for range in arg_scratch_map
            append!(all_scratch, collect(range))
        end
        
        return FunctionEvaluator(
            term.f,
            arg_evaluators,
            start_position,
            all_scratch,
            arg_scratch_map
        )
        
    elseif term isa InteractionTerm
        component_evaluators = AbstractEvaluator[]
        component_scratch_map = UnitRange{Int}[]
        component_widths = Int[]
        
        for comp in term.terms
            comp_width = width(comp)
            push!(component_widths, comp_width)
            comp_scratch = allocate_scratch!(scratch_allocator, comp_width)
            comp_start_pos = first(comp_scratch)
            
            # Pass categorical_levels through recursive call
            comp_eval = compile_term(comp, comp_start_pos, scratch_allocator, categorical_levels)
            
            push!(component_evaluators, comp_eval)
            push!(component_scratch_map, comp_scratch)
        end
        
        total_width = prod(component_widths)
        positions = collect(start_position:(start_position + total_width - 1))
        N = length(component_widths)
        pattern = compute_kronecker_pattern(component_widths)
        
        all_scratch = Int[]
        for range in component_scratch_map
            append!(all_scratch, collect(range))
        end
        
        return InteractionEvaluator{N}(
            component_evaluators,
            total_width,
            positions,
            all_scratch,
            component_scratch_map,
            pattern
        )
        
    elseif term isa ZScoredTerm
        underlying_width = width(term.term)
        underlying_scratch = allocate_scratch!(scratch_allocator, underlying_width)
        underlying_start_pos = first(underlying_scratch)
        
        # Pass categorical_levels through recursive call
        underlying_eval = compile_term(term.term, underlying_start_pos, scratch_allocator, categorical_levels)
        
        positions = collect(start_position:(start_position + underlying_width - 1))
        center = term.center isa Number ? Float64(term.center) : Float64(term.center[1])
        scale = term.scale isa Number ? Float64(term.scale) : Float64(term.scale[1])
        
        return ZScoreEvaluator(
            underlying_eval,
            center,
            scale,
            positions,
            collect(underlying_scratch),
            underlying_scratch
        )
        
    elseif term isa MatrixTerm
        sub_evaluators = AbstractEvaluator[]
        current_pos = start_position
        max_scratch = 0
        
        for sub_term in term.terms
            if width(sub_term) > 0
                sub_eval = compile_term(sub_term, current_pos, scratch_allocator, categorical_levels)
                next_pos = current_pos + output_width(sub_eval)
                push!(sub_evaluators, sub_eval)
                current_pos = next_pos
                
                sub_scratch = max_scratch_needed(sub_eval)
                max_scratch = max(max_scratch, sub_scratch)
            end
        end
        
        total_width = current_pos - start_position
        
        # FIXED: Create precomputed operations instead of storing evaluators
        constant_ops = PrecomputedConstantOp[]
        continuous_ops = PrecomputedContinuousOp[]
        categorical_evals = CategoricalEvaluator[]
        function_evals = FunctionEvaluator[]
        interaction_evals = InteractionEvaluator[]
        
        for eval in sub_evaluators
            if eval isa ConstantEvaluator
                # Precompute: extract fields at compilation time
                push!(constant_ops, PrecomputedConstantOp(eval.value, eval.position))
            elseif eval isa ContinuousEvaluator
                # Precompute: extract fields at compilation time
                push!(continuous_ops, PrecomputedContinuousOp(eval.column, eval.position))
            elseif eval isa CategoricalEvaluator
                push!(categorical_evals, eval)
            elseif eval isa FunctionEvaluator
                push!(function_evals, eval)
            elseif eval isa InteractionEvaluator
                push!(interaction_evals, eval)
            else
                error("Unknown evaluator type in matrix term: $(typeof(eval))")
            end
        end
        
        return CombinedEvaluator(
            constant_ops,
            continuous_ops,
            categorical_evals,
            function_evals,
            interaction_evals,
            total_width,
            max_scratch
        )
    else
        error("Unknown term type: $(typeof(term))")
    end
end

function compile_term(
    term::CategoricalTerm, start_position::Int = 1, 
    scratch_allocator::ScratchAllocator = ScratchAllocator()
)
    contrast_matrix = Matrix{Float64}(term.contrasts.matrix)
    n_contrasts = size(contrast_matrix, 2)
    positions = collect(start_position:(start_position + n_contrasts - 1))
    
    # Get pre-extracted level codes from global context
    level_codes = get_categorical_levels_for_column(term.sym)
    
    return CategoricalEvaluator(
        term.sym,
        contrast_matrix,
        size(contrast_matrix, 1),
        positions,
        level_codes  # NEW: Include pre-extracted level codes
    )
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
# 3. RECURSIVE EVALUATION
###############################################################################

"""
Recursively evaluate any evaluator into a pre-allocated output vector.
This is the core that makes the compositional approach work.
"""
function evaluate!(
    evaluator::AbstractEvaluator, output::AbstractVector{Float64}, 
    data, row_idx::Int, start_idx::Int=1
)
    
    if evaluator isa ConstantEvaluator
        @inbounds output[start_idx] = evaluator.value
        return start_idx + 1
        
    elseif evaluator isa ContinuousEvaluator
        @inbounds output[start_idx] = Float64(data[evaluator.column][row_idx])
        return start_idx + 1
        
    elseif evaluator isa CategoricalEvaluator
        return evaluate_categorical!(evaluator, output, data, row_idx, start_idx)
        
    elseif evaluator isa FunctionEvaluator
        return evaluate_function!(evaluator, output, data, row_idx, start_idx)
        
    elseif evaluator isa InteractionEvaluator
        return evaluate_interaction!(evaluator, output, data, row_idx, start_idx)
        
    elseif evaluator isa ZScoreEvaluator
        return evaluate_zscore!(evaluator, output, data, row_idx, start_idx)
        
    elseif evaluator isa CombinedEvaluator
        return evaluate_combined!(evaluator, output, data, row_idx, start_idx)
        
    else
        error("Unknown evaluator type: $(typeof(evaluator))")
    end
end

function evaluate_categorical!(
    eval::CategoricalEvaluator, output::AbstractVector{Float64}, 
    data, row_idx::Int, start_idx::Int
)
    @inbounds cat_val = data[eval.column][row_idx]
    @inbounds level_code = cat_val isa CategoricalValue ? levelcode(cat_val) : 1
    @inbounds level_code = clamp(level_code, 1, eval.n_levels)
    
    width = size(eval.contrast_matrix, 2)
    @inbounds for j in 1:width
        output[start_idx + j - 1] = eval.contrast_matrix[level_code, j]
    end
    
    return start_idx + width
end

function evaluate_function!(
    eval::FunctionEvaluator, output::AbstractVector{Float64}, 
    data, row_idx::Int, start_idx::Int
)
    n_args = length(eval.arg_evaluators)
    
    if n_args == 1
        # Unary function - most common case
        arg_eval = eval.arg_evaluators[1]
        
        if arg_eval isa ContinuousEvaluator
            @inbounds val = Float64(data[arg_eval.column][row_idx])
        elseif arg_eval isa ConstantEvaluator
            val = arg_eval.value
        else
            # Recursively evaluate complex argument
            temp_val = Vector{Float64}(undef, 1)
            evaluate!(arg_eval, temp_val, data, row_idx, 1)
            val = temp_val[1]
        end
        
        # Apply function with safety
        result = apply_function_safe(eval.func, val)
        @inbounds output[start_idx] = result
        
    elseif n_args == 2
        # Binary function
        val1, val2 = evaluate_two_args(eval.arg_evaluators, data, row_idx)
        result = apply_function_safe(eval.func, val1, val2)
        @inbounds output[start_idx] = result
        
    else
        # General case - evaluate all arguments
        args = Vector{Float64}(undef, n_args)
        for (i, arg_eval) in enumerate(eval.arg_evaluators)
            if arg_eval isa ContinuousEvaluator
                @inbounds args[i] = Float64(data[arg_eval.column][row_idx])
            elseif arg_eval isa ConstantEvaluator
                args[i] = arg_eval.value
            else
                temp_val = Vector{Float64}(undef, 1)
                evaluate!(arg_eval, temp_val, data, row_idx, 1)
                args[i] = temp_val[1]
            end
        end
        
        result = apply_function_safe(eval.func, args...)
        @inbounds output[start_idx] = result
    end
    
    return start_idx + 1
end

function evaluate_two_args(arg_evaluators::Vector{AbstractEvaluator}, data, row_idx::Int)
    val1 = if arg_evaluators[1] isa ContinuousEvaluator
        Float64(data[arg_evaluators[1].column][row_idx])
    elseif arg_evaluators[1] isa ConstantEvaluator
        arg_evaluators[1].value
    else
        temp = Vector{Float64}(undef, 1)
        evaluate!(arg_evaluators[1], temp, data, row_idx, 1)
        temp[1]
    end
    
    val2 = if arg_evaluators[2] isa ContinuousEvaluator
        Float64(data[arg_evaluators[2].column][row_idx])
    elseif arg_evaluators[2] isa ConstantEvaluator
        arg_evaluators[2].value
    else
        temp = Vector{Float64}(undef, 1)
        evaluate!(arg_evaluators[2], temp, data, row_idx, 1)
        temp[1]
    end
    
    return val1, val2
end

function evaluate_interaction!(eval::InteractionEvaluator, output::AbstractVector{Float64}, 
                              data, row_idx::Int, start_idx::Int)
    n_components = length(eval.components)
    component_widths = [output_width(comp) for comp in eval.components]
    
    # Evaluate each component into temporary buffers
    component_buffers = Vector{Vector{Float64}}(undef, n_components)
    
    for (i, component) in enumerate(eval.components)
        width = component_widths[i]
        component_buffers[i] = Vector{Float64}(undef, width)
        evaluate!(component, component_buffers[i], data, row_idx, 1)
    end
    
    # Compute Kronecker product
    compute_kronecker_product!(component_buffers, component_widths, 
                              view(output, start_idx:(start_idx + eval.total_width - 1)))
    
    return start_idx + eval.total_width
end

function evaluate_zscore!(eval::ZScoreEvaluator, output::AbstractVector{Float64}, 
                         data, row_idx::Int, start_idx::Int)
    # Evaluate underlying term
    underlying_width = output_width(eval.underlying)
    temp_buffer = Vector{Float64}(undef, underlying_width)
    evaluate!(eval.underlying, temp_buffer, data, row_idx, 1)
    
    # Apply Z-score transformation
    @inbounds for i in 1:underlying_width
        output[start_idx + i - 1] = (temp_buffer[i] - eval.center) / eval.scale
    end
    
    return start_idx + underlying_width
end

function evaluate_combined!(
    eval::CombinedEvaluator, output::AbstractVector{Float64}, 
    data, row_idx::Int, start_idx::Int
)
    current_idx = start_idx
    
    for sub_eval in eval.sub_evaluators
        current_idx = evaluate!(sub_eval, output, data, row_idx, current_idx)
    end
    
    return current_idx
end

function evaluate!(evaluator::ScaledEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    next_idx = evaluate!(evaluator.evaluator, output, data, row_idx, start_idx)
    @inbounds output[start_idx] *= evaluator.scale_factor
    return next_idx
end

function evaluate!(evaluator::ProductEvaluator, output::AbstractVector{Float64}, 
                  data, row_idx::Int, start_idx::Int=1)
    product = 1.0
    temp_buffer = Vector{Float64}(undef, 1)
    
    for component in evaluator.components
        evaluate!(component, temp_buffer, data, row_idx, 1)
        product *= temp_buffer[1]
    end
    
    @inbounds output[start_idx] = product
    return start_idx + 1
end

###############################################################################
# TESTING UTILITIES
###############################################################################

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
# 5. KRONECKER PRODUCT COMPUTATION
###############################################################################

function compute_kronecker_product!(
    component_buffers::Vector{Vector{Float64}}, 
    component_widths::Vector{Int}, 
    output::AbstractVector{Float64}
)
    n_components = length(component_buffers)
    
    if n_components == 1
        # Single component - just copy
        copy!(output, component_buffers[1])
        
    elseif n_components == 2
        # Two components - direct computation
        w1, w2 = component_widths[1], component_widths[2]
        buf1, buf2 = component_buffers[1], component_buffers[2]
        
        idx = 1
        @inbounds for j in 1:w2
            for i in 1:w1
                output[idx] = buf1[i] * buf2[j]
                idx += 1
            end
        end
        
    elseif n_components == 3
        # Three components - common case
        w1, w2, w3 = component_widths[1], component_widths[2], component_widths[3]
        buf1, buf2, buf3 = component_buffers[1], component_buffers[2], component_buffers[3]
        
        idx = 1
        @inbounds for k in 1:w3
            for j in 1:w2
                for i in 1:w1
                    output[idx] = buf1[i] * buf2[j] * buf3[k]
                    idx += 1
                end
            end
        end
        
    else
        # General case - recursive approach
        compute_general_kronecker!(component_buffers, component_widths, output)
    end
end

function compute_general_kronecker!(
    component_buffers::Vector{Vector{Float64}}, 
    component_widths::Vector{Int}, 
    output::AbstractVector{Float64}
)
    n_components = length(component_buffers)
    total_size = length(output)
    
    @inbounds for i in 1:total_size
        # Convert linear index to multi-dimensional indices
        indices = linear_to_multi_index(i - 1, component_widths) .+ 1
        
        # Compute product across all components
        product = 1.0
        for j in 1:n_components
            product *= component_buffers[j][indices[j]]
        end
        
        output[i] = product
    end
end

function linear_to_multi_index(linear_idx::Int, dimensions::Vector{Int})
    n_dims = length(dimensions)
    indices = Vector{Int}(undef, n_dims)
    
    remaining = linear_idx
    for i in 1:n_dims
        indices[i] = remaining % dimensions[i]
        remaining = remaining ÷ dimensions[i]
    end
    
    return indices
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