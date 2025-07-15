# compositional_compiler.jl
# True compositional approach for zero-allocation model matrix evaluation

###############################################################################
# 1. PRIMITIVE EVALUATORS - The atomic operations
###############################################################################

abstract type AbstractEvaluator end

# Primitive Evaluator Types
struct ConstantEvaluator <: AbstractEvaluator
    value::Float64
end

struct ContinuousEvaluator <: AbstractEvaluator
    column::Symbol
end

struct CategoricalEvaluator <: AbstractEvaluator
    column::Symbol
    contrast_matrix::Matrix{Float64}
    n_levels::Int
end

struct FunctionEvaluator <: AbstractEvaluator
    func::Function
    arg_evaluators::Vector{AbstractEvaluator}
    # Cache for argument buffers to avoid allocations
    arg_buffers::Vector{Vector{Float64}}
    
    function FunctionEvaluator(func, arg_evaluators)
        # Pre-allocate buffers for arguments
        arg_buffers = [Vector{Float64}(undef, 1) for _ in arg_evaluators]
        new(func, arg_evaluators, arg_buffers)
    end
end

struct InteractionEvaluator <: AbstractEvaluator
    components::Vector{AbstractEvaluator}
    component_widths::Vector{Int}
    total_width::Int
    # Pre-allocated scratch space
    component_buffers::Vector{Vector{Float64}}
    
    function InteractionEvaluator(components)
        component_widths = [output_width(comp) for comp in components]
        total_width = prod(component_widths)
        # Pre-allocate buffers for each component
        component_buffers = [Vector{Float64}(undef, w) for w in component_widths]
        new(components, component_widths, total_width, component_buffers)
    end
end

struct ZScoreEvaluator <: AbstractEvaluator
    underlying::AbstractEvaluator
    center::Float64
    scale::Float64
end

###############################################################################
# 4. TERM → EVALUATOR COMPILATION (RECURSIVE)
###############################################################################

"""
Convert a StatsModels term into an evaluator tree.
This is the recursive compilation step.
"""
function compile_term(term::AbstractTerm)
    if term isa InterceptTerm
        return hasintercept(term) ? ConstantEvaluator(1.0) : ConstantEvaluator(0.0)
        
    elseif term isa ConstantTerm
        return ConstantEvaluator(Float64(term.n))
        
    elseif term isa Union{ContinuousTerm, Term}
        return ContinuousEvaluator(term.sym)
        
    elseif term isa CategoricalTerm
        return CategoricalEvaluator(
            term.sym,
            Matrix{Float64}(term.contrasts.matrix),
            size(term.contrasts.matrix, 1)
        )
        
    elseif term isa FunctionTerm
        # Recursively compile arguments
        arg_evaluators = [compile_term(arg) for arg in term.args]
        return FunctionEvaluator(term.f, arg_evaluators)
        
    elseif term isa InteractionTerm
        # Recursively compile components
        component_evaluators = [compile_term(comp) for comp in term.terms]
        return InteractionEvaluator(component_evaluators)
        
    elseif term isa ZScoredTerm
        underlying_evaluator = compile_term(term.term)
        center = term.center isa Number ? Float64(term.center) : Float64(term.center[1])
        scale = term.scale isa Number ? Float64(term.scale) : Float64(term.scale[1])
        return ZScoreEvaluator(underlying_evaluator, center, scale)
        
    elseif term isa MatrixTerm
        # Compile each sub-term and create a combined evaluator
        sub_evaluators = [compile_term(t) for t in term.terms if width(t) > 0]
        return CombinedEvaluator(sub_evaluators)
        
    else
        @warn "Unknown term type: $(typeof(term)), using constant fallback"
        return ConstantEvaluator(1.0)
    end
end

"""
Evaluator that combines multiple sub-evaluators (for MatrixTerm).
"""
struct CombinedEvaluator <: AbstractEvaluator
    sub_evaluators::Vector{AbstractEvaluator}
    sub_widths::Vector{Int}
    total_width::Int
    
    function CombinedEvaluator(sub_evaluators)
        sub_widths = [output_width(eval) for eval in sub_evaluators]
        total_width = sum(sub_widths)
        new(sub_evaluators, sub_widths, total_width)
    end
end

"""
Get the output width (number of columns) for this evaluator.
"""
function output_width(evaluator::AbstractEvaluator)
    error("Not implemented for $(typeof(evaluator))")
end

output_width(eval::CombinedEvaluator) = eval.total_width
output_width(eval::ConstantEvaluator) = 1
output_width(eval::ContinuousEvaluator) = 1
output_width(eval::CategoricalEvaluator) = size(eval.contrast_matrix, 2)
output_width(eval::FunctionEvaluator) = 1
output_width(eval::InteractionEvaluator) = eval.total_width
output_width(eval::ZScoreEvaluator) = output_width(eval.underlying)

###############################################################################
# 5. COMPILED FORMULA INTERFACE
###############################################################################

"""
Extract all column names used in a formula term.
"""
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
