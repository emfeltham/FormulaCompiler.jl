###############################################################################
# DERIVATIVE CACHE AND COMPILATION
###############################################################################

const DERIVATIVE_CACHE = Dict{UInt64, Tuple{Vector{String}, Vector{Symbol}, Int}}()

struct CompiledDerivativeFormula{H}
    formula_val::Val{H}
    output_width::Int  # BREAKING CHANGE: Now same as original
    focal_variable::Symbol
    root_derivative_evaluator::PositionalDerivativeEvaluator
end

Base.length(cdf::CompiledDerivativeFormula) = cdf.output_width

function (cdf::CompiledDerivativeFormula)(row_vec::AbstractVector{Float64}, data, row_idx::Int)
    _derivative_modelrow_generated!(row_vec, cdf.formula_val, data, row_idx)
    return row_vec
end

function compile_derivative_formula(compiled_formula::CompiledFormula, focal_variable::Symbol; verbose=false)
    target_width = compiled_formula.output_width
    root_evaluator = compiled_formula.root_evaluator
    
    if verbose
        println("=== Compiling Full-Width Derivative Formula ===")
        println("Original formula width: $target_width")
        println("Focal variable: $focal_variable")
    end
    
    # SIMPLE: Just wrap the original system with position tracking
    positional_derivative_evaluator = PositionalDerivativeEvaluator(
        root_evaluator, 
        focal_variable, 
        target_width
    )
    
    instructions = generate_code_from_evaluator(positional_derivative_evaluator)
    
    if verbose
        println("Generated $(length(instructions)) full-width derivative instructions")
    end
    
    derivative_hash = hash((compiled_formula.formula_val, focal_variable, :full_width))
    DERIVATIVE_CACHE[derivative_hash] = (instructions, [focal_variable], target_width)
    
    return CompiledDerivativeFormula(
        Val(derivative_hash), 
        target_width,  # Same width as original
        focal_variable, 
        positional_derivative_evaluator
    )
end

@generated function _derivative_modelrow_generated!(
    row_vec::AbstractVector{Float64}, 
    ::Val{derivative_hash}, 
    data, 
    row_idx::Int
) where derivative_hash
    
    if !haskey(DERIVATIVE_CACHE, derivative_hash)
        return quote
            error("Derivative hash $($derivative_hash) not found in cache")
        end
    end
    
    instructions, _, _ = DERIVATIVE_CACHE[derivative_hash]
    
    try
        code_exprs = [Meta.parse(line) for line in instructions]
        return quote
            @inbounds begin
                $(code_exprs...)
            end
            return row_vec
        end
    catch e
        return quote
            error("Failed to parse derivative instructions: $($e)")
        end
    end
end
