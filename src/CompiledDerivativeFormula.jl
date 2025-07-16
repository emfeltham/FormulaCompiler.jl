# CompiledDerivativeFormula.jl

###############################################################################
# DERIVATIVE COMPILATION SYSTEM
###############################################################################

"""
    CompiledDerivativeFormula.jl - Enhanced @generated derivative workflow

BREAKING CHANGE: Now integrates with the main @generated formula compilation system.
All derivatives return full-width vectors matching the original formula.
"""

# DERIVATIVE_CACHE is now defined in derivatives.jl - remove duplicate here

"""
    CompiledDerivativeFormula{H} - UPDATED

BREAKING CHANGE: Now always has output_width == original_formula.output_width

# Fields
- `formula_val::Val{H}`: Hash-based identifier for @generated function dispatch
- `output_width::Int`: SAME as original formula width (BREAKING CHANGE)
- `focal_variable::Symbol`: Variable this derivative is with respect to
- `root_derivative_evaluator::PositionalDerivativeEvaluator`: Always positional
"""
struct CompiledDerivativeFormula{H}
    formula_val::Val{H}
    output_width::Int  # BREAKING CHANGE: Now same as original
    focal_variable::Symbol
    root_derivative_evaluator::PositionalDerivativeEvaluator  # Changed type
end

Base.length(cdf::CompiledDerivativeFormula) = cdf.output_width
variables(cdf::CompiledDerivativeFormula) = [cdf.focal_variable]

# Call interface - mirrors CompiledFormula exactly
function (cdf::CompiledDerivativeFormula)(row_vec::AbstractVector{Float64}, data, row_idx::Int)
    _derivative_modelrow_generated!(row_vec, cdf.formula_val, data, row_idx)
    return row_vec
end

"""
    compile_derivative_formula(compiled_formula::CompiledFormula, focal_variable::Symbol; verbose=false)

BREAKING CHANGE: Now returns full-width derivatives using @generated functions.

This replaces the previous scalar derivative system with a full-width derivative system
that produces vectors of the same length as the original formula. This enables:

1. **Drop-in replacement for marginal effects**: `dot(derivative_vec, coef(model))`
2. **Consistent vector operations**: All vectors have same length
3. **Position-aware derivatives**: Each model term gets its derivative in the right place

# Example
```julia
model = lm(@formula(y ~ x + log(z) + x*group), df)  # 4 parameters
compiled = compile_formula(model)                   # width = 4
dx_compiled = compile_derivative_formula(compiled, :x)  # width = 4

# Both vectors have same length:
model_vec = Vector{Float64}(undef, 4)      # [1.0, x_i, log(z_i), x_i*group_i]
deriv_vec = Vector{Float64}(undef, 4)      # [0.0, 1.0, 0.0, group_i]

compiled(model_vec, data, i)
dx_compiled(deriv_vec, data, i)

# Perfect alignment for marginal effects:
marginal_effect = dot(deriv_vec, coef(model))
```
"""
function compile_derivative_formula(compiled_formula::CompiledFormula, focal_variable::Symbol; verbose=false)
    target_width = compiled_formula.output_width
    root_evaluator = compiled_formula.root_evaluator
    
    if verbose
        println("=== Compiling Full-Width Derivative Formula ===")
        println("Original formula width: $target_width")
        println("Focal variable: $focal_variable")
        println("Using @generated compilation workflow")
    end
    
    # Create PositionalDerivativeEvaluator - this is the key integration point
    positional_derivative_evaluator = PositionalDerivativeEvaluator(
        root_evaluator, 
        focal_variable, 
        target_width
    )
    
    # Generate code instructions using EXISTING infrastructure from generators.jl
    # This is the key - we reuse generate_code_from_evaluator
    instructions = generate_code_from_evaluator(positional_derivative_evaluator)
    
    if verbose
        println("Generated $(length(instructions)) full-width derivative instructions:")
        for (i, instr) in enumerate(instructions[1:min(3, length(instructions))])
            println("  $i: $instr")
        end
        if length(instructions) > 3
            println("  ... ($(length(instructions) - 3) more)")
        end
    end
    
    # Cache for @generated function - same pattern as FORMULA_CACHE
    derivative_hash = hash((compiled_formula.formula_val, focal_variable, :full_width))
    DERIVATIVE_CACHE[derivative_hash] = (instructions, [focal_variable], target_width)
    
    if verbose
        println("Cached derivative with hash: $derivative_hash")
        println("Cache now contains $(length(DERIVATIVE_CACHE)) derivatives")
    end
    
    return CompiledDerivativeFormula(
        Val(derivative_hash), 
        target_width,  # BREAKING CHANGE: Same width as original
        focal_variable, 
        positional_derivative_evaluator
    )
end

# The @generated function is defined in derivatives.jl
