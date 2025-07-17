# derivative_modelrow.jl

###############################################################################
# CONVENIENT INTERFACES
###############################################################################

function modelrow!(row_vec::AbstractVector{Float64}, 
                             compiled_derivative::CompiledDerivativeFormula, 
                             data, row_idx::Int)
    compiled_derivative(row_vec, data, row_idx)
    return row_vec
end

function modelrow!(matrix::Matrix{Float64}, 
                             compiled_derivative::CompiledDerivativeFormula, 
                             data, row_indices::Vector{Int})
    for (i, row_idx) in enumerate(row_indices)
        modelrow!(view(matrix, i, :), compiled_derivative, data, row_idx)
    end
    return matrix
end

function marginal_effects!(effects_vec::Vector{Float64}, 
                          compiled_formula::CompiledFormula,
                          compiled_derivatives::Vector{CompiledDerivativeFormula},
                          coefficients::Vector{Float64}, 
                          data, row_idx::Int)
    
    deriv_vec = Vector{Float64}(undef, length(compiled_formula))
    
    for (i, compiled_deriv) in enumerate(compiled_derivatives)
        modelrow!(deriv_vec, compiled_deriv, data, row_idx)
        effects_vec[i] = dot(deriv_vec, coefficients)
    end
    
    return effects_vec
end

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

function clear_derivative_cache!()
    empty!(DERIVATIVE_CACHE)
    return nothing
end

function is_zero_derivative(evaluator::AbstractEvaluator, focal_variable::Symbol)
    if evaluator isa ConstantEvaluator
        return evaluator.value == 0.0
    elseif evaluator isa ContinuousEvaluator
        return false  # A variable is never always zero
    elseif evaluator isa CategoricalEvaluator
        return false  # Categorical values are never always zero
    elseif evaluator isa ZScoreEvaluator
        return is_zero_derivative(evaluator.underlying, focal_variable)
    elseif evaluator isa CombinedEvaluator
        return all(sub_eval -> is_zero_derivative(sub_eval, focal_variable), evaluator.sub_evaluators)
    elseif evaluator isa ScaledEvaluator
        return evaluator.scale_factor == 0.0 || is_zero_derivative(evaluator.evaluator, focal_variable)
    elseif evaluator isa ProductEvaluator
        return any(comp -> is_zero_derivative(comp, focal_variable), evaluator.components)
    elseif evaluator isa InteractionEvaluator
        # If ANY component is always zero, whole interaction is zero (0 * anything = 0)
        return any(comp -> is_zero_derivative(comp, focal_variable), evaluator.components)
    elseif evaluator isa FunctionEvaluator
        # A function evaluator is zero if it always evaluates to zero
        # This is hard to determine in general, so be conservative
        return false
    elseif evaluator isa ChainRuleEvaluator
        # Chain rule result f'(g) * g' is zero if g' is zero (conservative check)
        return is_zero_derivative(evaluator.inner_derivative, focal_variable)
    elseif evaluator isa ProductRuleEvaluator
        # Product rule f*g' + g*f' is zero if both derivatives are zero
        return is_zero_derivative(evaluator.left_derivative, focal_variable) && 
               is_zero_derivative(evaluator.right_derivative, focal_variable)
    elseif evaluator isa ForwardDiffEvaluator
        # Delegate to the underlying evaluator for ForwardDiff cases
        return is_zero_derivative(evaluator.original_evaluator, focal_variable)
    else
        # Conservative: assume non-zero for any remaining unknown cases
        return false
    end
end
