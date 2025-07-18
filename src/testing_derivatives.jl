###############################################################################
# HELPER FUNCTIONS FOR DERIVATIVE TESTING
###############################################################################

"""
Test derivative correctness for a single variable using finite differences.
"""
function test_single_variable_derivative(compiled::CompiledFormula, 
                                        focal_var::Symbol, 
                                        data::NamedTuple, 
                                        n_obs::Int, 
                                        tolerance::Float64,
                                        verbose::Bool)
    
    try
        derivative_compiled = compile_derivative_formula(compiled, focal_var)
        errors = Float64[]
        failed_rows = Int[]
        
        for row_idx in 1:min(n_obs, length(first(data)))
            error, success = test_derivative_at_observation(
                compiled, derivative_compiled, focal_var, data, row_idx, tolerance, verbose
            )
            
            push!(errors, error)
            if !success
                push!(failed_rows, row_idx)
            end
        end
        
        max_error = maximum(errors)
        mean_error = mean(errors)
        var_success = isempty(failed_rows)
        
        report = (
            success = var_success,
            max_error = max_error,
            mean_error = mean_error,
            failed_rows = failed_rows,
            num_tests = min(n_obs, length(first(data)))
        )
        
        return (var_success, report)
        
    catch e
        return (false, (success=false, error=string(e), max_error=Inf, mean_error=Inf, failed_rows=[], num_tests=0))
    end
end

"""
Test derivative correctness at a single observation using finite differences.
"""
function test_derivative_at_observation(compiled::CompiledFormula,
                                       derivative_compiled::CompiledDerivativeFormula,
                                       focal_var::Symbol,
                                       data::NamedTuple,
                                       row_idx::Int,
                                       tolerance::Float64,
                                       verbose::Bool)
    
    try
        # Get analytical derivative
        analytical_deriv = Vector{Float64}(undef, length(derivative_compiled))
        modelrow!(analytical_deriv, derivative_compiled, data, row_idx)
        
        # Get numerical derivative using finite differences
        numerical_deriv = compute_numerical_derivative(compiled, focal_var, data, row_idx)
        
        # Compare (derivatives should have same length)
        if length(analytical_deriv) != length(numerical_deriv)
            return (Inf, false)
        end
        
        # Compute error (use relative error for non-zero values)
        errors = Float64[]
        for i in 1:length(analytical_deriv)
            a_val = analytical_deriv[i]
            n_val = numerical_deriv[i]
            
            if abs(n_val) > 1e-12
                # Relative error for non-zero numerical values
                rel_error = abs(a_val - n_val) / abs(n_val)
                push!(errors, rel_error)
            else
                # Absolute error for near-zero numerical values
                abs_error = abs(a_val - n_val)
                push!(errors, abs_error)
            end
        end
        
        max_error = maximum(errors)
        success = max_error <= tolerance
        
        return (max_error, success)
        
    catch e
        return (Inf, false)
    end
end

"""
Compute numerical derivative using central finite differences.
"""
function compute_numerical_derivative(compiled::CompiledFormula, 
                                     focal_var::Symbol, 
                                     data::NamedTuple, 
                                     row_idx::Int;
                                     ε::Float64=sqrt(eps(Float64)))
    
    # Get current value of focal variable
    current_val = Float64(data[focal_var][row_idx])
    
    # Create modified data for finite differences
    original_column = data[focal_var]
    
    # Evaluate at x + ε
    modified_plus = copy(original_column)
    modified_plus[row_idx] = current_val + ε
    data_plus = merge(data, (focal_var => modified_plus,))
    
    result_plus = Vector{Float64}(undef, length(compiled))
    modelrow!(result_plus, compiled, data_plus, row_idx)
    
    # Evaluate at x - ε
    modified_minus = copy(original_column)
    modified_minus[row_idx] = current_val - ε
    data_minus = merge(data, (focal_var => modified_minus,))
    
    result_minus = Vector{Float64}(undef, length(compiled))
    modelrow!(result_minus, compiled, data_minus, row_idx)
    
    # Central difference: (f(x+ε) - f(x-ε)) / (2ε)
    numerical_derivative = (result_plus .- result_minus) ./ (2ε)
    
    return numerical_derivative
end

"""
Find all continuous variables in a compiled formula by analyzing the evaluator tree.
"""
function find_continuous_variables(compiled::CompiledFormula)
    continuous_vars = Symbol[]
    find_continuous_variables_recursive!(continuous_vars, compiled.root_evaluator)
    return unique(continuous_vars)
end

function find_continuous_variables_recursive!(vars::Vector{Symbol}, evaluator::AbstractEvaluator)
    if evaluator isa ContinuousEvaluator
        push!(vars, evaluator.column)
    elseif evaluator isa FunctionEvaluator
        for arg in evaluator.arg_evaluators
            find_continuous_variables_recursive!(vars, arg)
        end
    elseif evaluator isa InteractionEvaluator
        for comp in evaluator.components
            find_continuous_variables_recursive!(vars, comp)
        end
    elseif evaluator isa CombinedEvaluator
        for sub in evaluator.sub_evaluators
            find_continuous_variables_recursive!(vars, sub)
        end
    elseif evaluator isa ZScoreEvaluator
        find_continuous_variables_recursive!(vars, evaluator.underlying)
    elseif evaluator isa ScaledEvaluator
        find_continuous_variables_recursive!(vars, evaluator.evaluator)
    elseif evaluator isa ProductEvaluator
        for comp in evaluator.components
            find_continuous_variables_recursive!(vars, comp)
        end
    end
end
