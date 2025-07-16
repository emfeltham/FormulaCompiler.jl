# full_recursive_derivative_compilation.jl - Complete AST recursion for derivatives

###############################################################################
# ENHANCED RECURSIVE DERIVATIVE COMPILATION - Trust the AST Recursion
###############################################################################

"""
    Enhanced compute_derivative_evaluator that uses full AST recursion power.
    
    Philosophy: If the evaluator tree can represent it, we can differentiate it analytically.
    Only fall back to ForwardDiff for truly unknown function types.
"""
function compute_derivative_evaluator(evaluator::AbstractEvaluator, focal_variable::Symbol)
    
    # Base cases - leaf evaluators
    if evaluator isa ConstantEvaluator
        return ConstantEvaluator(0.0)  # ∂c/∂x = 0
        
    elseif evaluator isa ContinuousEvaluator
        if evaluator.column == focal_variable
            return ConstantEvaluator(1.0)  # ∂x/∂x = 1
        else
            return ConstantEvaluator(0.0)  # ∂y/∂x = 0
        end
        
    elseif evaluator isa CategoricalEvaluator
        return ConstantEvaluator(0.0)  # ∂(categorical)/∂x = 0 for continuous x
        
    # Composite cases - FULL RECURSIVE POWER
    elseif evaluator isa FunctionEvaluator
        return compute_function_derivative_recursive(evaluator, focal_variable)
        
    elseif evaluator isa InteractionEvaluator
        return compute_interaction_derivative_recursive(evaluator, focal_variable)
        
    elseif evaluator isa ZScoreEvaluator
        # ∂((g-center)/scale)/∂x = (1/scale) * ∂g/∂x
        underlying_derivative = compute_derivative_evaluator(evaluator.underlying, focal_variable)
        
        # Optimization: if underlying derivative is zero, whole thing is zero
        if is_zero_derivative(underlying_derivative, focal_variable)
            return ConstantEvaluator(0.0)
        else
            return ScaledEvaluator(underlying_derivative, 1.0 / evaluator.scale)
        end
        
    elseif evaluator isa CombinedEvaluator
        # Sum rule: ∂(f+g+h+...)/∂x = ∂f/∂x + ∂g/∂x + ∂h/∂x + ...
        return compute_sum_derivative_recursive(evaluator, focal_variable)
        
    # Handle derivative evaluator types recursively too
    elseif evaluator isa ChainRuleEvaluator
        return compute_chain_rule_derivative(evaluator, focal_variable)
        
    elseif evaluator isa ProductRuleEvaluator  
        return compute_product_rule_derivative(evaluator, focal_variable)
        
    elseif evaluator isa ScaledEvaluator
        # ∂(c*f)/∂x = c * ∂f/∂x
        inner_derivative = compute_derivative_evaluator(evaluator.evaluator, focal_variable)
        return ScaledEvaluator(inner_derivative, evaluator.scale_factor)
        
    else
        # Only fall back to ForwardDiff for truly unknown evaluator types
        @info "Using ForwardDiff fallback for unknown evaluator type: $(typeof(evaluator))"
        return ForwardDiffEvaluator(evaluator, focal_variable, NamedTuple())
    end
end

###############################################################################
# ENHANCED FUNCTION DERIVATIVES - Handle More Cases Analytically
###############################################################################

"""
    compute_function_derivative_recursive(evaluator::FunctionEvaluator, focal_variable::Symbol)
    
    Use full recursive power to handle function derivatives analytically.
    Supports unary functions, binary operations, and arbitrary nesting.
"""
function compute_function_derivative_recursive(evaluator::FunctionEvaluator, focal_variable::Symbol)
    func = evaluator.func
    args = evaluator.arg_evaluators
    n_args = length(args)
    
    # === UNARY FUNCTIONS ===
    if n_args == 1
        return compute_unary_function_derivative(func, args[1], focal_variable)
        
    # === BINARY OPERATIONS ===  
    elseif n_args == 2
        return compute_binary_function_derivative(func, args[1], args[2], focal_variable)
        
    # === N-ARY FUNCTIONS ===
    elseif n_args > 2
        # For now, most n-ary functions are extensions of binary (like +, *)
        # Handle them by treating as compositions of binary operations
        return compute_nary_function_derivative(func, args, focal_variable)
        
    else
        # Zero arguments - shouldn't happen but handle gracefully
        return ConstantEvaluator(0.0)
    end
end

"""
    compute_unary_function_derivative(func, arg_evaluator, focal_variable)
    
    Handle unary function derivatives using chain rule: ∂f(g)/∂x = f'(g) * ∂g/∂x
"""
function compute_unary_function_derivative(func::Function, arg_evaluator::AbstractEvaluator, focal_variable::Symbol)
    # Get derivative of inner function
    inner_derivative = compute_derivative_evaluator(arg_evaluator, focal_variable)
    
    # Optimization: if inner derivative is zero, whole thing is zero
    if is_zero_derivative(inner_derivative, focal_variable)
        return ConstantEvaluator(0.0)
    end
    
    # Get analytical derivative function if available
    derivative_func = get_standard_derivative_function(func)
    
    if derivative_func !== nothing
        # Chain rule: f'(g) * g'
        return ChainRuleEvaluator(derivative_func, arg_evaluator, inner_derivative)
    else
        # Unknown unary function - use ForwardDiff
        @info "Using ForwardDiff for unknown unary function: $func"
        return ForwardDiffEvaluator(FunctionEvaluator(func, [arg_evaluator]), focal_variable, NamedTuple())
    end
end

"""
    compute_binary_function_derivative(func, left_arg, right_arg, focal_variable)
    
    Handle binary operations analytically using appropriate derivative rules.
"""
function compute_binary_function_derivative(func::Function, left_arg::AbstractEvaluator, right_arg::AbstractEvaluator, focal_variable::Symbol)
    
    # Get derivatives of both arguments
    left_derivative = compute_derivative_evaluator(left_arg, focal_variable)
    right_derivative = compute_derivative_evaluator(right_arg, focal_variable)
    
    # Check if either argument has zero derivative (optimization)
    left_is_zero = is_zero_derivative(left_derivative, focal_variable)
    right_is_zero = is_zero_derivative(right_derivative, focal_variable)
    
    if func === (+) || func === (-)
        # Addition/Subtraction: ∂(f ± g)/∂x = ∂f/∂x ± ∂g/∂x
        if left_is_zero && right_is_zero
            return ConstantEvaluator(0.0)
        elseif left_is_zero
            return func === (+) ? right_derivative : ScaledEvaluator(right_derivative, -1.0)
        elseif right_is_zero
            return left_derivative
        else
            if func === (+)
                return CombinedEvaluator([left_derivative, right_derivative])
            else  # subtraction
                return CombinedEvaluator([left_derivative, ScaledEvaluator(right_derivative, -1.0)])
            end
        end
        
    elseif func === (*)
    # Multiplication: ∂(f*g)/∂x = f*∂g/∂x + g*∂f/∂x
    # Handle commutative nature - position doesn't matter
    
    if left_is_zero && right_is_zero
        return ConstantEvaluator(0.0)
    else
        # Check for unit derivatives (∂x/∂x = 1)
        left_is_one = (left_derivative isa ConstantEvaluator && left_derivative.value == 1.0)
        right_is_one = (right_derivative isa ConstantEvaluator && right_derivative.value == 1.0)
        
        # Check if arguments are constants
        left_is_constant = left_arg isa ConstantEvaluator
        right_is_constant = right_arg isa ConstantEvaluator
        
        # Case 1: One argument is constant, other has unit derivative
        # ∂(c*x)/∂x = c*1 + x*0 = c  OR  ∂(x*c)/∂x = x*0 + c*1 = c
        if left_is_constant && right_is_one
            return left_arg  # Return the constant
        elseif right_is_constant && left_is_one
            return right_arg  # Return the constant
            
        # Case 2: One argument is constant, other has zero derivative
        # ∂(c*f)/∂x = c*f' + f*0 = c*f'  OR  ∂(f*c)/∂x = f*0 + c*f' = c*f'
        elseif left_is_constant && right_is_zero
            return ConstantEvaluator(0.0)  # c*0 = 0
        elseif right_is_constant && left_is_zero
            return ConstantEvaluator(0.0)  # c*0 = 0
            
        # Case 3: One argument is constant, other has general derivative
        # ∂(c*f)/∂x = c*f'  OR  ∂(f*c)/∂x = c*f'
        elseif left_is_constant
            if left_arg.value == 1.0
                return right_derivative  # 1*f' = f'
            elseif left_arg.value == 0.0
                return ConstantEvaluator(0.0)  # 0*f' = 0
            else
                return ScaledEvaluator(right_derivative, left_arg.value)  # c*f'
            end
        elseif right_is_constant
            if right_arg.value == 1.0
                return left_derivative  # f'*1 = f'
            elseif right_arg.value == 0.0
                return ConstantEvaluator(0.0)  # f'*0 = 0
            else
                return ScaledEvaluator(left_derivative, right_arg.value)  # f'*c
            end
            
        # Case 4: Both derivatives are zero
        elseif left_is_zero && right_is_zero
            return ConstantEvaluator(0.0)
            
        # Case 5: One derivative is zero, other is not
        elseif left_is_zero
            # ∂(f*g)/∂x = f*g' + g*0 = f*g'
            product = ProductEvaluator([left_arg, right_derivative])
            return simplify_product_evaluator(product)
        elseif right_is_zero
            # ∂(f*g)/∂x = f*0 + g*f' = g*f'
            product = ProductEvaluator([right_arg, left_derivative])
            return simplify_product_evaluator(product)
            
        # Case 6: Both derivatives are unit (both are variables w.r.t. focal_variable)
        elseif left_is_one && right_is_one
            # ∂(f*g)/∂x = f*1 + g*1 = f + g
            return CombinedEvaluator([left_arg, right_arg])
            
        # Case 7: One derivative is unit, other is not
        elseif left_is_one
            # ∂(f*g)/∂x = f*g' + g*1 = f*g' + g
            term1 = ProductEvaluator([left_arg, right_derivative])
            return CombinedEvaluator([term1, right_arg])
        elseif right_is_one
            # ∂(f*g)/∂x = f*1 + g*f' = f + g*f'
            term2 = ProductEvaluator([right_arg, left_derivative])
            return CombinedEvaluator([left_arg, term2])
            
        # Case 8: General case - neither is constant, neither has unit/zero derivative
        else
            # ∂(f*g)/∂x = f*g' + g*f'
            term1 = ProductEvaluator([left_arg, right_derivative])
            term2 = ProductEvaluator([right_arg, left_derivative])
            return CombinedEvaluator([term1, term2])
        end
    end 
    elseif func === (/)
        # Division: ∂(f/g)/∂x = (g*∂f/∂x - f*∂g/∂x) / g²
        if left_is_zero && right_is_zero
            return ConstantEvaluator(0.0)
        elseif right_is_zero
            # ∂(f/g)/∂x = (g*f' - f*0) / g² = g*f' / g² = f'/g
            return FunctionEvaluator(/, [left_derivative, right_arg])
        elseif left_is_zero
            # ∂(f/g)/∂x = (g*0 - f*g') / g² = -f*g' / g²
            numerator = ProductEvaluator([ScaledEvaluator(left_arg, -1.0), right_derivative])
            denominator = ProductEvaluator([right_arg, right_arg])  # g²
            return FunctionEvaluator(/, [numerator, denominator])
        else
            return compute_division_derivative(left_arg, left_derivative, right_arg, right_derivative)
        end
        
    elseif func === (^)
        # Power: ∂(f^g)/∂x = f^g * (g'*ln(f) + g*f'/f)
        return compute_power_derivative(left_arg, left_derivative, right_arg, right_derivative)
        
    elseif func === max || func === min
        # Max/Min: Handle using conditional derivatives (more complex)
        return compute_extremum_derivative(func, left_arg, left_derivative, right_arg, right_derivative)
        
    else
        # Unknown binary function - use ForwardDiff
        @info "Using ForwardDiff for unknown binary function: $func"
        return ForwardDiffEvaluator(FunctionEvaluator(func, [left_arg, right_arg]), focal_variable, NamedTuple())
    end
end

"""
    compute_nary_function_derivative(func, args, focal_variable)
    
    Handle n-ary functions by reducing to binary operations where possible.
"""
function compute_nary_function_derivative(func::Function, args::Vector{AbstractEvaluator}, focal_variable::Symbol)
    
    if func === (+)
        # N-ary addition: ∂(f₁+f₂+...+fₙ)/∂x = ∂f₁/∂x + ∂f₂/∂x + ... + ∂fₙ/∂x
        derivatives = [compute_derivative_evaluator(arg, focal_variable) for arg in args]
        non_zero_derivatives = filter(!is_zero_evaluator, derivatives)
        
        if isempty(non_zero_derivatives)
            return ConstantEvaluator(0.0)
        elseif length(non_zero_derivatives) == 1
            return non_zero_derivatives[1]
        else
            return CombinedEvaluator(non_zero_derivatives)
        end
        
    elseif func === (*)
        # N-ary multiplication: Use generalized product rule
        return compute_nary_product_derivative(args, focal_variable)
        
    else
        # Unknown n-ary function - use ForwardDiff
        @info "Using ForwardDiff for unknown n-ary function: $func with $(length(args)) arguments"
        return ForwardDiffEvaluator(FunctionEvaluator(func, args), focal_variable, NamedTuple())
    end
end

###############################################################################
# N-WAY INTERACTION DERIVATIVES - Full Recursive Power
###############################################################################

"""
    compute_interaction_derivative_recursive(evaluator::InteractionEvaluator, focal_variable::Symbol)
    
    Handle N-way interactions using full product rule: ∂(f₁*f₂*...*fₙ)/∂x = Σᵢ (∂fᵢ/∂x * ∏ⱼ≠ᵢ fⱼ)
    
    This is the key function that unleashes the full power of recursive derivatives!
"""
function compute_interaction_derivative_recursive(evaluator::InteractionEvaluator, focal_variable::Symbol)
    components = evaluator.components
    n_components = length(components)
    
    if n_components == 0
        return ConstantEvaluator(0.0)
    elseif n_components == 1
        # Single component - just return its derivative
        return compute_derivative_evaluator(components[1], focal_variable)
    else
        # N-way product rule for interactions: ∂(f₁*f₂*...*fₙ)/∂x = Σᵢ (∂fᵢ/∂x * ∏ⱼ≠ᵢ fⱼ)
        # But stay in the interaction domain - don't mix with ProductEvaluator
        
        sum_terms = AbstractEvaluator[]
        
        for i in 1:n_components
            component_derivative = compute_derivative_evaluator(components[i], focal_variable)
            
            # Skip terms where derivative is zero (optimization)
            if is_zero_derivative(component_derivative, focal_variable)
                continue
            end
            
            # Get all other components: ∏ⱼ≠ᵢ fⱼ
            other_components = [components[j] for j in 1:n_components if j != i]
            
            if isempty(other_components)
                # Single component case: just the derivative
                push!(sum_terms, component_derivative)
            elseif length(other_components) == 1
                # Two components total: ∂fᵢ/∂x * fⱼ
                if component_derivative isa ConstantEvaluator && component_derivative.value == 1.0
                    # Optimization: 1 * fⱼ = fⱼ
                    push!(sum_terms, other_components[1])
                else
                    # Need to multiply derivative by the other component
                    # This creates a mathematical product, not a statistical interaction
                    push!(sum_terms, ProductEvaluator([component_derivative, other_components[1]]))
                end
            else
                # Multiple other components: ∂fᵢ/∂x * (∏ⱼ≠ᵢ fⱼ)
                # The product of other components stays as an interaction
                other_interaction = InteractionEvaluator(other_components)
                
                if component_derivative isa ConstantEvaluator && component_derivative.value == 1.0
                    # Optimization: 1 * (interaction) = interaction
                    push!(sum_terms, other_interaction)
                else
                    # Need to multiply derivative by the interaction
                    # This creates a mathematical product of (derivative * interaction)
                    push!(sum_terms, ProductEvaluator([component_derivative, other_interaction]))
                end
            end
        end
        
        # Return sum of all terms
        if isempty(sum_terms)
            return ConstantEvaluator(0.0)
        elseif length(sum_terms) == 1
            return sum_terms[1]
        else
            return CombinedEvaluator(sum_terms)
        end
    end
end

"""
    compute_nary_product_derivative(components, focal_variable)
    
    Implement the generalized product rule for N components:
    ∂(f₁*f₂*...*fₙ)/∂x = Σᵢ (∂fᵢ/∂x * ∏ⱼ≠ᵢ fⱼ)
"""
function compute_nary_product_derivative(components::Vector{AbstractEvaluator}, focal_variable::Symbol)
    n = length(components)
    sum_terms = AbstractEvaluator[]
    
    # For each component i, create term: ∂fᵢ/∂x * ∏ⱼ≠ᵢ fⱼ
    for i in 1:n
        component_derivative = compute_derivative_evaluator(components[i], focal_variable)
        
        # Optimization: skip terms where derivative is zero
        if is_zero_derivative(component_derivative, focal_variable)
            continue
        end
        
        # Get all other components: ∏ⱼ≠ᵢ fⱼ
        other_components = [components[j] for j in 1:n if j != i]
        
        if isempty(other_components)
            # Single component case: just the derivative
            push!(sum_terms, component_derivative)
        elseif length(other_components) == 1
            # Two-factor product: ∂fᵢ/∂x * fⱼ
            push!(sum_terms, ProductEvaluator([component_derivative, other_components[1]]))
        else
            # Multi-factor product: ∂fᵢ/∂x * (∏ⱼ≠ᵢ fⱼ)
            other_product = InteractionEvaluator(other_components)
            push!(sum_terms, ProductEvaluator([component_derivative, other_product]))
        end
    end
    
    # Return sum of all terms
    if isempty(sum_terms)
        return ConstantEvaluator(0.0)
    elseif length(sum_terms) == 1
        return sum_terms[1]
    else
        return CombinedEvaluator(sum_terms)
    end
end

###############################################################################
# SPECIALIZED BINARY OPERATION DERIVATIVES
###############################################################################

"""
    compute_division_derivative(f, f_prime, g, g_prime)
    
    Division rule: ∂(f/g)/∂x = (g*∂f/∂x - f*∂g/∂x) / g²
"""
function compute_division_derivative(f::AbstractEvaluator, f_prime::AbstractEvaluator, 
                                   g::AbstractEvaluator, g_prime::AbstractEvaluator)
    # Numerator: g*f' - f*g'
    term1 = ProductEvaluator([g, f_prime])
    term2 = ProductEvaluator([f, g_prime])
    numerator = CombinedEvaluator([term1, ScaledEvaluator(term2, -1.0)])
    
    # Denominator: g²
    denominator = InteractionEvaluator([g, g])
    
    # Result: numerator / denominator
    return FunctionEvaluator(/, [numerator, denominator])
end

"""
    compute_power_derivative(base, base_deriv, exponent, exp_deriv)
    
    Power rule: ∂(f^g)/∂x = f^g * (g'*ln(f) + g*f'/f)
"""
function compute_power_derivative(base::AbstractEvaluator, base_deriv::AbstractEvaluator,
                                exponent::AbstractEvaluator, exp_deriv::AbstractEvaluator)
    
    # Check for special cases
    if exponent isa ConstantEvaluator
        # Constant exponent: ∂(f^c)/∂x = c * f^(c-1) * ∂f/∂x
        c = exponent.value
        if c == 0.0
            return ConstantEvaluator(0.0)
        elseif c == 1.0
            return base_deriv
        else
            # c * f^(c-1) * f'
            new_exponent = ConstantEvaluator(c - 1.0)
            f_to_c_minus_1 = FunctionEvaluator(^, [base, new_exponent])
            coefficient = ConstantEvaluator(c)
            return ProductEvaluator([coefficient, f_to_c_minus_1, base_deriv])
        end
    elseif base isa ConstantEvaluator
        # Constant base: ∂(c^g)/∂x = c^g * ln(c) * ∂g/∂x
        c = base.value
        if c <= 0
            @warn "Power derivative with non-positive base $c, using ForwardDiff"
            return ForwardDiffEvaluator(FunctionEvaluator(^, [base, exponent]), :dummy, NamedTuple())
        else
            c_to_g = FunctionEvaluator(^, [base, exponent])
            ln_c = ConstantEvaluator(log(c))
            return ProductEvaluator([c_to_g, ln_c, exp_deriv])
        end
    else
        # General case: ∂(f^g)/∂x = f^g * (g'*ln(f) + g*f'/f)
        f_to_g = FunctionEvaluator(^, [base, exponent])
        ln_f = FunctionEvaluator(log, [base])
        f_prime_over_f = FunctionEvaluator(/, [base_deriv, base])
        
        # g'*ln(f) + g*f'/f
        term1 = ProductEvaluator([exp_deriv, ln_f])
        term2 = ProductEvaluator([exponent, f_prime_over_f])
        bracket_term = CombinedEvaluator([term1, term2])
        
        return ProductEvaluator([f_to_g, bracket_term])
    end
end

"""
    compute_extremum_derivative(func, left_arg, left_deriv, right_arg, right_deriv)
    
    Handle max/min derivatives (more complex due to non-differentiability at equality)
"""
function compute_extremum_derivative(func::Function, left_arg::AbstractEvaluator, left_deriv::AbstractEvaluator,
                                   right_arg::AbstractEvaluator, right_deriv::AbstractEvaluator)
    # For max/min, derivative is not well-defined when arguments are equal
    # Use subgradient approach or ForwardDiff for robustness
    @info "Using ForwardDiff for extremum function: $func (non-differentiable at equality)"
    return ForwardDiffEvaluator(FunctionEvaluator(func, [left_arg, right_arg]), :dummy, NamedTuple())
end

###############################################################################
# SUM RULE WITH OPTIMIZATION
###############################################################################

"""
    compute_sum_derivative_recursive(evaluator::CombinedEvaluator, focal_variable::Symbol)
    
    Sum rule with optimization: ∂(f+g+h+...)/∂x = ∂f/∂x + ∂g/∂x + ∂h/∂x + ...
    Skip zero terms for efficiency.
"""
function compute_sum_derivative_recursive(evaluator::CombinedEvaluator, focal_variable::Symbol)
    derivative_terms = AbstractEvaluator[]
    
    for sub_evaluator in evaluator.sub_evaluators
        sub_derivative = compute_derivative_evaluator(sub_evaluator, focal_variable)
        
        # Optimization: skip zero derivatives
        if !is_zero_derivative(sub_derivative, focal_variable)
            push!(derivative_terms, sub_derivative)
        end
    end
    
    # Return optimized sum
    if isempty(derivative_terms)
        return ConstantEvaluator(0.0)
    elseif length(derivative_terms) == 1
        return derivative_terms[1]
    else
        return CombinedEvaluator(derivative_terms)
    end
end

###############################################################################
# HIGHER-ORDER DERIVATIVE SUPPORT
###############################################################################

"""
    compute_chain_rule_derivative(evaluator::ChainRuleEvaluator, focal_variable::Symbol)
    
    Handle derivatives of derivatives (for higher-order derivatives).
"""
function compute_chain_rule_derivative(evaluator::ChainRuleEvaluator, focal_variable::Symbol)
    # ∂(f'(g) * g')/∂x = f''(g) * g' * g' + f'(g) * g''
    # This gets complex quickly, so for now use ForwardDiff
    @info "Using ForwardDiff for derivative of ChainRuleEvaluator (higher-order)"
    return ForwardDiffEvaluator(evaluator, focal_variable, NamedTuple())
end

"""
    compute_product_rule_derivative(evaluator::ProductRuleEvaluator, focal_variable::Symbol)
    
    Handle derivatives of product rule expressions.
"""
function compute_product_rule_derivative(evaluator::ProductRuleEvaluator, focal_variable::Symbol)
    # This is derivative of (f*g' + g*f'), which gets quite complex
    # For now use ForwardDiff for higher-order derivatives
    @info "Using ForwardDiff for derivative of ProductRuleEvaluator (higher-order)"
    return ForwardDiffEvaluator(evaluator, focal_variable, NamedTuple())
end

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

"""
    is_zero_evaluator(evaluator::AbstractEvaluator) -> Bool
    
    Enhanced check for zero evaluators, including more patterns.
"""
function is_zero_evaluator(evaluator::AbstractEvaluator)
    if evaluator isa ConstantEvaluator
        return evaluator.value == 0.0
    elseif evaluator isa ScaledEvaluator
        return evaluator.scale_factor == 0.0 || is_zero_evaluator(evaluator.evaluator)
    elseif evaluator isa CombinedEvaluator
        return all(is_zero_evaluator, evaluator.sub_evaluators)
    elseif evaluator isa ProductEvaluator
        return any(is_zero_evaluator, evaluator.components)
    else
        return false
    end
end

"""
    is_zero_derivative(evaluator::AbstractEvaluator, focal_variable::Symbol) -> Bool

Check if an evaluator always evaluates to zero.
Used for optimization in derivative computation - if a derivative evaluator is always zero, 
we can skip it in sums or use ConstantEvaluator(0.0).

# Examples
- ConstantEvaluator(0.0) → true (always evaluates to 0)
- ConstantEvaluator(5.0) → false (evaluates to 5, not 0)
- ContinuousEvaluator(:x) → false (evaluates to x, not 0)
- ProductEvaluator([ConstantEvaluator(0.0), anything]) → true (0 * anything = 0)
"""
function is_zero_derivative(evaluator::AbstractEvaluator, focal_variable::Symbol)
    if evaluator isa ConstantEvaluator
        return evaluator.value == 0.0  # Check if the constant VALUE is zero
        
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

"""
    get_standard_derivative_function(func::Function) -> Union{Function, Nothing}

Return the analytical derivative function for standard mathematical functions.
Returns `nothing` if no analytical derivative is available.

# Supported Functions
- `log` → `x -> 1/x`
- `exp` → `x -> exp(x)` 
- `sqrt` → `x -> 1/(2*sqrt(x))`
- `sin` → `x -> cos(x)`
- `cos` → `x -> -sin(x)`
- `abs` → `x -> sign(x)` (except at x=0)
"""
function get_standard_derivative_function(func::Function)
    if func === log
        return x -> inv(x)
    elseif func === exp
        return x -> exp(x)
    elseif func === sqrt
        return x -> 1/(2*sqrt(x))
    elseif func === sin
        return x -> cos(x)
    elseif func === cos
        return x -> -sin(x)
    elseif func === tan
        return x -> 1 + tan(x)^2  # sec²(x) = 1 + tan²(x)
    elseif func === atan
        return x -> 1/(1 + x^2)
    elseif func === sinh
        return x -> cosh(x)
    elseif func === cosh
        return x -> sinh(x)
    elseif func === tanh
        return x -> 1 - tanh(x)^2  # sech²(x) = 1 - tanh²(x)
    elseif func === log10
        return x -> 1/(x * log(10))
    elseif func === log2
        return x -> 1/(x * log(2))
    elseif func === sign
        return x -> 0.0  # Derivative of sign is 0 (except at 0 where undefined)   
    elseif func === abs
        # Note: derivative undefined at x=0, but we'll use sign function
        return x -> x == 0 ? 0.0 : sign(x)
    else
        return nothing
    end
end

"""
    simplify_product_evaluator(evaluator::ProductEvaluator) -> AbstractEvaluator

Simplify ProductEvaluator by handling common cases like multiplication by 1 or 0.
"""
function simplify_product_evaluator(evaluator::ProductEvaluator)
    components = evaluator.components
    
    # Check for multiplication by 0
    for comp in components
        if comp isa ConstantEvaluator && comp.value == 0.0
            return ConstantEvaluator(0.0)
        end
    end
    
    # Filter out multiplication by 1
    non_unit_components = AbstractEvaluator[]
    for comp in components
        if !(comp isa ConstantEvaluator && comp.value == 1.0)
            push!(non_unit_components, comp)
        end
    end
    
    # Return simplified form
    if isempty(non_unit_components)
        return ConstantEvaluator(1.0)  # All components were 1
    elseif length(non_unit_components) == 1
        return non_unit_components[1]  # Only one non-unit component
    else
        return ProductEvaluator(non_unit_components)  # Multiple components remain
    end
end