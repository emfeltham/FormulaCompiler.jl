
function generate_chain_rule_code!(instructions::Vector{String}, eval::ChainRuleEvaluator, pos::Int)
    # For chain rule f'(g(x)) * g'(x)
    
    # Generate variable for inner function value
    inner_var = next_var("inner")
    generate_single_component_code!(instructions, eval.inner_evaluator, inner_var)
    
    # Generate variable for inner derivative
    inner_deriv_var = next_var("inner_deriv")
    generate_single_component_code!(instructions, eval.inner_derivative, inner_deriv_var)
    
    # Apply derivative function and multiply by inner derivative
    # For now, we'll need to handle this case by case for standard functions
    push!(instructions, "@inbounds row_vec[$pos] = $(eval.derivative_func)($inner_var) * $inner_deriv_var")
    
    return pos + 1
end

function generate_product_rule_code!(instructions::Vector{String}, eval::ProductRuleEvaluator, pos::Int)
    # For product rule f*g' + g*f'
    
    # Generate variables for all components
    f_var = next_var("f")
    generate_single_component_code!(instructions, eval.left_evaluator, f_var)
    
    f_prime_var = next_var("f_prime")
    generate_single_component_code!(instructions, eval.left_derivative, f_prime_var)
    
    g_var = next_var("g")
    generate_single_component_code!(instructions, eval.right_evaluator, g_var)
    
    g_prime_var = next_var("g_prime")
    generate_single_component_code!(instructions, eval.right_derivative, g_prime_var)
    
    # Apply product rule: f*g' + g*f'
    push!(instructions, "@inbounds row_vec[$pos] = $f_var * $g_prime_var + $g_var * $f_prime_var")
    
    return pos + 1
end