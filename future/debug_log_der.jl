# Quick fix for the log derivative issue

# The problem is likely in the code generation for function derivatives
# Let's override the specific function that handles log derivatives

function fix_log_derivative_issue!()
    """
    Apply a targeted fix for the log derivative issue.
    The problem is that the derivative is returning z instead of 1/z.
    """
    
    println("Applying log derivative fix...")
    
    # Override the function derivative inline generation
    @eval FormulaCompiler begin
        function generate_function_derivative_inline!(instructions::Vector{String}, 
                                                     evaluator::FunctionEvaluator, 
                                                     focal_variable::Symbol, 
                                                     pos::Int)
            
            func = evaluator.func
            args = evaluator.arg_evaluators
            
            if length(args) == 1 && args[1] isa ContinuousEvaluator && args[1].column == focal_variable
                var_name = args[1].column
                data_expr = "Float64(data.$var_name[row_idx])"
                
                if func === log
                    # FIXED: ∂log(x)/∂x = 1/x (not x!)
                    push!(instructions, "@inbounds row_vec[$pos] = 1.0 / ($data_expr)")
                    
                elseif func === exp
                    push!(instructions, "@inbounds row_vec[$pos] = exp($data_expr)")
                    
                elseif func === sqrt
                    push!(instructions, "@inbounds row_vec[$pos] = 0.5 / sqrt($data_expr)")
                    
                elseif func === sin
                    push!(instructions, "@inbounds row_vec[$pos] = cos($data_expr)")
                    
                elseif func === cos
                    push!(instructions, "@inbounds row_vec[$pos] = -sin($data_expr)")
                    
                else
                    push!(instructions, "@inbounds row_vec[$pos] = 1.0  # Unknown function")
                end
                
            elseif length(args) == 2 && func === (^) && args[2] isa ConstantEvaluator && 
                   args[1] isa ContinuousEvaluator && args[1].column == focal_variable
                var_name = args[1].column
                c = args[2].value
                data_expr = "Float64(data.$var_name[row_idx])"
                
                if c == 0.0
                    push!(instructions, "@inbounds row_vec[$pos] = 0.0")
                elseif c == 1.0
                    push!(instructions, "@inbounds row_vec[$pos] = 1.0")
                elseif c == 2.0
                    push!(instructions, "@inbounds row_vec[$pos] = 2.0 * $data_expr")
                else
                    push!(instructions, "@inbounds row_vec[$pos] = $c * ($data_expr)^$(c-1.0)")
                end
                
            else
                push!(instructions, "@inbounds row_vec[$pos] = 0.0  # Complex function")
            end
        end
    end
    
    # Also fix the ChainRuleEvaluator creation to ensure correct derivative function
    @eval FormulaCompiler begin
        function compute_unary_function_derivative_fixed(func::Function, arg_evaluator::AbstractEvaluator, focal_variable::Symbol)
            inner_derivative = compute_derivative_evaluator(arg_evaluator, focal_variable)
            
            if is_zero_derivative(inner_derivative, focal_variable)
                return ConstantEvaluator(0.0)
            end
            
            if func === log
                # FIXED: Ensure derivative function is correct
                derivative_func = x -> begin
                    if x <= 0.0
                        return 0.0  # Handle domain error
                    else
                        return 1.0 / x  # Correct derivative
                    end
                end
                return ChainRuleEvaluator(derivative_func, arg_evaluator, inner_derivative)
                
            elseif func === exp
                return ChainRuleEvaluator(x -> exp(x), arg_evaluator, inner_derivative)
                
            elseif func === sqrt
                derivative_func = x -> begin
                    if x <= 0.0
                        return 0.0
                    else
                        return 0.5 / sqrt(x)
                    end
                end
                return ChainRuleEvaluator(derivative_func, arg_evaluator, inner_derivative)
                
            else
                return ForwardDiffEvaluator(FunctionEvaluator(func, [arg_evaluator]), focal_variable)
            end
        end
    end
    
    println("✓ Log derivative fix applied")
end

# Test the fix
function test_log_fix()
    println("Testing log derivative fix...")
    
    # Apply the fix
    fix_log_derivative_issue!()
    
    # Test with simple case
    df = DataFrame(
        y = [1.0, 2.0],
        z = [1.1, 2.1]
    )
    
    model = lm(@formula(y ~ log(z)), df)
    compiled = compile_formula(model)
    dz_compiled = compile_derivative_formula(compiled, :z)
    
    data = Tables.columntable(df)
    deriv_vec = Vector{Float64}(undef, length(dz_compiled))
    
    # Test evaluation
    modelrow!(deriv_vec, dz_compiled, data, 1)
    
    expected = [0.0, 1.0 / 1.1]
    error = maximum(abs.(deriv_vec .- expected))
    
    println("Result: $deriv_vec")
    println("Expected: $expected")
    println("Error: $error")
    
    return error < 1e-6
end

# Export the fix function
export fix_log_derivative_issue!, test_log_fix

fix_log_derivative_issue!()
test_log_fix()