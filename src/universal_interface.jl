# New file: src/universal_interface.jl
"""
    compile_formula_auto(model)

Automatically choose the best compilation approach for the given formula.
Falls back to compositional compiler for complex formulas.
"""
function compile_formula_auto(model)
    rhs = fixed_effects_form(model).rhs
    
    # Quick complexity check
    is_complex = has_complex_interactions(rhs) || has_nested_functions(rhs)
    
    if is_complex
        println("Detected complex formula, using compositional compiler...")
        return compile_formula_compositional_efficient(model)
    else
        try
            # Try the simpler approach first (if you want to keep it)
            return compile_formula_generated(model)
        catch e
            println("Simple compiler failed, falling back to compositional...")
            return compile_formula_compositional_efficient(model)
        end
    end
end

function has_complex_interactions(rhs)
    # Check if any interaction terms have complex components
    for term in rhs.terms
        if term isa InteractionTerm
            for component in term.terms
                if !(component isa Union{ContinuousTerm, Term, CategoricalTerm})
                    return true  # Has function terms or other complex components
                end
            end
        end
    end
    return false
end

function has_nested_functions(rhs)
    # Check for nested function calls
    function check_term(term)
        if term isa FunctionTerm
            for arg in term.args
                if arg isa FunctionTerm
                    return true  # Nested function
                end
            end
        end
        return false
    end
    
    return any(check_term(term) for term in rhs.terms)
end
