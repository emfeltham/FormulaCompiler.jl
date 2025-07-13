# Fix for component variable extraction bug

"""
Debug the component extraction to see exactly what's happening
"""
function debug_interaction_components(model, df)
    println("🔍 DEBUGGING INTERACTION COMPONENT EXTRACTION")
    
    ws = AMEWorkspace(model, df)
    
    # Find the 3-way interaction term
    interaction_term = nothing
    for (term, range) in ws.mapping.term_info
        if 8 in range
            interaction_term = term
            break
        end
    end
    
    println("Found term: $interaction_term")
    println("Term type: $(typeof(interaction_term))")
    
    if interaction_term isa InteractionTerm
        println("\\nInteraction details:")
        println("  interaction_term.terms: $(interaction_term.terms)")
        println("  length: $(length(interaction_term.terms))")
        
        for (i, component) in enumerate(interaction_term.terms)
            println("  Component $i:")
            println("    Value: $component")
            println("    Type: $(typeof(component))")
            
            # Try to extract the symbol
            if hasfield(typeof(component), :sym)
                println("    Symbol: $(component.sym)")
            else
                println("    No sym field")
                println("    Fields: $(fieldnames(typeof(component)))")
            end
        end
        
        # Test our current collection function
        println("\\n🧪 Testing collect_termvars_recursive:")
        vars = collect_termvars_recursive(interaction_term)
        println("  Result: $vars")
        
        # Test manual extraction
        println("\\n🧪 Manual extraction attempt:")
        manual_vars = Symbol[]
        for component in interaction_term.terms
            if component isa Term
                push!(manual_vars, component.sym)
                println("  Found Term with sym: $(component.sym)")
            elseif component isa ContinuousTerm
                push!(manual_vars, component.sym) 
                println("  Found ContinuousTerm with sym: $(component.sym)")
            elseif component isa CategoricalTerm
                push!(manual_vars, component.sym)
                println("  Found CategoricalTerm with sym: $(component.sym)")
            else
                println("  Unknown component type: $(typeof(component))")
                # Try to find sym field anyway
                try
                    if hasfield(typeof(component), :sym)
                        sym = getfield(component, :sym)
                        push!(manual_vars, sym)
                        println("    Found sym via getfield: $sym")
                    end
                catch e
                    println("    Could not extract sym: $e")
                end
            end
        end
        println("  Manual result: $manual_vars")
        
        return interaction_term, manual_vars
    end
    
    return nothing, Symbol[]
end

# Check what the issue is
interaction_term, manual_vars = debug_interaction_components(model, df)

# Now let's fix the collect_termvars_recursive function if needed
"""
Fixed version of _collect_vars_recursive! for InteractionTerm
"""
function _collect_vars_recursive_fixed!(vars::Set{Symbol}, term::InteractionTerm)
    println("🔧 _collect_vars_recursive_fixed! called with InteractionTerm")
    println("   term.terms: $(term.terms)")
    
    for subterm in term.terms
        println("   Processing subterm: $subterm ($(typeof(subterm)))")
        _collect_vars_recursive_fixed!(vars, subterm)
    end
    
    println("   Final vars: $vars")
end

function _collect_vars_recursive_fixed!(vars::Set{Symbol}, term::Term)
    println("🔧 _collect_vars_recursive_fixed! called with Term: $(term.sym)")
    push!(vars, term.sym)
end

function _collect_vars_recursive_fixed!(vars::Set{Symbol}, term::ContinuousTerm)
    println("🔧 _collect_vars_recursive_fixed! called with ContinuousTerm: $(term.sym)")
    push!(vars, term.sym)
end

function _collect_vars_recursive_fixed!(vars::Set{Symbol}, term::CategoricalTerm)
    println("🔧 _collect_vars_recursive_fixed! called with CategoricalTerm: $(term.sym)")
    push!(vars, term.sym)
end

# Fallback for other term types
function _collect_vars_recursive_fixed!(vars::Set{Symbol}, term::AbstractTerm)
    println("🔧 _collect_vars_recursive_fixed! fallback for $(typeof(term))")
    try
        if hasfield(typeof(term), :sym)
            sym = getfield(term, :sym)
            println("   Found sym: $sym")
            push!(vars, sym)
        else
            println("   No sym field found")
        end
    catch e
        println("   Error extracting sym: $e")
    end
end

# Test the fixed version
function test_fixed_collection(interaction_term)
    if interaction_term !== nothing
        println("\\n🧪 TESTING FIXED COLLECTION:")
        vars = Set{Symbol}()
        _collect_vars_recursive_fixed!(vars, interaction_term)
        println("Fixed collection result: $(collect(vars))")
        return collect(vars)
    end
    return Symbol[]
end

fixed_vars = test_fixed_collection(interaction_term)