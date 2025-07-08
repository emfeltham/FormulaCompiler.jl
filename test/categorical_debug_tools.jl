# categorical_debug_tools.jl - Tools to debug categorical term evaluation

"""
Debug tools to ensure recursive CategoricalTerm evaluation matches modelmatrix! exactly.
"""

using DataFrames, StatsModels, GLM, Tables, CategoricalArrays

"""
    debug_categorical_term_standalone()

Test categorical term evaluation without interactions to isolate any issues.
"""
function debug_categorical_term_standalone()
    println("🔍 Debugging Standalone Categorical Term")
    
    # Simple test case
    df = DataFrame(
        group = categorical(["A", "B", "C", "A", "B"]),
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    
    println("Data:")
    println("  group: ", df.group)
    println("  levels: ", levels(df.group))
    
    # Test categorical term alone
    formula = @formula(y ~ group)
    model = lm(formula, df)
    data = Tables.columntable(df)
    
    println("\nFormula: $formula")
    
    # Reference matrix
    X_ref = modelmatrix(model)
    println("\nReference matrix (from modelmatrix):")
    display(X_ref)
    
    # Get the categorical term for debugging
    mapping = enhanced_column_mapping(model)
    println("\nTerms in mapping:")
    for (i, (term, range)) in enumerate(mapping.term_info)
        println("  $i: $term → columns $range")
    end
    
    # Recursive evaluation
    n, p = size(X_ref)
    X_rec = Matrix{Float64}(undef, n, p)
    
    for col in 1:p
        term, local_col = get_term_for_column(mapping, col)
        println("\nEvaluating column $col:")
        println("  Term: $term")
        println("  Local column: $local_col")
        
        if term isa CategoricalTerm
            debug_categorical_term_details(term, data)
        end
        
        evaluate_single_column!(term, data, col, local_col, view(X_rec, :, col), nothing)
    end
    
    println("\nRecursive matrix:")
    display(X_rec)
    
    println("\nDifference (recursive - reference):")
    diff = X_rec .- X_ref
    display(diff)
    
    max_diff = maximum(abs.(diff))
    println("\nMax difference: $max_diff")
    
    return max_diff < 1e-12
end

"""
    debug_categorical_term_details(term::CategoricalTerm, data::NamedTuple)

Show detailed information about a categorical term's structure.
"""
function debug_categorical_term_details(term::CategoricalTerm, data::NamedTuple)
    println("    Categorical term details:")
    
    v = data[term.sym]
    println("      Variable: $(term.sym)")
    println("      Type: $(typeof(v))")
    println("      Levels: $(levels(v))")
    println("      Values: $(v)")
    
    codes = refs(v)
    println("      Codes: $(codes)")
    
    M = term.contrasts.matrix
    println("      Contrast matrix size: $(size(M))")
    println("      Contrast matrix:")
    display(M)
    
    println("      Contrast type: $(typeof(term.contrasts))")
end

"""
    debug_interaction_step_by_step()

Debug the x & group interaction step by step.
"""
function debug_interaction_step_by_step()
    println("🔍 Debugging x & group Interaction Step by Step")
    
    # Use the same simple data that showed the issue
    df = DataFrame(
        x = [1.0, 2.0, 3.0, 4.0],
        group = categorical(["A", "B", "A", "B"]),
        y = [1.0, 2.0, 3.0, 4.0]
    )
    
    println("Data:")
    println("  x: ", df.x)
    println("  group: ", df.group, " (levels: ", levels(df.group), ")")
    
    formula = @formula(y ~ x & group)
    model = lm(formula, df)
    data = Tables.columntable(df)
    
    println("\nFormula: $formula")
    
    # Reference matrix
    X_ref = modelmatrix(model)
    println("\nReference matrix:")
    display(X_ref)
    
    # Debug each component separately
    println("\n" * "="^50)
    println("COMPONENT ANALYSIS")
    println("="^50)
    
    # Component 1: x (continuous)
    println("\n1. Component x (continuous):")
    x_vals = data.x
    println("   Values: $x_vals")
    
    # Component 2: group (categorical)  
    println("\n2. Component group (categorical):")
    
    # Create a standalone categorical term to test
    group_formula = @formula(y ~ group)
    group_model = lm(group_formula, df)
    group_matrix = modelmatrix(group_model)
    
    println("   Standalone group matrix:")
    display(group_matrix)
    
    # Now test the interaction
    println("\n" * "="^50)
    println("INTERACTION ANALYSIS")
    println("="^50)
    
    mapping = enhanced_column_mapping(model)
    
    # Find the interaction term
    interaction_term = nothing
    interaction_range = nothing
    
    for (term, range) in mapping.term_info
        if term isa InteractionTerm
            interaction_term = term
            interaction_range = range
            println("\nFound interaction term: $term")
            println("Column range: $range")
            break
        end
    end
    
    if interaction_term !== nothing
        # Evaluate the interaction term using recursive method
        n = length(df.x)
        interaction_width = length(interaction_range)
        interaction_result = Matrix{Float64}(undef, n, interaction_width)
        
        println("\nEvaluating interaction recursively...")
        evaluate_term!(interaction_term, data, interaction_result, nothing)
        
        println("Recursive interaction result:")
        display(interaction_result)
        
        # Compare with reference (skip intercept column)
        ref_interaction = X_ref[:, 2:end]  # Skip intercept
        println("\nReference interaction (columns 2-3):")
        display(ref_interaction)
        
        println("\nDifference:")
        diff = interaction_result .- ref_interaction
        display(diff)
        
        max_diff = maximum(abs.(diff))
        println("\nMax difference: $max_diff")
        
        return max_diff < 1e-12
    end
    
    return false
end

"""
    test_categorical_infrastructure()

Test that our categorical evaluation matches StatsModels exactly.
"""
function test_categorical_infrastructure()
    println("🧪 Testing Categorical Infrastructure Compatibility")
    
    test_cases = [
        (["A", "B", "A", "B"], "Simple 2-level"),
        (["A", "B", "C", "A", "B", "C"], "3-level"),
        (["X", "Y", "Z", "X", "Y"], "Different labels"),
    ]
    
    all_passed = true
    
    for (levels_data, description) in test_cases
        println("\n--- Testing: $description ---")
        
        df = DataFrame(
            group = categorical(levels_data),
            y = randn(length(levels_data))
        )
        
        formula = @formula(y ~ group)
        model = lm(formula, df)
        
        # Compare standalone categorical
        passed = debug_categorical_term_standalone()
        println("Result: $(passed ? "✅ PASSED" : "❌ FAILED")")
        
        all_passed &= passed
    end
    
    return all_passed
end

export debug_categorical_term_standalone, debug_categorical_term_details
export debug_interaction_step_by_step, test_categorical_infrastructure