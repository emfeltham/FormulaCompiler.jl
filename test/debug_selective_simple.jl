# Simplified debugging that avoids the _cols! type issues

"""
    trace_selective_update_bug(model, df, variable::Symbol)

Simplified debugging that focuses on the key issue without calling _cols! directly.
"""
function trace_selective_update_bug(model, df, variable::Symbol)
    println("🔍 TRACING SELECTIVE UPDATE BUG FOR :$variable")
    
    # Setup
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, nrow(df))
    
    # Create perturbation
    orig_values = ws.base_data[variable]
    h = std(orig_values) * 1e-6
    pert_vector = orig_values .+ h
    pert_data = create_perturbed_data(ws.base_data, variable, pert_vector)
    
    println("Step size h = $h")
    println("Variable $variable should affect columns: $(ws.variable_plans[variable])")
    
    # --- GROUND TRUTH: Full matrix construction ---
    println("\\n📊 GROUND TRUTH (full matrix construction):")
    X_ground_truth = similar(ws.base_matrix)
    modelmatrix!(ipm, pert_data, X_ground_truth)
    
    # --- OUR APPROACH: Selective update ---
    println("\\n🎯 OUR APPROACH (selective update):")
    X_our_approach = copy(ws.base_matrix)
    
    # Let's trace what modelmatrix_with_base! is supposed to do step by step
    println("\\n🔍 Step-by-step analysis of modelmatrix_with_base!:")
    
    # Step 1: Identify affected columns
    changed_vars = [variable]
    total_cols = size(X_our_approach, 2)
    changed_cols = Set{Int}()
    
    for var in changed_vars
        var_cols = get_variable_columns_flat(ws.mapping, var)
        union!(changed_cols, var_cols)
        println("  Variable :$var affects columns: $var_cols")
    end
    
    changed_cols = sort(collect(changed_cols))
    unchanged_cols = get_unchanged_columns(ws.mapping, changed_vars, total_cols)
    
    println("  Total changed columns: $changed_cols")
    println("  Unchanged columns: $unchanged_cols")
    
    # Step 2: Call the actual function
    println("\\n🎯 Calling modelmatrix_with_base!...")
    try
        modelmatrix_with_base!(ipm, pert_data, X_our_approach, ws.base_matrix, [variable], ws.mapping)
        println("  ✅ modelmatrix_with_base! completed successfully")
    catch e
        println("  ❌ modelmatrix_with_base! failed: $e")
        return
    end
    
    # Step 3: Compare results column by column
    println("\\n📊 COLUMN-BY-COLUMN COMPARISON:")
    
    for col in 1:size(X_ground_truth, 2)
        col_diff = maximum(abs.(X_our_approach[:, col] - X_ground_truth[:, col]))
        
        # Get term info
        term_name = "unknown"
        try
            for (term, range) in ws.mapping.term_info
                if col in range
                    term_name = string(term)
                    break
                end
            end
        catch
        end
        
        if col_diff > 1e-10
            println("  ❌ Column $col ($term_name): DIFFERS by $col_diff")
            
            # Show sample values
            println("    Ground truth [1:3]: $(X_ground_truth[1:3, col])")
            println("    Our result [1:3]:   $(X_our_approach[1:3, col])")
            println("    Base matrix [1:3]:  $(ws.base_matrix[1:3, col])")
            
            # Check if this column was supposed to be updated
            if col in changed_cols
                println("    🔍 This column SHOULD have been updated")
            else
                println("    ⚠️  This column should NOT have been changed")
            end
        else
            status = col in changed_cols ? "updated" : "unchanged"
            println("  ✅ Column $col ($term_name): OK ($status)")
        end
    end
    
    return X_ground_truth, X_our_approach
end

"""
    compare_data_structures(orig_data, pert_data, variable::Symbol)

Compare the original and perturbed data to ensure perturbation is correct.
"""
function compare_data_structures(orig_data, pert_data, variable::Symbol)
    println("\\n🔍 DATA STRUCTURE COMPARISON:")
    
    for (key, orig_vals) in pairs(orig_data)
        pert_vals = pert_data[key]
        
        if key == variable
            diff = maximum(abs.(pert_vals - orig_vals))
            println("  :$key (PERTURBED): max change = $diff")
            println("    Original [1:3]: $(orig_vals[1:3])")
            println("    Perturbed [1:3]: $(pert_vals[1:3])")
        else
            if orig_vals === pert_vals
                println("  :$key: SHARED (same object)")
            elseif all(orig_vals .== pert_vals)
                println("  :$key: IDENTICAL (different objects)")
            else
                println("  :$key: ❌ DIFFERENT! (unexpected)")
            end
        end
    end
end

"""
    investigate_column_8_specifically(model, df)

Focus specifically on column 8 (the 3-way interaction) to understand the bug.
"""
function investigate_column_8_specifically(model, df)
    println("\\n🎯 INVESTIGATING COLUMN 8 (3-way interaction) SPECIFICALLY:")
    
    ws = AMEWorkspace(model, df)
    
    # Find the term for column 8
    term_8 = nothing
    for (term, range) in ws.mapping.term_info
        if 8 in range
            term_8 = term
            break
        end
    end
    
    println("Column 8 term: $term_8")
    
    if term_8 isa InteractionTerm
        println("Interaction components: $(term_8.terms)")
        
        # Check what variables this interaction depends on
        component_vars = Symbol[]
        for component in term_8.terms
            if component isa Term
                push!(component_vars, component.sym)
            end
        end
        println("Component variables: $component_vars")
        
        # Check if our variable mapping is correct
        vars_affecting_col8 = Symbol[]
        for (var, cols) in ws.variable_plans
            if 8 in cols
                push!(vars_affecting_col8, var)
            end
        end
        println("Variables that should affect column 8: $vars_affecting_col8")
        
        # These should match!
        missing_vars = setdiff(component_vars, vars_affecting_col8)
        extra_vars = setdiff(vars_affecting_col8, component_vars)
        
        if !isempty(missing_vars)
            println("❌ MISSING from variable plans: $missing_vars")
        end
        if !isempty(extra_vars)
            println("❌ EXTRA in variable plans: $extra_vars")
        end
        if isempty(missing_vars) && isempty(extra_vars)
            println("✅ Variable mapping looks correct")
        end
    end
end

function run_debugging()
    println("🚀 RUNNING SIMPLIFIED DEBUGGING")
    
    # First check the data structures
    ws = AMEWorkspace(model, df)
    orig_data = ws.base_data
    h = std(orig_data[:x]) * 1e-6
    pert_data = create_perturbed_data(orig_data, :x, orig_data[:x] .+ h)
    
    compare_data_structures(orig_data, pert_data, :x)
    
    # Check column 8 mapping
    investigate_column_8_specifically(model, df)
    
    # Run the main debugging
    X_truth, X_ours = trace_selective_update_bug(model, df, :x)
    
    return X_truth, X_ours
end

#####

run_debugging()