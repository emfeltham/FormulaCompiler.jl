# Deep debugging of InteractionTerm evaluation in _cols!

"""
    trace_interaction_term_evaluation(model, df)

Trace exactly what happens when _cols! evaluates the 3-way interaction.
"""
function trace_interaction_term_evaluation(model, df)
    println("🔬 DEEP TRACE: InteractionTerm Evaluation")
    
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, nrow(df))
    
    # Get the 3-way interaction term
    interaction_term = nothing
    for (term, range) in ws.mapping.term_info
        if term isa InteractionTerm && length(term.terms) == 3
            interaction_term = term
            break
        end
    end
    
    if interaction_term === nothing
        println("❌ No 3-way interaction found")
        return
    end
    
    println("Found interaction term: $interaction_term")
    println("Components: $(interaction_term.terms)")
    
    # Set up test data
    orig_values = ws.base_data[:x]
    h = std(orig_values) * 1e-6
    pert_data = create_perturbed_data(ws.base_data, :x, orig_values .+ h)
    
    println("\\n📊 Input data for interaction evaluation:")
    println("  x [1:3]: $(pert_data[:x][1:3])")
    println("  d [1:3]: $(pert_data[:d][1:3])")
    println("  z [1:3]: $(pert_data[:z][1:3])")
    
    # Test 1: Direct evaluation of the interaction term
    println("\\n🧪 TEST 1: Direct interaction evaluation")
    test_direct_interaction_evaluation(interaction_term, pert_data, ipm)
    
    # Test 2: Component-by-component evaluation
    println("\\n🧪 TEST 2: Component-by-component evaluation")
    test_component_evaluation(interaction_term, pert_data, ipm)
    
    # Test 3: Trace the selective update path
    println("\\n🧪 TEST 3: Selective update path trace")
    test_selective_update_path(interaction_term, pert_data, ws, ipm)
    
    return interaction_term
end

"""
Test direct evaluation of the interaction term
"""
function test_direct_interaction_evaluation(interaction_term, pert_data, ipm)
    println("  📋 Creating temp matrix for interaction evaluation...")
    
    # Create temporary matrix (3 rows for debugging, 1 column for interaction)
    temp_matrix = Matrix{Float64}(undef, 3, 1)
    
    # Initialize counters like _cols_selective! does
    fn_i = Ref(1)
    int_i = Ref(1)
    
    println("  🧮 Calling _cols! directly on interaction term...")
    
    try
        # This is the critical call - let's see what it produces
        _cols!(interaction_term, pert_data, temp_matrix, 1, ipm, fn_i, int_i)
        
        println("  ✅ _cols! completed successfully")
        println("  📊 Result [1:3]: $(temp_matrix[1:3, 1])")
        
        # Compare with manual computation
        x_vals = pert_data[:x][1:3]
        d_vals = [d == true ? 1.0 : 0.0 for d in pert_data[:d][1:3]]
        z_vals = pert_data[:z][1:3]
        manual_result = x_vals .* d_vals .* z_vals
        
        println("  🧮 Manual x*d*z [1:3]: $manual_result")
        
        diff = maximum(abs.(temp_matrix[1:3, 1] - manual_result))
        println("  📏 Difference: $diff")
        
        if diff < 1e-12
            println("  ✅ _cols! result matches manual computation!")
        else
            println("  ❌ _cols! result differs from manual computation")
            println("    This suggests _cols! for InteractionTerm has a bug")
        end
        
    catch e
        println("  ❌ _cols! failed with error: $e")
        println("  📚 Stacktrace:")
        for (i, frame) in enumerate(stacktrace(catch_backtrace()))
            println("    $i: $frame")
            if i > 5 break end  # Limit output
        end
    end
end

"""
Test evaluation of individual components
"""
function test_component_evaluation(interaction_term, pert_data, ipm)
    println("  🔍 Testing individual component evaluation...")
    
    components = interaction_term.terms
    println("  📝 Components: $components")
    
    # Test each component individually
    for (i, component) in enumerate(components)
        println("  \\n  Component $i: $component ($(typeof(component)))")
        
        try
            # Create temp matrix for this component
            comp_width = width(component)
            println("    Width: $comp_width")
            
            if comp_width > 0
                comp_matrix = Matrix{Float64}(undef, 3, comp_width)
                fn_i, int_i = Ref(1), Ref(1)
                
                # Evaluate this component
                _cols!(component, pert_data, comp_matrix, 1, ipm, fn_i, int_i)
                
                println("    Result [1:3]: $(comp_matrix[1:3, :])")
                
                # Compare with expected data
                if component isa Term || component isa ContinuousTerm
                    if hasfield(typeof(component), :sym)
                        sym = component.sym
                        if haskey(pert_data, sym)
                            expected = pert_data[sym][1:3]
                            actual = comp_matrix[1:3, 1]
                            diff = maximum(abs.(actual - expected))
                            println("    Expected from data: $expected")
                            println("    Difference: $diff")
                        end
                    end
                elseif component isa CategoricalTerm
                    println("    (Categorical component - converted to dummy coding)")
                end
            else
                println("    Zero width component")
            end
            
        catch e
            println("    ❌ Component evaluation failed: $e")
        end
    end
end

"""
Test the full selective update path to see where it goes wrong
"""
function test_selective_update_path(interaction_term, pert_data, ws, ipm)
    println("  🛤️  Tracing selective update path...")
    
    # Simulate what _cols_selective! does
    affected_cols = ws.variable_plans[:x]  # Should include column 8
    println("  📍 Affected columns: $affected_cols")
    
    # Find where the interaction term should be placed
    interaction_range = nothing
    for (term, range) in ws.mapping.term_info
        if term === interaction_term
            interaction_range = range
            break
        end
    end
    
    if interaction_range === nothing
        println("  ❌ Could not find interaction term in mapping")
        return
    end
    
    println("  📍 Interaction range: $interaction_range")
    
    # Check if column 8 is in the affected columns and the interaction range
    col_8_affected = 8 in affected_cols
    col_8_in_range = 8 in interaction_range
    
    println("  🎯 Column 8 affected by :x? $col_8_affected")
    println("  🎯 Column 8 in interaction range? $col_8_in_range")
    
    if !col_8_affected
        println("  ❌ BUG: Column 8 not marked as affected by :x")
    end
    
    if !col_8_in_range
        println("  ❌ BUG: Column 8 not in interaction term range")
    end
    
    # Simulate the selective copying logic
    j = first(interaction_range)
    w = width(interaction_term)
    term_cols = collect(j:(j + w - 1))
    cols_to_update = intersect(term_cols, affected_cols)
    
    println("  📊 Term columns: $term_cols")
    println("  📊 Columns to update: $cols_to_update")
    
    if 8 ∉ cols_to_update
        println("  ❌ BUG: Column 8 not in columns to update!")
    else
        println("  ✅ Column 8 correctly identified for update")
    end
    
    # Now test the actual copying logic
    println("  \\n  🔄 Testing copy logic...")
    
    # Create temp matrix like _cols_selective! does
    temp_matrix = Matrix{Float64}(undef, size(ws.base_matrix, 1), w)
    
    fn_i, int_i = Ref(1), Ref(1)
    _cols!(interaction_term, pert_data, temp_matrix, 1, ipm, fn_i, int_i)
    
    println("  📊 Temp matrix column 1 [1:3]: $(temp_matrix[1:3, 1])")
    
    # Test the copying step
    target_matrix = copy(ws.base_matrix)
    
    for col in cols_to_update
        local_col = col - j + 1
        println("  📋 Copying column $col (local $local_col)")
        
        if local_col >= 1 && local_col <= w
            target_matrix[:, col] = temp_matrix[:, local_col]
            println("    ✅ Copied temp_matrix[:, $local_col] -> target[:, $col]")
            println("    📊 Result [1:3]: $(target_matrix[1:3, col])")
        else
            println("    ❌ Invalid local column index: $local_col")
        end
    end
    
    # Compare result with ground truth
    X_ground_truth = similar(ws.base_matrix)
    modelmatrix!(ipm, pert_data, X_ground_truth)
    
    col_8_diff = maximum(abs.(target_matrix[:, 8] - X_ground_truth[:, 8]))
    println("  📏 Final column 8 difference: $col_8_diff")
    
    if col_8_diff < 1e-12
        println("  ✅ MYSTERIOUS: The logic works when traced step by step!")
        println("      This suggests a bug in the actual _cols_selective! implementation")
    else
        println("  ❌ Still wrong - bug is in the core logic")
    end
end

# Run the comprehensive trace
println("🚀 STARTING COMPREHENSIVE INTERACTION TRACE")
interaction_term = trace_interaction_term_evaluation(model, df)