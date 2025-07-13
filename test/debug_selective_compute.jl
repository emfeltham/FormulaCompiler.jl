# Fixed version that properly handles categorical variables

"""
Let's manually check what the 3-way interaction SHOULD be (FIXED)
"""
function manual_interaction_check_fixed(model, df)
    println("🧮 MANUAL 3-WAY INTERACTION CHECK (FIXED)")
    
    ws = AMEWorkspace(model, df)
    
    # Get the perturbed data
    orig_values = ws.base_data[:x]
    h = std(orig_values) * 1e-6
    pert_data = create_perturbed_data(ws.base_data, :x, orig_values .+ h)
    
    # Extract the component values
    x_vals = pert_data[:x]
    d_vals = pert_data[:d]  
    z_vals = pert_data[:z]
    
    println("Sample component values (first 3 rows):")
    println("  x: $(x_vals[1:3])")
    println("  d: $(d_vals[1:3]) (type: $(typeof(d_vals)))")
    println("  z: $(z_vals[1:3])")
    
    # Convert categorical to numeric properly
    if d_vals isa CategoricalArray
        # For Bool categorical: true -> 1.0, false -> 0.0
        d_numeric = [d == true ? 1.0 : 0.0 for d in d_vals]
        println("  d_numeric: $(d_numeric[1:3])")
    else
        d_numeric = Float64.(d_vals)
        println("  d_numeric: $(d_numeric[1:3])")
    end
    
    # Manual computation of 3-way interaction
    manual_interaction = x_vals .* d_numeric .* z_vals
    
    println("Manual 3-way interaction [1:3]: $(manual_interaction[1:3])")
    
    # Compare with ground truth
    X_ground_truth = similar(ws.base_matrix)
    ipm = InplaceModeler(model, nrow(df))
    modelmatrix!(ipm, pert_data, X_ground_truth)
    
    println("Ground truth column 8 [1:3]: $(X_ground_truth[1:3, 8])")
    
    # Check if they match
    diff = maximum(abs.(manual_interaction - X_ground_truth[:, 8]))
    println("Difference between manual and ground truth: $diff")
    
    if diff < 1e-10
        println("✅ Manual computation matches ground truth!")
    else
        println("❌ Manual computation differs from ground truth")
        println("   This suggests the 3-way interaction isn't just x*d*z")
        # Show more details
        println("   First few manual values: $(manual_interaction[1:5])")
        println("   First few ground truth: $(X_ground_truth[1:5, 8])")
    end
    
    return manual_interaction, X_ground_truth[:, 8]
end

"""
Enhanced selective update tracing
"""
function trace_selective_update_enhanced(model, df)
    println("\\n🎯 ENHANCED SELECTIVE UPDATE TRACING")
    
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, nrow(df))
    
    # Get the perturbed data
    orig_values = ws.base_data[:x]
    h = std(orig_values) * 1e-6
    pert_data = create_perturbed_data(ws.base_data, :x, orig_values .+ h)
    
    # Show what data is being passed to modelmatrix_with_base!
    println("Data being passed to selective update:")
    println("  x [1:3]: $(pert_data[:x][1:3])")
    println("  d [1:3]: $(pert_data[:d][1:3])")  
    println("  z [1:3]: $(pert_data[:z][1:3])")
    
    # Apply selective update
    X_selective = copy(ws.base_matrix)
    println("\\nBase matrix column 8 [1:3]: $(ws.base_matrix[1:3, 8])")
    
    modelmatrix_with_base!(ipm, pert_data, X_selective, ws.base_matrix, [:x], ws.mapping)
    
    println("Selective result column 8 [1:3]: $(X_selective[1:3, 8])")
    
    # Detailed analysis
    println("\\n🔍 DETAILED ANALYSIS:")
    
    # Check if column 8 equals any of the input components
    x_diff = maximum(abs.(X_selective[:, 8] - pert_data[:x]))
    println("Max diff from x values: $x_diff")
    
    # Check if it's just copying the base matrix
    base_diff = maximum(abs.(X_selective[:, 8] - ws.base_matrix[:, 8]))
    println("Max diff from base matrix: $base_diff")
    
    if x_diff < 1e-12
        println("❌ BUG FOUND: Column 8 is EXACTLY the x values!")
        println("   The selective update is copying x instead of computing x*d*z")
    elseif base_diff < 1e-12
        println("❌ BUG FOUND: Column 8 wasn't updated at all!")
        println("   The selective update failed to change the 3-way interaction")
    else
        println("   Column 8 was changed, but not correctly computed")
    end
    
    return X_selective[:, 8]
end

# Fixed test run
println("🚀 RUNNING FIXED COMPREHENSIVE DEBUG")

manual_result, ground_truth = manual_interaction_check_fixed(model, df)
selective_result = trace_selective_update_enhanced(model, df)

# Final comparison
println("\\n📊 FINAL COMPARISON (first 3 values):")
println("Manual x*d*z:     $(manual_result[1:3])")
println("Ground truth:     $(ground_truth[1:3])")  
println("Selective result: $(selective_result[1:3])")

# Identify the exact bug
println("\\n🎯 BUG IDENTIFICATION:")
manual_vs_truth = maximum(abs.(manual_result - ground_truth))
selective_vs_truth = maximum(abs.(selective_result - ground_truth))
selective_vs_manual = maximum(abs.(selective_result - manual_result))

println("Manual vs ground truth: $manual_vs_truth")
println("Selective vs ground truth: $selective_vs_truth") 
println("Selective vs manual: $selective_vs_manual")

if manual_vs_truth < 1e-10
    println("✅ Manual computation is correct")
    if selective_vs_truth > 1e-6
        println("❌ CONFIRMED: Selective update has a computation bug")
    end
else
    println("❌ Manual computation logic needs refinement")
end