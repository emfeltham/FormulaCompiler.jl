# =============================================================================
# EXAMPLE USAGE AND TESTS
# =============================================================================

# """
#     demo_usage()

# Demonstrate the basic usage of EfficientModelMatrices.
# """
# function demo_usage()
#     println("=== EfficientModelMatrices.jl Demo ===")
    
#     # Create sample data
#     n = 1000
#     df = DataFrame(
#         x1 = randn(n),
#         x2 = randn(n),
#         x3 = randn(n),
#         cat = repeat(["A", "B", "C"], n÷3 + 1)[1:n],
#         y = randn(n)
#     )
    
#     # Create a complex formula
#     formula = @formula(y ~ x1 + x2 + cat + x1&x2 + x1&cat)
    
#     println("=== Method 1: Build from formula ===")
#     # Build cached model matrix from formula
#     cached_mm1 = cached_modelmatrix(formula, df)
#     println("Matrix size: ", size(cached_mm1))
    
#     println("\n=== Method 2: Reuse existing model matrix (RECOMMENDED) ===")
#     # Fit a model first
#     using GLM
#     model = glm(formula, df, Normal())
    
#     # Create cached matrix from existing model matrix (much faster!)
#     cached_mm2 = cached_modelmatrix(model)
#     println("Matrix size: ", size(cached_mm2))
#     println("Matrices are identical: ", cached_mm1.matrix ≈ cached_mm2.matrix)
    
#     # Show dependency information
#     println("\nDependency analysis:")
#     for var in [:x1, :x2, :cat]
#         deps = get_dependency_info(cached_mm2, var)
#         println("  $var affects columns: $deps")
#     end
    
#     # Test selective update
#     println("\nTesting selective update...")
#     df_new = copy(df)
#     df_new.x1 = randn(n)  # Only change x1
    
#     # Time the update
#     println("Timing comparison:")
#     t1 = @elapsed update!(cached_mm2, df_new; changed_vars=[:x1])
#     println("  Update time (selective): $(round(t1*1000, digits=2)) ms")
    
#     # Compare with full recomputation
#     t2 = @elapsed cached_mm_full = cached_modelmatrix(formula, df_new)
#     println("  Full rebuild time: $(round(t2*1000, digits=2)) ms")
    
#     # Compare with model refit
#     t3 = @elapsed model_new = glm(formula, df_new, Normal())
#     println("  Model refit time: $(round(t3*1000, digits=2)) ms")
    
#     println("  Speedup vs rebuild: $(round(t2/t1, digits=1))x")
#     println("  Speedup vs refit: $(round(t3/t1, digits=1))x")
    
#     return cached_mm2
# end

# function demo_margins_integration()
#     println("\n=== Integration with Margins-style computation ===")
    
#     using GLM
    
#     # Create sample data
#     n = 1000
#     df = DataFrame(
#         x1 = randn(n),
#         x2 = randn(n),
#         cat = repeat(["A", "B", "C"], n÷3 + 1)[1:n],
#         y = randn(n)
#     )
    
#     # Fit model
#     model = glm(@formula(y ~ x1 + x2 + cat + x1&x2), df, Normal())
    
#     # Create cached matrix from existing model matrix
#     cached_mm = cached_modelmatrix(model)
    
#     # Simulate marginal effects computation
#     println("Computing marginal effects for x1...")
    
#     # Method 1: Using cached matrix (efficient)
#     df_pert = copy(df)
#     h = 0.001
#     df_pert.x1 .+= h
    
#     t1 = @elapsed begin
#         update!(cached_mm, df_pert; changed_vars=[:x1])
#         # In real margins computation, you'd use this matrix for predictions
#         X_pert = cached_mm.matrix
#         # effect = compute_marginal_effect(X_base, X_pert, coef(model), h)
#     end
    
#     # Method 2: Full matrix rebuild (inefficient)
#     t2 = @elapsed begin
#         X_pert_full = modelmatrix(model, df_pert)
#         # effect = compute_marginal_effect(modelmatrix(model), X_pert_full, coef(model), h)
#     end
    
#     println("  Cached update time: $(round(t1*1000, digits=2)) ms")
#     println("  Full rebuild time: $(round(t2*1000, digits=2)) ms")
#     println("  Speedup: $(round(t2/t1, digits=1))x")
    
#     # Verify results are identical
#     X_pert_full = modelmatrix(formula(model), df_pert)
#     println("  Results identical: ", cached_mm.matrix ≈ X_pert_full)
    
#     return cached_mm
# end
