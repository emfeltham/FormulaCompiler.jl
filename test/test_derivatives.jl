# test_derivatives.jl
# julia --project="." test/test_derivatives.jl > test/test_derivatives.txt 2>&1
# Correctness tests for derivatives

using Test
using Random
using FormulaCompiler
using DataFrames, Tables, GLM, MixedModels, CategoricalArrays
using LinearAlgebra: dot

# ====== PHASE 1: Manual Baseline Infrastructure ======

"""
    compute_manual_jacobian(compiled, data, row, vars; h=1e-8)

Compute Jacobian using basic finite differences without any 
FormulaCompiler derivative infrastructure. This is our ground truth.
"""
function compute_manual_jacobian(compiled, data, row, vars; h=1e-8)
    n_terms = length(compiled)
    n_vars = length(vars)
    J = Matrix{Float64}(undef, n_terms, n_vars)
    
    # Base evaluation
    y_base = Vector{Float64}(undef, n_terms)
    compiled(y_base, data, row)
    
    for (j, var) in enumerate(vars)
        # Get current value
        val = data[var][row]
        
        # Only perturb if numeric (continuous variables)
        if val isa Number
            # Create new arrays to avoid mutation
            vals_plus = copy(data[var])
            vals_minus = copy(data[var])
            vals_plus[row] = val + h
            vals_minus[row] = val - h
            
            # Create new named tuples with modified arrays
            data_plus = merge(data, NamedTuple{(var,)}((vals_plus,)))
            data_minus = merge(data, NamedTuple{(var,)}((vals_minus,)))
            
            # Evaluate at perturbed points
            y_plus = Vector{Float64}(undef, n_terms)
            y_minus = Vector{Float64}(undef, n_terms)
            compiled(y_plus, data_plus, row)
            compiled(y_minus, data_minus, row)
            
            # Central difference
            J[:, j] = (y_plus .- y_minus) ./ (2h)
        else
            # Non-numeric variables have zero derivative
            J[:, j] .= 0.0
        end
    end
    
    return J
end

@testset "Derivative correctness" begin
    # Fix random seed for reproducibility
    Random.seed!(06515)
    
    n = 300

    @testset "ForwardDiff and FD fallback" begin
        # Data and model
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            z = abs.(randn(n)) .+ 0.1,
            group3 = categorical(rand(["A", "B", "C"], n)),
        )
        data = Tables.columntable(df)
        model = lm(@formula(y ~ 1 + x + z + x & group3), df)
        compiled = compile_formula(model, data)

        # Choose continuous vars
        vars = [:x, :z]

        # Build derivative evaluator (ForwardDiff)
        de = build_derivative_evaluator(compiled, data; vars=vars)
        J = Matrix{Float64}(undef, length(compiled), length(vars))

        # Warmup to trigger Dual-typed caches
        derivative_modelrow!(J, de, 1)
        derivative_modelrow!(J, de, 2)

        # FD fallback comparison (standalone FD for correctness baseline)
        # Use same row (3) for both AD and FD
        derivative_modelrow!(J, de, 3)
        J_fd = similar(J)
        derivative_modelrow_fd!(J_fd, compiled, data, 3; vars=vars)
        
        # PHASE 2: Manual baseline comparison
        J_manual = compute_manual_jacobian(compiled, data, 3, vars)
        
        # Compare all methods against manual baseline
        # println("\n=== Manual Baseline Comparison (Row 3) ===")
        # println("Group value at row 3: ", data.group3[3])
        # println("Max |AD - Manual|: ", maximum(abs.(J .- J_manual)))
        # println("Max |FD - Manual|: ", maximum(abs.(J_fd .- J_manual)))
        # println("J (AD) first column: ", J[:, 1])
        # println("J_manual first column: ", J_manual[:, 1])
        
        # Test against manual baseline (ground truth)
        @test isapprox(J, J_manual; rtol=1e-5, atol=1e-8) 
        @test isapprox(J_fd, J_manual; rtol=1e-5, atol=1e-8)
        
        # Original test (AD vs FD) - now less critical since we test against ground truth
        @test isapprox(J, J_fd; rtol=1e-5, atol=1e-10)

        # Discrete contrast: swap group level at row
        Δ = Vector{Float64}(undef, length(compiled))
        row = 5
        contrast_modelrow!(Δ, compiled, data, row; var=:group3, from="A", to="B")
        # Validate against manual override with OverrideVector
        # Pass raw override values; create_override_data will wrap appropriately
        data_from = FormulaCompiler.create_override_data(data, Dict{Symbol,Any}(:group3 => "A"))
        data_to   = FormulaCompiler.create_override_data(data, Dict{Symbol,Any}(:group3 => "B"))
        y_from = modelrow(compiled, data_from, row)
        y_to = modelrow(compiled, data_to, row)
        @test isapprox(Δ, y_to .- y_from; rtol=0, atol=0)

        # Marginal effects: η = Xβ (test both AD and FD backends)
        β = coef(model)
        gη_ad = Vector{Float64}(undef, length(vars))
        gη_fd = Vector{Float64}(undef, length(vars))
        
        # Test AD backend
        marginal_effects_eta!(gη_ad, de, β, row; backend=:ad)
        # Check consistency with J' * β
        Jrow = Matrix{Float64}(undef, length(compiled), length(vars))
        derivative_modelrow!(Jrow, de, row)
        gη_ref = transpose(Jrow) * β
        @test isapprox(gη_ad, gη_ref; rtol=0, atol=0)
        
        # Test FD backend (allow reasonable tolerance for numerical differences)
        marginal_effects_eta!(gη_fd, de, β, row; backend=:fd)
        
        # DIAGNOSTIC: Check what's happening with marginal effects
        # println("\n=== Marginal Effects Diagnostic (Row $row) ===")
        # println("gη_ad (AD):  ", gη_ad)
        # println("gη_fd (FD):  ", gη_fd)
        # println("gη_ref:      ", gη_ref)
        # println("Difference (FD-AD): ", gη_fd .- gη_ad)
        
        # Additional diagnostic: Check the Jacobians directly
        J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
        J_fd = Matrix{Float64}(undef, length(compiled), length(vars))
        derivative_modelrow!(J_ad, de, row)
        derivative_modelrow_fd!(J_fd, de, row)
        # println("\n=== Jacobian Comparison (Row $row) ===")
        # println("Group value at row $row: ", data.group3[row])
        # println("Max |J_ad - J_fd|: ", maximum(abs.(J_ad .- J_fd)))
        # println("J_ad first column: ", J_ad[:, 1])
        # println("J_fd first column: ", J_fd[:, 1])
        # println("Difference first column: ", J_fd[:, 1] .- J_ad[:, 1])
        
        @test isapprox(gη_fd, gη_ref; rtol=1e-3, atol=1e-5)
        
        # Test μ marginal effects with both backends (allow reasonable tolerance)
        gμ_ad = Vector{Float64}(undef, length(vars))
        gμ_fd = Vector{Float64}(undef, length(vars))
        marginal_effects_mu!(gμ_ad, de, β, row; link=LogitLink(), backend=:ad)
        marginal_effects_mu!(gμ_fd, de, β, row; link=LogitLink(), backend=:fd)
        @test isapprox(gμ_ad, gμ_fd; rtol=1e-3, atol=1e-5)
    end

    @testset "Single-column FD and parameter gradients" begin
        # Data and model for testing
        n = 200
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            z = abs.(randn(n)) .+ 0.1,
            group3 = categorical(rand(["A", "B", "C"], n)),
        )
        data = Tables.columntable(df)
        model = lm(@formula(y ~ 1 + x + z + x & group3), df)
        compiled = compile_formula(model, data)
        vars = [:x, :z]
        β = coef(model)
        
        # Build evaluator
        de = build_derivative_evaluator(compiled, data; vars=vars)
        test_row = 5
        
        # Test single-column FD Jacobian
        @testset "fd_jacobian_column!" begin
            # Get full Jacobian for comparison
            J_full = Matrix{Float64}(undef, length(compiled), length(vars))
            derivative_modelrow!(J_full, de, test_row)
            
            # Test each variable column
            for (i, var) in enumerate(vars)
                Jk = Vector{Float64}(undef, length(compiled))
                fd_jacobian_column!(Jk, de, test_row, var)
                
                # Should match corresponding column from full AD Jacobian
                @test isapprox(Jk, J_full[:, i]; rtol=1e-6, atol=1e-8)
            end
            
            # Test against standalone FD Jacobian
            J_fd_standalone = Matrix{Float64}(undef, length(compiled), length(vars))
            derivative_modelrow_fd!(J_fd_standalone, compiled, data, test_row; vars=vars)
            
            for (i, var) in enumerate(vars)
                Jk = Vector{Float64}(undef, length(compiled))
                fd_jacobian_column!(Jk, de, test_row, var)
                @test isapprox(Jk, J_fd_standalone[:, i]; rtol=1e-6, atol=1e-8)
            end
        end
        
        # Test η parameter gradients
        @testset "me_eta_grad_beta!" begin
            for var in vars
                var_idx = findfirst(==(var), vars)
                
                # Get reference from AD Jacobian
                J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
                derivative_modelrow!(J_ad, de, test_row)
                ref_grad = J_ad[:, var_idx]  # For η, gradient is just the Jacobian column
                
                # Test our function
                gβ = Vector{Float64}(undef, length(compiled))
                me_eta_grad_beta!(gβ, de, β, test_row, var)
                
                @test isapprox(gβ, ref_grad; rtol=1e-6, atol=1e-8)
            end
        end
        
        # Test μ parameter gradients 
        @testset "me_mu_grad_beta!" begin
            # Test with LogitLink (has non-trivial second derivative)
            # Create data appropriate for logit: y in (0,1)
            df_logit = copy(df)
            df_logit.y = rand(size(df, 1))  # y ∈ (0,1) for LogitLink
            data_logit = Tables.columntable(df_logit)
            glm_model = glm(@formula(y ~ 1 + x + z + x & group3), df_logit, Normal(), LogitLink())
            compiled_glm = compile_formula(glm_model, data_logit)
            de_glm = build_derivative_evaluator(compiled_glm, data_logit; vars=vars)
            β_glm = coef(glm_model)
            
            for var in vars
                # Test our implementation
                gβ = Vector{Float64}(undef, length(compiled_glm))
                me_mu_grad_beta!(gβ, de_glm, β_glm, test_row, var; link=LogitLink())
                
                # Verify against manual computation using AD components
                # Get J_k
                Jk = Vector{Float64}(undef, length(compiled_glm))
                fd_jacobian_column!(Jk, de_glm, test_row, var)
                
                # Get X_row and η
                X_row = Vector{Float64}(undef, length(compiled_glm))
                compiled_glm(X_row, data, test_row)
                η = dot(β_glm, X_row)
                
                # Manual chain rule: gβ = g'(η) * J_k + (J_k' * β) * g''(η) * X_row
                g_prime = FormulaCompiler._dmu_deta(LogitLink(), η)
                g_double_prime = FormulaCompiler._d2mu_deta2(LogitLink(), η)
                Jk_dot_beta = dot(Jk, β_glm)
                
                ref_grad = g_prime .* Jk .+ Jk_dot_beta .* g_double_prime .* X_row
                
                @test isapprox(gβ, ref_grad; rtol=1e-6, atol=1e-8)
            end
        end
    end

    @testset "GLM(Logit) and MixedModels" begin
        # Data
        df = DataFrame(
            y = rand([0, 1], n),
            x = randn(n),
            z = abs.(randn(n)) .+ 0.1,
            group3 = categorical(rand(["A", "B", "C"], n)),
            g = categorical(rand(1:20, n)),
        )
        data = Tables.columntable(df)

        # row
        r = 3
        # GLM (Logit)
        glm_model = glm(@formula(y ~ 1 + x + z + x & group3), df, Binomial(), LogitLink())
        compiled_glm = compile_formula(glm_model, data)
        vars = [:x, :z]
        # Note: AD Jacobian allocation caps are environment-dependent (see DERIVATIVE_PLAN.md).
        de_glm = build_derivative_evaluator(compiled_glm, data; vars=vars)
        J = Matrix{Float64}(undef, length(compiled_glm), length(vars))
        derivative_modelrow!(J, de_glm, r)  # warm path
        # FD compare
        J_fd = similar(J)
        derivative_modelrow_fd!(J_fd, compiled_glm, data, r; vars=vars)
        @test isapprox(J, J_fd; rtol=1e-6, atol=1e-8)
        J - J_fd

        # MixedModels (fixed effects only)
        mm = fit(MixedModel, @formula(y ~ 1 + x + z + (1|g)), df; progress=false)
        compiled_mm = compile_formula(mm, data)
        de_mm = build_derivative_evaluator(compiled_mm, data; vars=vars)
        Jmm = Matrix{Float64}(undef, length(compiled_mm), length(vars))
        derivative_modelrow!(Jmm, de_mm, 2)
        Jmm_fd = similar(Jmm)
        derivative_modelrow_fd!(Jmm_fd, compiled_mm, data, 3; vars=vars)
        @test isapprox(Jmm, Jmm_fd; rtol=1e-6, atol=1e-8)
    end

    @testset "FD backend robustness and edge cases" begin
        # Test data with various scales and edge cases
        n = 100
        df = DataFrame(
            y = randn(n),
            x_tiny = randn(n) * 1e-6,      # Very small scale
            x_large = randn(n) * 1e6,      # Very large scale  
            x_zero = zeros(n),              # Constant zero
            x_normal = randn(n),           # Normal scale
            group4 = categorical(rand(["A", "B", "C", "D"], n)),  # 4 levels
            group2 = categorical(rand(["X", "Y"], n)),             # 2 levels
        )
        data = Tables.columntable(df)
        
        # Complex model with multiple interactions
        model = lm(@formula(y ~ 1 + x_tiny + x_large + x_normal + 
                           x_tiny & group4 + x_normal & group2 + 
                           x_large & group4), df)
        compiled = compile_formula(model, data)
        vars = [:x_tiny, :x_large, :x_normal]
        de = build_derivative_evaluator(compiled, data; vars=vars)
        
        @testset "Different step sizes" begin
            # Test various step sizes for FD
            test_row = 5
            J_auto = Matrix{Float64}(undef, length(compiled), length(vars))
            J_small = Matrix{Float64}(undef, length(compiled), length(vars))
            J_large = Matrix{Float64}(undef, length(compiled), length(vars))
            
            # Auto step (default)
            derivative_modelrow_fd!(J_auto, de, test_row)
            
            # Small step
            derivative_modelrow_fd!(J_small, compiled, data, test_row; vars=vars, step=1e-8)
            
            # Larger step  
            derivative_modelrow_fd!(J_large, compiled, data, test_row; vars=vars, step=1e-4)
            
            # All should be reasonably close (allowing for step size effects)
            @test isapprox(J_auto, J_small; rtol=1e-3, atol=1e-6)
            @test isapprox(J_auto, J_large; rtol=1e-2, atol=1e-5)
        end
        
        @testset "All categorical combinations" begin
            # Test that FD works correctly for all categorical levels
            rows_to_test = [1, 10, 25, 50, 75, 90]  # Sample across dataset
            
            for test_row in rows_to_test
                # Get the categorical values for this row
                group4_val = data.group4[test_row]
                group2_val = data.group2[test_row]
                
                # Compute Jacobian with FD
                J_fd = Matrix{Float64}(undef, length(compiled), length(vars))
                derivative_modelrow_fd!(J_fd, de, test_row)
                
                # Compute manual baseline for comparison
                J_manual = compute_manual_jacobian(compiled, data, test_row, vars)
                
                # Should match manual computation (allow for FD numerical error)
                @test isapprox(J_fd, J_manual; rtol=1e-3, atol=1e-6)
            end
        end
        
        @testset "Extreme variable scales with FD" begin
            # Focus on how FD handles different variable scales
            test_row = 15
            
            # Get Jacobians
            J_fd = Matrix{Float64}(undef, length(compiled), length(vars))
            derivative_modelrow_fd!(J_fd, de, test_row)
            J_manual = compute_manual_jacobian(compiled, data, test_row, vars)
            
            # Check each variable separately
            for (i, var) in enumerate(vars)
                var_scale = abs(data[var][test_row])
                
                # The relative error should be reasonable regardless of scale
                if var_scale > 1e-10  # Avoid issues with truly zero values
                    col_diff = abs.(J_fd[:, i] .- J_manual[:, i])
                    max_expected_val = maximum(abs.(J_manual[:, i]))
                    
                    if max_expected_val > 1e-10
                        rel_error = maximum(col_diff) / max_expected_val
                        @test rel_error < 1e-2  # More lenient for extreme scales
                    end
                end
            end
        end
        
        @testset "FD marginal effects consistency across rows" begin
            # Test that FD marginal effects are consistent across different rows
            β = coef(model)
            
            # Test multiple rows with FD backend
            test_rows = [5, 15, 25, 35, 45]
            
            for row in test_rows
                # Compute marginal effects with FD
                gη_fd = Vector{Float64}(undef, length(vars))
                marginal_effects_eta!(gη_fd, de, β, row; backend=:fd)
                
                # Compute reference using manual Jacobian
                J_manual = compute_manual_jacobian(compiled, data, row, vars)
                gη_manual = transpose(J_manual) * β
                
                # Should match
                @test isapprox(gη_fd, gη_manual; rtol=1e-4, atol=1e-7)
            end
        end
        
        @testset "Many variables FD scaling" begin
            # Test FD with more variables to stress the generated functions
            df_many = DataFrame(
                y = randn(n),
                x1 = randn(n), x2 = randn(n), x3 = randn(n), x4 = randn(n),
                x5 = randn(n), x6 = randn(n), x7 = randn(n), x8 = randn(n),
                group = categorical(rand(["A", "B"], n)),
            )
            data_many = Tables.columntable(df_many)
            
            # Model with many variables
            model_many = lm(@formula(y ~ 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + 
                                   x1 & group + x4 & group), df_many)
            compiled_many = compile_formula(model_many, data_many)
            vars_many = [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8]
            de_many = build_derivative_evaluator(compiled_many, data_many; vars=vars_many)
            
            # Test FD with many variables
            test_row = 10
            J_fd_many = Matrix{Float64}(undef, length(compiled_many), length(vars_many))
            derivative_modelrow_fd!(J_fd_many, de_many, test_row)
            
            # Compare against manual
            J_manual_many = compute_manual_jacobian(compiled_many, data_many, test_row, vars_many)
            @test isapprox(J_fd_many, J_manual_many; rtol=1e-5, atol=1e-8)
            
            # Test marginal effects too
            β_many = coef(model_many)
            gη_fd_many = Vector{Float64}(undef, length(vars_many))
            marginal_effects_eta!(gη_fd_many, de_many, β_many, test_row; backend=:fd)
            
            gη_manual_many = transpose(J_manual_many) * β_many
            @test isapprox(gη_fd_many, gη_manual_many; rtol=1e-4, atol=1e-7)
        end
    end
end
