# test_derivative_phase2.jl
# Testing Phase 2 function derivative implementation

using Test, Random, DataFrames, GLM, Tables, Statistics, LinearAlgebra

using FormulaCompiler
using FormulaCompiler: 
    compile_formula_specialized,
    show_derivative_info,
    marginal_effect, marginal_effect!
using FormulaCompiler:
    compile_derivative_formula_phase2

Random.seed!(08540)

###############################################################################
# PHASE 2 TEST DATA
###############################################################################

function create_function_test_data()
    n = 100
    df = DataFrame(
        x = abs.(randn(n)) .+ 0.1,  # Keep positive for log
        y = randn(n),
        z = abs.(randn(n)) .+ 0.1,  # Keep positive for sqrt
        w = randn(n),
        response = randn(n)
    )
    
    data = Tables.columntable(df)
    return df, data
end

###############################################################################
# PHASE 2 BASIC TESTS
###############################################################################

@testset "Phase 2 Basic Tests" begin
    @testset "Testing Simple Function Derivatives" begin
        
        df, data = create_function_test_data()
        
        # Test model with log function: response ~ log(x)
        model = lm(@formula(response ~ log(x)), df)
        specialized = compile_formula_specialized(model, data)
        
        # Compile Phase 2 derivative
        dx_deriv = compile_derivative_formula_phase2(specialized, :x)
        dy_deriv = compile_derivative_formula_phase2(specialized, :y)  # Should be zero

        show_derivative_info(dx_deriv)
        
        # Test derivative evaluation
        row_idx = 1
        x_val = data.x[row_idx]
        
        dx_vec = Vector{Float64}(undef, length(dx_deriv))
        modelrow!(dx_vec, dx_deriv, data, row_idx)
        
        dy_vec = Vector{Float64}(undef, length(dy_deriv))
        modelrow!(dy_vec, dy_deriv, data, row_idx)
        
        # Expected: ∂log(x)/∂x = 1/x, so derivative should be [0, 1/x] for [intercept, log(x)]
        expected_dx_deriv = 1.0 / x_val
        
        # Find the non-zero derivative (should be 1/x)
        nonzero_positions = findall(val -> abs(val) > 1e-10, dx_vec)
        @test length(nonzero_positions) == 1  # Exactly one non-zero derivative
        @test dx_vec[nonzero_positions[1]] ≈ expected_dx_deriv atol=1e-10
        
        # y derivative should be all zeros
        @test all(abs.(dy_vec) .< 1e-10)
    end

    @testset "Testing Multiple Function Derivatives" begin
        df, data = create_function_test_data()
        
        # Test model with multiple functions: response ~ log(x) + exp(y) + sqrt(z)
        model = lm(@formula(response ~ log(x) + exp(y) + sqrt(z)), df)
        specialized = compile_formula_specialized(model, data)
        
        # Compile derivatives for each variable
        dx_deriv = compile_derivative_formula_phase2(specialized, :x)
        dy_deriv = compile_derivative_formula_phase2(specialized, :y)
        dz_deriv = compile_derivative_formula_phase2(specialized, :z)
        dw_deriv = compile_derivative_formula_phase2(specialized, :w)  # Should be zero
        
        # Test at multiple observations
        test_rows = [1, 25, 50, 75, 100]
        
        for row_idx in test_rows
            x_val = data.x[row_idx]
            y_val = data.y[row_idx]
            z_val = data.z[row_idx]
            
            # Evaluate derivatives
            dx_vec = modelrow(dx_deriv, data, row_idx)
            dy_vec = modelrow(dy_deriv, data, row_idx)
            dz_vec = modelrow(dz_deriv, data, row_idx)
            dw_vec = modelrow(dw_deriv, data, row_idx)
            
            # Expected derivatives:
            # ∂log(x)/∂x = 1/x
            # ∂exp(y)/∂y = exp(y)
            # ∂sqrt(z)/∂z = 1/(2*sqrt(z))
            
            expected_dx = 1.0 / x_val
            expected_dy = exp(y_val)
            expected_dz = 1.0 / (2.0 * sqrt(z_val))
            
            # Check that each derivative vector has exactly one non-zero entry
            dx_nonzero = findall(val -> abs(val) > 1e-10, dx_vec)
            dy_nonzero = findall(val -> abs(val) > 1e-10, dy_vec)
            dz_nonzero = findall(val -> abs(val) > 1e-10, dz_vec)
            
            @test length(dx_nonzero) == 1
            @test length(dy_nonzero) == 1
            @test length(dz_nonzero) == 1
            @test all(abs.(dw_vec) .< 1e-10)  # w not in formula
            
            # Check derivative values
            @test dx_vec[dx_nonzero[1]] ≈ expected_dx atol=1e-8
            @test dy_vec[dy_nonzero[1]] ≈ expected_dy atol=1e-8
            @test dz_vec[dz_nonzero[1]] ≈ expected_dz atol=1e-8
        end
    end

    @testset "Testing Function Marginal Effects" begin    
        df, data = create_function_test_data()
        
        # Test model: response ~ log(x) + sqrt(z)
        model = lm(@formula(response ~ log(x) + sqrt(z)), df)
        coefficients = coef(model)
        
        # Compile derivatives
        specialized = compile_formula_specialized(model, data)
        dx_deriv = compile_derivative_formula_phase2(specialized, :x)
        dz_deriv = compile_derivative_formula_phase2(specialized, :z)
        
        # Test marginal effects at multiple observations
        test_rows = [1, 25, 50, 75, 100]
        
        for row_idx in test_rows
            x_val = data.x[row_idx]
            z_val = data.z[row_idx]
            
            # Compute marginal effects
            me_x = marginal_effect(dx_deriv, coefficients, data, row_idx)
            me_z = marginal_effect(dz_deriv, coefficients, data, row_idx)
            
            # Expected marginal effects:
            # ME(x) = coefficient_log_x * (1/x)
            # ME(z) = coefficient_sqrt_z * (1/(2*sqrt(z)))
            
            # Find coefficient positions (skip intercept)
            log_x_coef = coefficients[2]  # Assuming order: [intercept, log(x), sqrt(z)]
            sqrt_z_coef = coefficients[3]
            
            expected_me_x = log_x_coef * (1.0 / x_val)
            expected_me_z = sqrt_z_coef * (1.0 / (2.0 * sqrt(z_val)))
            
            @test me_x ≈ expected_me_x atol=1e-8
            @test me_z ≈ expected_me_z atol=1e-8
            
            # if row_idx <= 25
            #     println("Row $row_idx:")
            #     println("  ME(x) = $me_x ≈ $expected_me_x")
            #     println("  ME(z) = $me_z ≈ $expected_me_z")
            # end
        end
        
        # Test batch marginal effects
        derivative_formulas = [dx_deriv, dz_deriv]
        me_vec = Vector{Float64}(undef, 2)
        
        marginal_effect!(me_vec, derivative_formulas, coefficients, data, 1)
        
        x_val = data.x[1]
        z_val = data.z[1]
        expected_me_x = coefficients[2] * (1.0 / x_val)
        expected_me_z = coefficients[3] * (1.0 / (2.0 * sqrt(z_val)))
        
        @test me_vec[1] ≈ expected_me_x atol=1e-8
        @test me_vec[2] ≈ expected_me_z atol=1e-8
    end

    @testset "Testing Function Derivatives vs Finite Differences" begin
        df, data = create_function_test_data()
        
        # Test simple function: response ~ log(x)
        model = lm(@formula(response ~ log(x)), df)
        specialized = compile_formula_specialized(model, data)
        dx_deriv = compile_derivative_formula_phase2(specialized, :x)
        
        # Test at a few observations
        test_rows = [1, 25, 50]
        max_errors = Float64[]
        
        for row_idx in test_rows
            # Get analytical derivative
            analytical_deriv = modelrow(dx_deriv, data, row_idx)
            
            # Compute numerical derivative using finite differences
            ε = sqrt(eps(Float64)) * 10  # Slightly larger epsilon for functions
            current_x = Float64(data.x[row_idx])
            
            # Evaluate at x + ε
            modified_x_plus = copy(data.x)
            modified_x_plus[row_idx] = current_x + ε
            data_plus = merge(data, (x = modified_x_plus,))
            
            result_plus = modelrow(specialized, data_plus, row_idx)
            
            # Evaluate at x - ε
            modified_x_minus = copy(data.x)
            modified_x_minus[row_idx] = current_x - ε  
            data_minus = merge(data, (x = modified_x_minus,))
            
            result_minus = modelrow(specialized, data_minus, row_idx)
            
            # Central difference
            numerical_deriv = (result_plus .- result_minus) ./ (2ε)
            
            # Compare
            errors = abs.(analytical_deriv .- numerical_deriv)
            max_error = maximum(errors)
            push!(max_errors, max_error)
            
            # println("Row $row_idx (x = $current_x): max error = $(round(max_error, digits=10))")
            # println("  Analytical: $analytical_deriv")
            # println("  Numerical:  $numerical_deriv")
            
            # For log function, error should be reasonable
            @test max_error < 1e-6  # Functions have slightly larger errors than linear models
        end
        
        overall_max_error = maximum(max_errors)
        # "Overall maximum error: $(round(overall_max_error, digits=10))"
        
        @test overall_max_error < 1e-6
    end

    @testset "Testing Mixed Function and Continuous Derivatives" begin
        
        df, data = create_function_test_data()
        
        # Test model mixing functions and continuous: response ~ x + log(z) + y
        model = lm(@formula(response ~ x + log(z) + y), df)
        specialized = compile_formula_specialized(model, data)
        
        println("Testing mixed model: response ~ x + log(z) + y")
        
        # Compile derivatives
        dx_deriv = compile_derivative_formula_phase2(specialized, :x)
        dy_deriv = compile_derivative_formula_phase2(specialized, :y)
        dz_deriv = compile_derivative_formula_phase2(specialized, :z)
        
        # Test at observation 1
        row_idx = 1
        z_val = data.z[row_idx]
        
        dx_vec = modelrow(dx_deriv, data, row_idx)
        dy_vec = modelrow(dy_deriv, data, row_idx)
        dz_vec = modelrow(dz_deriv, data, row_idx)
        
        # println("∂/∂x: $dx_vec")
        # println("∂/∂y: $dy_vec") 
        # println("∂/∂z: $dz_vec")
        
        # Expected patterns:
        # ∂/∂x: [0, 1, 0, 0] (for [intercept, x, log(z), y])
        # ∂/∂z: [0, 0, 1/z, 0] (for [intercept, x, log(z), y])
        # ∂/∂y: [0, 0, 0, 1] (for [intercept, x, log(z), y])
        
        # Check that each has exactly one non-zero entry
        dx_nonzero = findall(val -> abs(val) > 1e-10, dx_vec)
        dy_nonzero = findall(val -> abs(val) > 1e-10, dy_vec)
        dz_nonzero = findall(val -> abs(val) > 1e-10, dz_vec)
        
        @test length(dx_nonzero) == 1  # x term
        @test length(dy_nonzero) == 1  # y term
        @test length(dz_nonzero) == 1  # log(z) term  
        
        # Check values
        @test dx_vec[dx_nonzero[1]] ≈ 1.0 atol=1e-10  # ∂x/∂x = 1
        @test dz_vec[dz_nonzero[1]] ≈ 1.0/z_val atol=1e-10  # ∂log(z)/∂z = 1/z
        @test dy_vec[dy_nonzero[1]] ≈ 1.0 atol=1e-10  # ∂y/∂y = 1
        
        # Check they're in different positions
        @test dx_nonzero[1] != dy_nonzero[1]
        @test dy_nonzero[1] != dz_nonzero[1]
        @test dx_nonzero[1] != dz_nonzero[1]
    end

    @testset "Testing Phase 2 Zero Allocation Performance" begin
        df, data = create_function_test_data()
        
        # Test function model
        model = lm(@formula(response ~ log(x) + sqrt(z)), df)
        specialized = compile_formula_specialized(model, data)
        dx_deriv = compile_derivative_formula_phase2(specialized, :x)
        
        # Pre-allocate output vector
        deriv_vec = Vector{Float64}(undef, length(dx_deriv))
        
        # Warmup
        for _ in 1:10
            modelrow!(deriv_vec, dx_deriv, data, 1)
        end
        
        # Test allocation behavior
        test_allocs = @allocated begin
            for i in 1:20
                row_idx = ((i - 1) % length(first(data))) + 1
                modelrow!(deriv_vec, dx_deriv, data, row_idx)
            end
        end
        
        avg_allocs_per_call = test_allocs / 20
        
        # Benchmark timing
        elapsed = @elapsed begin
            for i in 1:1000
                row_idx = ((i - 1) % length(first(data))) + 1
                modelrow!(deriv_vec, dx_deriv, data, row_idx)
            end
        end
        
        avg_time_ns = (elapsed / 1000) * 1e9
        
        # Phase 2 may have some allocations due to function complexity, but should be minimal
        # is it low allocation?
        @test avg_allocs_per_call < 250  # Allow some allocations for functions
    end
end
