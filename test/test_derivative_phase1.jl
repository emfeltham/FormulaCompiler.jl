# test_derivative_phase1.jl
# Testing Phase 1 derivative implementation

using Test, Random, DataFrames, GLM, Tables, Statistics, LinearAlgebra
using FormulaCompiler

using FormulaCompiler: 
    compile_formula_specialized,
    show_derivative_info,
    marginal_effect, marginal_effect!

Random.seed!(08540)

###############################################################################
# SETUP TEST DATA
###############################################################################

function create_test_data()
    n = 100    
    df = DataFrame(
        x = randn(n),
        y = randn(n),
        z = randn(n),
        w = randn(n),
        response = randn(n)
    )
    
    data = Tables.columntable(df)
    return df, data
end

###############################################################################
# BASIC DERIVATIVE TESTS
###############################################################################

@testset "Phase 1 Derivative Tests" begin
    @testset "Testing Simple Continuous Derivatives" begin
        df, data = create_test_data()
        
        # Test simple linear model: y ~ x + z
        model = lm(@formula(response ~ x + z), df)
        
        # Compile to specialized formula
        specialized = compile_formula_specialized(model, data)
        
        # Compile derivatives
        dx_deriv = compile_derivative_formula(specialized, :x)
        dz_deriv = compile_derivative_formula(specialized, :z)
        dy_deriv = compile_derivative_formula(specialized, :y)  # Should be all zeros
        
        # Show derivative info
        show_derivative_info(dx_deriv)
        
        # Test derivative evaluation
        row_idx = 1
        
        # ∂/∂x derivatives
        dx_vec = Vector{Float64}(undef, length(dx_deriv))
        modelrow!(dx_vec, dx_deriv, data, row_idx)
        
        # ∂/∂z derivatives  
        dz_vec = Vector{Float64}(undef, length(dz_deriv))
        modelrow!(dz_vec, dz_deriv, data, row_idx)
        
        # ∂/∂y derivatives (should be all zeros)
        dy_vec = Vector{Float64}(undef, length(dy_deriv))
        modelrow!(dy_vec, dy_deriv, data, row_idx)
        
        # For simple linear model y ~ intercept + x + z:
        # We expect exactly one 1.0 in each derivative vector (for the corresponding variable)
        # and zeros elsewhere
        
        @test count(val -> val ≈ 1.0, dx_vec) == 1  # Exactly one 1.0 for x
        @test count(val -> val ≈ 1.0, dz_vec) == 1  # Exactly one 1.0 for z
        @test all(dy_vec .≈ 0.0)  # All zeros for y (not in formula)
        
        # Check that derivatives sum to expected values
        @test sum(abs, dx_vec) ≈ 1.0  # Only one non-zero entry
        @test sum(abs, dz_vec) ≈ 1.0  # Only one non-zero entry
    end

    @testset "Testing Marginal Effects Computation" begin
        df, data = create_test_data()
        
        # Test model: y ~ x + z
        model = lm(@formula(response ~ x + z), df)
        coefficients = coef(model)
        
        # Compile specialized formula and derivatives
        specialized = compile_formula_specialized(model, data)
        dx_deriv = compile_derivative_formula(specialized, :x)
        dz_deriv = compile_derivative_formula(specialized, :z)
        
        # Test marginal effects at multiple observations
        for row_idx in [1, 10, 50, 100]
            # Compute marginal effects
            me_x = marginal_effect(dx_deriv, coefficients, data, row_idx)
            me_z = marginal_effect(dz_deriv, coefficients, data, row_idx)
            
            # println("Row $row_idx: ME(x) = $me_x, ME(z) = $me_z")
            
            # For simple linear model y ~ intercept + x + z:
            # ∂E[y]/∂x = coefficient of x = coefficients[2]
            # ∂E[y]/∂z = coefficient of z = coefficients[3]
            
            @test me_x ≈ coefficients[2] atol=1e-12
            @test me_z ≈ coefficients[3] atol=1e-12
        end
        
        # Test batch marginal effects
        derivative_formulas = [dx_deriv, dz_deriv]
        me_vec = Vector{Float64}(undef, 2)
        
        marginal_effect!(me_vec, derivative_formulas, coefficients, data, 1)
        
        @test me_vec[1] ≈ coefficients[2] atol=1e-12
        @test me_vec[2] ≈ coefficients[3] atol=1e-12
    end

    @testset "Testing Zero Allocation Performance" begin
        df, data = create_test_data()
        
        # Test model
        model = lm(@formula(response ~ x + z), df)
        specialized = compile_formula_specialized(model, data)
        dx_deriv = compile_derivative_formula(specialized, :x)
        
        # Pre-allocate output vector
        deriv_vec = Vector{Float64}(undef, length(dx_deriv))
        
        # Warmup
        for _ in 1:10
            modelrow!(deriv_vec, dx_deriv, data, 1)
        end
        
        # Test allocation behavior
        allocs_before = Base.gc_bytes()
        for i in 1:100
            row_idx = ((i - 1) % length(first(data))) + 1
            modelrow!(deriv_vec, dx_deriv, data, row_idx)
        end
        allocs_after = Base.gc_bytes()
        
        allocated_bytes = allocs_after - allocs_before
        
        # Benchmark timing
        elapsed = @elapsed begin
            for i in 1:1000
                row_idx = ((i - 1) % length(first(data))) + 1
                modelrow!(deriv_vec, dx_deriv, data, row_idx)
            end
        end
        
        avg_time_ns = (elapsed / 1000) * 1e9
        
        # Check for zero allocations in controlled test
        test_allocs = @allocated begin
            for i in 1:10
                modelrow!(deriv_vec, dx_deriv, data, 1)
            end
        end
        
        avg_allocs_per_call = test_allocs / 10
        is_zero_allocation = (avg_allocs_per_call == 0.0)  # ← Ensure boolean result
    end

    @testset "Testing Derivative Correctness vs Finite Differences" begin
        df, data = create_test_data()
        
        # Test simple model: y ~ x + z
        model = lm(@formula(response ~ x + z), df)
        specialized = compile_formula_specialized(model, data)
        dx_deriv = compile_derivative_formula(specialized, :x)
        
        # Test at a few observations
        test_rows = [1, 25, 50, 75, 100]
        max_errors = Float64[]
        
        for row_idx in test_rows
            # Get analytical derivative
            analytical_deriv = Vector{Float64}(undef, length(dx_deriv))
            modelrow!(analytical_deriv, dx_deriv, data, row_idx)
            
            # Compute numerical derivative using finite differences
            ε = sqrt(eps(Float64))
            current_x = Float64(data.x[row_idx])
            
            # Create modified data for x + ε
            modified_x_plus = copy(data.x)
            modified_x_plus[row_idx] = current_x + ε
            data_plus = merge(data, (x = modified_x_plus,))
            
            result_plus = Vector{Float64}(undef, length(specialized))
            modelrow!(result_plus, specialized, data_plus, row_idx)
            
            # Create modified data for x - ε
            modified_x_minus = copy(data.x)
            modified_x_minus[row_idx] = current_x - ε
            data_minus = merge(data, (x = modified_x_minus,))
            
            result_minus = Vector{Float64}(undef, length(specialized))
            modelrow!(result_minus, specialized, data_minus, row_idx)
            
            # Central difference: (f(x+ε) - f(x-ε)) / (2ε)
            numerical_deriv = (result_plus .- result_minus) ./ (2ε)
            
            # Compare analytical vs numerical
            errors = abs.(analytical_deriv .- numerical_deriv)
            max_error = maximum(errors)
            push!(max_errors, max_error)
            
            # Test that error is within acceptable tolerance
            @test max_error < 1e-10  # Should be very small for linear models
        end
        
        overall_max_error = maximum(max_errors)
        # "Overall maximum error: $(round(overall_max_error, digits=12))"
        
        @test overall_max_error < 1e-10
    end

    @testset "Testing Multiple Variables Derivatives" begin
        df, data = create_test_data()
        
        # Test model with multiple variables: y ~ x + y + z + w
        model = lm(@formula(response ~ x + y + z + w), df)
        specialized = compile_formula_specialized(model, data)
        
        variables = [:x, :y, :z, :w]
        derivative_formulas = [compile_derivative_formula(specialized, var) for var in variables]
            
        # Test at observation 1
        row_idx = 1
        expected_patterns = [
            [0.0, 1.0, 0.0, 0.0, 0.0],  # ∂/∂x: [intercept, x, y, z, w]
            [0.0, 0.0, 1.0, 0.0, 0.0],  # ∂/∂y
            [0.0, 0.0, 0.0, 1.0, 0.0],  # ∂/∂z
            [0.0, 0.0, 0.0, 0.0, 1.0],  # ∂/∂w
        ]
        
        for (i, (var, deriv_formula, expected)) in enumerate(zip(variables, derivative_formulas, expected_patterns))
            deriv_vec = Vector{Float64}(undef, length(deriv_formula))
            modelrow!(deriv_vec, deriv_formula, data, row_idx)
            
            @test deriv_vec ≈ expected
        end
        
        # Test marginal effects
        coefficients = coef(model)
        
        for (i, (var, deriv_formula)) in enumerate(zip(variables, derivative_formulas))
            me = marginal_effect(deriv_formula, coefficients, data, row_idx)
            expected_me = coefficients[i+1]  # Skip intercept
            
            @test me ≈ expected_me atol=1e-12
        end
    end

    @testset "Test Enhanced Formula (with Categorical) Derivatives" begin
        # Create data with categorical variable
        n = 100
        df = DataFrame(
            x = randn(n),
            y = randn(n),
            group = rand(["A", "B", "C"], n),
            response = randn(n)
        )
        
        data = Tables.columntable(df)
        
        # Test model with categorical: response ~ x + y + group
        model = lm(@formula(response ~ x + y + group), df)
        specialized = compile_formula_specialized(model, data)
        
        # Compile derivatives
        dx_deriv = compile_derivative_formula(specialized, :x)
        dy_deriv = compile_derivative_formula(specialized, :y)
        
        # Test derivative evaluation
        row_idx = 1
        
        dx_vec = Vector{Float64}(undef, length(dx_deriv))
        modelrow!(dx_vec, dx_deriv, data, row_idx)
        
        dy_vec = Vector{Float64}(undef, length(dy_deriv))
        modelrow!(dy_vec, dy_deriv, data, row_idx)
        
        # For model response ~ intercept + x + y + group_B + group_C:
        # ∂/∂x should be [0, 1, 0, 0, 0] (1 for x term, 0 elsewhere)
        # ∂/∂y should be [0, 0, 1, 0, 0] (1 for y term, 0 elsewhere)
        
        # Find positions of x and y terms
        x_position = findfirst(val -> val ≈ 1.0, dx_vec)
        y_position = findfirst(val -> val ≈ 1.0, dy_vec)
        
        @test x_position !== nothing
        @test y_position !== nothing
        @test x_position != y_position
        
        # Check that only one position is 1.0 for each derivative
        @test count(val -> val ≈ 1.0, dx_vec) == 1
        @test count(val -> val ≈ 1.0, dy_vec) == 1
        
        # Check that categorical terms have zero derivatives
        @test sum(abs, dx_vec) ≈ 1.0  # Only the x term contributes
        @test sum(abs, dy_vec) ≈ 1.0  # Only the y term contributes
        
        # Test marginal effects
        coefficients = coef(model)
        me_x = marginal_effect(dx_deriv, coefficients, data, row_idx)
        me_y = marginal_effect(dy_deriv, coefficients, data, row_idx)
        
        # For linear model, marginal effects should equal coefficients
        @test me_x ≈ coefficients[x_position] atol=1e-12
        @test me_y ≈ coefficients[y_position] atol=1e-12
    end
end
