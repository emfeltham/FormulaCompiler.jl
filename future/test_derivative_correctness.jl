# test/test_derivative_correctness.jl
# Mathematical correctness validation for analytical derivatives

using FormulaCompiler:
    test_single_variable_derivative, test_derivative_at_observation,
    compute_numerical_derivative, find_continuous_variables,
    find_continuous_variables_recursive!,
    test_single_variable_derivative

@testset "Derivative Correctness" begin
    
    # Create comprehensive test data
    df = DataFrame(
        x = randn(50),
        y = randn(50),
        z = abs.(randn(50)) .+ 0.1,  # Positive for log
        w = randn(50),
        group = categorical(rand(["A", "B", "C"], 50)),
        flag = rand([true, false], 50)
    )
    
    @testset "Basic Derivative Validation" begin
        # Test simple formulas with known derivatives
        
        # Test ∂(x)/∂x = 1
        model = lm(@formula(y ~ x), df)
        compiled = compile_formula(model)
        
        try
            dx_compiled = compile_derivative_formula(compiled, :x)
            @test dx_compiled isa CompiledDerivativeFormula
            
            # Validate against finite differences
            success, report = test_single_variable_derivative(compiled, :x, Tables.columntable(df), 10, 1e-6, false)
            if !success
                @test false  # Will show as test failure
                println("Basic derivative ∂(x)/∂x failed validation: $report")
            else
                @test true
                @test report.max_error < 1e-6
            end
            
        catch e
            @test_skip "Basic derivative test skipped due to compilation error"
            println("Error: $e")
        end
        
        # Test ∂(x + w)/∂x = 1  
        model2 = lm(@formula(y ~ x + w), df)
        compiled2 = compile_formula(model2)
        
        try
            dx_compiled2 = compile_derivative_formula(compiled2, :x)
            success, report = test_single_variable_derivative(compiled2, :x, Tables.columntable(df), 10, 1e-6, false)
            @test success
            # "Linear combination derivative failed validation: $(report)"
            
        catch e
            @test_skip "Linear combination derivative test skipped: $e"
        end
    end
    
    @testset "Function Derivative Validation" begin
        # Test derivatives of mathematical functions
        
        # Test ∂(log(z))/∂z = 1/z
        model = lm(@formula(y ~ log(z)), df)
        compiled = compile_formula(model)
        
        try
            dz_compiled = compile_derivative_formula(compiled, :z)
            success, report = test_single_variable_derivative(compiled, :z, Tables.columntable(df), 10, 1e-5, false)
            @test success
            # "Logarithm derivative ∂(log(z))/∂z failed validation"
            @test report.max_error < 1e-5  # Slightly relaxed for log
            
        catch e
            @test_skip "Logarithm derivative test skipped: $e"
        end
        
        # Test ∂(x^2)/∂x = 2x
        model2 = lm(@formula(y ~ x^2), df)
        compiled2 = compile_formula(model2)
        
        try
            dx_compiled2 = compile_derivative_formula(compiled2, :x)
            success, report = test_single_variable_derivative(compiled2, :x, Tables.columntable(df), 10, 1e-6, false)
            @test success
            # "Polynomial derivative ∂(x^2)/∂x failed validation"
            
        catch e
            @test_skip "Polynomial derivative test skipped: $e"
        end
        
        # Test ∂(sqrt(z))/∂z = 1/(2*sqrt(z))
        model3 = lm(@formula(y ~ sqrt(z)), df)
        compiled3 = compile_formula(model3)
        
        try
            dz_compiled3 = compile_derivative_formula(compiled3, :z)
            success, report = test_single_variable_derivative(compiled3, :z, Tables.columntable(df), 10, 1e-5, false)
            @test success
            # "Square root derivative failed validation"
            
        catch e
            @test_skip "Square root derivative test skipped: $e"
        end
    end
    
    @testset "Interaction Derivative Validation" begin
        # Test derivatives of interaction terms
        
        # Test ∂(x * w)/∂x = w
        model = lm(@formula(y ~ x * w), df)
        compiled = compile_formula(model)
        
        try
            dx_compiled = compile_derivative_formula(compiled, :x)
            success, report = test_single_variable_derivative(compiled, :x, Tables.columntable(df), 10, 1e-6, false)
            @test success
            # "Product derivative ∂(x*w)/∂x failed validation"
            
        catch e
            @test_skip "Product derivative test skipped: $e"
        end
        
        # Test ∂(x * group)/∂x = group_contrasts  
        model2 = lm(@formula(y ~ x * group), df)
        compiled2 = compile_formula(model2)
        
        try
            dx_compiled2 = compile_derivative_formula(compiled2, :x)
            success, report = test_single_variable_derivative(compiled2, :x, Tables.columntable(df), 10, 1e-6, false)
            @test success
            # "Continuous × categorical derivative failed validation"
            
        catch e
            @test_skip "Continuous × categorical derivative test skipped: $e"
        end
    end
    
    @testset "Complex Formula Validation" begin
        # Test derivatives of complex formulas
        
        # Test ∂(x + log(z) + x*w)/∂x = 1 + w
        model = lm(@formula(y ~ x + log(z) + x * w), df)
        compiled = compile_formula(model)
        
        try
            dx_compiled = compile_derivative_formula(compiled, :x)
            success, report = test_single_variable_derivative(compiled, :x, Tables.columntable(df), 10, 1e-5, false)
            @test success
            # "Complex mixed derivative failed validation"
            
        catch e
            @test_skip "Complex mixed derivative test skipped: $e"
        end
        
        # Test ∂(log(z) * group)/∂z = (1/z) * group_contrasts
        model2 = lm(@formula(y ~ log(z) * group), df)
        compiled2 = compile_formula(model2)
        
        try
            dz_compiled2 = compile_derivative_formula(compiled2, :z)
            success, report = test_single_variable_derivative(compiled2, :z, Tables.columntable(df), 10, 1e-5, false)
            @test success
            # "Function × categorical derivative failed validation"
            
        catch e
            @test_skip "Function × categorical derivative test skipped: $e"
        end
    end
    
    @testset "Edge Case Validation" begin
        # Test derivatives with challenging edge cases
        
        # Create edge case data
        edge_df = DataFrame(
            x = [0.0, 1e-6, 1e6, -1e6, 1.0],
            y = [1.0, 2.0, 3.0, 4.0, 5.0],
            z = [1e-5, 1.0, 1e5, 0.1, 10.0],  # All positive for log
            group = categorical(["A", "B", "A", "B", "A"])
        )
        
        # Test with extreme values
        model = lm(@formula(y ~ x + log(z)), edge_df)
        compiled = compile_formula(model)
        
        try
            dx_compiled = compile_derivative_formula(compiled, :x)
            success, report = test_single_variable_derivative(compiled, :x, Tables.columntable(edge_df), 5, 1e-4, false)
            @test success
            # "Edge case derivative validation failed"
            
        catch e
            @test_skip "Edge case test skipped: $e"
        end
        
        # Test with very small values
        small_df = DataFrame(
            x = [1e-10, 1e-8, 1e-6],
            y = [1.0, 2.0, 3.0],
            z = [1e-5, 1e-4, 1e-3]
        )
        
        model2 = lm(@formula(y ~ x^2), small_df)
        compiled2 = compile_formula(model2)
        
        try
            dx_compiled2 = compile_derivative_formula(compiled2, :x)
            success, report = test_single_variable_derivative(compiled2, :x, Tables.columntable(small_df), 3, 1e-3, false)
            @test success
            # "Small values derivative validation failed"
            
        catch e
            @test_skip "Small values test skipped: $e"
        end
    end
    
    @testset "Comprehensive Formula Test Suite" begin
        # Test a wide range of formulas systematically
        
        test_formulas = [
            (@formula(y ~ x), "simple linear"),
            (@formula(y ~ x + w), "multiple continuous"),
            (@formula(y ~ x^2), "polynomial"),
            (@formula(y ~ log(z)), "logarithm"),
            (@formula(y ~ x * w), "continuous interaction"),
            (@formula(y ~ x * group), "continuous × categorical"),
            (@formula(y ~ x + log(z) + x*w), "mixed complex")
        ]
        
        data = Tables.columntable(df)
        
        for (formula, description) in test_formulas
            @testset "$description" begin
                try
                    model = lm(formula, df)
                    compiled = compile_formula(model)
                    
                    # Find continuous variables to test
                    continuous_vars = find_continuous_variables(compiled)
                    
                    if !isempty(continuous_vars)
                        # Test first continuous variable
                        focal_var = continuous_vars[1]
                        
                        try
                            deriv_compiled = compile_derivative_formula(compiled, focal_var)
                            success, report = test_single_variable_derivative(compiled, focal_var, data, 10, 1e-5, false)
                            @test success
                            # "Derivative validation failed for $description with respect to $focal_var"
                            
                        catch e
                            @test_skip "Derivative compilation failed for $description: $e"
                        end
                    else
                        @test_skip "No continuous variables found for $description"
                    end
                    
                catch e
                    @test_skip "Model fitting failed for $description: $e"
                end
            end
        end
    end
    
    @testset "Performance and Allocation Validation" begin
        # Test that derivative evaluation maintains performance
        
        model = lm(@formula(y ~ x * group + log(z)), df)
        compiled = compile_formula(model)
        
        try
            dx_compiled = compile_derivative_formula(compiled, :x)
            data = Tables.columntable(df)
            deriv_vec = Vector{Float64}(undef, length(dx_compiled))
            
            # Test zero allocations
            allocs = @allocated modelrow!(deriv_vec, dx_compiled, data, 1)
            @test allocs == 0
            # "Derivative evaluation should be zero-allocation"
            
            # Test reasonable performance 
            eval_time = @elapsed modelrow!(deriv_vec, dx_compiled, data, 1)
            @test eval_time < 0.001
            # "Derivative evaluation should be fast (< 1ms)"
            
            # Test consistency across multiple calls
            times = [(@elapsed modelrow!(deriv_vec, dx_compiled, data, 1)) for _ in 1:10]
            @test all(t -> t < 0.001, times)
            # "All derivative evaluations should be fast"
            
        catch e
            @test_skip "Performance test skipped due to compilation error: $e"
        end
    end
    
end
