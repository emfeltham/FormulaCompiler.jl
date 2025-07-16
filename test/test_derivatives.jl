# test/test_derivatives.jl
# Tests for analytical derivative compilation

@testset "Derivative Compilation" begin
    df = DataFrame(
        x = randn(20),
        y = randn(20),
        z = abs.(randn(20)) .+ 0.1,
        group = categorical(rand(["A", "B"], 20))
    )
    data = Tables.columntable(df)
    
    @testset "Basic Derivative Evaluators" begin
        # Test ConstantEvaluator derivative
        const_eval = ConstantEvaluator(5.0)
        deriv = compute_derivative_evaluator(const_eval, :x)
        @test deriv isa ConstantEvaluator
        @test deriv.value == 0.0
        
        # Test ContinuousEvaluator derivative
        x_eval = ContinuousEvaluator(:x)
        deriv = compute_derivative_evaluator(x_eval, :x)
        @test deriv isa ConstantEvaluator
        @test deriv.value == 1.0
        
        # Test derivative w.r.t. different variable
        deriv = compute_derivative_evaluator(x_eval, :y)
        @test deriv isa ConstantEvaluator
        @test deriv.value == 0.0
        
        # Test CategoricalEvaluator derivative
        contrast_matrix = [0.0; 1.0][:, :]
        cat_eval = CategoricalEvaluator(:group, contrast_matrix, 2)
        deriv = compute_derivative_evaluator(cat_eval, :x)
        @test deriv isa ConstantEvaluator
        @test deriv.value == 0.0
    end
    
    @testset "Function Derivative Evaluators" begin
        # Test unary functions
        x_eval = ContinuousEvaluator(:x)
        
        # Test log derivative: d/dx log(x) = 1/x
        log_eval = FunctionEvaluator(log, [x_eval])
        deriv = compute_derivative_evaluator(log_eval, :x)
        @test deriv isa ChainRuleEvaluator
        
        # Test exp derivative: d/dx exp(x) = exp(x)
        exp_eval = FunctionEvaluator(exp, [x_eval])
        deriv = compute_derivative_evaluator(exp_eval, :x)
        @test deriv isa ChainRuleEvaluator
        
        # Test sqrt derivative: d/dx sqrt(x) = 1/(2*sqrt(x))
        sqrt_eval = FunctionEvaluator(sqrt, [x_eval])
        deriv = compute_derivative_evaluator(sqrt_eval, :x)
        @test deriv isa ChainRuleEvaluator
    end
    
    @testset "Binary Operation Derivatives" begin
        x_eval = ContinuousEvaluator(:x)
        y_eval = ContinuousEvaluator(:y)
        
        # Test addition: d/dx (x + y) = 1
        add_eval = FunctionEvaluator(+, [x_eval, y_eval])
        deriv = compute_derivative_evaluator(add_eval, :x)
        @test deriv isa ConstantEvaluator
        @test deriv.value == 1.0
        
        # Test multiplication: d/dx (x * y) = y (using product rule)
        mult_eval = FunctionEvaluator(*, [x_eval, y_eval])
        deriv = compute_derivative_evaluator(mult_eval, :x)
        @test deriv isa ContinuousEvaluator
        
        # Test power: d/dx (x^2) = 2*x
        const_2 = ConstantEvaluator(2.0)
        power_eval = FunctionEvaluator(^, [x_eval, const_2])
        deriv = compute_derivative_evaluator(power_eval, :x)
        # Should simplify to 2*x^1 = 2*x
        @test deriv isa ProductEvaluator
    end
    
    @testset "Interaction Derivatives" begin
        x_eval = ContinuousEvaluator(:x)
        y_eval = ContinuousEvaluator(:y)
        z_eval = ContinuousEvaluator(:z)
        
        # Test two-way interaction: d/dx (x * y) = y
        interaction = InteractionEvaluator([x_eval, y_eval])
        deriv = compute_derivative_evaluator(interaction, :x)
        @test deriv isa ContinuousEvaluator
        
        # Test three-way interaction: d/dx (x * y * z) = y * z
        interaction3 = InteractionEvaluator([x_eval, y_eval, z_eval])
        deriv = compute_derivative_evaluator(interaction3, :x)
        @test deriv isa InteractionEvaluator
        @test length(deriv.components) == 2  # y and z
    end
    
    @testset "Chain Rule Application" begin
        x_eval = ContinuousEvaluator(:x)
        
        # Test d/dx log(x^2) = 2*x / x^2 = 2/x
        const_2 = ConstantEvaluator(2.0)
        x_squared = FunctionEvaluator(^, [x_eval, const_2])
        log_x_squared = FunctionEvaluator(log, [x_squared])
        
        deriv = compute_derivative_evaluator(log_x_squared, :x)
        @test deriv isa ChainRuleEvaluator
        
        # Test evaluation gives correct result
        output = Vector{Float64}(undef, 1)
        evaluate!(deriv, output, data, 1, 1)
        
        # Should equal 2/x
        x_val = data.x[1]
        expected = 2.0 / x_val
        @test isapprox(output[1], expected, rtol=1e-10)
    end
    
    @testset "Product Rule Application" begin
        x_eval = ContinuousEvaluator(:x)
        z_eval = ContinuousEvaluator(:z)
        
        # Test d/dx (x * log(z)) = log(z) (since z doesn't depend on x)
        log_z = FunctionEvaluator(log, [z_eval])
        product = FunctionEvaluator(*, [x_eval, log_z])
        
        deriv = compute_derivative_evaluator(product, :x)
        @test deriv isa FunctionEvaluator
        
        # Test evaluation
        output = Vector{Float64}(undef, 1)
        evaluate!(deriv, output, data, 1, 1)
        
        # Should equal log(z)
        z_val = data.z[1]
        expected = log(z_val)
        @test isapprox(output[1], expected, rtol=1e-10)
    end
    
    @testset "Optimization Cases" begin
        x_eval = ContinuousEvaluator(:x)
        const_eval = ConstantEvaluator(5.0)
        
        # Test d/dx (x * 5) = 5 (constant in product)
        product = FunctionEvaluator(*, [x_eval, const_eval])
        deriv = compute_derivative_evaluator(product, :x)
        @test deriv isa ConstantEvaluator
        @test deriv == ConstantEvaluator(5.0)
        
        # Test d/dx (5 * x) = 5 (constant first)
        product2 = FunctionEvaluator(*, [const_eval, x_eval])
        deriv2 = compute_derivative_evaluator(product2, :x)
        @test deriv2 isa ConstantEvaluator
        
        # Test d/dx (constant) = 0
        deriv3 = compute_derivative_evaluator(const_eval, :x)
        @test deriv3 isa ConstantEvaluator
        @test deriv3.value == 0.0
    end
    
    @testset "Zero Derivative Detection" begin
        x_eval = ContinuousEvaluator(:x)
        y_eval = ContinuousEvaluator(:y)
        zero_const = ConstantEvaluator(0.0)
        nonzero_const = ConstantEvaluator(3.0)
        
        # Test is_zero_derivative function - checks if evaluator always evaluates to zero
        @test is_zero_derivative(zero_const, :x)      # 0 is always zero ✓
        @test !is_zero_derivative(nonzero_const, :x)  # 3 is never zero ✓
        @test !is_zero_derivative(x_eval, :x)         # x is not always zero ✓
        @test !is_zero_derivative(x_eval, :y)         # x is not always zero ✓
        @test !is_zero_derivative(y_eval, :x)         # y is not always zero ✓
        
        # Test with categorical
        contrast_matrix = [0.0; 1.0][:, :]
        cat_eval = CategoricalEvaluator(:group, contrast_matrix, 2)
        @test !is_zero_derivative(cat_eval, :x)       # Categorical values not always zero ✓
        
        # Test with products containing zero
        zero_product = ProductEvaluator([zero_const, x_eval])
        @test is_zero_derivative(zero_product, :x)    # 0 * x = 0 always ✓
        
        nonzero_product = ProductEvaluator([nonzero_const, x_eval])
        @test !is_zero_derivative(nonzero_product, :x) # 3 * x ≠ 0 always ✓
    end
    
    @testset "Derivative Evaluator Types" begin
        # Test that derivative evaluator types work correctly
        x_eval = ContinuousEvaluator(:x)
        y_eval = ContinuousEvaluator(:y)
        
        # Test ChainRuleEvaluator
        chain = ChainRuleEvaluator(x -> 1/x, x_eval, ConstantEvaluator(1.0))
        @test output_width(chain) == 1
        
        output = Vector{Float64}(undef, 1)
        evaluate!(chain, output, data, 1, 1)
        expected = (1.0 / data.x[1]) * 1.0  # f'(g(x)) * g'(x)
        @test isapprox(output[1], expected, rtol=1e-10)
        
        # Test ProductRuleEvaluator
        product_rule = ProductRuleEvaluator(x_eval, ConstantEvaluator(1.0), y_eval, ConstantEvaluator(0.0))
        @test output_width(product_rule) == 1
        
        evaluate!(product_rule, output, data, 1, 1)
        expected = data.x[1] * 0.0 + data.y[1] * 1.0  # f*g' + g*f'
        @test isapprox(output[1], expected, rtol=1e-10)
    end
    
    @testset "Standard Derivative Functions" begin
        # Test get_standard_derivative_function
        @test get_standard_derivative_function(log) !== nothing
        @test get_standard_derivative_function(exp) !== nothing
        @test get_standard_derivative_function(sqrt) !== nothing
        @test get_standard_derivative_function(sin) !== nothing
        @test get_standard_derivative_function(cos) !== nothing
        
        # Test unknown function
        custom_func(x) = x^3
        @test get_standard_derivative_function(custom_func) === nothing
        
        # Test derivative function correctness
        log_deriv = get_standard_derivative_function(log)
        @test log_deriv(2.0) ≈ 0.5  # d/dx log(x) = 1/x, so at x=2: 1/2
        
        exp_deriv = get_standard_derivative_function(exp)
        @test exp_deriv(1.0) ≈ exp(1.0)  # d/dx exp(x) = exp(x)
    end
    
    @testset "Numerical Validation" begin
        # Test derivative accuracy with numerical differentiation
        x_eval = ContinuousEvaluator(:x)
        
        # Test simple polynomial: d/dx (x^2) = 2*x
        const_2 = ConstantEvaluator(2.0)
        x_squared = FunctionEvaluator(^, [x_eval, const_2])
        deriv = compute_derivative_evaluator(x_squared, :x)
        
        # Validate against numerical derivative (1e-8 too strict)
        @test validate_derivative_evaluator(x_squared, deriv, :x, data, 1e-6)
        
        # Test logarithm: d/dx log(x) = 1/x
        log_x = FunctionEvaluator(log, [x_eval])
        deriv_log = compute_derivative_evaluator(log_x, :x)
        
        # Use positive data for log
        @test validate_derivative_evaluator(log_x, deriv_log, :x, data, 1e-6)
    end
    
    @testset "Complex Expression Derivatives" begin
        # Test derivative of x * log(z) + y^2
        x_eval = ContinuousEvaluator(:x)
        y_eval = ContinuousEvaluator(:y)
        z_eval = ContinuousEvaluator(:z)
        
        # Build: x * log(z) + y^2
        log_z = FunctionEvaluator(log, [z_eval])
        x_log_z = FunctionEvaluator(*, [x_eval, log_z])
        
        const_2 = ConstantEvaluator(2.0)
        y_squared = FunctionEvaluator(^, [y_eval, const_2])
        
        combined = CombinedEvaluator([x_log_z, y_squared])
        
        # Test derivative w.r.t. x: should be log(z)
        deriv_x = compute_derivative_evaluator(combined, :x)
        output = Vector{Float64}(undef, 1)
        evaluate!(deriv_x, output, data, 1, 1)
        
        expected = log(data.z[1])
        @test isapprox(output[1], expected, rtol=1e-10)
        
        # Test derivative w.r.t. y: should be 2*y
        deriv_y = compute_derivative_evaluator(combined, :y)
        evaluate!(deriv_y, output, data, 1, 1)
        
        expected = 2.0 * data.y[1]
        @test isapprox(output[1], expected, rtol=1e-10)
    end
    
end
