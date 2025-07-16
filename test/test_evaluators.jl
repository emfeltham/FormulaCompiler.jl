# test/test_evaluators.jl
# Tests for core evaluator functionality

@testset "Core Evaluators" begin
    
    # Create test data
    df = DataFrame(
        x = [1.0, 2.0, 3.0, 4.0],
        y = [10.0, 20.0, 30.0, 40.0],
        z = [0.1, 0.2, 0.3, 0.4],
        group = categorical(["A", "B", "A", "B"]),
        flag = [true, false, true, false]
    )
    data = Tables.columntable(df)
    
    @testset "ConstantEvaluator" begin
        eval = ConstantEvaluator(5.0)
        @test output_width(eval) == 1
        
        output = Vector{Float64}(undef, 1)
        next_idx = evaluate!(eval, output, data, 1, 1)
        @test output[1] == 5.0
        @test next_idx == 2
    end
    
    @testset "ContinuousEvaluator" begin
        eval = ContinuousEvaluator(:x)
        @test output_width(eval) == 1
        
        output = Vector{Float64}(undef, 1)
        next_idx = evaluate!(eval, output, data, 2, 1)
        @test output[1] == 2.0
        @test next_idx == 2
    end
    
    @testset "CategoricalEvaluator" begin
        # Create dummy contrast matrix (2 levels, 1 contrast column)
        contrast_matrix = [0.0; 1.0][:, :]  # Convert to matrix
        eval = CategoricalEvaluator(:group, contrast_matrix, 2)
        @test output_width(eval) == 1
        
        output = Vector{Float64}(undef, 1)
        # Test first row (group="A", level 1)
        next_idx = evaluate!(eval, output, data, 1, 1)
        @test output[1] == 0.0  # First level maps to 0.0
        @test next_idx == 2
        
        # Test second row (group="B", level 2) 
        evaluate!(eval, output, data, 2, 1)
        @test output[1] == 1.0  # Second level maps to 1.0
    end
    
    @testset "FunctionEvaluator" begin
        # Test unary function
        x_eval = ContinuousEvaluator(:x)
        log_eval = FunctionEvaluator(log, [x_eval])
        @test output_width(log_eval) == 1
        
        output = Vector{Float64}(undef, 1)
        evaluate!(log_eval, output, data, 1, 1)
        @test output[1] ≈ log(1.0)
        
        # Test binary function
        y_eval = ContinuousEvaluator(:y)
        add_eval = FunctionEvaluator(+, [x_eval, y_eval])
        evaluate!(add_eval, output, data, 1, 1)
        @test output[1] ≈ 1.0 + 10.0
    end
    
    @testset "InteractionEvaluator" begin
        x_eval = ContinuousEvaluator(:x)
        y_eval = ContinuousEvaluator(:y)
        interaction = InteractionEvaluator([x_eval, y_eval])
        @test output_width(interaction) == 1  # 1×1 = 1
        
        output = Vector{Float64}(undef, 1)
        evaluate!(interaction, output, data, 1, 1)
        @test output[1] ≈ 1.0 * 10.0
    end
    
    @testset "ZScoreEvaluator" begin
        x_eval = ContinuousEvaluator(:x)
        zscore_eval = ZScoreEvaluator(x_eval, 2.5, 1.0)  # center=2.5, scale=1.0
        @test output_width(zscore_eval) == 1
        
        output = Vector{Float64}(undef, 1)
        evaluate!(zscore_eval, output, data, 1, 1)
        @test output[1] ≈ (1.0 - 2.5) / 1.0  # (x - center) / scale
    end
    
    @testset "CombinedEvaluator" begin
        x_eval = ContinuousEvaluator(:x)
        const_eval = ConstantEvaluator(1.0)
        combined = CombinedEvaluator([const_eval, x_eval])
        @test output_width(combined) == 2
        
        output = Vector{Float64}(undef, 2)
        next_idx = evaluate!(combined, output, data, 1, 1)
        @test output[1] == 1.0  # constant
        @test output[2] == 1.0  # x value
        @test next_idx == 3
    end
    
    @testset "ScaledEvaluator" begin
        x_eval = ContinuousEvaluator(:x)
        scaled = ScaledEvaluator(x_eval, 2.0)
        @test output_width(scaled) == 1
        
        output = Vector{Float64}(undef, 1)
        evaluate!(scaled, output, data, 1, 1)
        @test output[1] ≈ 1.0 * 2.0
    end
    
    @testset "ProductEvaluator" begin
        x_eval = ContinuousEvaluator(:x)
        y_eval = ContinuousEvaluator(:y)
        product = ProductEvaluator([x_eval, y_eval])
        @test output_width(product) == 1
        
        output = Vector{Float64}(undef, 1)
        evaluate!(product, output, data, 1, 1)
        @test output[1] ≈ 1.0 * 10.0
    end
    
end
