# test/test_modelrow.jl
# Tests for modelrow! and modelrow interfaces

@testset "ModelRow Interfaces" begin
    df = DataFrame(
        x = randn(50),
        y = randn(50),
        z = abs.(randn(50)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 50))
    )
    data = Tables.columntable(df)
    model = lm(@formula(y ~ x * group + log(z)), df)
    
    @testset "Zero-Allocation modelrow!" begin
        # Test with pre-compiled formula
        compiled = compile_formula(model)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        # Test single row evaluation
        result = modelrow!(row_vec, compiled, data, 1)
        @test result === row_vec  # Returns same vector
        
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test zero allocations
        allocs = @allocated modelrow!(row_vec, compiled, data, 1)
        @test allocs == 0
    end
    
    @testset "Convenient modelrow! with Model" begin
        row_vec = Vector{Float64}(undef, size(modelmatrix(model), 2))
        
        # Test cached version
        result = modelrow!(row_vec, model, data, 1; cache=true)
        @test result === row_vec
        
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test non-cached version
        result = modelrow!(row_vec, model, data, 1; cache=false)
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "Allocating modelrow" begin
        # Test single row
        result = modelrow(model, data, 1)
        @test result isa Vector{Float64}
        
        expected = modelmatrix(model)[1, :]
        @test isapprox(result, expected, rtol=1e-12)
        
        # Test multiple rows
        result = modelrow(model, data, [1, 2, 3])
        @test result isa Matrix{Float64}
        @test size(result) == (3, size(modelmatrix(model), 2))
        
        expected = modelmatrix(model)[1:3, :]
        @test isapprox(result, expected, rtol=1e-12)
    end
    
    @testset "ModelRowEvaluator" begin
        # Test object-based interface
        evaluator = ModelRowEvaluator(model, df);
        
        # Test single evaluation
        result = evaluator(1);
        @test result isa Vector{Float64}
        
        expected = modelmatrix(model)[1, :]
        @test isapprox(result, expected, rtol=1e-12)
        
        # Test zero allocations
        allocs = @allocated evaluator(1);
        @test allocs == 0
        
        # Test evaluation into provided vector
        row_vec = Vector{Float64}(undef, length(expected))
        result = evaluator(row_vec, 1);
        @test result === row_vec
        @test isapprox(result, expected, rtol=1e-12)
    end
    
    @testset "Cache Management" begin
        # Test cache clearing
        model1 = lm(@formula(y ~ x), df)
        model2 = lm(@formula(y ~ x + z), df)
        
        # Use models to populate cache
        row_vec1 = Vector{Float64}(undef, 2)
        row_vec2 = Vector{Float64}(undef, 3)
        
        modelrow!(row_vec1, model1, data, 1)
        modelrow!(row_vec2, model2, data, 1)
        
        # Clear cache
        clear_model_cache!()
        
        # Should still work (will recompile)
        modelrow!(row_vec1, model1, data, 1)
        expected = modelmatrix(model1)[1, :]
        @test isapprox(row_vec1, expected, rtol=1e-12)
    end
    
    @testset "Error Handling" begin
        compiled = compile_formula(model)
        
        # Test wrong vector size
        small_vec = Vector{Float64}(undef, 1)
        @test_throws AssertionError modelrow!(small_vec, compiled, data, 1)
        
        # Test invalid row index
        row_vec = Vector{Float64}(undef, length(compiled))
        @test_throws AssertionError modelrow!(row_vec, compiled, data, 1000)
    end
    
    @testset "Consistency Across Interfaces" begin
        # Test that all interfaces give same results
        row_idx = 5
        
        # Pre-compiled version
        compiled = compile_formula(model)
        row_vec1 = Vector{Float64}(undef, length(compiled))
        modelrow!(row_vec1, compiled, data, row_idx)
        
        # Cached model version
        row_vec2 = Vector{Float64}(undef, length(compiled))
        modelrow!(row_vec2, model, data, row_idx; cache=true)
        
        # Non-cached model version
        row_vec3 = Vector{Float64}(undef, length(compiled))
        modelrow!(row_vec3, model, data, row_idx; cache=false)
        
        # Allocating version
        row_vec4 = modelrow(model, data, row_idx)
        
        # Object-based version
        evaluator = ModelRowEvaluator(model, df)
        row_vec5 = evaluator(row_idx)
        
        # All should be identical
        @test row_vec1 == row_vec2 == row_vec3 == row_vec4 == row_vec5
        
        # And match original model matrix
        expected = modelmatrix(model)[row_idx, :]
        @test isapprox(row_vec1, expected, rtol=1e-12)
    end
    
end
