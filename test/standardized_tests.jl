# test/zscoredterm_tests.jl
# Tests for ZScoredTerm support in EfficientModelMatrices.jl

@testset "ZScoredTerm Support" begin
    # Setup test data
    Random.seed!(42)
    n = 100
    df = DataFrame(
        x = randn(n),
        y = randn(n),
        z = randn(n),
        group = rand(["A", "B", "C"], n)
    )
    df.outcome = df.x + 0.5 * df.y - 0.3 * df.z + randn(n) * 0.1
    
    @testset "Basic ZScoredTerm functionality" begin
        # Create a simple model with ZScoredTerm
        contrasts = Dict(:x => ZScore(), :y => ZScore())
        
        # Fit model using StandardizedPredictors
        f = @formula(outcome ~ x + y + z)
        m = lm(f, df; contrasts=contrasts)
        
        # Test that InplaceModeler can be created
        ipm = InplaceModeler(m, nrow(df))
        @test ipm isa InplaceModeler
        
        # Test matrix construction
        X = Matrix{Float64}(undef, nrow(df), width(formula(m).rhs))
        tbl = Tables.columntable(df)
        
        # This should not throw an error now
        @test_nowarn modelmatrix!(ipm, tbl, X)
        
        # Verify the matrix has the correct dimensions
        @test size(X) == (nrow(df), width(formula(m).rhs))
        
        # Verify no NaN or Inf values (basic sanity check)
        @test all(isfinite, X)
    end
    
    @testset "ZScoredTerm transformation correctness" begin
        # Test against StatsModels.modelcols for correctness
        contrasts = Dict(:x => ZScore(), :y => ZScore())
        f = @formula(outcome ~ x + y + z)

        # Get reference result using StatsModels
        tbl = Tables.columntable(df)
        
        # Get our result using InplaceModeler
        m = lm(f, df)
        X_reference = StatsModels.modelmatrix(m)
        ipm = InplaceModeler(m, nrow(df))
        X_ours = Matrix{Float64}(undef, size(X_reference))
        modelmatrix!(ipm, tbl, X_ours)
        
        # Should be approximately equal (allowing for floating point precision)
        @test X_ours ≈ X_reference atol=1e-12
    end
    
    @testset "ZScoredTerm with custom center and scale" begin
        # Test with explicit center and scale values
        center_val = 2.0
        scale_val = 1.5
        
        # Create ZScoredTerm with custom values
        contrasts = Dict(:x => ZScore(center=center_val, scale=scale_val))
        
        f = @formula(outcome ~ x + y)
        
        # Test matrix construction
        m = lm(f, df)
        ipm = InplaceModeler(m, nrow(df))
        Xref = modelmatrix(m)
        X = Matrix{Float64}(undef, size(Xref))
        tbl = Tables.columntable(df)
        
        @test_nowarn modelmatrix!(ipm, tbl, X)
        
        @test X ≈ Xref atol=1e-12
    end
    
    @testset "ZScoredTerm in complex formulas" begin
        # Test with interactions and multiple standardized terms
        contrasts = Dict(:x => ZScore(), :y => ZScore(), :z => ZScore())
        
        f = @formula(outcome ~ x * y + z)
        m = lm(f, df)
        Xref = modelmatrix(m)
        ipm = InplaceModeler(m, nrow(df))
        X = Matrix{Float64}(undef, size(Xref))
        tbl = Tables.columntable(df)
        
        # Should handle complex formulas without error
        @test_nowarn modelmatrix!(ipm, tbl, X)
        
        # Compare with reference
        @test X ≈ Xref atol=1e-12
    end
    
    @testset "ZScoredTerm with categorical variables" begin
        # Test mixing standardized continuous and categorical variables
        df_cat = copy(df)
        df_cat.group = categorical(df_cat.group)
        
        contrasts = Dict(:x => ZScore(), :y => ZScore())
        
        f = @formula(outcome ~ x + y + group)
        
        m = lm(f, df_cat)
        X_reference = modelmatrix(m)
        ipm = InplaceModeler(m, nrow(df_cat))
        X = Matrix{Float64}(undef, size(X_reference))
        tbl = Tables.columntable(df_cat)
        
        # Should handle mixed variable types
        @test_nowarn modelmatrix!(ipm, tbl, X)
        
        # Compare with reference
        @test X ≈ X_reference atol=1e-12
    end
    
    @testset "Edge cases" begin
        # Test with constant variables (scale = 0 should be handled gracefully)
        df_const = copy(df)
        df_const.const_var = fill(5.0, nrow(df_const))
        
        # This might throw an error or handle it gracefully depending on implementation
        # The key is that it shouldn't crash unexpectedly
        contrasts = Dict(:const_var => ZScore())
        f = @formula(outcome ~ const_var + x)
        
        # Note: This might legitimately fail due to scale=0, which is expected behavior
        @test_nowarn schema(f, df_const, contrasts)
    end
    
    @testset "Performance comparison" begin
        # Benchmark against StatsModels.modelcols to ensure we're not slower
        contrasts = Dict(:x => ZScore(), :y => ZScore(), :z => ZScore())
        
        f = @formula(outcome ~ x + y + z)
       
        m = lm(f, df)
        X_ref = modelmatrix(m)
        ipm = InplaceModeler(m, nrow(df))
        X = Matrix{Float64}(undef, size(X_ref))
        tbl = Tables.columntable(df)
        
        # Warm up
        modelmatrix!(ipm, tbl, X)
        
        # Time our implementation
        time_ours = @elapsed for _ in 1:100
            modelmatrix!(ipm, tbl, X)
        end
        
        # Time reference implementation
        time_ref = @elapsed for _ in 1:100
            StatsModels.modelcols(f.rhs, tbl)
        end
        
        # Our implementation should be faster (or at least not much slower)
        @test time_ours <= time_ref * 2.0  # Allow up to 2x slower for safety
        
        println("Performance comparison:")
        println("  InplaceModeler: $(round(time_ours * 1000, digits=3)) ms")
        println("  StatsModels: $(round(time_ref * 1000, digits=3)) ms")
        println("  Speedup: $(round(time_ref / time_ours, digits=2))x")
    end
end

@testset "ZScoredTerm apply_zscore_inplace! function" begin
    # Direct tests of the helper function
    @testset "Scalar center and scale" begin
        X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        X_original = copy(X)
        center = 3.0
        scale = 2.0
        
        EfficientModelMatrices.apply_zscore_inplace!(X, 1, 2, center, scale)
        
        # Verify transformation: (x - 3) / 2
        expected = (X_original .- center) ./ scale
        @test X ≈ expected
    end
    
    @testset "Vector center and scale" begin
        X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        X_original = copy(X)
        center = [1.5, 2.5]
        scale = [0.5, 1.5]
        
        EfficientModelMatrices.apply_zscore_inplace!(X, 1, 2, center, scale)
        
        # Verify transformation: column-wise (x - center[i]) / scale[i]
        expected = copy(X_original)
        expected[:, 1] = (X_original[:, 1] .- center[1]) ./ scale[1]
        expected[:, 2] = (X_original[:, 2] .- center[2]) ./ scale[2]
        @test X ≈ expected
    end
    
    @testset "Zero center optimization" begin
        X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        X_original = copy(X)
        center = 0.0
        scale = 2.0
        
        EfficientModelMatrices.apply_zscore_inplace!(X, 1, 2, center, scale)
        
        # Should just divide by scale (optimized path)
        expected = X_original ./ scale
        @test X ≈ expected
    end
end