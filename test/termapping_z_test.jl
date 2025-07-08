# test_zscoredterm_integration.jl - Test ZScoredTerm support in Strategy 4

@testset "ZScoredTerm Integration Tests" begin
    # Create test data
    n = 100
    df = DataFrame(
        outcome = randn(n),
        x = randn(n) .* 2 .+ 3,  # mean=3, std≈2
        y = randn(n) .* 0.5 .+ 1, # mean=1, std≈0.5
        z = categorical(rand(["A", "B", "C"], n))
    )
    
    @testset "Basic ZScoredTerm with EfficientModelMatrices" begin
        # Test with automatic center and scale
        contrasts = Dict(:x => ZScore())
        f = @formula(outcome ~ x + y + z)
        
        # Create schema with ZScore contrasts
        sch = schema(f, df, contrasts)
        f_applied = apply_schema(f, sch)
        
        # Fit model
        m = lm(f_applied, df)
        
        # Test that model matrix construction works
        ipm = InplaceModeler(m, nrow(df))
        Xref = modelmatrix(m)
        X = Matrix{Float64}(undef, size(Xref))
        tbl = Tables.columntable(df)
        
        @test_nowarn modelmatrix!(ipm, tbl, X)
        @test X ≈ Xref atol=1e-12
        
        # Test that column mapping works
        mapping = build_enhanced_mapping(m)
        @test haskey(mapping.symbol_to_ranges, :x)
        @test haskey(mapping.symbol_to_ranges, :y)
        @test haskey(mapping.symbol_to_ranges, :z)
        
        # Test that affected columns are computed correctly
        x_cols = get_all_variable_columns(mapping, :x)
        @test !isempty(x_cols)
        @test all(col -> 1 <= col <= size(Xref, 2), x_cols)
    end
    
    @testset "ZScoredTerm with custom center and scale" begin
        # Test with explicit center and scale values
        center_val = 2.0
        scale_val = 1.5
        
        # Create ZScoredTerm with custom values
        contrasts = Dict(:x => ZScore(center=center_val, scale=scale_val))
        f = @formula(outcome ~ x + y)
        
        # Create schema and apply
        sch = schema(f, df, contrasts)
        f_applied = apply_schema(f, sch)
        
        # Test matrix construction
        m = lm(f_applied, df)
        ipm = InplaceModeler(m, nrow(df))
        Xref = modelmatrix(m)
        X = Matrix{Float64}(undef, size(Xref))
        tbl = Tables.columntable(df)
        
        @test_nowarn modelmatrix!(ipm, tbl, X)
        @test X ≈ Xref atol=1e-12
        
        # Verify Z-score transformation was applied correctly
        x_col_idx = 2  # Assuming intercept is column 1, x is column 2
        x_transformed = X[:, x_col_idx]
        x_expected = (df.x .- center_val) ./ scale_val
        @test x_transformed ≈ x_expected atol=1e-12
    end
    
    @testset "Strategy 4 AME with ZScoredTerm" begin
        # Test that Strategy 4 AME computation works with ZScoredTerm
        contrasts = Dict(:x => ZScore())
        f = @formula(outcome ~ x + y + z)
        
        sch = schema(f, df, contrasts)
        f_applied = apply_schema(f, sch)
        m = lm(f_applied, df)
        
        # Test AME computation
        @test_nowarn margins(m, :x, df)
        
        # Test that results are sensible
        result = margins(m, :x, df)
        @test haskey(result.result, :x)
        @test result.result[:x] isa Real
        @test isfinite(result.result[:x])
        
        # Test multiple variables
        result_multi = margins(m, [:x, :y], df)
        @test haskey(result_multi.result, :x)
        @test haskey(result_multi.result, :y)
        @test result_multi.result[:x] isa Real
        @test result_multi.result[:y] isa Real
    end
    
    @testset "Column mapping with ZScoredTerm interactions" begin
        # Test more complex formulas with interactions
        contrasts = Dict(:x => ZScore(), :y => ZScore())
        f = @formula(outcome ~ x + y + x&y + z)
        
        sch = schema(f, df, contrasts)
        f_applied = apply_schema(f, sch)
        m = lm(f_applied, df)
        
        # Test column mapping
        mapping = build_enhanced_mapping(m)
        
        # Both x and y should affect the interaction term
        x_cols = get_all_variable_columns(mapping, :x)
        y_cols = get_all_variable_columns(mapping, :y)
        
        @test !isempty(x_cols)
        @test !isempty(y_cols)
        
        # Check that interaction columns are included
        @test length(x_cols) >= 2  # At least main effect + interaction
        @test length(y_cols) >= 2  # At least main effect + interaction
        
        # Test AME computation with interactions
        @test_nowarn margins(m, :x, df)
        result = margins(m, :x, df)
        @test isfinite(result.result[:x])
    end
    
    @testset "Phase 1 evaluation with ZScoredTerm" begin
        # Test single column evaluation
        contrasts = Dict(:x => ZScore(center=0.0, scale=2.0))
        f = @formula(outcome ~ x + y)
        
        sch = schema(f, df, contrasts)
        f_applied = apply_schema(f, sch)
        m = lm(f_applied, df)
        
        mapping = build_enhanced_mapping(m)
        tbl = Tables.columntable(df)
        
        # Get the ZScoredTerm for x
        x_cols = get_all_variable_columns(mapping, :x)
        @test !isempty(x_cols)
        
        x_col = first(x_cols)
        term, local_col = get_term_for_column(mapping, x_col)
        
        # Test single column evaluation
        output = Vector{Float64}(undef, nrow(df))
        @test_nowarn evaluate_single_column!(term, tbl, x_col, local_col, output)
        
        # Verify the transformation was applied
        if isa(term, ZScoredTerm)
            expected = (df.x .- term.center) ./ term.scale
            @test output ≈ expected atol=1e-12
        end
    end
    
    @testset "Efficiency analysis with ZScoredTerm" begin
        # Test that efficiency statistics work
        contrasts = Dict(:x => ZScore())
        f = @formula(outcome ~ x + y + z + x&z)
        
        sch = schema(f, df, contrasts)
        f_applied = apply_schema(f, sch)
        m = lm(f_applied, df)
        
        # Create workspace
        ws = AMEWorkspace(nrow(df), width(fixed_effects_form(m).rhs), df, m)
        
        # Test efficiency reporting
        @test haskey(ws.affected_cols_cache, :x)
        @test haskey(ws.efficiency_stats, :x)
        
        stats = get_computation_stats(ws, :x)
        @test stats.variable == :x
        @test stats.affected_columns > 0
        @test stats.total_columns > 0
        @test 0.0 < stats.efficiency_ratio <= 1.0
        @test stats.estimated_speedup >= 1.0
    end
    
    @testset "Error handling with ZScoredTerm" begin
        # Test graceful handling of edge cases
        
        # Very small scale value
        contrasts = Dict(:x => ZScore(center=0.0, scale=1e-10))
        f = @formula(outcome ~ x)
        
        sch = schema(f, df, contrasts)
        f_applied = apply_schema(f, sch)
        
        # Should not error during model fitting
        @test_nowarn m = lm(f_applied, df)
        m = lm(f_applied, df)
        
        # AME computation should handle numerical issues gracefully
        result = margins(m, :x, df)
        # Result might be Inf or very large, but should not error
        @test result.result[:x] isa Real
    end
end

@testset "ZScoredTerm Performance Comparison" begin
    # Compare Strategy 4 performance with and without ZScoredTerm
    
    # Create larger dataset for meaningful timing
    n = 1000
    p_vars = 20
    
    # Create data with many variables
    data_dict = Dict(:outcome => randn(n))
    for i in 1:p_vars
        data_dict[Symbol("x$i")] = randn(n)
    end
    df_large = DataFrame(data_dict)
    
    # Formula with some ZScoredTerms
    terms_regular = [term(Symbol("x$i")) for i in 1:10]
    terms_zscore = [term(Symbol("x$i")) for i in 11:20]
    f_complex = term(:outcome) ~ sum(terms_regular) + sum(terms_zscore)
    
    # Create contrasts for Z-scoring
    contrasts = Dict(Symbol("x$i") => ZScore() for i in 11:20)
    
    sch = schema(f_complex, df_large, contrasts)
    f_applied = apply_schema(f_complex, sch)
    m = lm(f_applied, df_large)
    
    # Test that Strategy 4 works efficiently
    @test_nowarn margins(m, Symbol("x15"), df_large)  # ZScored variable
    @test_nowarn margins(m, Symbol("x5"), df_large)   # Regular variable
    
    # Get efficiency stats
    ws = AMEWorkspace(nrow(df_large), width(fixed_effects_form(m).rhs), df_large, m)
    
    zscore_stats = get_computation_stats(ws, Symbol("x15"))
    regular_stats = get_computation_stats(ws, Symbol("x5"))
    
    @test zscore_stats.strategy_used == "selective"
    @test regular_stats.strategy_used == "selective"
    
    # Both should show efficiency gains
    @test zscore_stats.estimated_speedup > 1.0
    @test regular_stats.estimated_speedup > 1.0
    
    println("ZScored variable efficiency: $(zscore_stats.efficiency_ratio) ($(zscore_stats.estimated_speedup)x speedup)")
    println("Regular variable efficiency: $(regular_stats.efficiency_ratio) ($(regular_stats.estimated_speedup)x speedup)")
end