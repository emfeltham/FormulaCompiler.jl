# test_standardized_predictors.jl - Integration tests for StandardizedPredictors.jl

using StandardizedPredictors
import StandardizedPredictors.ZScoredTerm

# Required StatsModels interface methods for ZScoredTerm
StatsModels.width(t::ZScoredTerm) = StatsModels.width(t.term)
StatsModels.termvars(t::ZScoredTerm) = StatsModels.termvars(t.term)

# Mock constructor for testing
function zscore_term(term, center, scale)
    return ZScoredTerm(term, center, scale)
end

@testset "StandardizedPredictors Integration Tests" begin

    @testset "Basic ZScoredTerm Detection" begin
        # Create a simple ZScoredTerm
        base_term = ContinuousTerm(:x, 5.0, 2.0, 1.0, 10.0)
        zscored = zscore_term(base_term, 5.0, 2.0)
        
        # Test variable collection
        vars = collect_termvars_recursive(zscored)
        @test :x in vars
        @test length(vars) == 1
        
        # Test column mapping
        mapping = build_column_mapping(zscored)
        @test mapping.total_columns == 1
        @test haskey(mapping.symbol_to_ranges, :x)
        @test mapping.symbol_to_ranges[:x] == [1:1]
        
        # Test variable extraction
        x_ranges = get_variable_ranges(mapping, :x)
        @test length(x_ranges) == 1
        @test x_ranges[1] == 1:1
    end

    @testset "ZScoredTerm in Complex Formula" begin
        # Create terms: intercept + x + zscore(y) + z + zscore(x) & z
        intercept = InterceptTerm{true}()
        x_term = ContinuousTerm(:x, 0.0, 1.0, -2.0, 2.0)
        y_term = ContinuousTerm(:y, 0.0, 1.0, -2.0, 2.0)
        z_term = ContinuousTerm(:z, 0.0, 1.0, -2.0, 2.0)
        
        # Create standardized versions
        y_zscored = zscore_term(y_term, 0.0, 1.0)
        x_zscored = zscore_term(x_term, 0.0, 1.0)
        
        # Create interaction with standardized term
        interaction = InteractionTerm((x_zscored, z_term))
        
        # Combine into matrix term
        matrix_term = MatrixTerm((intercept, x_term, y_zscored, z_term, interaction))
        mapping = build_column_mapping(matrix_term)
        
        # Should have: intercept(1) + x(1) + zscore(y)(1) + z(1) + zscore(x)&z(1) = 5 columns
        @test mapping.total_columns == 5
        
        # Check variable detection
        @test haskey(mapping.symbol_to_ranges, :x)
        @test haskey(mapping.symbol_to_ranges, :y)
        @test haskey(mapping.symbol_to_ranges, :z)
        
        # x should appear in two places: x term and zscore(x)&z interaction
        x_ranges = get_variable_ranges(mapping, :x)
        @test length(x_ranges) == 2
        
        # y should appear only in zscore(y) term
        y_ranges = get_variable_ranges(mapping, :y)
        @test length(y_ranges) == 1
        
        # z should appear in two places: z term and zscore(x)&z interaction
        z_ranges = get_variable_ranges(mapping, :z)
        @test length(z_ranges) == 2
        
        # Test detailed analysis
        analysis = analyze_formula_structure(mapping)
        @test analysis[:x]["appears_in_terms"] == 2
        @test analysis[:y]["appears_in_terms"] == 1
        @test analysis[:z]["appears_in_terms"] == 2
        
        # Test term information extraction
        x_terms = get_terms_involving_variable(mapping, :x)
        @test length(x_terms) == 2
        
        # One should be the regular ContinuousTerm, one should involve ZScoredTerm
        term_types = [typeof(term) for (term, range) in x_terms]
        @test any(t -> t <: ContinuousTerm, term_types)  # regular x term
        @test any(t -> t <: InteractionTerm, term_types)  # zscore(x) & z interaction
    end

    @testset "Multiple ZScoredTerms Same Variable" begin
        # Test case: x + zscore(x, center=0, scale=1) + zscore(x, center=5, scale=2)
        # This represents the same variable standardized in different ways
        
        base_term = ContinuousTerm(:x, 2.5, 4.0, 0.0, 5.0)
        zscored1 = zscore_term(base_term, 0.0, 1.0)  # standard normal
        zscored2 = zscore_term(base_term, 5.0, 2.0)  # different centering/scaling
        
        matrix_term = MatrixTerm((base_term, zscored1, zscored2))
        mapping = build_column_mapping(matrix_term)
        
        @test mapping.total_columns == 3
        @test haskey(mapping.symbol_to_ranges, :x)
        
        # x should appear in 3 ranges (original + 2 standardized versions)
        x_ranges = get_variable_ranges(mapping, :x)
        @test length(x_ranges) == 3
        
        # All ranges should be single columns and non-overlapping
        all_x_cols = get_all_variable_columns(mapping, :x)
        @test length(all_x_cols) == 3
        @test all_x_cols == [1, 2, 3]  # Should be consecutive
        
        # Test term analysis
        x_terms = get_terms_involving_variable(mapping, :x)
        @test length(x_terms) == 3
        
        # Should have one ContinuousTerm and two ZScoredTerms
        term_types = [typeof(term) for (term, range) in x_terms]
        @test count(t -> t <: ContinuousTerm, term_types) == 1
        @test count(t -> t <: ZScoredTerm, term_types) == 2
    end

    @testset "Nested ZScoredTerms in Functions and Interactions" begin
        # Test complex case: log(zscore(x)) + zscore(y) & group + zscore(x) & zscore(y)
        
        x_term = ContinuousTerm(:x, 0.0, 1.0, -3.0, 3.0)
        y_term = ContinuousTerm(:y, 0.0, 1.0, -3.0, 3.0)
        
        # Create group term (categorical)
        contrasts = StatsModels.DummyCoding()
        levels = ["A", "B"]
        contrasts_matrix = StatsModels.ContrastsMatrix(contrasts, levels)
        group_term = CategoricalTerm(:group, contrasts_matrix)
        
        # Create standardized terms
        x_zscored = zscore_term(x_term, 0.0, 1.0)
        y_zscored = zscore_term(y_term, 0.0, 1.0)
        
        # Create function term with nested ZScoredTerm: log(zscore(x))
        log_zscored_x = FunctionTerm(log, [x_zscored], :(log(zscore(x))))
        
        # Create interactions
        interaction1 = InteractionTerm((y_zscored, group_term))  # zscore(y) & group
        interaction2 = InteractionTerm((x_zscored, y_zscored))   # zscore(x) & zscore(y)
        
        # Combine all terms
        complex_term = MatrixTerm((log_zscored_x, interaction1, interaction2))
        mapping = build_column_mapping(complex_term)
        
        # Should detect all variables
        @test haskey(mapping.symbol_to_ranges, :x)
        @test haskey(mapping.symbol_to_ranges, :y)
        @test haskey(mapping.symbol_to_ranges, :group)
        
        # x appears in: log(zscore(x)) + zscore(x) & zscore(y)
        x_ranges = get_variable_ranges(mapping, :x)
        @test length(x_ranges) == 2
        
        # y appears in: zscore(y) & group + zscore(x) & zscore(y)
        y_ranges = get_variable_ranges(mapping, :y)
        @test length(y_ranges) == 2
        
        # group appears in: zscore(y) & group
        group_ranges = get_variable_ranges(mapping, :group)
        @test length(group_ranges) == 1
        
        # Test that nested ZScoredTerms are properly detected
        x_terms = get_terms_involving_variable(mapping, :x)
        @test length(x_terms) >= 2
        
        # Should include both FunctionTerm and InteractionTerm
        term_types = [typeof(term) for (term, range) in x_terms]
        @test any(t -> t <: FunctionTerm, term_types)      # log(zscore(x))
        @test any(t -> t <: InteractionTerm, term_types)   # zscore(x) & zscore(y)
        
        # Test analysis
        analysis = analyze_formula_structure(mapping)
        @test analysis[:x]["appears_in_terms"] == 2
        @test analysis[:y]["appears_in_terms"] == 2
        @test analysis[:group]["appears_in_terms"] == 1
    end

    @testset "Real Modeling Workflow with StandardizedPredictors" begin
        # Simulate a realistic workflow where some predictors are standardized
        Random.seed!(12345)
        n = 100
        
        df = DataFrame(
            x = 3 .* randn(n),           # Will be standardized
            y = 2 .* randn(n) .+ 5,      # Will be standardized  
            z = randn(n),                # Will remain raw
            group = categorical(rand(["Control", "Treatment"], n))
        )
        
        # Create response with known relationships
        df.outcome = 1.0 .+ 
                    0.8 .* ((df.x .- mean(df.x)) ./ std(df.x)) .+     # standardized x effect
                    0.5 .* df.z .+                                     # raw z effect
                    (df.group .== "Treatment") .* 0.7 .+               # group effect
                    0.3 .* ((df.y .- mean(df.y)) ./ std(df.y)) .* df.z .+  # standardized y * z interaction
                    0.1 .* randn(n)
        
        # Create terms manually (simulating what would happen with StandardizedPredictors.jl)
        x_std = zscore_term(ContinuousTerm(:x, mean(df.x), var(df.x), minimum(df.x), maximum(df.x)), 
                           mean(df.x), std(df.x))
        y_std = zscore_term(ContinuousTerm(:y, mean(df.y), var(df.y), minimum(df.y), maximum(df.y)), 
                           mean(df.y), std(df.y))
        z_raw = ContinuousTerm(:z, mean(df.z), var(df.z), minimum(df.z), maximum(df.z))
        
        # Create group term
        group_levels = levels(df.group)
        contrasts = StatsModels.DummyCoding()
        contrasts_matrix = StatsModels.ContrastsMatrix(contrasts, group_levels)
        group_term = CategoricalTerm(:group, contrasts_matrix)
        
        # Create interaction: zscore(y) & z
        interaction = InteractionTerm((y_std, z_raw))
        
        # Full model: intercept + zscore(x) + z + group + zscore(y) & z
        intercept = InterceptTerm{true}()
        full_rhs = MatrixTerm((intercept, x_std, z_raw, group_term, interaction))
        
        # Test column mapping
        mapping = build_column_mapping(full_rhs)
        
        # Should have: intercept(1) + zscore(x)(1) + z(1) + group(1) + zscore(y)&z(1) = 5 columns
        @test mapping.total_columns == 5
        
        # Test variable detection
        @test haskey(mapping.symbol_to_ranges, :x)
        @test haskey(mapping.symbol_to_ranges, :y)
        @test haskey(mapping.symbol_to_ranges, :z)
        @test haskey(mapping.symbol_to_ranges, :group)
        
        # x should appear only in zscore(x) term
        x_ranges = get_variable_ranges(mapping, :x)
        @test length(x_ranges) == 1
        
        # y should appear only in zscore(y) & z interaction
        y_ranges = get_variable_ranges(mapping, :y)
        @test length(y_ranges) == 1
        
        # z should appear in z term and zscore(y) & z interaction
        z_ranges = get_variable_ranges(mapping, :z)
        @test length(z_ranges) == 2
        
        # Test that this would work for margins computation
        # If we want marginal effects of x, only need to recompute column 2
        x_cols = get_all_variable_columns(mapping, :x)
        @test x_cols == [2]  # Only affects one column
        
        # If we want marginal effects of z, need to recompute columns 3 and 5
        z_cols = get_all_variable_columns(mapping, :z)
        @test Set(z_cols) == Set([3, 5])  # Affects z main effect and interaction
        
        # Calculate efficiency for margins computation
        total_cols = mapping.total_columns
        x_efficiency = 1.0 - (length(x_cols) / total_cols)
        z_efficiency = 1.0 - (length(z_cols) / total_cols)
        
        @test x_efficiency == 0.8  # 80% savings for x (1 out of 5 columns)
        @test z_efficiency == 0.6  # 60% savings for z (2 out of 5 columns)
        
        println("Standardized predictors efficiency test:")
        println("  x marginal effects: $(round(x_efficiency * 100))% computational savings")
        println("  z marginal effects: $(round(z_efficiency * 100))% computational savings")
        
        # Test detailed analysis
        analysis = analyze_formula_structure(mapping)
        @test analysis[:x]["appears_in_terms"] == 1
        @test analysis[:y]["appears_in_terms"] == 1
        @test analysis[:z]["appears_in_terms"] == 2
        @test analysis[:group]["appears_in_terms"] == 1
        
        # Verify term types are correctly identified
        x_terms = get_terms_involving_variable(mapping, :x)
        y_terms = get_terms_involving_variable(mapping, :y)
        z_terms = get_terms_involving_variable(mapping, :z)
        
        @test length(x_terms) == 1
        @test length(y_terms) == 1
        @test length(z_terms) == 2
        
        # x should involve a ZScoredTerm
        @test any(typeof(term) <: ZScoredTerm for (term, range) in x_terms)
        
        # z should involve both ContinuousTerm and InteractionTerm
        z_term_types = [typeof(term) for (term, range) in z_terms]
        @test any(t -> t <: ContinuousTerm, z_term_types)
        @test any(t -> t <: InteractionTerm, z_term_types)
    end

end
