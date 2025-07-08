# test_column_mapping.jl - Comprehensive tests for column mapping functionality

@testset "Column Mapping Tests" begin

    @testset "Basic Term Types" begin
        # Test individual term types work correctly
        
        # Simple continuous term
        t1 = ContinuousTerm(:x, 0.0, 1.0, -1.0, 1.0)
        mapping1 = build_column_mapping(t1)
        
        @test mapping1.total_columns == 1
        @test haskey(mapping1.symbol_to_ranges, :x)
        @test mapping1.symbol_to_ranges[:x] == [1:1]
        @test get_all_variable_columns(mapping1, :x) == [1]
        
        # Categorical term (width = 2 for 3-level factor with dummy coding)
        contrasts = StatsModels.DummyCoding()
        levels = ["a", "b", "c"]
        contrasts_matrix = StatsModels.ContrastsMatrix(contrasts, levels)
        t2 = CategoricalTerm(:z, contrasts_matrix)
        mapping2 = build_column_mapping(t2)
        
        @test mapping2.total_columns == 2
        @test haskey(mapping2.symbol_to_ranges, :z)
        @test mapping2.symbol_to_ranges[:z] == [1:2]
        @test get_all_variable_columns(mapping2, :z) == [1, 2]
        
        # Intercept term
        t3 = InterceptTerm{true}()
        mapping3 = build_column_mapping(t3)
        
        @test mapping3.total_columns == 1
        @test isempty(mapping3.symbol_to_ranges)  # No variables in intercept
        
        # Constant term
        t4 = ConstantTerm(1)
        mapping4 = build_column_mapping(t4)
        
        @test mapping4.total_columns == 1
        @test isempty(mapping4.symbol_to_ranges)  # No variables in constant
    end

    @testset "Matrix Terms and Combinations" begin
        # Test combinations of terms in MatrixTerm
        
        t_intercept = InterceptTerm{true}()
        t_x = ContinuousTerm(:x, 0.0, 1.0, -1.0, 1.0)
        t_y = ContinuousTerm(:y, 0.0, 1.0, -1.0, 1.0)
        
        # Create a MatrixTerm with multiple components
        matrix_term = MatrixTerm((t_intercept, t_x, t_y))
        mapping = build_column_mapping(matrix_term)
        
        @test mapping.total_columns == 3  # intercept + x + y
        @test get_all_variable_columns(mapping, :x) == [2]
        @test get_all_variable_columns(mapping, :y) == [3]
        
        # Check that column ranges don't overlap
        x_ranges = get_variable_ranges(mapping, :x)
        y_ranges = get_variable_ranges(mapping, :y)
        @test length(x_ranges) == 1
        @test length(y_ranges) == 1
        @test x_ranges[1] != y_ranges[1]  # Different ranges
        
        # Test with categorical variable
        contrasts = StatsModels.DummyCoding()
        levels = ["low", "high"]
        contrasts_matrix = StatsModels.ContrastsMatrix(contrasts, levels)
        t_cat = CategoricalTerm(:group, contrasts_matrix)
        
        matrix_term2 = MatrixTerm((t_intercept, t_x, t_cat))
        mapping2 = build_column_mapping(matrix_term2)
        
        @test mapping2.total_columns == 3  # intercept + x + group (1 dummy)
        @test get_all_variable_columns(mapping2, :x) == [2]
        @test get_all_variable_columns(mapping2, :group) == [3]
    end

    @testset "Function Terms and Nested Calls" begin
        # Test function terms with various complexities
        
        # Simple function: log(x)
        t_x = ContinuousTerm(:x, 0.0, 1.0, -1.0, 1.0)
        f_term1 = FunctionTerm(log, [t_x], :(log(x)))
        mapping1 = build_column_mapping(f_term1)
        
        @test mapping1.total_columns == 1
        @test haskey(mapping1.symbol_to_ranges, :x)
        @test get_all_variable_columns(mapping1, :x) == [1]
        
        # Nested function: log(1 + x)
        t_const = ConstantTerm(1)
        inner_sum = FunctionTerm(+, [t_const, t_x], :(1 + x))
        f_term2 = FunctionTerm(log, [inner_sum], :(log(1 + x)))
        mapping2 = build_column_mapping(f_term2)
        
        @test mapping2.total_columns == 1
        @test haskey(mapping2.symbol_to_ranges, :x)
        @test get_all_variable_columns(mapping2, :x) == [1]
        
        # Multiple variables in function: x + y
        t_y = ContinuousTerm(:y, 0.0, 1.0, -1.0, 1.0)
        f_term3 = FunctionTerm(+, [t_x, t_y], :(x + y))
        mapping3 = build_column_mapping(f_term3)
        
        @test mapping3.total_columns == 1
        @test haskey(mapping3.symbol_to_ranges, :x)
        @test haskey(mapping3.symbol_to_ranges, :y)
        @test get_all_variable_columns(mapping3, :x) == [1]
        @test get_all_variable_columns(mapping3, :y) == [1]  # Same column since it's one function
        
        # Test that both variables are detected in the same term
        x_terms = get_terms_involving_variable(mapping3, :x)
        y_terms = get_terms_involving_variable(mapping3, :y)
        @test length(x_terms) == 1
        @test length(y_terms) == 1
        @test x_terms[1][2] == y_terms[1][2]  # Same range
    end

    @testset "Interaction Terms" begin
        # Test various interaction patterns
        
        t_x = ContinuousTerm(:x, 0.0, 1.0, -1.0, 1.0)
        t_y = ContinuousTerm(:y, 0.0, 1.0, -1.0, 1.0)
        
        # Simple interaction: x & y
        interaction1 = InteractionTerm((t_x, t_y))
        mapping1 = build_column_mapping(interaction1)
        
        @test mapping1.total_columns == 1  # x*y creates 1 column (both continuous)
        @test haskey(mapping1.symbol_to_ranges, :x)
        @test haskey(mapping1.symbol_to_ranges, :y)
        @test get_all_variable_columns(mapping1, :x) == [1]
        @test get_all_variable_columns(mapping1, :y) == [1]
        
        # Interaction with categorical
        contrasts = StatsModels.DummyCoding()
        levels = ["low", "med", "high"]
        contrasts_matrix = StatsModels.ContrastsMatrix(contrasts, levels)
        t_cat = CategoricalTerm(:group, contrasts_matrix)
        
        interaction2 = InteractionTerm((t_x, t_cat))
        mapping2 = build_column_mapping(interaction2)
        
        @test mapping2.total_columns == 2  # x * group(3 levels) = 2 columns
        @test haskey(mapping2.symbol_to_ranges, :x)
        @test haskey(mapping2.symbol_to_ranges, :group)
        @test get_all_variable_columns(mapping2, :x) == [1, 2]
        @test get_all_variable_columns(mapping2, :group) == [1, 2]
        
        # Three-way interaction
        t_z = ContinuousTerm(:z, 0.0, 1.0, -1.0, 1.0)
        interaction3 = InteractionTerm((t_x, t_y, t_z))
        mapping3 = build_column_mapping(interaction3)
        
        @test mapping3.total_columns == 1  # x*y*z = 1 column (all continuous)
        @test get_all_variable_columns(mapping3, :x) == [1]
        @test get_all_variable_columns(mapping3, :y) == [1]
        @test get_all_variable_columns(mapping3, :z) == [1]
    end

    @testset "Complex Formula with Multiple Variable Appearances" begin
        # Test the complex case: variable appears in multiple different terms
        # Simulating: x + x^2 + inv(x) & a + inv(x) & a & b
        
        t_x = ContinuousTerm(:x, 0.0, 1.0, -1.0, 1.0)
        t_a = ContinuousTerm(:a, 0.0, 1.0, -1.0, 1.0)
        t_b = ContinuousTerm(:b, 0.0, 1.0, -1.0, 1.0)
        
        # x term (simple)
        term1 = t_x
        
        # x^2 term (function)
        term2 = FunctionTerm(^, [t_x, ConstantTerm(2)], :(x^2))
        
        # inv(x) function
        inv_x = FunctionTerm(inv, [t_x], :(inv(x)))
        
        # inv(x) & a interaction
        term3 = InteractionTerm((inv_x, t_a))
        
        # inv(x) & a & b interaction  
        term4 = InteractionTerm((inv_x, t_a, t_b))
        
        # a term (simple)
        term5 = t_a
        
        # Combine all terms
        complex_matrix = MatrixTerm((term1, term2, term3, term4, term5))
        mapping = build_column_mapping(complex_matrix)
        
        # Check that x appears in multiple ranges
        x_ranges = get_variable_ranges(mapping, :x)
        @test length(x_ranges) >= 3  # At least in x, x^2, inv(x)&a, inv(x)&a&b
        
        # Check that a appears in multiple ranges
        a_ranges = get_variable_ranges(mapping, :a)
        @test length(a_ranges) >= 3  # At least in a, inv(x)&a, inv(x)&a&b
        
        # Check that b only appears in one interaction
        b_ranges = get_variable_ranges(mapping, :b)
        @test length(b_ranges) == 1  # Only in inv(x)&a&b
        
        # Verify total columns is reasonable
        @test mapping.total_columns >= 5  # At least x + x^2 + inv(x)&a + inv(x)&a&b + a
        
        # Test analysis structure
        analysis = analyze_formula_structure(mapping)
        @test haskey(analysis, :x)
        @test haskey(analysis, :a)
        @test haskey(analysis, :b)
        
        @test analysis[:x]["appears_in_terms"] >= 3
        @test analysis[:a]["appears_in_terms"] >= 3
        @test analysis[:b]["appears_in_terms"] == 1
        
        # Verify we can get terms involving each variable
        x_term_info = get_terms_involving_variable(mapping, :x)
        @test length(x_term_info) >= 3
        
        # Check that column indices don't have gaps or overlaps inappropriately
        all_x_cols = get_all_variable_columns(mapping, :x)
        all_a_cols = get_all_variable_columns(mapping, :a)
        @test length(all_x_cols) >= 3  # x participates in multiple columns
        @test length(all_a_cols) >= 3  # a participates in multiple columns
        
        # Verify columns are within bounds
        @test all(1 ≤ col ≤ mapping.total_columns for col in all_x_cols)
        @test all(1 ≤ col ≤ mapping.total_columns for col in all_a_cols)
    end

end
