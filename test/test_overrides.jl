# test_overrides.jl
# Comprehensive CounterfactualVector system testing

using Test, Random
using FormulaCompiler
using DataFrames, GLM, Tables, CategoricalArrays, StatsModels
using BenchmarkTools

Random.seed!(06515)

@testset "CounterfactualVector System Tests" begin

    @testset "NumericCounterfactualVector Basic Functionality" begin
        # Test numeric counterfactual
        base_vec = [1.0, 2.0, 3.0, 4.0, 5.0]
        cf_vec = FormulaCompiler.NumericCounterfactualVector{Float64}(base_vec, 3, 99.0)

        @test cf_vec[1] == 1.0     # Original value
        @test cf_vec[2] == 2.0     # Original value
        @test cf_vec[3] == 99.0    # Counterfactual value
        @test cf_vec[4] == 4.0     # Original value
        @test cf_vec[5] == 5.0     # Original value
        @test length(cf_vec) == 5

        # Test memory efficiency (much smaller than copying entire array)
        regular_size = sizeof(copy(base_vec))
        cf_size = sizeof(cf_vec)
        @test cf_size < regular_size  # Should be smaller

        # Test iteration
        expected = [1.0, 2.0, 99.0, 4.0, 5.0]
        for (i, val) in enumerate(cf_vec)
            @test val == expected[i]
        end

        # Test fast access
        access_time = @elapsed begin
            for i in 1:1000
                val = cf_vec[mod1(i, 5)]
            end
        end
        @test access_time < 0.01  # Should be very fast
    end

    @testset "CounterfactualVector Construction" begin
        test_data = (
            x = [1.0, 2.0, 3.0, 4.0, 5.0],
            y = [10.0, 20.0, 30.0, 40.0, 50.0],
            z = [100, 200, 300, 400, 500]
        )

        # Test basic CounterfactualVector creation from dispatch
        cf_x = FormulaCompiler.counterfactualvector(test_data.x, 2)
        @test cf_x isa FormulaCompiler.NumericCounterfactualVector{Float64}
        @test cf_x[1] == 1.0
        @test cf_x[2] == 0.0  # Default replacement
        @test cf_x[3] == 3.0

        # Test with specific replacement
        cf_x_specific = FormulaCompiler.NumericCounterfactualVector{Float64}(test_data.x, 3, 99.0)
        @test cf_x_specific[3] == 99.0

        # Test integer counterfactual
        cf_z = FormulaCompiler.counterfactualvector(test_data.z, 2)
        @test cf_z isa FormulaCompiler.NumericCounterfactualVector{Int64}
        @test cf_z[1] == 100
        @test cf_z[2] == 0    # Default replacement
        @test cf_z[3] == 300
    end

    @testset "CategoricalCounterfactualVector Creation" begin
        # Create test categorical data
        original_data = categorical(["A", "B", "A", "C", "B"], levels=["A", "B", "C"])

        # Test categorical counterfactual creation - now returns reference indices (UInt32)
        cf_cat = FormulaCompiler.counterfactualvector(original_data, 2)
        @test cf_cat isa FormulaCompiler.CategoricalCounterfactualVector
        @test length(cf_cat) == 5
        @test cf_cat[1] == UInt32(1)  # Original ref index for "A"
        @test cf_cat[2] == UInt32(1)  # Default replacement (first ref index)
        @test cf_cat[3] == UInt32(1)  # Original ref index for "A"

        # Test with specific categorical replacement
        cf_cat_b = FormulaCompiler.CategoricalCounterfactualVector{String, UInt32}(
            original_data, 2, original_data[1]  # Using first element as template
        )
        @test cf_cat_b[1] == original_data.refs[1]  # Compare ref indices
        @test cf_cat_b[3] == original_data.refs[3]  # Compare ref indices

        # Test type dispatch
        @test FormulaCompiler.counterfactualvector(original_data, 1) isa FormulaCompiler.CategoricalCounterfactualVector
    end

    @testset "BoolCounterfactualVector Creation" begin
        test_data = (
            x = [1.0, 2.0, 3.0, 4.0, 5.0],
            group = categorical(["A", "B", "A", "C", "B"], levels=["A", "B", "C"]),
            treatment = [true, false, true, true, false]
        )

        # Test boolean counterfactual creation
        cf_bool = FormulaCompiler.counterfactualvector(test_data.treatment, 2)
        @test cf_bool isa FormulaCompiler.BoolCounterfactualVector
        @test cf_bool[1] == true   # Original
        @test cf_bool[2] == false  # Default replacement
        @test cf_bool[3] == true   # Original

        # Test with specific boolean replacement
        cf_bool_true = FormulaCompiler.BoolCounterfactualVector(test_data.treatment, 2, true)
        @test cf_bool_true[1] == true  # Original
        @test cf_bool_true[2] == true  # Replacement
        @test cf_bool_true[3] == true  # Original

        # Test type dispatch for boolean vectors
        @test FormulaCompiler.counterfactualvector(test_data.treatment, 1) isa FormulaCompiler.BoolCounterfactualVector
    end

    @testset "Formula Integration with CounterfactualVectors" begin
        # Create realistic test data
        n = 100
        df = DataFrame(
            x = randn(n),
            y = randn(n),
            group = categorical(rand(["A", "B", "C"], n), levels=["A", "B", "C"]),
            treatment = rand([true, false], n)
        )

        data = Tables.columntable(df)

        # Test with simple continuous formula
        model_continuous = lm(@formula(y ~ x), df)
        compiled = compile_formula(model_continuous, data)

        # Create counterfactual data with modified x values
        cf_data = merge(data, (
            x = FormulaCompiler.NumericCounterfactualVector{Float64}(data.x, 1, 5.0),
        ))

        output_orig = Vector{Float64}(undef, length(compiled))
        output_cf = Vector{Float64}(undef, length(compiled))

        # Test that it executes with both original and counterfactual data
        @test_nowarn compiled(output_orig, data, 1)
        @test_nowarn compiled(output_cf, cf_data, 1)
        @test length(output_orig) == 2  # Intercept + x coefficient
        @test length(output_cf) == 2

        # Results should differ due to different x value
        @test output_orig != output_cf

        # Test with categorical formula
        model_cat = lm(@formula(y ~ x + group), df)
        compiled_cat = compile_formula(model_cat, data)

        # Create counterfactual data with modified group
        # Find a different group value for the counterfactual
        different_group = nothing
        for val in data.group
            if val != data.group[1]
                different_group = val
                break
            end
        end

        # If all groups are the same, create a different categorical value
        if different_group === nothing
            # Create a categorical value from the existing pool but different level
            pool_levels = levels(data.group)
            different_level = pool_levels[findfirst(x -> x != string(data.group[1]), pool_levels)]
            different_group = categorical([different_level], levels=pool_levels)[1]
        end

        cf_cat_data = merge(data, (
            group = FormulaCompiler.CategoricalCounterfactualVector{String, UInt32}(
                data.group, 1, different_group
            ),
        ))

        output_cat_orig = Vector{Float64}(undef, length(compiled_cat))
        output_cat_cf = Vector{Float64}(undef, length(compiled_cat))

        @test_nowarn compiled_cat(output_cat_orig, data, 1)
        @test_nowarn compiled_cat(output_cat_cf, cf_cat_data, 1)
        @test length(output_cat_orig) == 4  # Intercept + x + 2 group dummies

        # The outputs should differ in the categorical parts
        @test output_cat_orig != output_cat_cf
    end

    @testset "CounterfactualVector Loop Patterns" begin
        test_data = (
            x = [1.0, 2.0, 3.0, 4.0, 5.0],
            y = [10.0, 20.0, 30.0, 40.0, 50.0],
            group = categorical(["A", "B", "A", "C", "B"], levels=["A", "B", "C"])
        )

        # Test loop pattern for population analysis (replacement for scenario grids)
        x_values = [1.0, 2.0, 3.0]
        group_values = ["A", "B"]

        # Create a simple compiled formula for testing
        df = DataFrame(y = randn(5), x = test_data.x, group = test_data.group)
        model = lm(@formula(y ~ x + group), df)
        compiled = compile_formula(model, test_data)

        results = []

        # Population analysis using loops (replacement for scenario grid)
        for x_val in x_values
            for group_val in group_values
                # Create counterfactual data for this combination
                cf_data = merge(test_data, (
                    x = FormulaCompiler.NumericCounterfactualVector{Float64}(test_data.x, 1, x_val),
                    group = FormulaCompiler.CategoricalCounterfactualVector{String, UInt32}(
                        test_data.group, 1, categorical([group_val], levels=["A", "B", "C"])[1]
                    ),
                ))

                # Evaluate for this combination
                output = Vector{Float64}(undef, length(compiled))
                compiled(output, cf_data, 1)
                push!(results, (x = x_val, group = group_val, result = copy(output)))
            end
        end

        @test length(results) == 6  # 3 × 2 combinations

        # Test all combinations were evaluated
        x_vals_found = [r.x for r in results]
        group_vals_found = [r.group for r in results]
        @test sort(unique(x_vals_found)) == sort(x_values)
        @test sort(unique(group_vals_found)) == sort(group_values)

        # Results should vary across combinations
        result_vectors = [r.result for r in results]
        @test length(unique(result_vectors)) > 1  # Should have different results
    end

    @testset "CounterfactualVector Performance" begin
        # Create large test data
        n = 10000
        large_data = (
            x = randn(n),
            y = randn(n),
            group = categorical(rand(["A", "B", "C"], n)),
            treatment = rand([true, false], n)
        )

        # Test that counterfactual creation is fast
        creation_time = @elapsed begin
            cf_x = FormulaCompiler.NumericCounterfactualVector{Float64}(large_data.x, 1000, 5.0)
            cf_group = FormulaCompiler.CategoricalCounterfactualVector{String, UInt32}(
                large_data.group, 500, large_data.group[1]
            )
        end
        @test creation_time < 0.01  # Should be very fast

        # Test memory efficiency - CounterfactualVectors should be much smaller than copies
        cf_vectors = [
            FormulaCompiler.NumericCounterfactualVector{Float64}(large_data.x, i, Float64(i))
            for i in 1:10
        ]

        # Calculate memory usage
        original_size = sizeof(large_data.x)
        cf_overhead = sum(sizeof(cf) for cf in cf_vectors)
        naive_copy_size = original_size * length(cf_vectors)

        savings = (naive_copy_size - cf_overhead) / naive_copy_size
        @test savings > 0.90  # Should save >90% of memory

        # Test data access performance
        cf_x = cf_vectors[1]
        access_time = @elapsed begin
            for i in 1:1000
                val = cf_x[rand(1:n)]
            end
        end
        @test access_time < 0.01  # Should be very fast
    end

    @testset "CounterfactualVector Edge Cases" begin
        # Test with single-level categorical
        single_level = categorical(["A", "A", "A"], levels=["A"])
        cf_single = FormulaCompiler.CategoricalCounterfactualVector{String, UInt32}(
            single_level, 2, single_level[1]
        )
        # CategoricalCounterfactualVector returns reference indices (UInt32) for zero allocations
        # These convert to Int for use in extract_level_code
        @test cf_single[1] == UInt32(1)  # Returns ref index, not CategoricalValue
        @test cf_single[2] == UInt32(1)  # Override position also returns ref index

        # Test with empty data (edge case)
        empty_x = Float64[]
        empty_group = categorical(String[], levels=["A", "B"])

        # These should work but have length 0
        @test length(empty_x) == 0
        @test length(empty_group) == 0

        # Test boundary conditions
        test_data = [1.0, 2.0, 3.0]

        # Test first position
        cf_first = FormulaCompiler.NumericCounterfactualVector{Float64}(test_data, 1, 99.0)
        @test cf_first[1] == 99.0
        @test cf_first[2] == 2.0

        # Test last position
        cf_last = FormulaCompiler.NumericCounterfactualVector{Float64}(test_data, 3, 99.0)
        @test cf_last[1] == 1.0
        @test cf_last[3] == 99.0

        # Test that bounds checking works
        @test_throws BoundsError cf_first[0]
        @test_throws BoundsError cf_first[4]
    end

    @testset "Integer CounterfactualVector Support" begin
        # Test CounterfactualVector system with integer continuous variables
        n = 100
        df = DataFrame(
            y = randn(n),
            int_age = rand(18:80, n),        # Integer continuous
            int_score = rand(0:1000, n),     # Integer test score
            float_x = randn(n),              # Float for comparison
            group = categorical(rand(["A", "B", "C"], n))
        )
        data = Tables.columntable(df)
        model = lm(@formula(y ~ int_age + int_score + float_x + group), df)
        compiled = compile_formula(model, data)

        @testset "Basic integer counterfactuals" begin
            # Test integer-to-integer counterfactual
            cf_age = FormulaCompiler.NumericCounterfactualVector{Int64}(data.int_age, 1, 25)
            cf_score = FormulaCompiler.NumericCounterfactualVector{Int64}(data.int_score, 1, 500)

            cf_data = merge(data, (
                int_age = cf_age,
                int_score = cf_score
            ))

            output_orig = Vector{Float64}(undef, length(compiled))
            output_cf = Vector{Float64}(undef, length(compiled))

            compiled(output_orig, data, 1)
            compiled(output_cf, cf_data, 1)

            @test cf_data.int_age[1] == 25
            @test cf_data.int_score[1] == 500
            @test output_orig != output_cf  # Should be different
        end

        @testset "Type-flexible counterfactuals" begin
            # Test integer column with float counterfactual (non-integer)
            cf_age_float = FormulaCompiler.NumericCounterfactualVector{Float64}(
                convert(Vector{Float64}, data.int_age), 1, 25.5
            )

            @test cf_age_float[1] == 25.5
            @test typeof(cf_age_float[1]) == Float64

            # Test integer column with integer float counterfactual
            cf_age_int_float = FormulaCompiler.NumericCounterfactualVector{Int64}(data.int_age, 1, 30)

            @test cf_age_int_float[1] == 30
            @test typeof(cf_age_int_float[1]) == Int64

            # Test float column with counterfactual
            cf_x_int = FormulaCompiler.NumericCounterfactualVector{Float64}(data.float_x, 1, 42.0)

            @test cf_x_int[1] == 42.0
            @test typeof(cf_x_int[1]) == Float64
        end

        @testset "Integer loop patterns" begin
            # Test loop pattern for integer variables (replacement for scenario grids)
            age_values = [25, 45, 65]
            score_values = [200, 500, 800]
            group_values = ["A", "B"]

            results = []
            test_row = 5

            for age_val in age_values
                for score_val in score_values
                    for group_val in group_values
                        # Create counterfactual data for this combination
                        cf_data = merge(data, (
                            int_age = FormulaCompiler.NumericCounterfactualVector{Int64}(data.int_age, test_row, age_val),
                            int_score = FormulaCompiler.NumericCounterfactualVector{Int64}(data.int_score, test_row, score_val),
                            group = FormulaCompiler.CategoricalCounterfactualVector{String, UInt32}(
                                data.group, test_row, categorical([group_val], levels=["A", "B", "C"])[1]
                            ),
                        ))

                        # Evaluate for this combination
                        output = Vector{Float64}(undef, length(compiled))
                        compiled(output, cf_data, test_row)
                        push!(results, (age = age_val, score = score_val, group = group_val, result = copy(output)))
                    end
                end
            end

            @test length(results) == 18  # 3 × 3 × 2 = 18 combinations

            # Test all combinations were evaluated
            age_vals_found = [r.age for r in results]
            score_vals_found = [r.score for r in results]
            group_vals_found = [r.group for r in results]
            @test sort(unique(age_vals_found)) == sort(age_values)
            @test sort(unique(score_vals_found)) == sort(score_values)
            @test sort(unique(group_vals_found)) == sort(group_values)

            # Results should vary across combinations
            result_vectors = [r.result for r in results]
            @test length(unique(result_vectors)) > 1  # Should have different results
        end

        @testset "Integer counterfactuals with derivatives" begin
            # Test that derivative evaluator works with data that has integer counterfactuals
            vars = [:int_age, :float_x]
            de_fd = derivativeevaluator_fd(compiled, data, vars)

            # Create counterfactual data with integer modifications
            cf_data = merge(data, (
                int_age = FormulaCompiler.NumericCounterfactualVector{Int64}(data.int_age, 1, 35),
                int_score = FormulaCompiler.NumericCounterfactualVector{Int64}(data.int_score, 1, 750)
            ))

            # This should not error - testing basic compatibility
            J = Matrix{Float64}(undef, length(compiled), length(vars))
            derivative_modelrow!(J, de_fd, 1)

            @test size(J) == (length(compiled), length(vars))
            @test all(isfinite.(J))
        end
    end
end