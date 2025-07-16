# test/test_regression.jl
# Regression tests for known working cases and edge cases

@testset "Regression Tests" begin
    
    Random.seed!(42)
    
    # Create comprehensive test dataset
    df = DataFrame(
        x = randn(100),
        y = randn(100),
        z = abs.(randn(100)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 100)),
        flag = rand([true, false], 100),
        cat2 = categorical(rand(["X", "Y"], 100)),
        cat3 = categorical(rand(["P", "Q", "R"], 100))
    )
    
    @testset "Known Working Cases" begin
        # Test cases from the original test suite
        
        test_cases = [
            (@formula(y ~ cat2 * cat3), "cat 2 x cat 3"),
            (@formula(y ~ cat2 * flag), "cat 2 x bool"),
            (@formula(y ~ cat2 * (x^2)), "cat 2 x continuous"),
            (@formula(y ~ flag * (x^2)), "binary x continuous"),
            (@formula(y ~ group * (x^2)), "cat >2 x continuous"),
            (@formula(y ~ group * flag), "cat >2 x bool"),
            (@formula(y ~ group * cat2), "cat >2 x cat 2"),
            (@formula(y ~ group * cat3), "cat >2 x cat >2"),
            (@formula(y ~ x * z * group), "three-way continuous x categorical"),
            (@formula(y ~ (x > 0) * group), "boolean function x categorical"),
            (@formula(y ~ log(z) * group * cat2), "function x cat >2 x cat 2"),
        ]
        
        data = Tables.columntable(df)
        
        for (formula, description) in test_cases
            @testset "$description" begin
                model = lm(formula, df)
                compiled = compile_formula(model)
                
                # Test correctness on first few rows
                row_vec = Vector{Float64}(undef, length(compiled))
                for i in 1:min(5, nrow(df))
                    compiled(row_vec, data, i)
                    expected = modelmatrix(model)[i, :]
                    @test isapprox(row_vec, expected, rtol=1e-12) "Failed on row $i for $description"
                end
                
                # Test zero allocations
                allocs = @allocated compiled(row_vec, data, 1)
                @test allocs == 0 "Non-zero allocations for $description"
                
                # Test performance
                eval_time = @elapsed compiled(row_vec, data, 1)
                @test eval_time < 0.001 "Slow evaluation for $description"
            end
        end
    end
    
    @testset "Edge Cases from Original Code" begin
        # Test specific edge cases that have been problematic
        
        # Test empty/minimal models
        @testset "Minimal Models" begin
            # Intercept only
            model = lm(@formula(y ~ 1), df)
            compiled = compile_formula(model)
            @test length(compiled) == 1
            
            row_vec = Vector{Float64}(undef, 1)
            compiled(row_vec, Tables.columntable(df), 1)
            @test row_vec[1] == 1.0
            
            # No intercept, single variable
            model = lm(@formula(y ~ 0 + x), df)
            compiled = compile_formula(model)
            @test length(compiled) == 1
            
            row_vec = Vector{Float64}(undef, 1)
            compiled(row_vec, Tables.columntable(df), 1)
            @test row_vec[1] == df.x[1]
        end
        
        # Test specific categorical configurations
        @testset "Categorical Edge Cases" begin
            # Single-level categorical (should work but be trivial)
            df_single = DataFrame(
                x = [1.0, 2.0, 3.0],
                y = [1.0, 2.0, 3.0],
                single_cat = categorical(["A", "A", "A"])
            )
            
            model = lm(@formula(y ~ x + single_cat), df_single)
            compiled = compile_formula(model)
            
            row_vec = Vector{Float64}(undef, length(compiled))
            compiled(row_vec, Tables.columntable(df_single), 1)
            expected = modelmatrix(model)[1, :]
            @test isapprox(row_vec, expected, rtol=1e-12)
            
            # Binary categorical with interaction
            model = lm(@formula(y ~ x * cat2), df)
            compiled = compile_formula(model)
            
            row_vec = Vector{Float64}(undef, length(compiled))
            compiled(row_vec, Tables.columntable(df), 1)
            expected = modelmatrix(model)[1, :]
            @test isapprox(row_vec, expected, rtol=1e-12)
        end
        
        # Test function edge cases
        @testset "Function Edge Cases" begin
            # Test with zero/negative values where appropriate
            df_pos = copy(df)
            df_pos.z = abs.(df_pos.z) .+ 1e-10  # Ensure positive for log
            
            model = lm(@formula(y ~ log(z) + sqrt(z)), df_pos)
            compiled = compile_formula(model)
            
            row_vec = Vector{Float64}(undef, length(compiled))
            compiled(row_vec, Tables.columntable(df_pos), 1)
            expected = modelmatrix(model)[1, :]
            @test isapprox(row_vec, expected, rtol=1e-12)
            
            # Test with extreme values
            df_extreme = DataFrame(
                x = [1e-10, 1e10],
                y = [1.0, 2.0],
                z = [1e-5, 1e5]
            )
            
            model = lm(@formula(y ~ x + log(z)), df_extreme)
            compiled = compile_formula(model)
            
            row_vec = Vector{Float64}(undef, length(compiled))
            for i in 1:2
                compiled(row_vec, Tables.columntable(df_extreme), i)
                expected = modelmatrix(model)[i, :]
                @test isapprox(row_vec, expected, rtol=1e-12)
            end
        end
    end
    
    @testset "Complex Interaction Patterns" begin
        # Test complex interaction patterns that have been problematic
        
        # Three-way interaction with mixed types
        model = lm(@formula(y ~ x * group * flag), df)
        compiled = compile_formula(model)
        
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        for i in 1:min(10, nrow(df))
            compiled(row_vec, data, i)
            expected = modelmatrix(model)[i, :]
            @test isapprox(row_vec, expected, rtol=1e-12)
        end
        
        # Four-way interaction
        model = lm(@formula(y ~ x * z * group * flag), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Interaction with functions
        model = lm(@formula(y ~ log(z) * sqrt(abs(x) + 1) * group), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "Formula Parsing Edge Cases" begin
        # Test edge cases in formula parsing
        
        # Parentheses and precedence
        model = lm(@formula(y ~ (x + z) * group), df)
        compiled = compile_formula(model)
        
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Multiple function applications
        model = lm(@formula(y ~ log(exp(x)) + sqrt(z^2)), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Nested interactions
        model = lm(@formula(y ~ x * (group + cat2)), df)
        compiled = compile_formula(model)
        
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "Data Type Edge Cases" begin
        # Test with different data types
        
        # Test with Float32
        df_f32 = DataFrame(
            x = Float32.(randn(20)),
            y = Float32.(randn(20)),
            group = categorical(rand(["A", "B"], 20))
        )
        
        model = lm(@formula(y ~ x * group), df_f32)
        compiled = compile_formula(model)
        
        data = Tables.columntable(df_f32)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-6)  # Slightly relaxed tolerance
        
        # Test with integers
        df_int = DataFrame(
            x = rand(1:10, 20),
            y = randn(20),
            group = categorical(rand(["A", "B"], 20))
        )
        
        model = lm(@formula(y ~ x * group), df_int)
        compiled = compile_formula(model)
        
        data = Tables.columntable(df_int)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test with missing values (after complete cases)
        df_missing = DataFrame(
            x = [1.0, 2.0, missing, 4.0],
            y = [1.0, 2.0, 3.0, 4.0],
            group = categorical(["A", "B", "A", "B"])
        )
        
        df_complete = df_missing[completecases(df_missing), :]
        model = lm(@formula(y ~ x * group), df_complete)
        compiled = compile_formula(model)
        
        data = Tables.columntable(df_complete)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "Performance Regression" begin
        # Test that performance hasn't regressed
        
        # Standard benchmark case
        model = lm(@formula(y ~ x * group + log(z)), df)
        compiled = compile_formula(model)
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        # Should compile quickly
        compile_time = @elapsed compile_formula(model)
        @test compile_time < 0.5
        
        # Should evaluate quickly
        eval_time = @elapsed compiled(row_vec, data, 1)
        @test eval_time < 0.001
        
        # Should have zero allocations
        allocs = @allocated compiled(row_vec, data, 1)
        @test allocs == 0
        
        # Should be consistent across runs
        times = [(@elapsed compiled(row_vec, data, i)) for i in 1:10]
        @test std(times) < mean(times)  # Low variance
    end
    
    @testset "Correctness Regression" begin
        # Test specific cases that have been fixed
        
        # Test case from original usage.jl
        df_usage = DataFrame(
            x = randn(1000),
            y = randn(1000),
            z = abs.(randn(1000)) .+ 0.1,
            group = categorical(rand(["A", "B", "C"], 1000)),
            bool = rand([false, true], 1000),
            group2 = categorical(rand(["C", "D", "X"], 1000)),
            group3 = categorical(rand(["E", "F", "G"], 1000)),
            cat2a = categorical(rand(["X", "Y"], 1000)),
            cat2b = categorical(rand(["P", "Q"], 1000))
        )
        
        # Test specific problematic cases
        problematic_cases = [
            @formula(y ~ cat2a * cat2b),
            @formula(y ~ cat2a * bool),
            @formula(y ~ group2 * group3),
            @formula(y ~ x * z * group),
            @formula(y ~ log(z) * group2 * cat2a)
        ]
        
        data = Tables.columntable(df_usage)
        
        for formula in problematic_cases
            model = lm(formula, df_usage)
            compiled = compile_formula(model)
            
            row_vec = Vector{Float64}(undef, length(compiled))
            compiled(row_vec, data, 1)
            expected = modelmatrix(model)[1, :]
            @test isapprox(row_vec, expected, rtol=1e-12)
        end
    end
    
    @testset "Cache Consistency" begin
        # Test that caching doesn't break anything
        
        model1 = lm(@formula(y ~ x + z), df)
        model2 = lm(@formula(y ~ x * group), df)
        
        # Compile both
        compiled1 = compile_formula(model1)
        compiled2 = compile_formula(model2)
        
        # Compile again (should use cache)
        compiled1_cached = compile_formula(model1)
        compiled2_cached = compile_formula(model2)
        
        # Should give same results
        data = Tables.columntable(df)
        row_vec1 = Vector{Float64}(undef, length(compiled1))
        row_vec2 = Vector{Float64}(undef, length(compiled2))
        row_vec1_cached = Vector{Float64}(undef, length(compiled1_cached))
        row_vec2_cached = Vector{Float64}(undef, length(compiled2_cached))
        
        compiled1(row_vec1, data, 1)
        compiled2(row_vec2, data, 1)
        compiled1_cached(row_vec1_cached, data, 1)
        compiled2_cached(row_vec2_cached, data, 1)
        
        @test row_vec1 == row_vec1_cached
        @test row_vec2 == row_vec2_cached
        
        # Clear cache and test again
        clear_model_cache!()
        
        compiled1_fresh = compile_formula(model1)
        row_vec1_fresh = Vector{Float64}(undef, length(compiled1_fresh))
        compiled1_fresh(row_vec1_fresh, data, 1)
        
        @test row_vec1 == row_vec1_fresh
    end
    
end
