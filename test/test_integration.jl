# test/test_integration.jl
# Integration tests with various Julia packages and model types

@testset "Integration Tests" begin
    
    Random.seed!(42)
    n = 100
    df = DataFrame(
        x = randn(n),
        y = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], n)),
        flag = rand([true, false], n)
    )
    
    @testset "GLM Integration" begin
        # Test LinearModel
        linear_model = lm(@formula(y ~ x + z), df)
        compiled = compile_formula(linear_model)
        
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        # Test multiple rows
        for i in 1:min(10, n)
            compiled(row_vec, data, i)
            expected = modelmatrix(linear_model)[i, :]
            @test isapprox(row_vec, expected, rtol=1e-12)
        end
        
        # Test GeneralizedLinearModel
        df_binary = copy(df)
        df_binary.y_binary = df_binary.y .> 0
        
        logit_model = glm(@formula(y_binary ~ x + z), df_binary, Binomial(), LogitLink())
        compiled_logit = compile_formula(logit_model)
        
        data_binary = Tables.columntable(df_binary)
        row_vec_logit = Vector{Float64}(undef, length(compiled_logit))
        
        compiled_logit(row_vec_logit, data_binary, 1)
        expected_logit = modelmatrix(logit_model)[1, :]
        @test isapprox(row_vec_logit, expected_logit, rtol=1e-12)
    end
    
    @testset "StandardizedPredictors Integration" begin
        # Test ZScore standardization
        contrasts = Dict(:x => ZScore(), :z => ZScore())
        
        model = lm(@formula(y ~ x + z), df, contrasts=contrasts)
        compiled = compile_formula(model)
        
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test that standardization is applied correctly
        @test abs(row_vec[2]) < 10  # x should be standardized
    end
    
    @testset "CategoricalArrays Integration" begin
        # Test different categorical configurations
        df_cat = DataFrame(
            x = randn(50),
            y = randn(50),
            cat_ordered = categorical(rand(["Low", "Medium", "High"], 50), ordered=true),
            cat_unordered = categorical(rand(["Red", "Blue", "Green"], 50)),
            cat_binary = categorical(rand(["Yes", "No"], 50))
        )
        
        # Test ordered categorical
        model_ordered = lm(@formula(y ~ x + cat_ordered), df_cat)
        compiled_ordered = compile_formula(model_ordered)
        
        data_cat = Tables.columntable(df_cat)
        row_vec = Vector{Float64}(undef, length(compiled_ordered))
        
        compiled_ordered(row_vec, data_cat, 1)
        expected = modelmatrix(model_ordered)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test unordered categorical
        model_unordered = lm(@formula(y ~ x + cat_unordered), df_cat)
        compiled_unordered = compile_formula(model_unordered)
        
        row_vec2 = Vector{Float64}(undef, length(compiled_unordered))
        compiled_unordered(row_vec2, data_cat, 1)
        expected2 = modelmatrix(model_unordered)[1, :]
        @test isapprox(row_vec2, expected2, rtol=1e-12)
        
        # Test binary categorical
        model_binary = lm(@formula(y ~ x + cat_binary), df_cat)
        compiled_binary = compile_formula(model_binary)
        
        row_vec3 = Vector{Float64}(undef, length(compiled_binary))
        compiled_binary(row_vec3, data_cat, 1)
        expected3 = modelmatrix(model_binary)[1, :]
        @test isapprox(row_vec3, expected3, rtol=1e-12)
    end
    
    @testset "Tables.jl Integration" begin
        # Test with different table types
        
        # Test with DataFrame
        model = lm(@formula(y ~ x + group), df)
        compiled = compile_formula(model)
        
        # Test with columntable
        data_ct = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data_ct, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test with rowtable
        data_rt = Tables.rowtable(df)
        data_rt_converted = Tables.columntable(data_rt)
        compiled(row_vec, data_rt_converted, 1)
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test with NamedTuple
        row_as_nt = (x=df.x[1], y=df.y[1], z=df.z[1], group=df.group[1], flag=df.flag[1])
        single_row_data = (
            x = [row_as_nt.x],
            y = [row_as_nt.y], 
            z = [row_as_nt.z],
            group = [row_as_nt.group],
            flag = [row_as_nt.flag]
        )
        compiled(row_vec, single_row_data, 1)
        @test isapprox(row_vec, expected, rtol=1e-12)
    end
    
    @testset "StatsModels Integration" begin
        # Test various formula constructs
        
        # Test with explicit intercept
        model1 = lm(@formula(y ~ 1 + x), df)
        compiled1 = compile_formula(model1)
        
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled1))
        compiled1(row_vec, data, 1)
        expected1 = modelmatrix(model1)[1, :]
        @test isapprox(row_vec, expected1, rtol=1e-12)
        
        # Test without intercept
        model2 = lm(@formula(y ~ 0 + x + group), df)
        compiled2 = compile_formula(model2)
        
        row_vec2 = Vector{Float64}(undef, length(compiled2))
        compiled2(row_vec2, data, 1)
        expected2 = modelmatrix(model2)[1, :]
        @test isapprox(row_vec2, expected2, rtol=1e-12)
        
        # Test with & interaction
        model3 = lm(@formula(y ~ x & group), df)
        compiled3 = compile_formula(model3)
        
        row_vec3 = Vector{Float64}(undef, length(compiled3))
        compiled3(row_vec3, data, 1)
        expected3 = modelmatrix(model3)[1, :]
        @test isapprox(row_vec3, expected3, rtol=1e-12)
        
        # Test with * interaction (main effects + interaction)
        model4 = lm(@formula(y ~ x * group), df)
        compiled4 = compile_formula(model4)
        
        row_vec4 = Vector{Float64}(undef, length(compiled4))
        compiled4(row_vec4, data, 1)
        expected4 = modelmatrix(model4)[1, :]
        @test isapprox(row_vec4, expected4, rtol=1e-12)
    end
    
    @testset "Function Term Integration" begin
        # Test various mathematical functions
        
        # Test logarithm
        model_log = lm(@formula(y ~ log(z)), df)
        compiled_log = compile_formula(model_log)
        
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled_log))
        compiled_log(row_vec, data, 1)
        expected_log = modelmatrix(model_log)[1, :]
        @test isapprox(row_vec, expected_log, rtol=1e-12)
        
        # Test square root
        model_sqrt = lm(@formula(y ~ sqrt(z)), df)
        compiled_sqrt = compile_formula(model_sqrt)
        
        row_vec_sqrt = Vector{Float64}(undef, length(compiled_sqrt))
        compiled_sqrt(row_vec_sqrt, data, 1)
        expected_sqrt = modelmatrix(model_sqrt)[1, :]
        @test isapprox(row_vec_sqrt, expected_sqrt, rtol=1e-12)
        
        # Test polynomial
        model_poly = lm(@formula(y ~ x + x^2 + x^3), df)
        compiled_poly = compile_formula(model_poly)
        
        row_vec_poly = Vector{Float64}(undef, length(compiled_poly))
        compiled_poly(row_vec_poly, data, 1)
        expected_poly = modelmatrix(model_poly)[1, :]
        @test isapprox(row_vec_poly, expected_poly, rtol=1e-12)
        
        # Test trigonometric functions
        model_trig = lm(@formula(y ~ sin(x) + cos(x)), df)
        compiled_trig = compile_formula(model_trig)
        
        row_vec_trig = Vector{Float64}(undef, length(compiled_trig))
        compiled_trig(row_vec_trig, data, 1)
        expected_trig = modelmatrix(model_trig)[1, :]
        @test isapprox(row_vec_trig, expected_trig, rtol=1e-12)
        
        # Test boolean functions
        model_bool = lm(@formula(y ~ (x > 0) + (z < 1)), df)
        compiled_bool = compile_formula(model_bool)
        
        row_vec_bool = Vector{Float64}(undef, length(compiled_bool))
        compiled_bool(row_vec_bool, data, 1)
        expected_bool = modelmatrix(model_bool)[1, :]
        @test isapprox(row_vec_bool, expected_bool, rtol=1e-12)
    end
    
    @testset "Complex Formula Integration" begin
        # Test highly complex formulas
        
        # Test nested functions
        model_nested = lm(@formula(y ~ log(x^2 + 1) + sqrt(abs(z - 1))), df)
        compiled_nested = compile_formula(model_nested)
        
        data = Tables.columntable(df)
        row_vec = Vector{Float64}(undef, length(compiled_nested))
        compiled_nested(row_vec, data, 1)
        expected_nested = modelmatrix(model_nested)[1, :]
        @test isapprox(row_vec, expected_nested, rtol=1e-12)
        
        # Test functions in interactions
        model_func_int = lm(@formula(y ~ log(z) * group + x^2 * flag), df)
        compiled_func_int = compile_formula(model_func_int)
        
        row_vec_func_int = Vector{Float64}(undef, length(compiled_func_int))
        compiled_func_int(row_vec_func_int, data, 1)
        expected_func_int = modelmatrix(model_func_int)[1, :]
        @test isapprox(row_vec_func_int, expected_func_int, rtol=1e-12)
        
        # Test three-way interactions with functions
        model_3way = lm(@formula(y ~ x * log(z) * group), df)
        compiled_3way = compile_formula(model_3way)
        
        row_vec_3way = Vector{Float64}(undef, length(compiled_3way))
        compiled_3way(row_vec_3way, data, 1)
        expected_3way = modelmatrix(model_3way)[1, :]
        @test isapprox(row_vec_3way, expected_3way, rtol=1e-12)
    end
    
    @testset "Error Handling Integration" begin
        # Test graceful handling of edge cases
        
        # Test with missing values (should work with complete cases)
        df_missing = copy(df)
        df_complete = df_missing[completecases(df_missing), :]
        
        model = lm(@formula(y ~ x + z), df_complete)
        compiled = compile_formula(model)
        
        data = Tables.columntable(df_complete)
        row_vec = Vector{Float64}(undef, length(compiled))
        compiled(row_vec, data, 1)
        expected = modelmatrix(model)[1, :]
        @test isapprox(row_vec, expected, rtol=1e-12)
        
        # Test with extreme values
        df_extreme = DataFrame(
            x = [1e-10, 1e10, -1e10],
            y = [1.0, 2.0, 3.0],
            z = [1e-5, 1e5, 1e-5]
        )
        
        model_extreme = lm(@formula(y ~ x + log(z)), df_extreme)
        compiled_extreme = compile_formula(model_extreme)
        
        data_extreme = Tables.columntable(df_extreme)
        row_vec_extreme = Vector{Float64}(undef, length(compiled_extreme))
        compiled_extreme(row_vec_extreme, data_extreme, 1)
        expected_extreme = modelmatrix(model_extreme)[1, :]
        @test isapprox(row_vec_extreme, expected_extreme, rtol=1e-12)
    end
    
    @testset "Performance Consistency" begin
        # Test that compiled formulas maintain performance across model types
        
        models = [
            lm(@formula(y ~ x), df),
            lm(@formula(y ~ x + z), df),
            lm(@formula(y ~ x * group), df),
            lm(@formula(y ~ x + z + group + flag), df),
            lm(@formula(y ~ log(z) * group), df)
        ]
        
        data = Tables.columntable(df)
        
        for model in models
            compiled = compile_formula(model)
            row_vec = Vector{Float64}(undef, length(compiled))
            
            # Test that evaluation is consistently fast
            time1 = @elapsed compiled(row_vec, data, 1)
            time2 = @elapsed compiled(row_vec, data, 2)
            time3 = @elapsed compiled(row_vec, data, 3)
            
            # Times should be very small and consistent
            @test time1 < 0.001  # Less than 1ms
            @test time2 < 0.001
            @test time3 < 0.001
            
            # Test zero allocations
            allocs = @allocated compiled(row_vec, data, 1)
            @test allocs == 0
        end
    end
    
end
